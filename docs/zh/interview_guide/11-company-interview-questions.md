# 11. 2025–2026 厂商真题精讲：把 slime 讲进面试

> **快照说明**：本文答案以 `main@3778dbf6`（v0.3.2，扫描日期 2026-08-29）源码为准。
>
> **来源说明**：题目收集自 2025–2026 公开渠道，按可信度分三档标注：
>
> - **[真题]**：牛客等平台的公开面经原题（字节、百度、阿里、腾讯、商汤等），文字保留原意；
> - **[官方口径]**：slime 作者朱子霖公开分享（青稞 Talk 第 68 期实录、微信公众号文章）中的设计解释，最接近"标准答案"；
> - **[高频方向]**：从多篇技术博客、论文和框架对比文中总结的必考方向，没有逐字原题，但在 RL infra/算法面试中反复出现。
>
> 面经属于二手信息，问法会因面试官而异；重要的不是背题，而是**每道题都能落到 slime 的具体机制上**——这正是这份指南前十章训练的能力。

## 0. slime 在面试中的三种出场方式

1. **简历上写了 slime**（你研究过它、给它提过 PR）：面试官会顺着简历追问框架细节、你改过什么、为什么这么改。
2. **RL infra 岗**：面试官不一定点名 slime，但"训推分离、权重同步、异步 RL、MoE RL"这些问题的最佳答案往往就是 slime/verl 的实现，主动用 slime 举例是强加分项。
3. **大模型算法岗**：GRPO/DAPO/GSPO/CISPO、训推不一致、长尾 rollout 是 2025–2026 算法面的绝对高频。答到"算法在框架里怎么实现"这一层，就能和只会背公式的候选人拉开差距。

每道题的参考回答遵循前十章的四层结构：**结论 → 机制 → 取舍 → 验证**。

## 1. 真题全景：谁在考什么

| 公司/场景 | 岗位 | 公开面经中的高频主题 |
|---|---|---|
| 字节跳动（含 Seed 方向） | 大模型算法 | Agentic RL 过程打分、rollout 长尾与 GPU 利用率、MoE 路由不一致、DAPO/GSPO 细节、clip 之后为什么取 min |
| 百度（文心） | RL 后训练 | GRPO 数据流、三个 π 的区别、on/off-policy、大 batch 的 off-policy 缓解、KL 公式与平滑、GAE 的 λ |
| 阿里（含国际/Qwen 方向） | 大模型算法 | GRPO vs SFT 选型、advantage 的意义、重要性采样失效条件、序列级奖励的 token 级分配、多轮工具调用挑战 |
| 腾讯（混元） | 大模型对齐 | RLHF 全流程、SFT 之后为什么还要 RL、GRPO 相对 PPO 的改进、ZeRO/显存估算 |
| 商汤 | 大模型算法 | PPO 涉及哪些模型、GRPO 四种长度-正误组合的倾向排序、训推不一致来源、MoE 的问题 |
| 智谱（slime 东家） | RL infra / 算法 | 框架设计取舍（作者公开分享即官方口径）、Megatron+SGLang 集成、大 MoE RL 实践 |
| MiniMax / 月之暗面 / 快手 / 美团等 | 算法/infra | CISPO（MiniMax 自家算法）、MoE 路由、K2 的 RL 变体、Agent 工程、部署框架选型 |

出题逻辑基本是三条线：**算法线**（GRPO 家族与变体）、**系统线**（训推分离/权重同步/异步）、**一致性线**（训推 mismatch/MoE 路由/确定性）。slime 恰好在三条线上都有一手实现，下面逐题精讲。

## 2. 算法层真题：GRPO 家族

### Q1 [真题·字节] GRPO 中 rollout 的长尾问题导致 GPU 使用率低，有什么工程上的解决方案？

**结论**：核心思路只有三类——**超发再截断、断点续推、彻底异步**。slime 三类都有内置实现。

**机制**（按侵入性从低到高）：

1. **Dynamic sampling 超发 + 先到先得**：一次发出比需求更多的 prompt 组，先完成的组先收集，凑够 `rollout_batch_size` 个有效组就停止，不等最慢的。slime 的默认 rollout 循环就是这样写的：`--over-sampling-batch-size` 控制超发粒度，`--dynamic-sampling-filter-path` 按 DAPO 语义丢弃全对/全错组（[arguments.py#L440](https://github.com/THUDM/slime/blob/3778dbf6d1a533ab478ecf5ddaa11449a47752b2/slime/utils/arguments.py#L440)）。
2. **Partial rollout（APRIL 思路）**：超发后被 abort 的未完成轨迹不丢弃，连同已生成的 token 一起放回 Data Buffer，下一轮从断点继续生成，复用已算的前缀。slime 的 `--partial-rollout` 就是该机制（[arguments.py#L468](https://github.com/THUDM/slime/blob/3778dbf6d1a533ab478ecf5ddaa11449a47752b2/slime/utils/arguments.py#L468)）；APRIL 论文报告其在 GRPO/DAPO/GSPO 上最多提升 rollout 吞吐约 49.5%，且已集成进 slime。
3. **跨 batch 常驻异步生成池（fully-async）**：训练 batch 不再绑定"同一批发出的请求"，后台 worker 维持恒定并发，完成的组进队列，训练侧按需取。slime 的 `examples/fully_async` + `slime/rollout/fully_async_rollout.py` 实现了这个模式，队列有背压控制，多余完成组留在队列供下一轮使用（[fully_async_rollout.py#L148](https://github.com/THUDM/slime/blob/3778dbf6d1a533ab478ecf5ddaa11449a47752b2/slime/rollout/fully_async_rollout.py#L148)）。
4. 补充手段：`train_async.py` 的 N/N+1 流水让训练与下一轮生成重叠；MoE 大模型还可以用 FP8 rollout、投机解码（MTP）直接加速生成本身。

**取舍**：超发浪费算力换时延；partial rollout 的续推段是旧权重生成的，引入轨迹内混合策略；fully-async 引入不固定的 staleness。所以答题时要补一句：**吞吐方案必须配 off-policy 修正/监控**（TIS、OPSM、staleness 指标，见 Q15–Q17）。

**验证**：看 `perf/rollout_time`、生成长度 P50/P95/P99、abort/requeue 比例；开启后对比相同有效 token 数下的端到端 round time 与 reward 曲线。

**追问预演**："先到先得会不会引入分布偏差？"——会，短回答更容易先完成，采样分布偏向简单样本。作者的公开回答 [官方口径] 是：配合 DAPO 式动态过滤（简单题很快全对而被过滤）与课程学习，实践上未见问题，但要监控。

### Q2 [真题·字节] GRPO 训练 MoE 模型时，rollout 和训练的专家路由不一样的原因是什么？解决方案有什么？

**结论**：根因是**离散路由放大了训推两套引擎的数值差异**；解法分"对齐路由"（routing replay）和"算法容错"（TIS/OPSM/GSPO）两派，slime 两派都实现了。

**机制**：

- **为什么会不一致**：rollout 由 SGLang 执行，训练侧由 Megatron 重算 logprob。两套引擎的 kernel 实现、算子归约顺序、精度（BF16/FP8）、top-k 的 tie-breaking 都可能有微小差异。Dense 模型里这只是 logprob 的小数点误差；MoE 里 router 是 `top-k` 的**离散选择**，score 差 1e-6 就可能翻转专家选择，token 走完全不同的专家网络，logprob 出现跳变，重要性采样比率 `π_train/π_rollout` 失真甚至爆炸，训练不稳或崩溃。
- **解法一：Routing Replay（对齐派）**。训练时不再自己算路由，而是记录并重放 rollout 时的专家选择，从根上消除路由差异。slime 提供两个开关：`--use-routing-replay`（GSPO 论文 [arXiv:2507.18071](https://arxiv.org/abs/2507.18071) 提出的训练侧 routing replay）和 `--use-rollout-routing-replay`（R3，[arXiv:2510.11370](https://arxiv.org/abs/2510.11370)，直接重放 rollout 引擎的路由决策），见 [arguments.py#L1103](https://github.com/THUDM/slime/blob/3778dbf6d1a533ab478ecf5ddaa11449a47752b2/slime/utils/arguments.py#L1103)。实现上 slime 还处理了一个细节：SGLang 的确定性 top-k 用 `sorted=False`，Megatron 默认 `sorted=True`，专家集合相同但**归约顺序**不同也会改变 BF16 累加结果，slime 的 [routing_replay.py](https://github.com/THUDM/slime/blob/3778dbf6d1a533ab478ecf5ddaa11449a47752b2/slime/utils/routing_replay.py#L52) 专门对齐了 top-k 顺序。
- **解法二：算法容错派**。TIS（截断重要性采样）用 `exp(logp_train − logp_rollout)` 加权并截断极端值；OPSM（Off-Policy Sequence Masking）直接把 mismatch 过大的整条序列 mask 掉不参与梯度；GSPO 把 ratio 从 token 级改为序列几何平均，天然抹平单 token 的路由抖动。slime 对应 `--use-tis`、`--use-opsm`、`--advantage-estimator gspo`。
- **解法三：极致确定性（2026 年新方向）**。slime 与 SGLang 配合提供了确定性推理 patch 和 Megatron 侧 DeepEP 对齐模块（`slime/backends/megatron_utils/alignment/`），目标是在匹配的版本、配置和硬件路径下实现训推 bitwise 对齐，从源头降低乃至消除已观测到的 mismatch。它不是对所有模型、算子和部署组合的无条件保证，回答时要同时说明适用边界。

**取舍**：replay 要传输/存储路由信息、且训练梯度仍会流过 router（只重放离散选择）；TIS/OPSM 丢失部分样本效率；确定性对齐牺牲部分 kernel 性能。

**验证**：监控 `train/train_rollout_logprob_abs_diff`、TIS 的 `tis_clipfrac`、OPSM 的 `opsm_clipfrac`；第一步（on-policy 起点）应看 train/rollout logprob 绝对差是否接近 0。它是 mismatch 指标，不是 `train/ppo_kl`（old-current signed log-ratio），更不是 `rollout/kl`（current-reference）。

### Q3 [真题·字节] GSPO 具体采取了什么方案缓解这个问题？

**结论**：GSPO 把重要性比率从 token 级改成**序列级几何平均**，一个 token 的路由翻转/数值抖动被整条序列平均稀释，MoE 下更稳。

**机制**：对有效 token 集 \(M\)，GSPO 的比率是

$$
r_{seq}(\theta)=\exp\Big(\frac{1}{|M|}\sum_{t\in M}(\log\pi_\theta(y_t)-\log\pi_{old}(y_t))\Big),
$$

再把同一个 \(r_{seq}\) 广播给该序列所有 token 进入 clipped surrogate。slime 的实现在 [`compute_gspo_kl`](https://github.com/THUDM/slime/blob/3778dbf6d1a533ab478ecf5ddaa11449a47752b2/slime/utils/ppo_utils.py#L95)：先对完整序列求平均 log-ratio（CP 并行时先 all-gather 完整 response，避免每个 rank 用局部片段算出不同的序列比率），再扩展回 token。

**取舍**：序列级平均也可能掩盖个别 token 的严重偏移；clip 发生在序列粒度上，一条序列要么全保留要么全被 clip，粒度更粗。GSPO 论文同时提出了 routing replay 作为配套（对应 slime `--use-routing-replay`），说明作者们也不认为单靠序列平均就够。

**追问预演**："GSPO 和 GRPO 共用什么？"——slime 里两者共用同一套组内 reward 归一化 advantage，仅 ratio 粒度不同，这也是"estimator 名 ≠ 完整 recipe"的好例子（详见 [05 算法篇](05-algorithms-and-losses.md)）。

### Q4 [真题·字节] GRPO/PPO 为什么要在 clip 之后做 min？

**结论**：`min` 让目标函数成为原始目标的**悲观下界**（pessimistic bound），clip 只削弱"继续偏离旧策略的激励"，而不会奖励偏离。

**机制**：目标是 \(\min(r_t A_t,\ \text{clip}(r_t,1-\epsilon,1+\epsilon)A_t)\)。分四象限想：

- \(A>0\) 且 \(r\) 已超过 \(1+\epsilon\)：clip 项变常数、梯度为 0，min 选中它 → 不再鼓励继续加大 \(r\)；
- \(A>0\) 且 \(r<1-\epsilon\)：min 选中未 clip 项 → 保留"把概率拉回来"的梯度；
- \(A<0\) 且 \(r<1-\epsilon\)：min 选中未 clip 项（更负）→ 同样不奖励逃逸；
- 若只用 clip 不用 min，\(A<0\) 时 ratio 掉出下界反而没有惩罚梯度，策略可以"作弊"式逃离。

**slime 对应**：[`compute_policy_loss`](https://github.com/THUDM/slime/blob/3778dbf6d1a533ab478ecf5ddaa11449a47752b2/slime/utils/ppo_utils.py#L125) 实现了标准 clipped surrogate（`eps_clip`/`eps_clip_high` 分控上下界），并支持 Dual-Clip PPO 的第三重下界 `--eps-clip-c`（对 \(A<0\) 且 ratio 极大的 token 再加一层 `c·A` 下限，防止负优势 × 巨大 ratio 的梯度爆炸；[arXiv:1912.09729](https://arxiv.org/pdf/1912.09729)）。注意 dual-clip 参数在 2026 年 8 月的 [PR #2247](https://github.com/THUDM/slime/pull/2247) 才真正接通——如果你读过旧版源码，这是展示"我持续跟踪代码演进"的好素材。

### Q5 [真题·字节] DAPO 相较于 GRPO 的 clip 有什么区别？逐 token 更新和 batch 内更新有什么区别？

**结论**：DAPO 的四个改动是 **clip-higher、token-level loss、dynamic sampling、overlong 处理**；"逐 token vs batch 级"本质是 **loss 归一化分母**选序列还是选全局 token。

**机制与 slime 对应**：

| DAPO 改动 | 动机 | slime 中的对应开关 |
|---|---|---|
| Clip-Higher（上界放宽，如 0.28） | 对称 clip 压制低概率 token 的探索，熵坍缩 | `--eps-clip-high` 单独调上界 |
| Token-level policy loss | GRPO 按序列平均导致长短回答权重失衡 | `--calculate-per-token-loss` 切换为全局 token 加权 |
| Dynamic sampling | 全对/全错组 advantage 为 0，浪费算力 | `--dynamic-sampling-filter-path` + 超发循环 |
| Overlong reward shaping / 过滤 | 截断样本的 reward 噪声 | filter hub / 自定义 reward 后处理 |

**逐 token vs batch 内更新的区别**（百度也考过变体）：GRPO 原式先对每条序列内 token 求平均、再对序列平均——每条序列等权，长序列内单 token 权重被稀释（\(1/|y_i|\)）；DAPO 把分母换成 batch 内全部有效 token 数——每个 token 等权，长序列整体影响更大，梯度不会因为回答变长而被摊薄，适合长链推理。slime 默认是**每逻辑 rollout 等权**（分母是该 rollout 的有效 token 总数，跨 micro-batch 预计算 `rollout_mask_sums`），开 `--calculate-per-token-loss` 后变成全局 token 等权——正好对应两种口径，能把这道题答到实现层。

### Q6 [真题·百度] GRPO 里的 \(\pi_\theta\)、\(\pi_{\theta_{old}}\)、\(\pi_{rollout}\) 分别是什么？

**结论**：三者分别是**正在优化的当前策略、本轮更新前的参考快照（ratio 分母）、真正生成数据的行为策略**；理想 on-policy 下三者数值接近，工程里必须分开对待。

**机制**（slime 的四种 logprob 是这道题的完美素材，详见 [04 数据篇](04-data-pipeline.md)）：

| 概念 | slime 字段 | 谁算的 | 用途 |
|---|---|---|---|
| \(\pi_{rollout}\)（behavior） | `rollout_log_probs` | SGLang 采样时返回 | 训推 mismatch 监控、TIS 修正 |
| \(\pi_{\theta_{old}}\) | `log_probs` | Megatron 在 optimizer 更新前 forward 重算 | PPO/GRPO ratio 的分母 |
| \(\pi_\theta\) | 训练中实时 forward | Megatron | ratio 的分子、entropy |
| （追问会带出）\(\pi_{ref}\) | `ref_log_probs` | 冻结参考模型 | KL 约束 |

**关键点**：为什么 \(\pi_{\theta_{old}}\) 不直接用 \(\pi_{rollout}\)？因为两套引擎数值有差异（见 Q2），且一轮数据可能做多个 optimizer step——第一步时 old=current（ratio=1，`train/ppo_kl`=0），后续步 ratio 才偏离。slime 提供 `--use-rollout-logprobs` 直接把 behavior logprob 当 old 用（省一次 forward，但把训推 mismatch 直接引入 ratio），默认则用 Megatron 重算——这个取舍本身就是面试官想听的。

### Q7 [真题·百度] 讲一下 GRPO 训练的数据流。

**结论**：用 slime 的管线回答，比背论文伪代码有区分度得多。

**机制**（一条 prompt 的完整旅程）：

```text
JSONL/Parquet 行 → Dataset(prompt/label/metadata)
→ 每个 prompt 深拷贝 n_samples_per_prompt 份（组）
→ SGLang 异步并发生成（返回 token + behavior logprob）
→ reward（规则 verifier / RM，组内可 --group-rm 批量打分）
→ dynamic filter（可选：丢全对全错组，超发补齐）
→ 组内 reward 减均值（可选除 std）→ 广播为 token 级 advantage
→ 按 rollout_id 切训练 step → micro-batch 打包 → 按 DP rank 分发
→ Megatron 重算 old logprob → clipped surrogate（+ KL/entropy 可选）
→ optimizer step → 新权重同步回 SGLang → 下一轮
```

要点强调三处：组的建立在数据源层（`group_index`）；归一化发生在 rollout 后处理层，训练侧拿到的已是 advantage；GRPO 无 critic，所以不存在 GAE/value 分支。追问"训练 step 怎么切"可讲 slime 按 distinct `rollout_id` 计 GBS 的口径（[04 数据篇第 7 节](04-data-pipeline.md)）。

### Q8 [真题·百度] GRPO 是 on-policy 还是 off-policy？batch size 非常大时，如何缓解 off-policy 问题？

**结论**：GRPO 名义上按 on-policy 设计，但工程实现里几乎总有轻度 off-policy；缓解手段按"控制 staleness"和"修正分布"两类回答。

**机制**：off-policy 的三个来源——①一批数据做多个 optimizer step（第 2 步起 current≠behavior）；②大 batch 意味着一次生成覆盖多个训练 step，后面的 step 用的是更旧的数据；③异步流水/权重同步间隔引入的策略滞后。缓解：

1. **控制侧**：减少每批 optimizer step 数；缩短权重同步间隔（slime 同步驱动每轮都同步，`train_async.py` 用 `--update-weights-interval` 控制）；限制最大 staleness（fully-async 场景监控队列年龄）。
2. **修正侧**：PPO clip 本身就是一阶防线；TIS 截断重要性采样纠正 behavior→train 偏移（`--use-tis`）；OPSM 把偏移过大的序列整条 mask（`--use-opsm`）；必要时用 `--use-rollout-logprobs` 让 ratio 直接以 behavior 为分母，语义更接近标准 IS。
3. **观测侧**：`train/ppo_kl`（old vs current 的有符号 log-ratio）、`train/train_rollout_logprob_abs_diff`（默认 train-old vs rollout；启用 `use_rollout_logprobs` 后为 current vs rollout）、`pg_clipfrac`、`tis_clipfrac`，以及 rollout 权重版本与训练 step 的差距分布。

### Q9 [真题·阿里] 为什么需要 advantage？直接用 reward 不行吗？重要性采样在新旧策略差别很大时还有用吗？

**结论**：advantage 是"减 baseline 的 reward"，作用是**降方差、不改变梯度期望**；IS 在策略差异大时理论上仍无偏但方差爆炸，实践必须 clip/截断，等价于接受一点偏差换方差。

**机制**：策略梯度 \(\nabla J=\mathbb E[\nabla\log\pi\cdot\Psi]\) 中 \(\Psi\) 可以是原始 reward，也可以是减去任意与动作无关 baseline 的量——期望不变、方差大幅下降。GRPO 的组内均值就是一个"免费 critic"：同 prompt 采 n 个回答，用组均值当 baseline（slime 在 rollout 后处理阶段完成，[`get_grpo_returns`](https://github.com/THUDM/slime/blob/3778dbf6d1a533ab478ecf5ddaa11449a47752b2/slime/utils/ppo_utils.py#L361) 只做广播）。若直接用 raw reward：全正的 reward 会让所有采样到的动作都被强化，收敛慢且不稳。

IS 部分：权重 \(w=\pi_{new}/\pi_{old}\) 的方差随两策略 KL 指数增长；当差异大时少数样本拿到巨大权重，梯度被单点主导。所以 PPO clip、TIS 截断、OPSM 掩码本质都是"有偏但可控"的方差控制。可以补充 slime 的实践锚点：每轮第一步的 `train/ppo_kl`（old-current signed log-ratio）严格为 0，后续步的该指标和 clipfrac 才是“更新后策略差异是否过大”的直接读数；它不能替代 train/rollout mismatch 指标。

### Q10 [真题·阿里] 序列级 reward 如何分配到每个 token（credit assignment）？

**结论**：主流做法四种，按"是否学习"和"粒度"区分：**广播（GRPO）、折扣回传（REINFORCE++）、学习分配（PPO/GAE）、过程奖励（PRM/step-level）**。

**机制与 slime 对应**：

| 方案 | 分配方式 | slime 实现 |
|---|---|---|
| 广播 | 组归一化后的序列标量复制给每个 response token | `get_grpo_returns`（GRPO/GSPO/CISPO 共用） |
| 折扣回传 | terminal reward + 稠密 KL 惩罚，从后向前 \(G_t=r_t+\gamma G_{t+1}\) | [`get_reinforce_plus_plus_returns`](https://github.com/THUDM/slime/blob/3778dbf6d1a533ab478ecf5ddaa11449a47752b2/slime/utils/ppo_utils.py#L371)（已向量化） |
| 学习分配 | critic 给每个 token 估 value，GAE 算逐 token advantage | PPO 分支 + `chunked_gae` |
| 过程奖励 | 对中间步骤单独打分（PRM、规则 step 检查） | 自定义 RM / reward 后处理 hook 表达 |

追问"广播是不是等于没有 credit assignment？"——组相对基线在**序列间**做了分配（好回答整体加强），token 间靠 clip + 多步更新的隐式信号；这正是 GRPO 在长程任务上弱于 PPO/PRM 的原因之一，也是字节问"Agentic RL 过程打分"（Q22）的引子。

### Q11 [真题·商汤] GRPO 中"正确且短、正确且长、错误且长、错误且短"四种情况，模型倾向排序是什么？为什么？

**结论**：标准 GRPO（序列内平均）下，模型的偏好排序是**正确且短 > 正确且长 > 错误且长 > 错误且短**——长度偏置来自 \(1/|y_i|\) 归一化。

**机制**：序列内平均让每个 token 的梯度权重是 \(A_i/|y_i|\)。\(A>0\)（正确）时，短回答的每 token 强化信号更大 → 正确答案越短越受偏爱；\(A<0\)（错误）时，长回答的每 token 惩罚被稀释 → 错误时"说得越长罚得越轻"，模型宁可错得冗长。叠加组内 std 归一化还有第二重偏置：简单题（组内几乎全对，std 小）的 advantage 被放大。这是 Dr. GRPO 论文批评的两点，也是 DAPO 改 token-level loss 的动机。

**slime 对应**：默认 reducer 是每逻辑 rollout 等权（保留 \(1/|y|\) 语义，但分母按整个 rollout 的有效 token 数预计算，fan-out 不会重复计权）；`--calculate-per-token-loss` 切到 DAPO 式全局 token 加权；`--grpo-std-normalization` 可关掉 std 除法（Dr. GRPO 风格）。能把三个开关和两重偏置对上，这题就是满分答案。

### Q12 [真题·商汤/多家] RL 的训推不一致有了解过吗？哪些方面可能产生？

**结论**：训推不一致 = **同一份权重在 rollout 引擎和训练引擎下给出不同的 token 概率**；来源分五层，MoE 是重灾区。

**机制**（分层背下来）：

1. **kernel/算子层**：两套引擎的 GEMM/attention 实现、归约顺序不同，浮点非确定性；
2. **精度层**：FP8 rollout vs BF16 训练（slime 大 MoE recipe 的标准配置）、KV cache 量化；
3. **采样层**：temperature/top-p/top-k 截断改变了实际采样分布，而训练侧重算的是完整分布下的 logprob；
4. **MoE 路由层**：top-k 离散翻转（见 Q2，影响最大）；
5. **系统层**：权重同步时机（stale weights）、chat template/tokenizer 不一致、多轮拼接错位。

**slime 的完整应对栈**（这是把答案抬到框架层的机会）：监控（`train_rollout_logprob_abs_diff`、mismatch 指标）→ 算法修正（TIS/OPSM）→ 路由对齐（两种 routing replay）→ 极致方案（`alignment/` 确定性模块 + SGLang deterministic patch，GLM-5 训练实践）。参考 [08 调试篇](08-debugging-reliability-and-performance.md) 的排查路线。

### Q13 [真题·腾讯混元] DeepSeek 用的 GRPO 相比 GPT 的 PPO 做了哪些改进？

**结论**：一句话——**用组内相对基线替换 critic**，把"四模型系统"（actor/critic/RM/ref）简化为"两模型 + verifier"。

**机制**：①去 critic：省一半训练显存与 value loss 调参；②组采样：同 prompt 采 n 个回答，reward 减组均值（可除 std）当 advantage；③KL 处理：GRPO 论文把 KL 作为 loss 项（而非 PPO 的 reward shaping），且用 k3 低方差估计。适配场景：数学/代码这类**可验证 reward**（规则 verifier 免费且无限），配合 pass@k 天然对齐。代价：必须能对同 prompt 采多响应（推理成本 ×n）、组内奖励全同时无信号（zero-std 问题）、序列级 credit assignment 更粗。

**slime 对应**：`--advantage-estimator grpo` + `--n-samples-per-prompt`；PPO 是唯一自动创建 critic 的选项；KL 的三条路径（reward shaping / kl_loss / OPD）是独立开关——注意 slime 当前实现中 GRPO 内置分支的 `kl_coef` 不参与 returns，想要 GRPO+KL 应该用 `--use-kl-loss`（详见 [05 算法篇第 3 节](05-algorithms-and-losses.md)，这是很好的"读过源码"证据）。

### Q14 [高频方向] GRPO、GSPO、CISPO、K2 变体怎么对比？各厂为什么各选一条路？

**结论**：四者都是"免 critic 的组相对优势"家族，分歧在**对重要性比率的处理**——这直接映射到各家的模型架构痛点。

| 算法 | 出处 | ratio 处理 | 动机 |
|---|---|---|---|
| GRPO | DeepSeek | token 级 clip + min | 通用基线 |
| GSPO | Qwen（阿里） | 序列几何平均后 clip | Qwen3 MoE 训练稳定性（路由抖动被序列平均稀释） |
| CISPO | MiniMax（M1） | **clip 权重、不 clip 更新**：对 ratio 做 stop-gradient 截断，token 梯度保留 | 长 CoT 中低概率关键 token（"Wait"/"Recheck"类反思词）被 PPO clip 永久杀死梯度的问题 |
| K2 变体 | 月之暗面 | GRPO 相对基线 + 显式 KL 正则的混合 | MoE 规模下的稳定性折中 |

**CISPO 细节**（MiniMax 面试必考自家算法）：\(L=-\text{sg}[\text{clip}(r_t)]\cdot A_t\log\pi_\theta(y_t)\)，被截断的 token 权重封顶但梯度仍从 \(\log\pi_\theta\) 流过；MiniMax 报告同等性能只需 DAPO 一半训练步数。slime 内置实现见 [`compute_cispo_loss`](https://github.com/THUDM/slime/blob/3778dbf6d1a533ab478ecf5ddaa11449a47752b2/slime/utils/ppo_utils.py#L152)，且参数校验会提醒 canonical 用法是 `eps_clip=1.0` 关掉下界、单调 `eps_clip_high`——slime 是少数把三家算法都收进同一套 loss 分派的框架，横向对比题可以直接拿它当"活教材"。

## 3. 系统与 Infra 真题

### Q15 [官方口径·智谱] 为什么 slime 选 SGLang 而不是 vLLM？

**结论**：作者公开答案有三层——**大 EP MoE 推理性能、server-based engine 设计哲学、社区协作效率**。

**机制**（据朱子霖青稞 Talk 实录整理）：

1. **性能**：智谱要训 355B 级 MoE，RL 的瓶颈在推理；SGLang 的大规模 EP（DeepEP/DP attention/FP8）MoE 推理长期领先，slime 用 `--sglang-` 前缀透传全部参数，上游优化零成本继承。
2. **Server-based vs engine-based**：多数旧框架 `from vllm import LLM` 内嵌 engine，slime 反过来让 RL 适配推理框架——SGLang 以独立 server + router 运行，数据生成方只需向 OpenAI-compatible endpoint 发请求。好处：自定义 rollout 变成"写 HTTP 客户端"，多轮/工具/外部 agent 环境接入不需要理解推理引擎内部；升级引擎不破坏接口。
3. **社区**：SGLang 国内社区活跃，slime 的需求（pause/continue 接口、权重更新接口、确定性 patch）能快速合入上游。

**加分补充**：单 rollout 后端是**刻意取舍**——多后端框架被迫抽象成"最小公共能力集"，牺牲各引擎的独有特性（PD 分离、hicache、投机解码）。生态项目 vime（vLLM rollout）和 Miles（RadixArk 企业版）证明这个架构可以被替换后端，但核心仓库不做这个抽象。

### Q16 [官方口径·智谱] 为什么训练后端只支持 Megatron？verl 都支持 FSDP 了。

**结论**：作者的回答是**正确性必须靠内部大规模验证背书**，而不是"写不写得出来"的问题。

**机制**：①Megatron 是唯一经过智谱内部数千卡、300B+ MoE 全流程验证的后端，开源出去的代码和内部跑的是同一套，社区反馈能回流；②如果发布一个内部从未大规模验证的 FSDP 后端，等于把风险转嫁给用户，"内外部使用同一套代码"的开源协作模型就被打破；③FSDP 系方案大多复用 HF modeling，在 MoE/超大规模上的正确性与性能都难对齐 Megatron。代价是上手门槛高（checkpoint 转换、并行配置），slime 的路线是降低 Megatron 使用门槛而非绕开它。

**对比句式**（面试可直接用）："verl 选的是覆盖面——FSDP 易用、Megatron 高性能，用户按规模换后端；slime 选的是单路径深度——所有人力押在一条经过 frontier model 验证的路径上。这是团队资源和目标场景决定的，不是谁对谁错。"

### Q17 [官方口径+真题变体] Ray 在 slime 里承担什么角色？千卡规模下 Ray 会不会成为瓶颈？

**结论**：Ray 主要做**控制面**（资源编排、进程生命周期、异步依赖表达），不承载分离部署的大权重广播或 Megatron 集合通信。colocate 是边界例外：Ray actor 调用会携带 tensor，但当前 CUDA 路径依赖 IPC handle 共享底层显存，而不是经 object store 复制整份权重。

**机制**：

- Ray 负责：placement group 锁定 GPU 拓扑、创建训练/rollout actor、`.remote()/ray.get` 表达同步异步依赖。
- Ray 不负责：**权重的数据面不经 Ray object store 搬运大权重**（当前 CUDA colocate 路径把 tensor 交给 engine，生命周期由 CUDA IPC handle 管理；分离部署由 Megatron 与 SGLang 建 NCCL 通信组 broadcast；跨集群走共享磁盘）；**Megatron 内部集合通信不走 Ray**；rollout 数据（token/mask/logprob，量级小）经 Ray object store 单点交给训练侧再内部广播。这里的 CUDA IPC/NCCL 是当前 GPU 实现边界，不代表任意 accelerator/backend 自动具备同一传输路径。
- 作者原话要点：曾有人担心 Ray 序列化大 tensor 的开销（GPU→CPU→序列化→反序列化→GPU），slime 的设计就是让这条路径上没有大 tensor，所以规模上去后 Ray 不是卡点。

**追问预演**："为什么不去掉 Ray？"——Python 生态里同时做资源分配+异步的工具，Ray 目前最成熟；slime 对 Ray 的使用很浅，未来有更好方案可替换。

### Q18 [高频方向] 训练完的权重怎么同步给推理引擎？开销多大？怎么保证不在生成中途换权重？

**结论**：在当前 CUDA GPU backend 下，slime 三条路径是 **colocate 的 tensor/CUDA IPC、分离的 NCCL 分块广播、跨集群的磁盘（全量或字节级 delta）**；非 CUDA accelerator 或非 NCCL backend 不能直接套用这组结论。355B 模型分钟级，同步前 pause 生成 + flush KV。

**机制**：

| 路径 | 适用 | 原理 |
|---|---|---|
| full + tensor（CUDA IPC） | CUDA colocate 同卡 | tensor 经 Ray actor 调用交给 engine，CUDA IPC handle 管理底层显存共享；训练侧另有 GPU collective/Gloo gather |
| full + NCCL | CUDA GPU 训推分离、同集群 | gather Megatron 分片 → 转 HF 命名 → 分 chunk 广播给各 engine；要求 NCCL backend/可建通信组 |
| full + disk | 跨集群/external engine/release-train | 写版本化 HF checkpoint，engine reload；容错最好（engine 挂了重启直接从盘加载） |
| delta + disk | 超大模型跨集群 | 对上一版本做字节级 diff，只发布差量，host 本地合并后 reload |

一致性协议：更新前 `pause_generation` + flush cache，发布后 resume，加分布式锁防并发 broadcast 死锁——正常协议旨在避免请求跨越两个权重版本；异常恢复仍要结合具体部署验证。量级 [官方口径]：355B 约 1–2 分钟（相对一轮 1 小时的 rollout 可忽略，具体取决于拓扑、带宽和版本），业界极致参考是 Kimi 的 checkpoint engine ~20s/TB。此外 `optimizer step ≠ weight sync`：前者更新训练侧参数，后者才把参数发布给 serving，频率可以不同（`--update-weights-interval`），这是面试常见混淆点。

### Q19 [真题变体·多家] 训推一体（colocate）和训推分离怎么选？

**结论**：卡不够选 colocate（分时复用），卡够且想重叠生成/训练选分离——用**端到端 round time 和成本**做决策，不看单阶段吞吐。

**机制**：colocate 下 actor 和 rollout 共用 GPU，靠 offload/onload 切换（训练时释放 KV cache，生成时释放优化器状态）；GPU 数 = max(训练需求, 推理需求)。分离下两边各占独立 GPU（相加），可用 `train_async.py` 让 train(N) 与 generate(N+1) 重叠，代价是固定一轮策略滞后。slime 的资源公式在 [placement_group.py](https://github.com/THUDM/slime/blob/3778dbf6d1a533ab478ecf5ddaa11449a47752b2/slime/ray/placement_group.py)：分离 = A+R，colocate = max(A,R)。硬约束：`train_async.py` 断言不支持 colocate（重叠与分时复用目标冲突）。还有第三种形态：external rollout engine（推理集群完全独立部署，slime 只管训练和发权重），适合 serving 团队独立运维的场景。

### Q20 [高频方向] 同步、异步、fully-async 的区别？staleness 怎么管？

**结论**：slime 的三档正好是标准答案的实物版——**同步（每轮先生成后训练）、N/N+1 流水（one-step off-policy）、fully-async（常驻生成池，staleness 不固定）**。

**机制**：详见 [03 执行模式篇](03-synchronous-and-asynchronous-execution.md)。答题抓三个锚点：

1. 判断"真异步"的标准是**关键路径上生成和训练是否同时在跑**，函数名带 async 不算（`async_train` 只是返回 ObjectRef）；
2. N/N+1 的滞后是**可解释的固定一轮**：generate(N+1) 用 W_N 提交，训练 N 后才同步 W_{N+1}，同步前 drain 在途生成；
3. fully-async 的 staleness 由并发池、队列积压、长尾分布共同决定，必须监控"样本的权重版本 vs 当前训练 step"分布，配 TIS/OPSM 修正，abort 的组回炉重推（abort + recycle）。

**跨框架视角**（对比题弹药）：AReaL 全异步 + staleness 超参 η 控制最大版本差；verl fully_async_policy 用 bounded queue + backpressure + partial rollout;slime 走 hybrid 路线（同步/流水/纯异步都支持，作者称之为给算法留自由度）；Seer 反向坚持极致同步，用 divided rollout + 全局 KV 池调度消灭长尾。

### Q21 [高频方向] RL 训练的 checkpoint 恢复要对齐哪些状态？

**结论**：四类状态必须在**同一 rollout 边界**对齐——模型/优化器/RNG、训练进度（rollout id）、数据源游标（offset/epoch/shuffle 状态）、权重同步版本。

**机制**：只恢复模型不恢复游标 → prompt 重复或跳过；游标超前模型 → 数据被"预支"。slime 的默认数据源把 offset/epoch/sample index 存到 `global_dataset_state_dict_{rollout_id}.pt`，与 Megatron checkpoint 按同一 rollout id 配对。边界情况是加分点：`train_async.py` 保存模型 N 时 generate(N+1) 可能已推进游标，fully-async 的在途任务/完成队列不持久化——所以**异步模式当前不承诺 exact resume**，生产要么接受近似恢复，要么自行把在途状态设计成可恢复事务。PPO 还要确认 actor/critic 两组 checkpoint 进度一致。

### Q22 [真题·字节] Agentic RL 场景下如何设计过程打分？反向传播时 token 分数怎么算？

**结论**：过程打分在 **reward 结构**层解决（结果分 + 过程分/PRM/格式分），token 级分配在 **advantage 与 mask**层解决——两层解耦是标准工程答案。

**机制**：

1. **打分设计**：结果 reward（任务成败，如测试通过）+ 过程 reward（步骤合法性、工具调用格式、中间结论验证）+ 惩罚项（超时、死循环）。多维 reward 可以是 dict，训练时按 key 选择或加权（slime `Sample.reward` 支持 dict + `--reward-key`）。防 reward hacking：过程分要防"刷格式"，evaluator 与训练环境隔离。
2. **token 级反传**：先用 `loss_mask` 把非模型 token（工具输出、环境观察）清零——**这些 token 不接收任何梯度**；模型 token 上，序列级 reward 按 Q10 的四种方式分配（GRPO 广播 / R++ 折扣 / PPO GAE / PRM 对分段分别赋值）。多段轨迹（subagent、compact 分段）在 slime 里 fan-out 成共享 `rollout_id` 的多条 Sample，loss 归一化按整个逻辑 rollout 的有效 token 计算，保证"一次执行只算一次"。
3. **实现锚点**：slime 的 `append_response_tokens(..., trainable=False)` 维护 token/logprob/mask 三者同步增长；`examples/coding_agent_rl` 展示了 sandbox + 干净 evaluator + fan-out 的完整结构。

### Q23 [官方口径] 多轮工具调用的 assistant mask（loss mask）怎么处理？

**结论**：作者的答案很直接——**框架不猜，用户随 response 返回等长的 0/1 mask 数组**；模型生成 token 为 1，工具/环境注入 token 为 0。

**机制**：不同模型的 chat template 和工具协议差异太大，框架层做统一 mask 推断不现实。slime 的契约是 `len(loss_mask) == response_length`，工具结果必须以不可训练方式追加。常见错误：把工具输出 decode 成字符串拼进 prompt 再 retokenize（破坏原始采样 token，logprob 对不上）；只设 `status=FAILED` 以为会自动剔除（不会，必须显式 filter/remove/全零 mask）。对标准模板 slime 也提供 `--loss-mask-type` 预置解析（当前 choices 为 `qwen`/`qwen3`/`qwen3_5`/`distill_qwen`），2026 年 8 月还修过 Qwen3 连续 tool response 的 tokenization 边界（[PR #2264](https://github.com/THUDM/slime/pull/2264)）——说明这类细节连维护者都会踩坑，面试时举这个例子非常真实。

### Q24 [官方口径] Agent 环境很重（每个 task 一个镜像）时，怎么和 RL 框架耦合？

**结论**：作者的方案是**反转控制流**——不是 RL 框架驱动 agent，而是 agent 环境作为独立系统向 slime 的 server-based engine 发请求，RL 侧只收集轨迹数据。

**机制**：agent 框架初始化后持续向 OpenAI-compatible endpoint 发请求跑自己的循环；slime 定期收集 agent 产生的轨迹 log，凑够训练数据就 pause 整个 server → 训练 → 更新权重 → continue。这样 agent 环境（K8s 镜像、沙箱集群、真实业务系统）完全不需要理解 RL 框架，只需加一种"存轨迹"的方式。这是 server-based engine 设计"涌现"出来的纯异步 agent 方案，也是 slime 与"以训练循环为中心"框架的最大差异点。风险要主动说：外部环境的故障域、轨迹与权重版本的对应关系、数据回收的一致性都要额外设计。

### Q25 [高频方向] slime、verl、OpenRLHF、AReaL、NeMo-RL 怎么选型？

**结论**：先问三个问题——**模型多大（尤其是不是 MoE）、任务长尾多严重、团队想不想改框架内核**，再对号入座。

**机制**（综合 2025-12 青稞对比文与各家公开资料）：

| 框架 | 定位 | 训练后端 | 推理后端 | 突出点 |
|---|---|---|---|---|
| slime（智谱） | SGLang-native、RL scaling | Megatron | SGLang（唯一） | 大 MoE/GLM 项目实践路径、hybrid 同步/异步、轻量易读；具体模型与拓扑按版本验证 |
| verl（字节） | 大一统、易用 | Megatron/FSDP | vLLM/SGLang | HybridFlow 数据流、社区最活跃、特性最全 |
| OpenRLHF | 早期标杆、易上手 | DeepSpeed 系 | vLLM | 教学与中小规模友好 |
| AReaL（蚂蚁） | 全异步 | Megatron/FSDP | vLLM/SGLang | staleness 超参化、异步吞吐极致 |
| NeMo-RL（NVIDIA） | NV 栈整合 | Megatron 系 | 内置 | replay buffer + in-flight 权重更新 |

选型口诀：**超大 MoE 且技术栈匹配 SGLang/Megatron → 优先评估 slime；要快速上手/全特性/多后端 → verl；极长尾 agent 且接受 off-policy → AReaL 或 slime 异步模式；学习源码 → slime（控制流较显式）**。加一句维度提醒：各框架在互相吸收特性（partial rollout、异步、routing replay 都在扩散），对比结论有时效性，答题时报出你核对过的时间点，并用自己的模型、硬件和任务做验证。

## 4. 手撕与白板题

### Q26 [真题·多家] 手写 GRPO 的 advantage 计算 + policy loss

面试标准版（numpy/torch 均可，先写公式再写码）：

```python
import torch

def grpo_advantages(rewards, group_size, std_norm=True, eps=1e-6):
    """rewards: [B]，B = num_prompts * group_size，同组连续排列"""
    r = rewards.view(-1, group_size)                # [P, G]
    adv = r - r.mean(dim=1, keepdim=True)           # 组内减均值
    if std_norm:
        adv = adv / (r.std(dim=1, keepdim=True) + eps)
    return adv.flatten()                            # 序列级标量，训练时广播到 token

def grpo_policy_loss(logp, logp_old, adv, mask, eps_lo=0.2, eps_hi=0.2):
    """logp/logp_old/mask: [B, T]; adv: [B] 广播到 token"""
    ratio = torch.exp(logp - logp_old)
    a = adv.unsqueeze(-1)
    l1 = ratio * a
    l2 = torch.clamp(ratio, 1 - eps_lo, 1 + eps_hi) * a
    per_tok = -torch.min(l1, l2) * mask
    # 序列内平均再对序列平均（GRPO 原式；DAPO 则改为全局 token 平均）
    return (per_tok.sum(-1) / mask.sum(-1).clamp(min=1)).mean()
```

写完主动说三个工程差异（对照 slime 真实实现）：①生产实现用 `ppo_kl = logp_old - logp` 再 `exp(-ppo_kl)`，数值上等价；这里记录的是 old-current signed log-ratio，不是 reference KL；②归一化分母要预计算并跨 micro-batch 保持一致（slime 的 `rollout_mask_sums`）；③零 std 组要么被 dynamic filter 丢弃要么 advantage 为 0 白占算力——这就是 zero-std 监控的意义。

### Q27 [真题·百度] KL 散度的公式，几种估计怎么写？怎么"平滑"？

三种蒙特卡洛估计（John Schulman 的 k1/k2/k3，slime 的 [`compute_approx_kl`](https://github.com/THUDM/slime/blob/3778dbf6d1a533ab478ecf5ddaa11449a47752b2/slime/utils/ppo_utils.py#L12) 全部实现）：

```python
log_ratio = logp - logp_ref            # log(π/π_ref)，逐 token
k1 = log_ratio                         # 无偏，方差大，可为负
k2 = 0.5 * log_ratio ** 2              # 有偏，低方差，恒非负
k3 = (-log_ratio).exp() - 1 + log_ratio  # 低方差且无偏地估计 KL(π||π_ref)，恒非负
```

"平滑/防数值爆炸"的答法：logprob 域计算（先 log_softmax 再相减，绝不先 exp）；`exp(clamp(log_ratio, -20, 20))` 可以作为你建议的额外溢出保护，但要明确 **v0.3.2 的 `compute_approx_kl` 当前直接对 log-ratio 做 `exp`，没有这层 clamp**。k3 恒非负，可避免 k1 单样本为负时让惩罚符号反转；追问 softmax 数值稳定就答减 max 的 log-sum-exp 技巧。

### Q28 [高频方向] 手写 TIS（截断重要性采样）修正

```python
def tis_correction(logp_train_old, logp_behavior, pg_loss, mask, c=2.0):
    """训推 mismatch 修正：behavior=SGLang 采样时的 logprob"""
    tis = torch.exp(logp_train_old - logp_behavior)   # 训练引擎 vs 行为策略
    tis_clipped = torch.clamp(tis, max=c)             # 单边截断，防大权重
    return pg_loss * tis_clipped * mask
```

要点：TIS 的 ratio 是 **train_old vs behavior**（跨引擎），与 PPO 的 **current vs old**（跨更新步）是两个正交的比率，slime 指标里分别叫 `tis` 和 `ois`。截断只保护尾部；若 `tis_clipfrac` 长期偏高，说明该修根因（同步频率/路由对齐）而不是继续调 c。

### Q29 [高频方向] 白板题：设计一个支持 10B–100B MoE 的 RL 训练系统

推荐画 slime 的架构再解释（面试官若熟悉会直接认出来，这本身就是信号）：

```text
            ┌─────────── control plane: Ray ───────────┐
            │  placement group / actor 生命周期 / 依赖   │
            └──────────────────────────────────────────┘
   ┌──────────────┐   Sample(tokens,reward,   ┌──────────────────┐
   │ Rollout 面    │   logprob,mask,ids)       │ Training 面       │
   │ SGLang×N     │ ────────────────────────▶ │ Megatron actors  │
   │ + router     │                           │ (+critic/ref可选)  │
   │ + DataBuffer │ ◀──────────────────────── │                  │
   └──────────────┘   权重: CUDA IPC/NCCL(GPU)/disk └─────────────┘
        ▲  自定义生成/RM/环境 hook                    │ checkpoint
        └── agent env / verifier / sandbox ──────────┘
```

按面试官追问逐层展开：数据契约（Sample/三个 ID/mask）→ 资源形态（colocate/分离/external 三选）→ 执行模式（同步/流水/fully-async）→ 权重同步协议（pause-flush-publish-resume + 版本）→ 一致性（TIS/OPSM/routing replay）→ 可靠性（checkpoint+游标、rollout health monitor、debug replay 二分法）。每层都能引用前十章的对应文档。

## 5. 把 slime 讲成加分项的策略

1. **报快照**：开口先说“我基于 2026 年 8 月 29 日扫描的 main/v0.3.2”，展示你知道框架在快速演进（比如 dual-clip 是 8 月才接通的、mbridge 是 8 月才内部化的）。
2. **用证据分层**：源码实现 / 测试覆盖 / README 声称 / 示例 recipe 四档表述不要混——这是资深工程师和背题者的最大区别。
3. **主动给边界**：每个亮点跟一句代价（单后端 ↔ 深度优化；异步 ↔ staleness；colocate ↔ 切换开销）。
4. **落到指标**：任何"会不会有问题"类追问，都用"我会看哪个指标、跑哪个最小实验"收尾（zero-std、clipfrac、logprob diff、rollout/train replay）。
5. **有 PR 更好**：如果你给 slime 提过 PR（哪怕是 bugfix + 回归测试），面试中它是比任何背题都硬的证据；讲清 bug 的触发路径、修复取舍、测试如何防回归即可。

## 6. 高频误区 checklist（面试前最后过一遍）

- [ ] GRPO 的"KL"是加在 loss 上的 k3 估计，不是 PPO 的 reward shaping；slime 里两条路径互斥。
- [ ] `π_old` 默认是训练引擎重算的，不是 SGLang 返回的 behavior logprob；混用要报 `--use-rollout-logprobs` 的取舍。
- [ ] 长度偏置排序题的根源是 \(1/|y|\)，答案是"正确且短 > 正确且长 > 错误且长 > 错误且短"。
- [ ] clip 之后取 min 是悲观下界，不是"双重保险"这种含糊说法。
- [ ] MoE 训推不一致的核心词是"离散路由放大数值差异"，解法先说 routing replay 再说 TIS/OPSM。
- [ ] 权重同步 ≠ optimizer step ≠ checkpoint 保存，三个频率三个用途。
- [ ] Ray 只是控制面；说"slime 用 Ray 传权重"是硬伤。
- [ ] fully-async 的 staleness 不是固定一轮；被 abort 的组会回炉，完成队列跨轮保温。
- [ ] "支持某模型"要分转换代码/recipe/CPU 测试/GPU E2E/生产验证五档。
- [ ] 框架对比有时效性，报出你核对的时间点。

---

继续深挖：[架构](02-architecture-and-control-flow.md) · [执行模式](03-synchronous-and-asynchronous-execution.md) · [数据管线](04-data-pipeline.md) · [算法](05-algorithms-and-losses.md) · [调试](08-debugging-reliability-and-performance.md) · [通用题库](10-interview-question-bank.md)
