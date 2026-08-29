# 01. 项目全景：slime 解决什么问题

> **适用源码快照**：本文基于 `v0.3.2@3778dbf6`（扫描日期 2026-08-29）。slime 仍在快速演进；面试中应先说明快照，再讨论能力边界，避免把后续版本或生态项目的能力误算到核心仓库。

## 一句话定位

slime 是面向大语言模型 RL scaling 的 post-training 框架。它不重新实现训练内核或推理内核，而是用 Ray 把 **Megatron 训练、SGLang rollout、数据缓冲、自定义生成/奖励以及权重同步** 组织成一个可扩展的在线训练闭环。项目自己的定位和两项核心能力可见 [README_zh.md](https://github.com/THUDM/slime/blob/3778dbf6d1a533ab478ecf5ddaa11449a47752b2/README_zh.md#L9)。

这句话有三个关键词：

1. **post-training**：重点是预训练之后的 SFT、RL、on-policy distillation 等流程，而不是通用预训练平台；
2. **RL scaling**：重点是多 GPU、多节点下训练与生成的吞吐、显存调度和一致性，而不只是提供一个 PPO 公式；
3. **编排层**：Ray 负责跨进程、跨节点控制；Megatron 和 SGLang 分别承担重计算的数据面。

## 为什么需要这样一个框架

在线 RL 不是“先生成一个静态数据集，再调用 trainer”这么简单。每一轮至少要完成：

```text
当前策略生成轨迹 -> 奖励/验证 -> 训练数据整形 -> 更新 actor（可选 critic）
        ^                                              |
        |--------------- 同步新权重 ------------------|
```

真正困难的是系统边界：

- rollout 常比训练慢，而且不同轨迹耗时差异很大；
- Megatron 参数是分片的，SGLang 需要 serving 侧可加载的权重布局；
- actor、critic、rollout 可能独占 GPU，也可能按阶段共享 GPU；
- checkpoint 不只包含模型，还涉及优化器、学习率调度器和数据集游标；
- 自定义 agent、工具调用或 verifier 不应迫使用户修改训练内核。

slime 的价值主要在这些连接处，而不是声称发明了新的分布式训练或推理引擎。

## 与 Ray、Megatron、SGLang 的关系

| 项目 | 在 slime 中的角色 | slime 增加的部分 |
| --- | --- | --- |
| Ray | 控制面：remote actor、ObjectRef、资源调度、placement group | 创建训练/rollout actor，固定 GPU 拓扑，表达同步与异步依赖 |
| Megatron | 训练数据面：模型并行、优化器、前后向、distributed checkpoint | 把 rollout batch 转为 Megatron 可消费的数据；管理 actor/critic；把分片权重转换并同步给 serving |
| SGLang | rollout 数据面：高吞吐生成、router、KV cache、serving 并行 | 启停/卸载引擎，接入数据源、奖励和自定义生成，并接收每轮新权重 |

Ray 不参与张量级训练算法；Megatron 不负责 agent 环境；SGLang 不负责优化器更新。边界清楚，是阅读架构时最重要的抓手。

源码上的对应关系也很直接：

- [placement_group.py](https://github.com/THUDM/slime/blob/3778dbf6d1a533ab478ecf5ddaa11449a47752b2/slime/ray/placement_group.py#L120) 分配 Ray 资源并创建训练组；
- [RolloutManager](https://github.com/THUDM/slime/blob/3778dbf6d1a533ab478ecf5ddaa11449a47752b2/slime/ray/rollout.py#L38) 管理 rollout 生命周期、数据源、生成/评估函数、health monitor 和权重更新锁；
- SGLang 的真实部署链是 `RolloutManager` → [`deployment.start_rollout_servers`](https://github.com/THUDM/slime/blob/3778dbf6d1a533ab478ecf5ddaa11449a47752b2/slime/backends/sglang_utils/deployment.py#L79) → [`resolve_sglang_config`](https://github.com/THUDM/slime/blob/3778dbf6d1a533ab478ecf5ddaa11449a47752b2/slime/backends/sglang_utils/sglang_config.py#L210) → normal / [PD、EPD](https://github.com/THUDM/slime/blob/3778dbf6d1a533ab478ecf5ddaa11449a47752b2/slime/backends/sglang_utils/disaggregation.py#L14) → [`ServerGroup.start_engines`](https://github.com/THUDM/slime/blob/3778dbf6d1a533ab478ecf5ddaa11449a47752b2/slime/backends/sglang_utils/engine_group.py#L63) → [`SGLangEngine.init`](https://github.com/THUDM/slime/blob/3778dbf6d1a533ab478ecf5ddaa11449a47752b2/slime/backends/sglang_utils/sglang_engine.py#L105)；
- [MegatronTrainRayActor](https://github.com/THUDM/slime/blob/3778dbf6d1a533ab478ecf5ddaa11449a47752b2/slime/backends/megatron_utils/actor.py#L56) 承担 actor/critic 的 Megatron 初始化、训练、保存和权重发布；
- [train.py](https://github.com/THUDM/slime/blob/3778dbf6d1a533ab478ecf5ddaa11449a47752b2/train.py#L9) 与 [train_async.py](https://github.com/THUDM/slime/blob/3778dbf6d1a533ab478ecf5ddaa11449a47752b2/train_async.py#L10) 只保留高层控制流。

这条拆分需要按职责理解：`slime/ray/rollout.py` 管“何时生成、回收、offload/onload、恢复与销毁”，`deployment.py` 管“按配置启动哪一种拓扑”，`engine_group.py` 管同构 engine group 的 placement 与 actor 生命周期；[`slime/observability/`](https://github.com/THUDM/slime/tree/3778dbf6d1a533ab478ecf5ddaa11449a47752b2/slime/observability) 只承载日志、指标、trace、profile 和 debug data 工具，不是部署层或 rollout lifecycle manager。

## 核心设计取舍

### 1. 深度集成一条主路径，而非统一所有后端

当前快照的 `--train-backend` 只有 `megatron` 一个合法值，见 [arguments.py](https://github.com/THUDM/slime/blob/3778dbf6d1a533ab478ecf5ddaa11449a47752b2/slime/utils/arguments.py#L1596)；核心 rollout 路径围绕 SGLang 构建。这样可以直接暴露 Megatron 并行参数与 SGLang serving 能力，减少“最小公分母”抽象。代价是：**slime 核心不是一个可随意替换训练/推理后端的通用适配层**。

README 提到的 vime 是基于 slime 数据流、改用 vLLM rollout 的独立生态项目，不等于当前 slime 核心同时原生支持 vLLM，见 [README_zh.md](https://github.com/THUDM/slime/blob/3778dbf6d1a533ab478ecf5ddaa11449a47752b2/README_zh.md#L126)。

### 2. 控制流显式，扩展点下沉到数据路径

顶层循环清楚地写出 generate、train、save、update weights、eval 的顺序。自定义行为主要通过函数路径注入：

- 替换整个 rollout：`--rollout-function-path`；
- 替换单样本生成：`--custom-generate-function-path`；
- 自定义 reward：`--custom-rm-path`；
- 生成后、打分前的轻量样本后处理：`--rollout-sample-hook-path`（可重复传入，[PR #2250](https://github.com/THUDM/slime/pull/2250) 新增）；
- 自定义 advantage、loss、样本到训练数据的转换。

这些参数可从 [arguments.py](https://github.com/THUDM/slime/blob/3778dbf6d1a533ab478ecf5ddaa11449a47752b2/slime/utils/arguments.py#L339) 和 [arguments.py](https://github.com/THUDM/slime/blob/3778dbf6d1a533ab478ecf5ddaa11449a47752b2/slime/utils/arguments.py#L1365) 看到。优点是 agent/tool/verifier 逻辑与训练 kernel 解耦；代价是扩展函数必须遵守 `Sample`、batch 形状、rollout id 等契约，灵活并不意味着没有接口约束。

### 3. 支持 colocate，也支持训推分离

- **colocate**：actor 与 rollout 映射到同一批 GPU，按阶段 offload/onload，节省卡数但无法重叠训练和生成；
- **disaggregated**：训练 GPU 与 rollout GPU 分离，资源更多，但可用 `train_async.py` 做流水重叠。

资源计算在 [placement_group.py](https://github.com/THUDM/slime/blob/3778dbf6d1a533ab478ecf5ddaa11449a47752b2/slime/ray/placement_group.py#L100)：colocate 取两者 GPU 数的最大值，分离模式则相加。它体现的是明确的成本—吞吐取舍，而不是某一种部署永远更优。

### 4. 保留策略新鲜度与吞吐之间的可调空间

同步驱动每轮先生成后训练，并且每轮都同步权重；异步驱动让 batch N+1 的生成与 batch N 的训练重叠，只有该驱动的常规路径才用 `--update-weights-interval` 控制发布频率，见 [train_async.py](https://github.com/THUDM/slime/blob/3778dbf6d1a533ab478ecf5ddaa11449a47752b2/train_async.py#L66)。间隔越大，behavior policy 通常越旧，因此算法侧需要关注 off-policy 程度、importance sampling 或 mask 策略。框架提供机制，但不会替用户证明某个异步配置仍满足特定算法的理论假设。

## 当前快照明确支持什么

以下说法可以从代码或仓库示例直接得到，但“支持”仍应理解为受具体模型、依赖版本和硬件拓扑约束：

- **训练内核**：Megatron；支持其张量、流水、上下文、专家等并行参数透传；
- **rollout 内核**：SGLang server + router，包括核心仓库管理的 engine 或 external SGLang engine；
- **算法入口**：`grpo`、`gspo`、`cispo`、`reinforce_plus_plus`、`reinforce_plus_plus_baseline`、`ppo`，以参数 choices 为准，见 [arguments.py](https://github.com/THUDM/slime/blob/3778dbf6d1a533ab478ecf5ddaa11449a47752b2/slime/utils/arguments.py#L952)；
- **训练损失**：policy loss、SFT loss、自定义 loss，见 [arguments.py](https://github.com/THUDM/slime/blob/3778dbf6d1a533ab478ecf5ddaa11449a47752b2/slime/utils/arguments.py#L926)；
- **PPO critic**：选择 PPO 会设置 `use_critic`；真正训练时才创建 critic，`num_rollout=0` 的 eval-only 路径不会创建，见 [arguments.py](https://github.com/THUDM/slime/blob/3778dbf6d1a533ab478ecf5ddaa11449a47752b2/slime/utils/arguments.py#L1913) 与 [placement_group.py](https://github.com/THUDM/slime/blob/3778dbf6d1a533ab478ecf5ddaa11449a47752b2/slime/ray/placement_group.py#L186)；
- **数据生成**：单轮、多轮、tool/environment、自定义 reward、dynamic sampling、partial rollout 等可通过扩展点表达；
- **执行形态**：同步、batch 级 N/N+1 流水、fully-async rollout；
- **权重更新**：full + NCCL、colocate tensor/CUDA IPC 路径、full/delta + disk；
- **工程能力**：训练 checkpoint、数据源游标保存、eval、debug rollout-only/train-only、profiling、故障恢复等。

模型列表应以当前 README 与可运行 recipe 为准，而不能从“参数透传”推导出所有 Megatron/SGLang 支持的模型都已经端到端验证。当前 README 列出的已支持家族见 [README_zh.md](https://github.com/THUDM/slime/blob/3778dbf6d1a533ab478ecf5ddaa11449a47752b2/README_zh.md#L30)。

## 不应宣称什么

技术面试中，以下表述更准确：

- 不说“支持任意训练后端”：当前 CLI 只接受 Megatron；
- 不说“原生支持所有推理引擎”：核心选择 SGLang，vLLM 路径属于独立生态项目；
- 不说“异步训练等于完全无 staleness”：batch 流水至少有一轮策略滞后，fully-async 的滞后还与队列有关；
- 不说“所有算法都完全 on-policy”：实际取决于同步间隔、生成队列和纠偏配置；
- 不说“所有模型/硬件组合都生产验证”：仓库有广泛 recipe 和 CI，但组合空间远大于测试矩阵；
- 不说“框架替代了 Ray/Megatron/SGLang”：它编排并扩展三者；
- 不说“它是完整 agent platform”：slime 提供 agentic rollout 接口，sandbox、工具服务和业务环境仍可能由用户或外部系统提供；
- 不说“checkpoint 与 rollout 权重同步是一回事”：前者用于持久化恢复，后者用于在线把当前 actor 发布给 serving。

## 与一般 RLHF 框架如何比较

与常见“一体化 trainer + 多后端适配”型 RLHF 框架相比，slime 的差异可概括为：

| 维度 | slime 的倾向 | 相应代价 |
| --- | --- | --- |
| 后端抽象 | 深度优化 Megatron + SGLang | 更换核心后端不是简单配置项 |
| 扩展方式 | 数据生成、reward、loss 等函数路径 | 用户要理解数据契约与分布式语义 |
| 系统规模 | 优先考虑多 GPU、多节点、MoE 和长 rollout | 小模型单机实验的上手成本未必最低 |
| 控制流 | 顶层 Python 循环显式 | 一些高级策略需要用户自行组合 |
| 训推资源 | colocate 与分离均可选 | 配置、显存和同步路径更复杂 |
| 异步能力 | 从 batch 流水到跨 batch 常驻生成池 | 需管理 staleness 与评估限制 |

这不是“全面优于其他框架”的结论。若目标是最少依赖、快速跑一个小模型实验，较轻的 trainer 可能更合适；若已有特定非 SGLang serving 栈，多后端框架可能迁移成本更低。slime 的优势最明显的场景，是团队明确选择 Megatron + SGLang，并愿意为大规模吞吐、拓扑控制和 agentic 数据生成承担更专业的系统配置。

## 面试回答模板

> slime 是一个面向 LLM 在线 RL 的 post-training 编排框架。Ray 负责控制面和 GPU placement，Megatron 负责 actor/critic 训练，SGLang 负责 rollout serving；slime 自己解决数据闭环、训练与生成的同步、权重格式转换/发布、checkpoint 和自定义 rollout 接口。它刻意深度优化 Megatron + SGLang，而不是抽象所有后端。同步模式强调策略新鲜度，分离部署下可用 N/N+1 或 fully-async rollout 提高利用率，但需要接受并管理 staleness。

这个回答同时交代了定位、组件、取舍和边界，比只背“高性能 RL 框架”更有区分度。
