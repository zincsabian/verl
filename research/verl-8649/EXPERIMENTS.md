# 实验登记与复现模板

当前尚无本项目的实测结果。每次运行复制适用模板，用 YYYYMMDD-Wn-序号 生成 Run ID。每个数值关联原始样本文件、代码 SHA 和复跑命令。大体积 trace/checkpoint 保存在持久卷，仓库记录可恢复的路径、文件名和摘要。

## 所有实验共用的运行卡

| 字段 | 本次填写 |
| --- | --- |
| Run ID / 任务 / 状态 | … / W… / planned、passed、failed 或 blocked |
| 研究问题与判据 | 待证假设、比较策略、预设容差或计时口径 |
| 代码与依赖 | baseline SHA、candidate SHA、开关、环境锁或镜像、Torch / Triton / Transformers / vLLM / CUDA / driver 版本 |
| 硬件 | GPU 型号/显存/卡数、rank/device mesh、互联、功耗/时钟策略和其他负载 |
| 可复现输入 | seed、模型或 config revision、数据集 revision/样本 ID、dtype、accumulation dtype、shape、温度、entropy 开关 |
| 命令与结果位置 | 精确命令、配置覆盖、工作目录、stdout/stderr、CSV/JSON、trace 路径、运行时间与 GPU 小时 |
| 结论 | 支持/反驳什么，最大限制，失败的首个信号，下一次改变哪一个变量 |

## A · 正确性

同一输入、权重、标签、mask 和上游梯度分别走 full 与 candidate。小规模用高精度参考；生产 dtype 的 atol/rtol 在对照前填写，记录近零分母处理方法。确认是否生成 entropy、输出 shape/dtype/顺序与 loss 归约。

| 张量或条件 | 样本与 dtype | atol/rtol | max abs / max rel / 最差位置 | 非有限数 / 判定 |
| --- | --- | --- | --- | --- |
| 有效 logprob、entropy、标量 loss | … | … | … | … |
| LM head 输入梯度 dH、权重梯度 dW | … | … | … | … |
| 空有效响应、混合长度、稀疏 mask、回退 | … | … | … | … |

## B · GPU microbenchmark

固定 T、H、V、后端与 dtype，改变实际选择的预测行数 K。逐组记录 prompt/response token 数和 K/T，多轮数据另记有效 mask 数。建议先做 10%、25%、50%、90%、近全量；另测 K=0 的定义。预热和编译/autotune 单列。用 CUDA events 或显式同步包围测量区间；前向、反向、组合分开测。记录梯度清零、输入分配、gather/scatter 是否计时，A/B 均用相同口径。每个配置保留逐次值与重复轮数，用中位数和分位数描述波动。

| Run / backend / variant | T,K,H,V / dtype | 预热/计时次数 | Fwd / Bwd / 组合 p50 ms | 组合 p95 ms | peak allocated / reserved MiB | 基线÷候选 |
| --- | --- | --- | --- | --- | --- | --- |
| … / full | … | … | … | … | … | 1.00 |
| … / selected | … | … | … | … | … | … |

显存重置统计的时点、常驻参数/梯度与增量峰值写在表下；OOM 同样保留 shape、失败阶段和可复跑命令。

## C · Nsight Systems / Nsight Compute

先保存无 profiler 的配对计时。在另一轮用相同配置采样。系统时间线标注 actor、backbone、head、gather/scatter、通信与同步边界。Compute 单独抓代表 kernel，记录过滤规则、replay 与采样设置。

| Run / 工具版本 | 完整采集命令 / NVTX 或 kernel filter | 窗口 / rank | 指标与观察 | 原始报告 |
| --- | --- | --- | --- | --- |
| … / nsys | … | 稳态步、head/actor | launch count、GPU busy、kernel 总时长、CPU gap、NCCL 重叠 | … .nsys-rep |
| … / ncu | … | full 与 selected 各一代表 shape | kernel 名称、延迟、SM/DRAM 吞吐、occupancy、主要 stall；保留单位 | … .ncu-rep |

结论填写：哪项开销减少、哪项新增；响应占比提高时何处收益消失；head 优化对 actor 实际贡献多少。若采样权限或计数器不可用，标明缺失项。

## D · Qwen3.5-9B 正式训练

固定模型和 tokenizer revision、数据、算法与精度。首先重放同一 rollout batch 隔离 actor update，再跑完整 rollout → 更新闭环。记录 FSDP2 mesh/sharding/offload、vLLM TP 与显存配额、batch/microbatch、累积步数、prompt/response 长度、生成参数、seed 和 warmup。配对计时用相同有效 token 量，记录所有 rank 中的最大显存。

| Run / 模型 / variant | 更新步数 / 有效 token | actor update p50/p95 s | rollout / 整轮 s | peak GPU MiB / rank | loss、KL、entropy、grad norm、非有限数 |
| --- | --- | --- | --- | --- | --- |
| … / 9B / full | … | … | … | … | … |
| … / 9B / selected | … | … | … | … | … |

另附训练配置、命令、成功 optimizer step 的日志、耗时分布和持久化位置。在线 rollout 的样本差异要单列。若 9B 遇到 OOM 或兼容性问题，记录失败配置、峰值与最小复现；4B smoke 单设 Run ID，9B 验收保持待办。
