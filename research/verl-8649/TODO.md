# 任务清单与验收

每个周节点下的两个任务可按晚拆分。勾选验收前，在该节点回填证据并更新 [SESSION_LOG](SESSION_LOG.md)。前置节点、验收、恢复点均绑定本节点；若受阻，写明原因和可以推进的并行节点。

## W1 · 固定 FSDP2 → fused output → PPO/GRPO loss 的调用链

**Dependencies**　建档。　**交付**　DESIGN.md。

- [ ] 在固定 SHA 上从 actor update 找到 engine、model、head 和 loss consumer，逐处记录 symbol 与输入输出。
- [ ] 用两条变长序列手算 next-token 对齐，列出 entropy、mask、SFT、distill、SP 的消费方。
- [ ] **Exit criteria**　DESIGN 有可点击的源码行链接、shape/坐标系和未决问题；能指出第一个 response token 的预测行。

| 节点记录 | 内容 |
| --- | --- |
| Context memory | 开始时从 [CONTEXT](CONTEXT.md) 摘录本节点适用的 SHA、约束与待证假设。 |
| Experiment results | 待执行。填 Run ID、结果位置和通过/失败/阻塞。 |
| Git commit | 待填写实现或证据提交 SHA。 |
| 下一晚的一个动作 | 打开固定 SHA 的 transformer_impl.py，从 fused 分支记录第一个生产者与消费者。 |

## W2 · 构造 packed response index 的 CPU oracle

**Dependencies**　W1。　**交付**　CPU tests、测试命令和结果。

- [ ] 用逐样本 padded+mask 循环作独立参考；记录目标 token、预测行和样本归属。
- [ ] 覆盖左右 padding、1-token response、混合长度、空有效响应、多轮 mask 孔洞和相邻样本边界。
- [ ] **Exit criteria**　索引 type/范围/顺序正确，参考值逐项一致；能用一条测试检测漏掉因果位移或跨样本读取。

| 节点记录 | 内容 |
| --- | --- |
| Context memory | 开始时从 [CONTEXT](CONTEXT.md) 摘录本节点适用的 SHA、约束与待证假设。 |
| Experiment results | 待执行。填 Run ID、结果位置和通过/失败/阻塞。 |
| Git commit | 待填写实现或证据提交 SHA。 |
| 下一晚的一个动作 | 先把两条手算序列写成一个 CPU 测试。 |

## W3 · 测量全量 LM head 前反向基线

**Dependencies**　W1。　**交付**　Baseline Run ID。

- [ ] 固定 seed、H/V、后端、dtype 和总 packed token 数，保存一份可重复生成的合成输入。
- [ ] 预热后分开测 forward、backward、组合和 peak memory；输出原始逐次计时。
- [ ] **Exit criteria**　同一配置可复跑；说明梯度清零、同步位置、样本数和离散度，留下原始数据路径。

| 节点记录 | 内容 |
| --- | --- |
| Context memory | 开始时从 [CONTEXT](CONTEXT.md) 摘录本节点适用的 SHA、约束与待证假设。 |
| Experiment results | 待执行。填 Run ID、结果位置和通过/失败/阻塞。 |
| Git commit | 待填写实现或证据提交 SHA。 |
| 下一晚的一个动作 | 从一个可装入单卡的 shape 测 full baseline。 |

## W4 · 由 engine 构建并传递选择索引

**Dependencies**　W2。　**交付**　Engine 修改与测试提交。

- [ ] 从 W1 的消费方反推启用条件；在 packing 信息仍可用处生成预测行，沿现有 forward 参数传递。
- [ ] 对禁用、元数据缺失及未支持请求保留 full 路径，并观测实际传入的 index。
- [ ] **Exit criteria**　逐样本索引与 CPU oracle 相符；测试验证支持路径收到 index，回退路径沿用原调用。

| 节点记录 | 内容 |
| --- | --- |
| Context memory | 开始时从 [CONTEXT](CONTEXT.md) 摘录本节点适用的 SHA、约束与待证假设。 |
| Experiment results | 待执行。填 Run ID、结果位置和通过/失败/阻塞。 |
| Git commit | 待填写实现或证据提交 SHA。 |
| 下一晚的一个动作 | 为 eligible 和 fallback 各写一个路径断言。 |

## W5 · 验证选择投影与反向传播

**Dependencies**　W3、W4。　**交付**　Correctness Report / Run ID。

- [ ] 共享同一 backbone 输出、权重和上游梯度，只选择 LM head 输入行，按原调用者所需恢复结果。
- [ ] 对照 full oracle 的 logprob、entropy、loss、dH、dW，按 dtype 预先设定阈值；记录最大误差位置。
- [ ] **Exit criteria**　输入输出 shape 与 mask 契约一致；FP32 小例与生产 dtype 对照通过，空选择的 loss/grad 语义有测试。

| 节点记录 | 内容 |
| --- | --- |
| Context memory | 开始时从 [CONTEXT](CONTEXT.md) 摘录本节点适用的 SHA、约束与待证假设。 |
| Experiment results | 待执行。填 Run ID、结果位置和通过/失败/阻塞。 |
| Git commit | 待填写实现或证据提交 SHA。 |
| 下一晚的一个动作 | 先用 FP32 小例同时比较 dH 与 dW。 |

## W6 · 检查后端与回退回归

**Dependencies**　W5。　**交付**　PR candidate、回归矩阵。

- [ ] 核实 Torch/Triton 各自支持的模式和 backward；在相同数据上比较支持路径。
- [ ] 构造 SFT、distillation、sequence parallel、需全量 logits 等用例；明确支持、回退或待测。
- [ ] **Exit criteria**　每个支持项有测试结果；回退可观察且与基线一致；无法运行的组合列明原因。

| 节点记录 | 内容 |
| --- | --- |
| Context memory | 开始时从 [CONTEXT](CONTEXT.md) 摘录本节点适用的 SHA、约束与待证假设。 |
| Experiment results | 待执行。填 Run ID、结果位置和通过/失败/阻塞。 |
| Git commit | 待填写实现或证据提交 SHA。 |
| 下一晚的一个动作 | 先在支持矩阵里填入 Torch 与 Triton 的实际入口。 |

## W7 · 解释 GPU 收益及其上限

**Dependencies**　W3、W6。　**交付**　Benchmark Report、.nsys-rep、.ncu-rep。

- [ ] 固定 T/H/V，对 selected rows/T 做 10%、25%、50%、90%、近全量扫描，计入 gather/scatter。
- [ ] 用 Nsight Systems 看时间线和 launch 数，用 Nsight Compute 看代表 kernel 的计算、带宽或占用。
- [ ] **Exit criteria**　报告配对原始数据、波动、负收益点和 break-even 区间；分开写 LM head 与 actor 范围。

| 节点记录 | 内容 |
| --- | --- |
| Context memory | 开始时从 [CONTEXT](CONTEXT.md) 摘录本节点适用的 SHA、约束与待证假设。 |
| Experiment results | 待执行。填 Run ID、结果位置和通过/失败/阻塞。 |
| Git commit | 待填写实现或证据提交 SHA。 |
| 下一晚的一个动作 | 先跑 prompt-heavy 与近全量两个对照点。 |

## W8 · 在正式模型上验证训练

**Dependencies**　W6。　**交付**　E2E Training Report / Run ID。

- [ ] 固定 Qwen3.5-9B revision、FSDP2 拓扑、rollout 配置和数据；先用同一 rollout batch 做 actor A/B。
- [ ] 跑 vLLM rollout → logprob → loss → backward → optimizer step；区分 actor update 与整轮耗时。
- [ ] **Exit criteria**　9B 有可复现配置、成功更新记录、分 rank 峰值显存与稳态计时；4B smoke 单独标记，9B 阻塞则延长。

| 节点记录 | 内容 |
| --- | --- |
| Context memory | 开始时从 [CONTEXT](CONTEXT.md) 摘录本节点适用的 SHA、约束与待证假设。 |
| Experiment results | 待执行。填 Run ID、结果位置和通过/失败/阻塞。 |
| Git commit | 待填写实现或证据提交 SHA。 |
| 下一晚的一个动作 | 先估算 9B 各组件显存并固定最小可行配置。 |

## W9 · 在独立实现冻结后对比 #8760

**Dependencies**　W7、W8。　**交付**　Fork PR Draft、对照表。

- [ ] 记录独立实现 SHA，再读取 #8760 的 diff、review、测试及版本。
- [ ] 逐项比较索引语义、接口、后端、回退、测试和性能；能对齐硬件/配置时再跑直接 A/B。
- [ ] **Exit criteria**　PR 草稿解释相同与不同决策、可比条件和剩余缺口，链接实现与原始证据。

| 节点记录 | 内容 |
| --- | --- |
| Context memory | 开始时从 [CONTEXT](CONTEXT.md) 摘录本节点适用的 SHA、约束与待证假设。 |
| Experiment results | 待执行。填 Run ID、结果位置和通过/失败/阻塞。 |
| Git commit | 待填写实现或证据提交 SHA。 |
| 下一晚的一个动作 | 先标记独立实现 SHA，再打开上游 Files changed。 |

## W10 · 完成个人 Fork PR、自审与总结

**Dependencies**　W9。　**交付**　Final Research Report、个人 PR。

- [ ] 明确 PR 的 base/head 都在个人 Fork；逐行审查并在固定 head SHA 重跑测试，查看 CI。
- [ ] 汇总训练验证、正负结果、局限、复现步骤和后续可独立贡献的缺口。
- [ ] **Exit criteria**　个人 PR URL、diff、自审、CI 状态、最终报告与可恢复的研发 Memory 齐全；上游合并不是验收条件。

| 节点记录 | 内容 |
| --- | --- |
| Context memory | 开始时从 [CONTEXT](CONTEXT.md) 摘录本节点适用的 SHA、约束与待证假设。 |
| Experiment results | 待执行。填 Run ID、结果位置和通过/失败/阻塞。 |
| Git commit | 待填写实现或证据提交 SHA。 |
| 下一晚的一个动作 | 从 PR_CHECKLIST 的复现入口开始最后一次自审。 |
