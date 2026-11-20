# 设计笔记 · PPO 响应位置词表投影

**状态**　待验证的研究设计。固定 [基线 a4e0dd2](https://github.com/zincsabian/verl/commit/a4e0dd2188e3dd458a2cbb097c03d975ca77f159)。源码发现按「符号、行号、输入/输出、消费者」补在此页；测量写入 [EXPERIMENTS](EXPERIMENTS.md)。

## 一组手算样本

约定每条样本从位置 0 开始，hidden 行 t 预测同一样本的 token t+1。两条无 padding 序列 A=[p0,p1,p2,r0,r1]、B=[q0,q1,s0] 串接后的起点是 0 和 5。响应目标的 packed 行分别是 [3,4] 和 [7]；预测这些 token 的 hidden 行应为 [2,3] 和 [6]。用 A 的末行 4 预测 B 的首 token 是跨样本错误。

单轮、连续响应且 prompt 长度 P≥1 时，样本起点 O、响应长度 R 对应预测区间 [O+P−1, O+P+R−1)。遇到空响应、无前驱 token、多轮工具输出、左右 padding 或稀疏 mask 时，从真实消费语义逐项推导；W2 的 reference 用逐样本枚举，不复用候选的索引生成逻辑。

## 张量契约与验证

| 边界 | 候选行为 | 要从源码或测试确认 |
| --- | --- | --- |
| 输入与 backbone | 完整运行所有 packed token，记最后一层输出为 H，形状示意 [T,D] | 真实 batch/packed 布局、offset、SP 切分和 dummy 行。 |
| LM head | 仅 gather 下游需要的 K 个预测行，与同一 W[V,D] 计算；labels 来自对应目标 token | 温度、dtype、entropy 开关、weight tying 和实际 backend；不能依据共享参数推断其总梯度来源。 |
| 输出消费 | 遵守现有 shape、顺序、dtype、padding 和有效位语义 | 谁需要完整张量或全量 logits？哪些位置会先记录统计再 mask？无消费者的位置如何表示？ |
| backward | 固定相同上游梯度，对比选中行 dH 和全量 dW；对 LM head 这一支，未选 hidden 行应收到零梯度 | 加入 entropy loss 与不同归约，检查梯度累积、空选择和重复 index 的语义。backbone 仍可把梯度传回 prompt。 |
| 回退 | 对未经验证的 SFT、distillation、sequence parallel 和全量输出请求走原路径 | 测试实际分支与基线输出；识别全局 packed 行到 rank 本地行的转换位置。 |

## W1 源码追踪起点

- [verl/workers/engine/fsdp/transformer_impl.py](https://github.com/zincsabian/verl/blob/a4e0dd2188e3dd458a2cbb097c03d975ca77f159/verl/workers/engine/fsdp/transformer_impl.py)
- [verl/workers/utils/padding.py](https://github.com/zincsabian/verl/blob/a4e0dd2188e3dd458a2cbb097c03d975ca77f159/verl/workers/utils/padding.py)
- [verl/models/transformers/qwen3_5.py](https://github.com/zincsabian/verl/blob/a4e0dd2188e3dd458a2cbb097c03d975ca77f159/verl/models/transformers/qwen3_5.py)
- [verl/utils/experimental/torch_functional.py](https://github.com/zincsabian/verl/blob/a4e0dd2188e3dd458a2cbb097c03d975ca77f159/verl/utils/experimental/torch_functional.py)
- [verl/utils/kernel/linear_cross_entropy.py](https://github.com/zincsabian/verl/blob/a4e0dd2188e3dd458a2cbb097c03d975ca77f159/verl/utils/kernel/linear_cross_entropy.py)

抓取一条 actor update，从 engine 输入一直跟到 PPO/GRPO loss 的读取点。逐处回填实际的文件与 symbol；同名函数或历史实现要与当前运行配置核对。

## W1 首次源码追踪 · 2026-11-20

基线为 a4e0dd2。下表是代码阅读结果。真实训练配置命中哪个分支、数值误差和性能仍需实测。

| 边界 | 已核实的行为 | 固定版本的源码 |
| --- | --- | --- |
| FSDP engine → 模型 | remove-padding 路径将 input_ids.values() 整理为 [1,T]，全局 roll(-1) 构造 labels；fused 分支显式传入 shift_labels。SP 时全局 roll 先于切分。 | [打包和 roll](https://github.com/zincsabian/verl/blob/a4e0dd2188e3dd458a2cbb097c03d975ca77f159/verl/workers/engine/fsdp/transformer_impl.py#L1134-L1155) · [传入 labels](https://github.com/zincsabian/verl/blob/a4e0dd2188e3dd458a2cbb097c03d975ca77f159/verl/workers/engine/fsdp/transformer_impl.py#L1269-L1286) |
| Qwen3.5 backbone → LM head | 普通 Torch fused 分支把 outputs[0] 的 hidden_states、lm_head.weight 和 rolled_labels 交给 FusedLinearForPPO，调用前没有选取响应行。top-k 另有分支。Triton 的 linear_cross_entropy 同样接收 hidden_states。 | [Torch backbone](https://github.com/zincsabian/verl/blob/a4e0dd2188e3dd458a2cbb097c03d975ca77f159/verl/models/transformers/qwen3_5.py#L596-L610) · [权重、top-k 与普通 head](https://github.com/zincsabian/verl/blob/a4e0dd2188e3dd458a2cbb097c03d975ca77f159/verl/models/transformers/qwen3_5.py#L614-L655) · [Triton head](https://github.com/zincsabian/verl/blob/a4e0dd2188e3dd458a2cbb097c03d975ca77f159/verl/models/transformers/qwen3_5.py#L696-L719) |
| 模型 → engine → loss callable | engine 取 log_probs、按需取 entropy，gather/unpad 后按原 input_ids.offsets() 重建 jagged tensor，再传给 loss_function。直接返回长度 K 的向量会破坏这处边界。 | [取输出](https://github.com/zincsabian/verl/blob/a4e0dd2188e3dd458a2cbb097c03d975ca77f159/verl/workers/engine/fsdp/transformer_impl.py#L1318-L1330) · [恢复 jagged 布局](https://github.com/zincsabian/verl/blob/a4e0dd2188e3dd458a2cbb097c03d975ca77f159/verl/workers/engine/fsdp/transformer_impl.py#L1400-L1415) · [实际调用](https://github.com/zincsabian/verl/blob/a4e0dd2188e3dd458a2cbb097c03d975ca77f159/verl/workers/engine/fsdp/transformer_impl.py#L1564-L1578) |
| packed 输出 → padded responses | no_padding_2_padding 从嵌套 offsets 或 attention_mask 求长度，检查总行数和非空 prompt。若 E 是样本结束偏移、R 是真实响应长度，读取半开区间 [E−R−1,E−1)，再右填充到响应宽度。此处未使用 response_mask 的逐元素取值。 | [长度和切片](https://github.com/zincsabian/verl/blob/a4e0dd2188e3dd458a2cbb097c03d975ca77f159/verl/workers/utils/padding.py#L119-L147) |

**对 W2 的约束**　先用“整个连续 response 窗口”的参考值做 oracle。多轮 response_mask 中为 0 的行也可能出现在该窗口；能否进一步跳过它们取决于后续消费者，不能仅凭 loss mask 决定。上面的 A/B 例子按源码公式恰好得到 [2,3,6]。

**下一晚唯一任务**　从 policy loss 中找到 no_padding_2_padding 的实际调用者，分别标出 logprob、entropy 在 response_mask 之前的读取点，保存调用者的行号链接。

## 设计决策记录

每项写明「日期 / SHA / 决策 / 曾考虑的方案 / 证据 / 何时重新评估」。先冻结自己的实现与测试，再打开 #8760 的 diff。
