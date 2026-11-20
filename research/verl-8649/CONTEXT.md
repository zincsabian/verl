# 研究状态 · verl #8649

## 恢复入口

| 项目 | 已知状态 |
| --- | --- |
| 工作区 | [个人研究分支](https://github.com/zincsabian/verl/tree/research/rl-fused-output-study)；周进度见 [Tracking Issue #1](https://github.com/zincsabian/verl/issues/1) |
| 对照基线 | Fork main 在建档时为 [a4e0dd2188e3dd458a2cbb097c03d975ca77f159](https://github.com/zincsabian/verl/commit/a4e0dd2188e3dd458a2cbb097c03d975ca77f159)。研究分支由该 SHA 创建。当时网页显示它落后上游 main 4 个提交。 |
| 主问题 | [#8649](https://github.com/verl-project/verl/issues/8649) 提出对下游消费的预测行执行词表投影。上游 Issue 页在 2026-11-20 将关联的 [PR #8760](https://github.com/verl-project/verl/pull/8760) 标为 Open。W9 重新核对状态与 SHA 后比较实现。 |
| 已有证据 | Issue 报告的时间来自 Ascend NPU。GPU 上 head 的占比、gather/scatter 成本和 actor 端收益均待本项目实测。 |
| 验证层级 | 手算/CPU oracle → 合成 LM head 前反向 → Torch/Triton 与回退 → Qwen3.5-9B FSDP2 + vLLM rollout。 |
| 研究节奏 | 每晚约 1 小时。先读最近一条 [SESSION_LOG](SESSION_LOG.md)，用 [TODO](TODO.md) 找到已就绪的一个任务。 |

## 当前待证假设

1. PPO 的消费点可从样本内 next-token 对齐和 packing 元数据完整恢复。要逐项核实 loss 之外的 logprob/entropy 消费方，以及多轮 mask 中的孔洞。
2. 对同一上游梯度，选择投影与 full 路径对有效位置给出数值等价的输出和梯度。LM head 输入的局部梯度与经过 backbone 后的 prompt 梯度要分别检查。
3. prompt 占比越高，可去掉的投影行越多；实际收益还取决于 head 的原始占比、选行和恢复输出的开销、后端以及并行配置。

## 下次恢复

W1 已核实 Qwen3.5 普通 fused head 收到完整 hidden_states，engine 按完整 offsets 重建输出，响应切片读取 [E−R−1,E−1)。[源码证据](DESIGN.md) 解释了首个响应为何对应最后一个 prompt 行。

下一晚唯一任务：从 policy loss 中找到 no_padding_2_padding 的实际调用者，分别标出 logprob、entropy 在 response_mask 之前的读取点，保存调用者的行号链接。 近期运行记录见 [SESSION_LOG](SESSION_LOG.md)。

## 第二期候选

[#8376](https://github.com/verl-project/verl/issues/8376) 关注高置信度 logprob 与梯度稳定性。启动前逐路径核对 [#8377](https://github.com/verl-project/verl/pull/8377) 和 [#8378](https://github.com/verl-project/verl/pull/8378)，用独立输入重现剩余缺口后再选题。
