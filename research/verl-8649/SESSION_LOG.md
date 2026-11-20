# 每晚研发日志

最新记录放在顶部。优先写 SHA、测试及 Run ID，下一晚只留一个主要动作。研究代码 SHA 与记录这份日志的文档提交可以不同。

## 2026-11-20 · W1 首次只读追踪

| 项目 | 记录 |
| --- | --- |
| 今天完成 | 核实 engine 的全局 roll、Qwen3.5 Torch/Triton head 的输入、jagged 输出契约和 response 窗口。源码 permalink 见 [DESIGN](DESIGN.md)。 |
| 当前 Git commit | 读取的训练代码固定在 [a4e0dd2](https://github.com/zincsabian/verl/commit/a4e0dd2188e3dd458a2cbb097c03d975ca77f159)。 |
| 测试通过/失败 | 已逐份对照 8 份远端 Markdown 原文。CPU/GPU 未运行；[2,3,6] 为手算例。 |
| 当前技术假设 | 可以跳过 prompt 投影；是否能跳过多轮 response 窗口中的 mask 孔洞仍待查。 |
| 实验结果位置 | 本次只有源码证据，尚无 GPU Run ID。 |
| 下一晚唯一主要任务 | 从 policy loss 中找到 no_padding_2_padding 的实际调用者，分别标出 logprob、entropy 在 response_mask 之前的读取点，保存调用者的行号链接。 |

## 2026-11-20 · 建立研究分支和追踪入口

| 项目 | 记录 |
| --- | --- |
| 今天完成 | 从 Fork main 创建研究分支，启用个人 Fork Issues 并创建 [Tracking Issue #1](https://github.com/zincsabian/verl/issues/1)；建立十周计划、逐周任务、设计与实验记录。 |
| 当前 Git commit | 本轮研究代码基线为 [a4e0dd2188e3dd458a2cbb097c03d975ca77f159](https://github.com/zincsabian/verl/commit/a4e0dd2188e3dd458a2cbb097c03d975ca77f159)。建档提交见[研究分支历史](https://github.com/zincsabian/verl/commits/research/rl-fused-output-study/)。 |
| 测试通过/失败 | 本轮为文档建档；训练、CPU 正确性和 GPU 性能均未执行。 |
| 当前假设 | 预测首个 response token 需要其前一行 hidden；需核对 packed/padded 转换及真正的消费边界。 |
| 实验结果位置 | 尚无 Run ID；登记模板见 [EXPERIMENTS](EXPERIMENTS.md)。 |
| 下一晚唯一任务 | 在固定基线定位 FSDP fused forward 的第一个生产者与消费者，保存源码 permalink。 |

## 复制此模板

### YYYY-MM-DD · Wn · 一句话任务

| 字段 | 内容 |
| --- | --- |
| 今天完成什么 | 改动、结论及证据链接 |
| 当前 Git commit | 源码基线、当前实现 SHA；若有未提交改动写明范围 |
| 测试通过或失败 | 命令、环境、通过/失败/跳过数、首个错误；没跑哪项及原因 |
| 当前技术假设 | 对应证据、可能推翻它的观察 |
| 实验结果位置 | Run ID、命令、原始日志、报告或持久卷相对路径 |
| 下一晚唯一主要任务 | 前置条件、打开哪个文件/运行哪个用例、结束时看到什么 |
