# verl 训练与推理研究

第一期从 [verl #8649](https://github.com/verl-project/verl/issues/8649) 出发，独立复现 PPO/GRPO fused output 在 prompt-heavy batch 上的词表投影开销。先用可手算的 packed batch 证明选对了行，再用合成张量和 Qwen3.5-9B 分别回答两个问题：LM head 节省了多少工作，actor update 最终快了多少？

[Tracking Issue #1](https://github.com/zincsabian/verl/issues/1) · [研究分支](https://github.com/zincsabian/verl/tree/research/rl-fused-output-study) · [源码基线 a4e0dd2](https://github.com/zincsabian/verl/commit/a4e0dd2188e3dd458a2cbb097c03d975ca77f159)

| 文件 | 何时使用 |
| --- | --- |
| [PLAN](verl-8649/PLAN.md) | 查看十周路线、依赖图和 GPU 资源安排 |
| [TODO](verl-8649/TODO.md) | 选择前置任务已验收的节点，逐项勾选并回填证据 |
| [CONTEXT](verl-8649/CONTEXT.md) | 恢复基线、研究边界、当前假设和下一步 |
| [DESIGN](verl-8649/DESIGN.md) | 固定源码定位、张量契约、索引与梯度的验证方法 |
| [EXPERIMENTS](verl-8649/EXPERIMENTS.md) | 登记正确性、微基准、Nsight 和正式训练结果 |
| [SESSION_LOG](verl-8649/SESSION_LOG.md) | 每晚记录提交、测试、假设、结果和次日唯一任务 |
| [PR_CHECKLIST](verl-8649/PR_CHECKLIST.md) | 检查个人 Fork PR、CI、自审和报告的证据 |

每晚从 CONTEXT 和最近一条 SESSION_LOG 开始。每次实验分配一个 Run ID，任务记录只引用提交和 Run ID。原始数据保存在可恢复的位置，仓库中保留复现所需的参数、摘要和路径。
