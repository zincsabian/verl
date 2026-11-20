# 个人 Fork PR 与最终验收

个人 PR 的 base 和 head 都选择 zincsabian/verl。打开 PR 后记录 URL、base SHA、head SHA 和审查日期。用最后一次验证过的 head 写报告。

## 独立性与实现

- [ ] 保存复现输入、基线 SHA、问题边界与个人设计的时间线。
- [ ] 固定独立实现 SHA，再比较 [上游 #8760](https://github.com/verl-project/verl/pull/8760) 的接口、索引、数值、回退与实测环境。
- [ ] 逐行检查 dispatch 默认值、dtype/device、padding、跨样本位移、空选择、SP 切分、额外内存和 autograd。
- [ ] 人工解释每一处必要改动。若考虑上游贡献，按当时的贡献指南核对重复工作，说明独立增量、测试与 AI 辅助情况。

## 正确性与性能证据

- [ ] CPU oracle 能检出 off-by-one 与跨样本错误；覆盖短响应、空有效行和稀疏 mask。
- [ ] 正常入口覆盖 forward/backward，logprob、entropy、loss、dH 和 dW 有原始误差与预设容差。
- [ ] Torch/Triton、SFT、distillation、SP 及全量输出请求有支持/回退测试矩阵。
- [ ] 同卡配对 benchmark 包含不同响应占比、前反向时延、峰值显存和原始重复测量。
- [ ] Nsight Systems/Compute 报告解释 kernel 数、瓶颈及小 shape 或近全量时的限制。
- [ ] Qwen3.5-9B FSDP2 + vLLM 完成训练更新，actor 和整轮用时分别记录；4B smoke 单独标识。

## 审查与报告

- [ ] 个人 PR 描述给出问题、设计、复现命令、正确性、性能与限制；链接 [Tracking Issue](https://github.com/zincsabian/verl/issues/1) 和所有报告。
- [ ] 保存自审中的问题、修正提交、CI 链接以及实际状态。跳过/不可用项写明原因和替代验证。
- [ ] 最终报告给出所测硬件、收益范围、退化点、可推广条件及未解决问题，并把有效研发状态回写 CONTEXT / SESSION_LOG。

| 最终证据 | URL / SHA / Run ID |
| --- | --- |
| 个人 PR / self-review / CI | 待 W10 |
| 正确性 / benchmark / profiler / 9B 训练 | 待 W5–W8 |
| 最终技术报告 / 尚待复查事项 | 待 W10 |
