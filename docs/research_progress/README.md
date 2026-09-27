# 研究进度与验证排查记录

集中保存研究进度快照、验证失败分析和受控复现结论。每份报告反映记录时的状态，不代表实时作业状态。

## Causal tail 研究

- [2026-09-27：对照 PLAN 的研究进度](CAUSAL_PLAN_PROGRESS_20260927.md)
- [119625：Ray 节点退出与代码评分 MemoryError 排查](CAUSAL_VALIDATION_119625_INVESTIGATION.md)
- [120031：检查点验证失败、资源分析及复现记录](CAUSAL_VALIDATION_120031_RECHECK.md)
- [LiveCodeBench 第 206–213 题：受控复现记录](LCB_206_213_INVESTIGATION.md)

实验设计与运行说明仍位于 [PLAN.md](../../experiments/causal/PLAN.md) 和 [实验 README](../../experiments/causal/README.md)。

## 相关持续学习故障排查

- [120036：Tooluse 训练阶段 Ray 心跳超时分析](120036_FAILURE_DIAGNOSIS.md)

120036 属于独立的 SDPO-SVC 持续学习实验，不计入 causal tail 的研究进度。

临时复现脚本和原始诊断材料保留在本地 `.runtime/diagnostics/` 或报告注明的 manager 路径；`.runtime/` 不纳入版本控制。报告中的历史配置与当前脚本可能不同。
