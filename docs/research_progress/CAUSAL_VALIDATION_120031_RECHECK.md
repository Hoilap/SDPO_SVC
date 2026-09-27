# 作业 120031 再次失败：2026-09-24 排查记录

## 当前结论

直接故障已确认：Raylet 的事件循环出现长时间延迟，GCS 连续五次健康检查超时，将节点判死，随后 Raylet 退出并导致 TaskRunner 的 `ActorDiedError`。

**尚未确认造成停顿的根因。此次不能归因为 LiveCodeBench 评分器，也没有内存耗尽的证据。** 第 93 步是训练步数；本次失败前只开始了 71 个验证批次，尚未进入代码题。

## 作业及代码

- 节点：`5500-node07` / `192.168.10.29`。
- 作业：`120031`，2026-09-23 17:06:16 开始，2026-09-24 01:13:26 结束，`FAILED / 1:0`。
- W&B：<https://wandb.ai/20040817dkn-facebook/SDPO/runs/1btekpmb>。
- W&B 元数据中的启动提交：`68479fd3964ee8ddaef407b24bfe5f52d833dabf`，包含上次评分器修复。
- 检查点：`outputs/causal_tail_seed1/118343/checkpoints/tail-seed1-SDPO-Math/global_step_93`，四个 rank 均记录模型保存成功。

远端路径均相对 `/share/home/dengkn/SDPO_SVC`。

## 失败位置

`logs/tail-causal-120031.out` 中：

- `test_gen_batch meta info:` 出现 71 次。
- `validation generation end` 出现 70 次。
- 未出现 `MemoryError`。
- 验证集共 1660 条，过滤后 1659 条，与上次一致；每批 8 题，每题采样 16 个回答。

按原验证顺序，AIME24 的 30 条、AIME25 的 30 条和 MATH500 的 500 条共占前 560 条，即前 70 批。第 71 批对应 **sciknoweval biology 的第 1–8 题**。这一定位依赖原数据顺序及先前确认的过滤位置（唯一过滤项在后面的 GPQA），不是日志直接打印出的题号。

最后打印的是第 71 批生成开始，未打印生成结束，因此更准确的表述是“生成调用尚未返回时发生节点失联”，不能称为“第 71 批评分失败”，也不能据此认定这 8 题本身触发故障。

W&B 本地二进制记录只保存到第 70 批开始，少于 Slurm 日志；本地资源记录只到 01:08:30。后续资源数据由 W&B API 的 `stream='system'` 补齐，不能把本地记录终点当作崩溃时间。

## 时间线与资源证据

| 时间（北京时间） | 证据 |
| --- | --- |
| 01:07:12 | W&B 本地输出记录第 70 批生成开始 |
| 01:08:30 | 节点内存使用率 64.6%，可用 182627.66 MiB |
| 01:09:30 | 使用率升至 69.96%，可用 154741.49 MiB |
| 01:11:26.681 | GCS 第一次健康检查超时，剩余 4 次 |
| 01:11:39.682–01:12:18.685 | 后续四次健康检查均超时 |
| 01:12:00 | 四张 GPU 利用率降至约 6%，显存占用仍约 63.7% |
| 01:12:18.685 | GCS 判定节点死亡 |
| 01:12:30 | GPU 利用率为 0；节点仍有 151026.55 MiB（约 147.5 GiB）可用内存 |
| 01:12:43 | Raylet 事件统计显示周期回调最大排队约 124.8 秒 |
| 01:12:46.040 | Raylet 因 GCS 已判死而致命退出 |

补充：

- TaskRunner RSS 约 1.5 GiB，未随节点内存变化明显增加；它不代表所有子进程之和。
- Plasma 对象存储仅使用约 `0.034 / 87.234 GB`，没有待创建对象或恢复对象，未见对象存储满载。
- `MemoryMonitor.CheckIsMemoryUsageAboveThreshold` 的 **Queueing time max** 达 124688.90 ms。这是回调等待执行的时间，不能解释成内存检测函数本身运行了 125 秒。
- 同时其他周期回调也有约 125 秒的排队；GCS 自身也出现约 36 秒排队延迟。
- 当前可读的内核日志没有此次故障时段的 OOM 或 GPU Xid 记录。不能据此绝对排除内核问题。

## I/O 线索及其限制

CPU 诊断任务 `120532`、`120549` 均完成，退出码为 0。

- `sar` 的 01:10–01:20 区间平均 I/O wait 为 9.64%，此前约 0.02–0.05%；01:20 时有 6 个 blocked 任务。
- 同一区间重大缺页约 333 次/秒，后台页回收约 5659 页/秒。
- swap-in 约 0.02 页/秒、swap-out 为 0，没有持续交换抖动证据。
- 本地磁盘 sda 的该区间平均利用率只有 0.23%，不足以认定本地磁盘满载。
- `/tmp` 位于本地 XFS；`/share` 是 Lustre。内核有 9 月 23 日 13:54 的 Lustre 超时记录，但**不是本次故障时间**，不能作为本次根因的直接证据。
- 01:10–01:20 的采样窗口还覆盖原作业退出及后续其他作业启动，不能把整个区间的 I/O/缺页变化都归到本次崩溃之前。

当前应优先调查节点级或进程级停顿、I/O 等待、调度和内存局部压力。现有数据不足以指认某个其他作业、Lustre 故障或 Ray 内部缺陷。

## 检查点复现

独立脚本位于本地及远端 `.runtime/diagnostics/120031-recheck/`：

- `reproduce.py`：读取原 W&B 完整配置，恢复原第 93 步模型与 extra state，仅运行验证。
- `reproduce.sbatch`：原节点、4 GPU、24 CPU、原内存申请，最长 4 小时；评分器仍为 8 并发。
- 保留原数据顺序、batch size、采样数；不修改健康检查阈值。
- 使用独立输出目录和 Ray 临时目录，W&B 离线记录；每 10 秒采集 `/proc/meminfo`、`/proc/vmstat`、负载及本用户进程 RSS/状态/等待位置。
- 正常退出或程序报错时归档 Ray 日志、资源采样和验证输出；强制 SIGKILL 无法保证 EXIT trap 执行，节点 `/tmp/sdpo-repro-<jobid>` 是额外取证位置。
- 复现不训练、不删除原检查点、不覆盖原评估结果。采样随机性及从检查点恢复的运行状态差异意味着不能保证逐 token 重现原回答。

本地语法检查及远端配置解析预检均通过。已提交 **120550**；提交本身不代表已复现，需检查调度和运行结果。提交时原节点 8 张 GPU 中有 6 张占用，4 卡复现存在资源等待。提交后的状态查询遇到 SSH 超时，因此本记录尚未确认复现作业已启动，也没有复现结果。

```bash
ssh -F ~/.ssh/config manager 'squeue -j 120550 -o "%.18i %.24j %.12T %.30R"'
ssh -F ~/.ssh/config manager 'tail -n 80 /share/home/dengkn/SDPO_SVC/logs/repro-120031-120550.err'
ssh -F ~/.ssh/config manager 'tail -n 80 /share/home/dengkn/SDPO_SVC/logs/repro-120031-120550.out'
```

若相同模型和验证顺序通过，应继续对照节点压力与原失联时段，而不是宣称问题已修复。若再次失败，优先对齐健康检查第一次超时、进程等待位置、页回收和可用内存的秒级时间线。

## 原始材料

- `logs/tail-causal-120031.{out,err}`。
- `outputs/causal_tail_seed1/118343/wandb/wandb/run-20260923_172853-1btekpmb/`。
- `.runtime/diagnostics/120031-recheck/{gcs_server.out,raylet.out,kernel-tail.txt}`。
- `logs/inspect-120031-120532.out`、`logs/io-120031-120549.out`。
- `.runtime/diagnostics/120031-recheck/preflight/resolved-config.yaml`。

本次只添加诊断材料和复现任务，未更改或推送生产代码。

## 后续对比：是否越来越不稳定

逐轮读取 W&B 配置和 requirements，确认 118343（GRPO、SDPO）、119101、119134、119625、120031 均采用 async vLLM，验证 batch=8、每题 n=16、TP=2、GPU memory utilization=0.55、agent workers=8、dataloader workers=8、最大回答长度 8192。

这些运行记录的依赖版本均为 Ray 2.53.0、PyTorch 2.6.0+cu124、Transformers 4.57.1、vLLM 0.8.5。当前没有升级这些依赖或增加推理批量导致退化的证据。

| 作业 | 实际故障 |
| --- | --- |
| 118343 的 SDPO 阶段 | 208 批生成完成，26544 个回答的指标已写入；随后保存检查点时报 Ray OutOfMemoryError，评分期间也有 MemoryError |
| 119101 | 10/93 步时取消；同名归档目录存的是更早的 118343 结果 |
| 119134 | 全部生成结束后，在 `_dump_generations` 的 `"\n".join(lines)` 写出路径发生 MemoryError |
| 119625 | 190 批开始、189 批结束后，Ray 节点健康检查超时；较早有评分 MemoryError |
| 120031 | 71 批开始、70 批结束，未进入代码题，Ray 健康检查超时；无对应评分 MemoryError |

因此这是不同阶段的故障，不能据此认定单一问题随重试不断恶化。已关闭验证明细整体导出，且将保存检查点提前至验证前；后者改变了验证入口时的内存状态，不能仅凭静态配置相同就认定各轮运行状态完全相同。评分器修复不涉及本次尚未进入的代码评分路径，不能解释所有节点失联。

### 新确认的 Slurm 内存调度问题

`scontrol show config` 返回 `SelectType=select/cons_tres`、`SelectTypeParameters=CR_CORE`。manager 上 cgroup 配置启用了 `ConstrainRAMSpace=yes`、`ConstrainSwapSpace=yes`，但限制单个作业使用上限不等于为它预留独占内存。

官方说明：采用 `CR_Core_Memory` 才同时把核心和内存作为可消耗资源调度，见 <https://slurm.schedmd.com/cons_tres.html>。

实际历史重叠也说明不能将 `--mem=460000` 解读为独占约 460 GB：118343 与 119010 在 9 月 20 日 00:08–06:17 同节点运行，两者均申请 460000M，超过节点配置的 510000M；120031 与申请 200G 的 119222 同节点运行。

这提供了共享节点压力不受内存请求总量约束的证据，但尚不能归因最近一次 Ray 停顿。早期 118343 同样共享节点；不能简单归咎于后来移除 `--exclusive`。

### 复现实验修正

120550 最新检查仍在排队（Priority），尚无复现结论。检查诊断脚本时发现它曾将 `validation_dump_generations` 设为 True，已在运行前恢复为 False，与 120031 一致，避免重新引入 119134 的导出内存错误。仍保留独立 W&B 离线监控、每 10 秒的进程/内存采样、Ray 日志和最终验证指标。

下一步应使用同一已保存检查点做只验证实验，并在资源允许时与独占节点或另一节点对照。当前 gpu_chen 分区只有 5500-node07，换节点需要另一个可用分区/节点授权；没有擅自修改生产作业的资源或集群配置。

### 用户授权迁移到 gpu_ai

按用户要求，已取消 gpu_chen 上仍在排队的 120550（sacct 确认 CANCELLED），使用 `reproduce_gpu_ai.sbatch` 新建复现任务 **120674**。新脚本只修改分区为 gpu_ai 并移除 5500-node07 的节点限制，保留 4 GPU、24 CPU、460000M 内存请求、4 小时时限、原检查点和资源监控。新日志为 `logs/repro-120031-120674.{out,err}`，归档目录为 `.runtime/diagnostics/120031-recheck/repro-120674`。此迁移没有修改 120032–120035 的生产依赖链。
