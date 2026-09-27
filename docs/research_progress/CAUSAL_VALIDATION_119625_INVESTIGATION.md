# Causal 验证任务 119625：Ray 节点退出与代码评分 MemoryError 排查记录

本文记录 2026 年 9 月 23 日对任务 `119625` 的排查、数据集定位、资源监控分析、受控复现和修复过程。所有时间均为北京时间（UTC+8）；题号和验证批次从 1 开始计数，代码中的数组索引从 0 开始。

## 1. 结论与证据边界

这次排查确认了两个问题，但尚未证明它们之间存在直接因果关系：

1. **训练退出的直接原因**：GCS 连续检测不到节点的健康响应，将节点标记为死亡；raylet 随后主动触发致命错误退出，最终出现 `ActorDiedError`。
2. **评分器中可复现的问题**：评测子进程继承较大的虚拟地址空间后，再设置绝对 `1 GiB` 的地址空间上限，可以让一个只执行 `print(0)` 的程序在回传结果时出现 `SEND ERROR: MemoryError`。这不需要整机内存耗尽。

尚未确定的部分：

- 原始训练中是哪一道题、哪个模型答案触发了每条 `MemoryError`。
- 原始错误是否就是上述内存限制机制导致。
- 故障前整机可用内存下降来自哪个进程或作业。
- 评分器的错误是否导致了最终 Ray 心跳超时。

第 184 批的集中评分报错与最终第 190 批期间的节点退出，必须分别分析。修复评分器不等于已经证明修复了节点退出的全部根因。

## 2. 环境与证据文件

| 项目 | 值 |
|---|---|
| 本地仓库 | `/home/dengkn/SDPO-CL` |
| manager 仓库 | `/share/home/dengkn/SDPO_SVC` |
| 原始训练作业 | `119625` |
| 计算节点 | `5500-node07` |
| Ray 节点 IP | `192.168.10.29` |
| TaskRunner PID | `34756` |
| raylet PID | `33205` |
| GCS PID | `33014` |
| W&B run | `20040817dkn-facebook/SDPO/i0k8rlmj` |
| Python 环境 | `/share/home/dengkn/miniconda3/envs/sdpo` |

主要证据均位于 manager 上：

```text
/share/home/dengkn/SDPO_SVC/logs/tail-causal-119625.out
/share/home/dengkn/SDPO_SVC/logs/tail-causal-119625.err

/share/home/dengkn/SDPO_SVC/.runtime/diagnostics/119625-120027/
  collection-time.txt
  kernel-window.log
  session_2026-09-22_11-23-46_891912_32776/
    gcs_server.out
    gcs_server.err
    raylet.out
    raylet.err
    debug_state.txt
    python-core-worker-8d2dc45646086b7ed8cfc517d2a6dc563d94d5c61a0dd13f8a9af872_34756.log
    worker-8d2dc45646086b7ed8cfc517d2a6dc563d94d5c61a0dd13f8a9af872-01000000-34756.err

/share/home/dengkn/SDPO_SVC/outputs/causal_tail_seed1/118343/wandb/wandb/
  run-20260922_113059-i0k8rlmj/
    run-i0k8rlmj.wandb
    files/config.yaml
    files/output.log
    logs/debug-internal.log
```

输出目录中的 `118343` 是复用的路径编号，不代表本次失败的 Slurm 作业编号。当前调查对象是 `119625`。

## 3. 从 ActorDiedError 追到节点健康检查失败

最初的 Python 堆栈停在：

```text
ray.get(runner.run.remote(config))
ray.exceptions.ActorDiedError
Owner's node has crashed.
```

这只能说明 Actor 所属节点已被 Ray 判定失效，不能直接判断 GPU 显存不足、系统 OOM 或应用代码异常。

检查 GCS 和 raylet 日志后得到以下顺序：

| 时间 | 事件 |
|---|---|
| 9 月 22 日 19:45:18—19:45:57 | 连续 4 次健康检查超时，剩余次数降至 1，之后恢复 |
| 9 月 23 日 00:24:49.436 | 健康检查超时，剩余 4 次 |
| 00:25:05.535 | 剩余 3 次 |
| 00:25:22.021 | 剩余 2 次 |
| 00:25:46.827 | 剩余 1 次 |
| 00:25:51.642 | 剩余 0 次，GCS 判定节点死亡 |
| 00:25:54.679 | raylet 因收到判死结果而触发致命错误退出 |
| 00:26:27 | Slurm 作业结束，状态 `FAILED / 1:0` |

关键原文：

```text
Health check failed ... status message Deadline Exceeded
Node is dead because the health check failed.
GCS consider this node to be dead.
```

raylet 最后一次 `DebugString()` 耗时 **26,939 ms**，此前多次只有 **0—1 ms**。事件统计也出现过几十秒的排队延迟，但这些最大值是累计统计，不能全部归到最终故障窗口。

最后一次 Plasma 使用量约为 `0.085 / 50.626 GB`，没有对象存储撑满的迹象。这不能用于证明整机内存充足。

启动时出现的 metrics exporter 连接错误早于最终故障约 13 小时，没有证据将其认定为直接原因。

## 4. 从原始训练日志发现 MemoryError

仅查看诊断目录中的 Ray 日志时，没有找到明确的应用层内存分配错误。继续检查原始训练输出，发现：

```text
tail-causal-119625.out:1304  [8] SEND ERROR: MemoryError:
tail-causal-119625.out:1310  [2] SEND ERROR: MemoryError:
...
tail-causal-119625.out:1320  [2] OUTER EXCEPTION: MemoryError:
```

共计 **39 条 `SEND ERROR: MemoryError`**，另有 **1 条 `OUTER EXCEPTION: MemoryError`**。

这些错误对应评分器 `verl/utils/reward_score/feedback/code.py` 的测试执行和结果回传逻辑。原实现将以下字段一起发送到父进程：

```python
record = {
    "test_idx": test_idx,
    "input": test_input,
    "expected": test_output,
    "actual": output_value,
    "passed": passed,
    "debug": test_debug,
    "time": time_elapsed,
}
send_conn.send(record)
```

日志里的 `[8]`、`[2]` 是测试用例索引，不是题号，也不是同时运行的进程数量。

## 5. 验证批次数如何映射到数据集

### 5.1 批次计数规则

`ray_trainer.py` 的 `_validate()` 每从验证 DataLoader 取出一批数据，就打印一次：

```text
test_gen_batch meta info: {..., 'validate': True, 'global_steps': 93}
```

因此：

- `global_steps: 93` 是训练步数。
- 第 184 次打印表示这次验证的第 184 批，不是训练第 184 步。
- `validation generation end` 表示当前批次生成调用返回，后续仍要评分；它不表示整个验证完成。

本次实际配置：

```text
val_batch_size = 8
val_kwargs.n = 16
validation_shuffle = False
val_max_samples = -1
max_prompt_length = 2048
apply_chat_template_kwargs = {enable_thinking: False}
```

每批 8 道题，每题生成 16 个答案，即完整批次生成 128 个答案。

### 5.2 重算过滤后的数据顺序

用当时的 tokenizer 和聊天模板参数重新计算提示长度，得到：

| 数据集 | 原始条数 | 过滤后条数 | 合并后的题号范围 |
|---|---:|---:|---|
| AIME24 | 30 | 30 | 1–30 |
| AIME25 | 30 | 30 | 31–60 |
| Math500 | 500 | 500 | 61–560 |
| Biology | 50 | 50 | 561–610 |
| Chemistry | 210 | 210 | 611–820 |
| Material | 94 | 94 | 821–914 |
| Physics | 80 | 80 | 915–994 |
| GPQA Diamond | 198 | 197 | 995–1191 |
| Tooluse | 68 | 68 | 1192–1259 |
| LiveCodeBench | 400 | 400 | 1260–1659 |

唯一被过滤的条目是 GPQA 原始索引 `127`，长度为 **2,809 tokens**。最终共 1,659 道题：

```text
ceil(1659 / 8) = 208 个验证批次
```

第 `b` 批对应合并数据中的题号：

```text
起始题号 = (b - 1) × 8 + 1
结束题号 = min(b × 8, 1659)
LiveCodeBench 内题号 = 合并数据题号 - 1259
```

| 验证批次 | 合并数据题号 | LiveCodeBench 内题号 | 观察到的现象 |
|---|---|---|---|
| 182 | 1449–1456 | 190–197 | 首次出现 MemoryError |
| 184 | 1465–1472 | 206–213 | 集中出现 MemoryError |
| 190 | 1513–1520 | 254–261 | 开始后没有对应的生成结束日志，随后节点退出 |

原始日志共计 **190 次批次开始、189 次生成结束**，所以最后中断的位置是第 190 批，不是第 184 批。

### 5.3 使用 W&B 日志时间戳核对阶段

Slurm 输出中的 MemoryError 没有独立时间戳。通过解析本地 `.wandb` 文件中的 `output_raw` 记录，可获得批次日志的记录时间：

| 时间 | 记录 |
|---|---|
| 9 月 22 日 23:34:27 | 第 184 批开始 |
| 23:36:39 | 第 184 批生成结束 |
| 随后的 Slurm 日志区间 | 集中出现 MemoryError |
| 23:44:07 | 第 185 批开始，程序继续运行 |
| 9 月 23 日 00:15:18 | 第 189 批开始 |
| 00:17:07 | 第 189 批生成结束 |
| 00:25:51 | GCS 判定节点死亡 |

批次日志的时间戳不能替代每条子进程错误的精确发生时间。能够确认的是：第 184 批生成已结束，评分阶段的错误之后程序继续进入了后续批次；不能把它直接描述为最终节点退出时的“模型推理 OOM”。

## 6. 系统日志和 W&B 资源指标

### 6.1 用 CPU 作业访问故障节点

从 manager 直接 SSH 到计算节点认证失败，于是提交只申请 CPU 的短任务到 `5500-node07`。

- `120250`：确认成功运行于目标节点，但 `journalctl` 权限不足；旧版 `dmesg` 不支持 `--time-format`。任务最终因检查命令返回非零而标记失败，不是一次新的节点故障。
- `120251`：改用 `dmesg -T`，并读取 `/var/log/sa/sa23`，任务完成。

结果：

- `journalctl` 和 `/var/log/messages` 无法以当前用户读取。
- 可读的 `dmesg` 中未匹配到本次故障时段的记录；匹配出的 OOM 是 **8 月 1 日**的旧事件。
- `sar` 显示 00:20—00:30 区间平均 I/O 等待为 **7.39%**，此前区间为 **0.02%**。
- 00:30 时有 3 个阻塞任务，5 分钟平均负载为 30.73。
- 该区间整机 CPU 平均空闲仍有 82.12%，不支持“整机 CPU 持续打满”的判断。

`sar` 的采样间隔为 10 分钟，不能精确解释几十秒的卡顿。没有发现本次 OOM kill 记录，也不能作为绝对排除 OOM 的证据。

### 6.2 本地 W&B 文件不完整，云端补齐最后几分钟

本地 `.wandb` 文件包含 1,532 条系统指标，最后一条是 **00:19:48**。随后通过只读 Public API 查询系统指标流：

```python
import wandb

run = wandb.Api(timeout=40).run("20040817dkn-facebook/SDPO/i0k8rlmj")
rows = run.history(samples=20000, stream="system", pandas=False)
```

云端返回 1,544 条系统指标，最后一条为 **00:25:50**。`history()` 是采样接口，返回条数和时间范围需要检查，不能默认等同于完整逐次采集数据。

[W&B run](https://wandb.ai/20040817dkn-facebook/SDPO/runs/i0k8rlmj) 中故障窗口的主要变化：

| 时间 | 整机内存使用率 | 整机可用内存 | 四卡 GPU 利用率 |
|---|---:|---:|---|
| 00:24:18 | 87.80% | 61.6 GiB | 约 90%、90%、91%、90% |
| 00:24:48 | 91.28% | 43.9 GiB | 约 94%、66%、84%、61% |
| 00:25:18 | 该次缺失 | 该次缺失 | 20%、0%、0%、0% |
| 00:25:50 | 93.48% | 32.9 GiB | 该次缺失 |

约 92 秒内整机可用内存减少 **28.7 GiB**，与心跳超时窗口重合。有记录的最后阶段 GPU 显存占用仍约 63.6%，没有采样到显存打满。

同时，被监控的 TaskRunner 进程 RSS 始终约 2.74 GiB，没有同步增长。内存变化可能来自其他 worker、评测子进程、同节点其他作业或系统层面，现有指标无法归属。

### 6.3 核实指标含义，避免误读

根据 manager 安装的 W&B 版本源码：

- `system.cpu`：被监控进程的 CPU 使用率，经 CPU 数量归一化，不是整机使用率。
- `system.cpu.<i>.cpu_percent`：整机各核使用率。
- `system.memory`：整机内存使用率。
- `system.proc.memory.availableMB`：虽然名称带 `proc`，实际取自整机 `virtual_memory().available`。
- `system.proc.memory.rssMB`：单个被监控进程的 RSS，不包含整个 Ray 进程树。
- `system.disk.in/out`：自监控开始以来的累计读写量，不能直接当作瞬时吞吐率，更不能据此确定共享文件系统的延迟。

缺失采样值不能当作零。资源曲线支持继续调查主机内存变化和阻塞，但没有给出直接的进程级根因。

## 7. 第 206–213 题是什么，是否要求多进程

这些题都来自普通算法竞赛，题面没有要求创建多个进程：

| 题号 | 题目 ID | 内容 | 测试用例数 |
|---|---|---|---:|
| 206 | `abc303_d` | Shift/Caps Lock 输入字符串的最短时间 | 15 |
| 207 | `abc303_e` | 从树中还原原来的星形图 | 15 |
| 208 | `abc304_a` | 从最年轻者开始，按圆桌顺序输出姓名 | 14 |
| 209 | `abc304_b` | 整数保留前三位，其余位清零 | 16 |
| 210 | `abc304_c` | 平面上病毒在距离范围内传播 | 15 |
| 211 | `abc304_d` | 蛋糕分块后草莓数的最小值和最大值 | 14 |
| 212 | `abc304_e` | 添加边后是否违反禁止连通条件 | 13 |
| 213 | `abc305_a` | 找到最近的饮水站 | 10 |

评分器会为测试用例创建进程。这八题共 112 个用例，每题生成 16 个答案，全部评分可能累计启动 `112 × 16 = 1792` 个测试进程；这是累计数量，不是同时运行数量。每个答案的测试并发上限也不等于整个节点的全局并发上限。

原始模型答案没有找到：配置中 `rollout_data_dir=null`、`log_val_generations=0`，这次失败验证没有留下完整的结果输出。因此，无法检查或逐字重放当时的 128 个答案，也不能绝对排除某个生成答案尝试使用特殊库或异常分配内存。

## 8. 受控复现：区分评分器机制和原始故障

### 8.1 实验一：用 print(0) 复现结果回传 MemoryError

作业 `120257` 在 `5500-node07` 上运行完成，申请 1 CPU、4 GiB 内存、无 GPU，最长 5 分钟。只运行经过检查的固定程序：

```python
print(0)
```

测试输入是 8 MiB 字符串，每次只启动一个测试子进程。实验针对修复前的评分器实现。

| 条件 | 父进程 VmSize | 父进程 RSS | 结果 |
|---|---:|---:|---|
| 较小父进程，子进程绝对上限 1 GiB | 191,792 kB | 32,708 kB | 通过 |
| 父进程预留 2 GiB 虚拟地址空间，子进程绝对上限仍为 1 GiB | 2,295,020 kB | 39,264 kB | `SEND ERROR: MemoryError`；父进程收到 EOF |
| 同样的大地址空间父进程，子进程上限 3 GiB | 2,295,020 kB | 39,440 kB | 通过 |

关键输出：

```text
[0] SEND ERROR: MemoryError:
RESULT large-virtual-parent-1GiB-cap [(False, 'Process Error: EOFError: ')]
```

2 GiB 通过 `mmap` 预留，没有填充数据，不代表增加了 2 GiB 的实际物理内存占用。

原代码的常量名是 `MAX_ADDITIONAL_MEMORY_BYTES`，但实际设置的是：

```python
resource.setrlimit(resource.RLIMIT_AS, (1 * GiB, 1 * GiB))
```

`RLIMIT_AS` 限制的是总虚拟地址空间，不是额外内存。继承的地址空间已经很大时，序列化结果所需的新分配可能失败。实验确认了这个机制可以产生与原日志相同的错误文字，但没有重现 Ray 节点退出。

### 8.2 实验二：八题真实输入的结果回传对照

作业 `120258` 同样运行于 `5500-node07`，17 秒完成。每题选择输入与输出合计最大的一个测试用例，使用固定打印标准答案的程序，只隔离测试数据与结果回传路径。

这些程序是传输对照用的固定输出程序，不是独立求解算法，也不是当时模型生成的答案。

| 题号 | 选中用例的输入字节数 | 绝对 1 GiB 上限 | 继承 VmSize + 1 GiB 上限 |
|---|---:|---|---|
| 206 | 300,034 | 通过 | 通过 |
| 207 | 2,577,757 | 通过 | 通过 |
| 208 | 1,671 | 通过 | 通过 |
| 209 | 10 | 通过 | 通过 |
| 210 | 17,658 | 通过 | 通过 |
| 211 | 7,954,913 | 通过 | 通过 |
| 212 | 7,844,548 | 通过 | 通过 |
| 213 | 4 | 通过 | 通过 |

两种限制下的 16 次对照全部通过，说明真实测试数据本身不足以稳定触发错误。地址空间上限低于继承的 VmSize，也不意味着所有分配立即失败：部分分配可以复用已有内存分配器中的空间。原始模型代码行为、分配历史和回传数据大小仍会影响结果。

实验还有一个资源限制：脚本先读取整个 parquet 再切片，父进程 RSS 达到约 5 GiB，超过申请的 4 GiB。不能假设此集群的 Slurm 内存申请值就是硬限制。后续实验应使用按列、按 row group 读取，或先抽取目标测试数据，再在干净的进程中运行。

### 8.3 复现日志和脚本

manager 上的结果日志：

```text
/share/home/dengkn/SDPO_SVC/logs/lcb-memory-repro-120257.out
/share/home/dengkn/SDPO_SVC/logs/lcb-memory-repro-120257.err
/share/home/dengkn/SDPO_SVC/logs/lcb-eight-repro-120258.out
/share/home/dengkn/SDPO_SVC/logs/lcb-eight-repro-120258.err
```

本地诊断脚本与实验记录位于：

```text
.runtime/diagnostics/lcb206-213/repro_memory_limit.py
.runtime/diagnostics/lcb206-213/repro_eight_inputs.py
.runtime/diagnostics/lcb206-213/synthetic-result.txt
docs/research_progress/LCB_206_213_INVESTIGATION.md
```

这些是本地诊断产物，未包含在修复提交中。脚本使用 manager 的绝对路径，并且实验结论基于修复前代码；在新版本上执行时不能期待重新出现同样的失败。该目录 README 中“未修改生产代码”和并发为 2 的说明，是复现阶段的历史状态，不代表下面修复完成后的状态。

## 9. 最终实施的修复

用户确认实施内存限制、结果回传、输出与错误分类三项修改，并要求测试并发为 **8**。

修复已提交并推送至 GitHub `main`：

[68479fd — Fix code evaluator memory budgets and bound result transport](https://github.com/Hoilap/SDPO_SVC/commit/68479fd)

### 9.1 修正地址空间预算

- 子进程在执行测试代码前读取 `/proc/self/statm`，得到当前 VmSize。
- 目标上限为 `当前 VmSize + 额外预算`，额外预算默认 1 GiB。
- 不放宽继承的有限 soft/hard limit；若无法容纳完整预算，明确报告 `memory_limit_setup`。
- 去掉同时设置同一数值的 `RLIMIT_DATA` 和 `RLIMIT_RSS`。
- 不再吞掉限制设置失败后继续执行。

这仍然是虚拟地址空间增长限制，不是实际物理内存或整个进程树的严格预算。

### 9.2 精简回传数据

- 子进程不再发送完整 `input`、`expected`。
- 父进程根据当前测试索引补回这些字段，保持反馈记录布局。
- `actual`、`debug` 回传预览各限制为 4,096 字符。
- 正常答案先进行完整比较，再生成反馈预览。

### 9.3 限制输出与区分错误

- stdout、stderr/debug 使用有写入上限的缓冲区。
- `CODE_REWARD_MAX_OUTPUT_CHARS` 可配置上限，默认每个缓冲区 `4 × 1024 × 1024` 字符；字符数不等于内存字节数。
- 超限标记保留，即使生成代码捕获异常，也不能将该测试判为通过。
- 回传失败时只尝试发送预先序列化的小错误标记；仍失败时由父进程识别 EOF、退出状态。
- 区分执行内存错误、结果准备/回传错误、限制设置失败、输出超限、超时和异常退出。
- 基础设施错误在反馈中单独展示，避免标成普通 Wrong Answer。

### 9.4 并发和验证

两个训练入口均设为：

```bash
export CODE_REWARD_MAX_CONCURRENCY=8
```

涉及文件：

- `verl/utils/reward_score/feedback/code.py`
- `tests/utils/reward_score/test_feedback_code.py`
- `experiments/causal/run.sh`
- `experiments/continual/run_sdpo_svc_cl.sh`

本地 20 项回归测试通过，覆盖：大地址空间继承、大输入、有限继承上限、回传精简、固定错误标记、stdout/stderr/debug 超限、完整答案比较、执行 MemoryError、设置失败、超时回收、异常退出，以及 stdin/functional/code 模式。Shell 语法检查和 `git diff --check` 通过。

推送使用 SSH 443 端口完成，因为 HTTPS 缺少交互凭据。该次操作仅完成 GitHub 推送，未执行 manager 拉取或重新启动训练。

## 10. 复查时可使用的命令

以下命令只读取现有证据；在 manager 上执行。

```bash
ssh -F /home/dengkn/.ssh/config manager

cd /share/home/dengkn/SDPO_SVC/.runtime/diagnostics/119625-120027/session_2026-09-22_11-23-46_891912_32776
grep -n 'gcs_health_check_manager.cc' gcs_server.out | tail -n 30
grep -n 'DebugString() time' raylet.out | tail -n 20

cd /share/home/dengkn/SDPO_SVC
grep -n 'MemoryError' logs/tail-causal-119625.out
grep -c 'test_gen_batch meta info:' logs/tail-causal-119625.out
grep -c 'validation generation end' logs/tail-causal-119625.out
nl -ba logs/tail-causal-119625.out | sed -n '1300,1322p'

sacct -j 119625,120251,120257,120258 \
  --format=JobID,State,ExitCode,NodeList,Start,End -P
```

内核与 `sar` 命令必须通过获准的节点访问方式，在 `5500-node07` 上执行，不能用 manager 的系统日志替代：

```bash
journalctl -k --since '2026-09-23 00:20:00' \
  --until '2026-09-23 00:27:00' --no-pager

dmesg -T | grep -iE 'Sep 23.*00:2[0-7]:|oom|out of memory|killed process|blocked for|I/O error|NVRM|Xid'

sar -u -r -W -q -f /var/log/sa/sa23 -s 00:10:00 -e 00:40:00
```

`sa23` 按日期编号，之后可能被轮换或覆盖；复查时应确认文件内的实际日期。`dmesg -T` 显示的墙钟时间也应与其他日志交叉核对。

## 11. 仍需验证的事项

要判断修复是否解决原始训练中断，应在新版本上重新观察验证，并额外保存：

- 批次号、数据集、题目 ID、答案编号及评分开始/结束时间。
- 评测子进程的基线 VmSize、实际地址空间上限和退出状态。
- TaskRunner、Ray worker、代码评测进程及其子进程的 RSS、CPU、I/O。
- 验证生成结果，或足以重放评分的目标批次答案。

评估标准应同时包括：评分器不再出现已复现的回传错误、测试结果保持正确、节点没有异常内存增长和心跳超时。仅仅不再看到 `SEND ERROR`，不足以证明整条训练故障链已经消除。
