# 120036 failure investigation

## Confirmed evidence

Read-only inspection of manager, 5500-node07, the local W&B datastore, and repository code. No new jobs submitted, no live jobs changed.

- Tooluse training (`datasets/tooluse/train.parquet`), run `9mno81sn`, failed after completed step 86/93. No validation batches started in this phase. Math and Science completed and produced checkpoints, metrics, and calibrated HF models.
- Local W&B datastore contains 514 system records and 85 history records. It ends at 2026-09-25 08:43:54 +08:00; stdout contains a later completed step than the datastore. Absence of later W&B records is not proof of when computation stopped.
- 08:43:24: available memory 184466.15 MiB, node memory usage 64.20%, tracked process RSS 1341.81 MiB.
- 08:43:54: available memory 167711.09 MiB, usage 67.44%, tracked process RSS 1407.51 MiB. The available-memory drop is approximately 16.36 GiB, whereas this process's RSS rises only 65.70 MiB. GPU utilization in the last sample remains 85.67% and 92.27%; this does not cover the subsequent failure interval.
- GCS health-check failures: 08:44:10.442, 08:44:23.447, 08:44:36.448, 08:44:49.450, 08:45:02.479. Node declared dead at 08:45:02.481. Raylet exits at 08:45:22.464.
- Final Raylet event statistics show several periodic callbacks with maximum queueing delays around 81 seconds. These are cumulative maxima, not measured execution durations of those callbacks.
- Node sar: CPU idle around 87% through the 08:40 sample; no sustained swap-in/out. The 08:40–08:50 interval has 4.92% I/O wait, 97.49 major faults/s and 8721.25 background page scans/s. This interval also includes job exit and 120032 starting at 08:46:06, so it cannot establish which activity preceded the failure.
- The queried kernel log has no matching entries in the inspected interval. Slurm logs do not contain an explicit CUDA OOM or MemoryError for this failure. Neither observation rules out cgroup limits, transient pressure, missing historical kernel records, or other process failures.
- Slurm reports `SelectTypeParameters=CR_CORE`. This is not consumable-memory scheduling. Other jobs overlapped on the node; their presence alone does not establish causation.

## Source locations

Remote root: `/share/home/dengkn/SDPO_SVC`.

- `logs/sdpo-svc-try-120036.{out,err}`
- `outputs/sdpo_svc_cl_try/120036/wandb/wandb/run-20260925_042621-9mno81sn/run-9mno81sn.wandb`
- On 5500-node07: `/tmp/ray/session_2026-09-25_04-21-42_098177_11153/logs/{gcs_server.out,raylet.out}`
- On 5500-node07: `/var/log/sa/sa25`

Cloud W&B API access through a key embedded in a script was rejected by automatic approval review. No key was extracted by that operation. Analysis instead used the existing offline datastore without authentication.

## Separate validation memory issue

`verl/trainer/ppo/ray_trainer.py::_validate` accumulates decoded inputs, outputs, ground truths and reward extra information for the entire pass, even when `validation_dump_generations=False`. The flag only suppresses the final export. With 1659 prompts and 16 samples, the pass collects 26544 responses. This is a demonstrated retention pattern, not proof that it caused 120036, which was not validating.

`perf/cpu_memory_used_gb` in `fsdp_workers.py` comes from `psutil.virtual_memory().used`: it is node-level memory, not the job's total RSS. W&B tracked-process RSS also excludes child workers. GPU allocator high-water marks are not instantaneous free-VRAM measurements.

## Investigation and repair sequence

1. Before another expensive retry, add flushed start/end markers around generation, reward, teacher/log-prob computation and actor update, with step, batch/sample IDs, timestamps and durations. Record the last in-flight phase, not only completed-step metrics.
2. Collect an independent 5-second process/cgroup monitor on node-local storage: per-worker RSS/PSS and state/wchan, allowed CPUs and NUMA nodes, cgroup memory usage/limits/failure counters and CPU throttling, node MemAvailable/page faults/reclaim/I/O, GPU utilization. Use cgroup v1 or v2 counters as available. W&B's 30-second sampling and node totals alone are insufficient. Retain Ray logs on TERM/EXIT and preserve a local copy if archiving fails.
3. On a detected stall, obtain Raylet and worker thread stacks plus process wait channels. This distinguishes CPU scheduling/cgroup throttling, kernel or filesystem waits, and application locks. CPU idle at node level does not establish free CPU capacity within the job's cpuset.
4. Change validation to retain numeric scores/UIDs and only bounded text previews; stream optional detailed outputs. Preserve existing metric semantics. Split evaluation into dataset-level resumable units, with model/config/dataset identity recorded and results written atomically. Verify aggregates against existing metrics; do not average per-dataset summaries blindly or change sampling settings unnoticed.
5. Save resumable training state periodically, including optimizer/RNG/dataloader state where required; a model-only checkpoint is not an exact continuation. Preserve the existing Science-SVC boundary for restarting Tooluse.
6. Run one controlled comparison after instrumentation: same checkpoint/data/settings on an exclusive node or a different node, if resources permit. Keep n=16 and generation length unchanged initially. If pressure is confirmed, reduce microbatch/scoring concurrency one setting at a time.
7. Do not use increased heartbeat tolerance or unlimited walltime as a claimed fix for Ray stalls. They do not address missing progress, memory retention or underlying node problems.

References: https://docs.ray.io/en/latest/ray-core/scheduling/memory-management.html and https://slurm.schedmd.com/cons_tres.html .
