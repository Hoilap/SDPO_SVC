# LiveCodeBench validation batch 184 investigation

Supporting scripts and results remain in the local, unversioned `.runtime/diagnostics/lcb206-213/` directory and the manager paths below. File names below refer to that directory. This report records the investigation-time configuration, not current runtime defaults.

Dataset: manager:/share/home/dengkn/SDPO_SVC/outputs/causal_tail_seed1/118343/data/normalized/livecodebench-v6.parquet
Rows are numbered from 1. No production code was changed.

| Row | Problem ID | Description | Tests |
|---|---|---|---:|
|206|abc303_d|Minimum typing cost with Shift/Caps Lock|15|
|207|abc303_e|Recover star components from a tree|15|
|208|abc304_a|Print circular seating order starting at youngest person|14|
|209|abc304_b|Truncate integer to leading three digits|16|
|210|abc304_c|Virus propagation between nearby points|15|
|211|abc304_d|Minimum/maximum strawberries per rectangular cake piece|14|
|212|abc304_e|Whether adding an edge violates forbidden connectivity|13|
|213|abc305_a|Nearest water station at a multiple of five|10|

The problems do not require multiprocessing. The evaluator launches a process
per test case. There are 112 cases across these eight questions; with 16 model
answers per question, a full evaluation can launch 1792 test processes in total,
not necessarily simultaneously. Current causal script limits test concurrency
per completion to two; historical concurrency was not established here.

## Controlled reproductions

Job 120257 on 5500-node07 completed successfully. See synthetic-result.txt and
repro_memory_limit.py. A trusted print(0) program reproduces exactly
SEND ERROR: MemoryError with an inherited 2 GiB virtual reservation and an
absolute 1 GiB child memory limit. It passes with a small parent and also with
a 3 GiB limit in the large-parent control. This establishes a possible failure
mechanism, not the historical cause of the Ray crash.

Job 120258 on 5500-node07 completed in 17 seconds. repro_eight_inputs.py chooses
one largest input+output test per question, and uses a trusted oracle stub that
prints its expected output. All eight passed both with the absolute 1 GiB cap
and with a cap of inherited VMS plus 1 GiB. These are transport controls, not
independently solved programs and not replays of original model completions.
An address-space limit below inherited VMS does not guarantee every allocation
fails: already allocated allocator arenas can still satisfy some allocations.

Largest test input bytes: 206=300034; 207=2577757; 208=1671; 209=10;
210=17658; 211=7954913; 212=7844548; 213=4.

The actual-input script reads the whole parquet before slicing. Recorded parent
RSS reached about 5 GiB despite the 4 GiB Slurm request; do not assume this
cluster enforces the requested memory as a hard limit. Both jobs have ended.
For further reproduction, use a projected/row-group read in a fresh process.

No original completions for this failed SDPO validation were found. Config has
rollout_data_dir=null, log_val_generations=0; the configured validation output
directory has no completed output for this run. Exact historical replay and
attribution of individual MemoryError lines to a question remain unavailable.

Remote logs:
- /share/home/dengkn/SDPO_SVC/logs/lcb-memory-repro-120257.out
- /share/home/dengkn/SDPO_SVC/logs/lcb-eight-repro-120258.out
