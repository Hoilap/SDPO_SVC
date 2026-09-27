# Causal tail pilot: PLAN audit, 2026-09-27 23:20 CST

Remote root: `/share/home/dengkn/SDPO_SVC/outputs/causal_tail_seed1/118343`.
Compared against `experiments/causal/PLAN.md`, source code, scheduler records,
intervention manifest, metrics JSON and Ray worker logs. This is a snapshot,
not a live status page. No jobs or experiment settings were changed.

## Progress

| Plan item | Evidence/status |
| --- | --- |
| One paired seed, <=3000 training samples per task, no SVC | Seed 42; dataset manifest cap 3000; causal runner has no SVC. Actual sample-ID pairing across all runs has not been independently audited. |
| GRPO/SDPO Math training | Both HF models exist. Original SDPO job 120031 failed during validation; checkpoint validation repro 121928 completed separately. |
| Eight intervention models | 120032 completed. Manifest has eight arms and 252 selected matrices, rank fraction 0.1. |
| Reconstruction error <1e-5 | Manifest maxima: GRPO 5.3406124e-9, SDPO 7.4667126e-9. |
| Eight post-Math evaluations | Official `after_math` metrics exist for GRPO-full and GRPO-no-tail, each with 26544 samples. Remaining six lack final results. |
| Resume evaluation | 122061 RUNNING; explicitly skipped the two completed arms. Ray TaskRunner PID 31926 local log records 11 batch starts / 10 generation ends. Slurm stdout currently lacks these progress messages. |
| Four Science continuations | 0/4 completed or started; science_hf and during_science results are empty. 120034 awaits afterok:122061. |
| Post-Science evaluations | Not started; 120035 awaits afterok:120034. Both pending jobs have UNLIMITED time limits. |
| Scores and causal contrasts | No summary artifacts yet. Full method-gap, random-tail and transplant contrasts cannot yet be computed. |

## Preliminary post-Math scores

Values below are `val-aux/<dataset>/score/mean@16`, multiplied by 100.
They are reward scores; particularly for LiveCodeBench they must not be
presented as exact-solution accuracy. The summarizer currently selects these
score metrics. Both score and acc should be labeled explicitly in analysis.

| Dataset | GRPO-full | GRPO-no-tail | no-tail minus full (points) |
| --- | ---: | ---: | ---: |
| AIME24+25 (combined math label) | 4.17 | 6.77 | +2.60 |
| MATH-500 | 31.10 | 33.89 | +2.79 |
| SciKnowEval | 16.99 | 19.67 | +2.68 |
| GPQA | 34.39 | 32.77 | -1.62 |
| ToolUse | 55.15 | 54.96 | -0.18 |
| LiveCodeBench | 38.28 | 40.28 | +2.00 |

Results are mixed. These two arms alone do not establish a GRPO-vs-SDPO gap,
direction specificity, transplant sufficiency or continual retention. No
significance assessment was performed. The pilot has only one training seed.

## Plan/reporting discrepancies to resolve

1. PLAN's introductory status "not started" is obsolete.
2. Both AIME files are normalized to `data_source=math`, so existing aggregate
   metrics cannot distinguish AIME24 and AIME25. Adding separate reporting
   should preserve scoring dispatch and comparability; do not silently change
   the dataset identity midway through the pilot.
3. Default Science frequency is 23 steps. For a 93-step run this yields
   0,23,46,69,92,93 evaluations, rather than exactly five planned milestones.
4. Manifest `exact_svd=false`: principal directions use randomized low-rank
   SVD. `tail=delta-principal`, so a small reconstruction residual alone does
   not certify the accuracy of the estimated leading singular subspace.
   Random-tail projection also makes spectral matching approximate; transplant
   code projects donor directions away from host principal before norm matching.
5. Repro 121928 metrics are outside the causal `evaluation/after_math` tree.
   They are useful evidence that SDPO checkpoint validation works, but should
   not be substituted automatically for the standardized SDPO-full evaluation
   without checking inference configuration and model provenance.
6. Jobs 120036 and 122068 are separate SVC continual-learning experiments.
   They do not satisfy the causal PLAN's no-SVC Science-continuation requirements.
