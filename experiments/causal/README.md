# Tail-direction causal experiment

Scientific design: [PLAN.md](PLAN.md).

The runner enforces `data.train_max_samples <= 3000` for both Math and Science.
The default is one paired seed (`SEED=42`) and no SVC.

## Preflight

```bash
bash experiments/causal/run.sh all --dry-run
python3 experiments/causal/prepare_manifest.py \
  --output-dir outputs/causal_tail_seed1/preflight/data
```

The preparation command validates the already prepared train/test Parquet
files and normalizes AIME, MATH-500, GPQA and LiveCodeBench for the common verl
evaluation schema. It never adds evaluation data to training.

## Run by stage

Use the same absolute `OUTPUT_ROOT` for every submitted stage:

```bash
export OUTPUT_ROOT=/users/$USER/SDPO/outputs/causal_tail_seed1/seed42

bash experiments/causal/run.sh prepare
sbatch --export=ALL,OUTPUT_ROOT="$OUTPUT_ROOT" experiments/causal/run.sh math
sbatch --export=ALL,OUTPUT_ROOT="$OUTPUT_ROOT" experiments/causal/run.sh surgery
sbatch --export=ALL,OUTPUT_ROOT="$OUTPUT_ROOT" experiments/causal/run.sh eval-math
sbatch --export=ALL,OUTPUT_ROOT="$OUTPUT_ROOT" experiments/causal/run.sh science
sbatch --export=ALL,OUTPUT_ROOT="$OUTPUT_ROOT" experiments/causal/run.sh eval-science
bash experiments/causal/run.sh summary
```

Submit a stage only after its predecessor succeeds. Alternatively, one long
allocation can execute everything sequentially:

```bash
sbatch --export=ALL,OUTPUT_ROOT="$OUTPUT_ROOT" experiments/causal/run.sh all
```

Useful safe overrides include `BASE_MODEL`, `SEED`, `TRAIN_SAMPLE_LIMIT`
(values above 3000 are rejected), `TAIL_DEVICE`, and `WANDB_MODE`.

## Outputs

- `data/dataset_manifest.json`: exact train/evaluation inputs;
- `math_hf/`: original GRPO/SDPO checkpoints after Math;
- `tail_arms/manifest.json`: derived checkpoint paths and per-matrix SVD stats;
- `science_hf/`: four final continuation checkpoints;
- `evaluation/`: raw validation samples and metric JSON files;
- `summary/scores.csv`: selected per-dataset scores;
- `summary/contrasts.json`: tail effects, method gaps and interactions.
