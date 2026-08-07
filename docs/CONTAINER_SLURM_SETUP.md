# SDPO_SVC：Slurm + Singularity 容器配置指南

本文档说明如何在 Slurm 集群上使用 Singularity 容器运行 SDPO_SVC。内容针对以下环境：

- 项目目录：`/share/home/dengkn/SDPO_SVC`
- 调度系统：Slurm
- GPU 分区：`gpu_chen`
- GPU：NVIDIA A800 80GB PCIe
- NVIDIA Driver：535.104.12
- 宿主系统：glibc 2.17、GCC 4.8.5
- 容器运行时：`singularity/4.0.2`
- Docker 命令存在，但普通用户无权访问 `/var/run/docker.sock`

宿主机 glibc 2.17 无法使用 `xformers==0.0.29.post2` 的官方 wheel。容器内使用较新的 Ubuntu/glibc，可以避开 xFormers、SciPy 和编译器兼容问题。因此，正式训练使用 Singularity；宿主 Conda 环境仅用于日常辅助操作，不承担 vLLM 训练环境。

## 1. 理解仓库里的 `docker/` 目录

`docker/` 目录保存的是 Dockerfile 构建配方，不是已经下载到本机的镜像。

针对 A800 和当前 `rollout.name=vllm` 配置，推荐使用：

```text
docker/verl0.4-cu124-torch2.6-fa2.7.4/
└── Dockerfile.app.vllm.mcore0.12
```

它对应的主要软件组合是：

```text
CUDA          12.4
PyTorch       2.6.0
FlashAttention 2.7.4.post1
vLLM          0.8.5.post1
```

优先拉取已构建镜像，无需从这些 Dockerfile 重新编译 CUDA、Apex、FlashAttention 和 Megatron：

```text
verlai/verl:app-verl0.4-vllm0.8.5-mcore0.12.2-te2.2
```

根目录的 `Dockerfile` 没有安装当前训练配置需要的 vLLM，不作为首选。

## 2. 目录规划

代码和持久化文件统一放在项目目录下：

```text
/share/home/dengkn/SDPO_SVC/
├── .runtime/
│   ├── containers/
│   │   ├── verl-vllm085.sif
│   │   ├── cache/
│   │   └── tmp/
│   ├── envs/
│   │   └── sdpo-vllm085/
│   ├── huggingface/
│   ├── checkpoints/
│   ├── logs/
│   └── wandb/
├── datasets/
├── logs/
├── verl/
└── run_sdpo_singularity.sbatch
```

初始化变量和目录：

```bash
cd /share/home/dengkn/SDPO_SVC

export PROJECT_ROOT="$(pwd -P)"
export STORAGE_ROOT="$PROJECT_ROOT/.runtime"
export IMAGE="$STORAGE_ROOT/containers/verl-vllm085.sif"
export CONTAINER_ENV="$STORAGE_ROOT/envs/sdpo-vllm085"

mkdir -p \
  "$STORAGE_ROOT/containers/cache" \
  "$STORAGE_ROOT/containers/tmp" \
  "$STORAGE_ROOT/envs" \
  "$STORAGE_ROOT/huggingface" \
  "$STORAGE_ROOT/checkpoints" \
  "$STORAGE_ROOT/logs" \
  "$STORAGE_ROOT/wandb" \
  "$PROJECT_ROOT/logs"
```

检查容量和配额：

```bash
df -h "$PROJECT_ROOT"
quota -s 2>/dev/null || true
```

建议至少预留 100 GB；如果需要频繁保存完整模型 checkpoint，应准备更多空间。若 `/share/home` 配额不足，应将 `.runtime/huggingface` 和 `.runtime/checkpoints` 迁移到集群 scratch 文件系统。

将运行产物加入 `.gitignore`：

```gitignore
.runtime/
logs/
```

## 3. 加载 Singularity

```bash
module load singularity/4.0.2
singularity --version
```

预期能看到 Singularity 4.0.2。不要使用宿主机 Docker：当前用户不属于 `docker` 组，访问 Docker daemon 会得到 `permission denied`。

## 4. 下载 SIF 镜像

设置 Singularity 的缓存和临时目录，避免默认缓存占满其他目录：

```bash
export SINGULARITY_CACHEDIR="$STORAGE_ROOT/containers/cache"
export SINGULARITY_TMPDIR="$STORAGE_ROOT/containers/tmp"
```

拉取镜像：

```bash
singularity pull \
  "$IMAGE" \
  docker://verlai/verl:app-verl0.4-vllm0.8.5-mcore0.12.2-te2.2
```

检查结果：

```bash
ls -lh "$IMAGE"
singularity inspect "$IMAGE" | head
```

`.sif` 是普通只读文件。只要 `/share/home` 是共享文件系统，管理节点下载一次后，计算节点可以直接读取，无需重复下载。

若主镜像 tag 不存在，可临时使用：

```bash
singularity pull \
  "$STORAGE_ROOT/containers/vllm085.sif" \
  docker://vllm/vllm-openai:v0.8.5
```

但该镜像只提供 vLLM 环境，可能缺少部分 verl 训练依赖，优先使用 `verlai/verl` 镜像。

## 5. 验证计算节点能访问镜像

```bash
srun \
  --partition=gpu_chen \
  --gres=gpu:1 \
  --time=00:10:00 \
  bash -lc '
    module load singularity/4.0.2
    ls -lh /share/home/dengkn/SDPO_SVC/.runtime/containers/verl-vllm085.sif
  '
```

如果计算节点找不到文件，说明该路径不是共享存储，需要把 SIF 移到管理节点和计算节点都能访问的文件系统。

## 6. 验证 GPU 和核心依赖

```bash
srun --partition=gpu_chen --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=00:15:00 bash -lc 'module load singularity/4.0.2; singularity exec --nv /share/home/dengkn/containers/verl-vllm085.sif python -c "import torch,vllm,xformers,ray; print(\"torch:\",torch.__version__); print(\"CUDA:\",torch.version.cuda); print(\"vllm:\",vllm.__version__); print(\"xformers:\",xformers.__version__); print(\"ray:\",ray.__version__); print(\"GPU:\",torch.cuda.get_device_name(0))"'
```

预期核心结果接近：

```text
torch: 2.6.0+cu124
CUDA: 12.4
vllm: 0.8.5.post1
xformers: 0.0.29.post2
ray: 2.47.1
GPU: NVIDIA A800 80GB PCIe
```

`--nv` 会把计算节点分配到的 NVIDIA GPU 和必要的 Driver 库注入容器。`nvidia-smi` 显示的 `CUDA Version: 12.2` 是宿主 Driver 能力信息，不等同于容器内 PyTorch 自带的 CUDA runtime 版本；最终以实际导入和 GPU 运算测试为准。

为避免后续路径混乱，统一设置：
```
export PROJECT_ROOT="/share/home/dengkn/SDPO_SVC"
export IMAGE="/share/home/dengkn/containers/verl-vllm085.sif"
export STORAGE_ROOT="$PROJECT_ROOT/.runtime"
export CONTAINER_ENV="$STORAGE_ROOT/envs/sdpo-vllm085"
```

## 7. 创建容器专用 Python venv

SIF 镜像是只读的。为当前 SDPO 源码创建一个位于共享目录的 venv，并继承镜像内已经安装的 PyTorch、vLLM、xFormers 和 Ray：

```bash
module load singularity/4.0.2

singularity exec \
  --bind "$PROJECT_ROOT:$PROJECT_ROOT" \
  --bind "$STORAGE_ROOT:$STORAGE_ROOT" \
  "$IMAGE" \
  python -m venv \
    --system-site-packages \
    "$CONTAINER_ENV"
```

安装当前仓库和论文额外依赖：

```bash
singularity exec \
  --bind "$PROJECT_ROOT:$PROJECT_ROOT" \
  --bind "$STORAGE_ROOT:$STORAGE_ROOT" \
  "$IMAGE" \
  bash -lc "
    source '$CONTAINER_ENV/bin/activate'

    python -m pip install --no-deps -e '$PROJECT_ROOT'

    python -m pip install \
      word2number==1.1 \
      latex2sympy2==1.5.4 \
      latex2sympy2_extended==1.10.2 \
      'math-verify[antlr4_9_3]==0.8.0'
  "
```

这个 venv 引用了镜像内部的 Python 和系统包，只能在同一个 Singularity 镜像内使用。不要在宿主 shell 中直接运行 `$CONTAINER_ENV/bin/python`。

## 8. 验证 SDPO 源码环境

```bash
srun \
  --partition=gpu_chen \
  --gres=gpu:1 \
  --cpus-per-task=8 \
  --mem=32G \
  --time=00:15:00 \
  bash -lc '
    module load singularity/4.0.2

    PROJECT_ROOT=/share/home/dengkn/SDPO_SVC
    STORAGE_ROOT=$PROJECT_ROOT/.runtime
    IMAGE=$STORAGE_ROOT/containers/verl-vllm085.sif
    CONTAINER_ENV=$STORAGE_ROOT/envs/sdpo-vllm085

    singularity exec --nv \
      --bind "$PROJECT_ROOT:$PROJECT_ROOT" \
      --bind "$STORAGE_ROOT:$STORAGE_ROOT" \
      "$IMAGE" \
      bash -lc "
        source '$CONTAINER_ENV/bin/activate'
        cd '$PROJECT_ROOT'

        python -c 'import torch,vllm,xformers,ray,verl; print(torch.__version__); print(vllm.__version__); print(xformers.__version__); print(verl.__file__); print(torch.cuda.get_device_name(0))'
      "
  '
```

## 9. 准备训练数据

仓库提供的是 JSON 数据，训练配置需要 `train.parquet` 和 `test.parquet`。

处理 ToolUse：

```bash
singularity exec \
  --bind "$PROJECT_ROOT:$PROJECT_ROOT" \
  --bind "$STORAGE_ROOT:$STORAGE_ROOT" \
  "$IMAGE" \
  bash -lc "
    source '$CONTAINER_ENV/bin/activate'
    cd '$PROJECT_ROOT'
    python data/preprocess.py --data_source datasets/tooluse
  "
```

处理 SciKnowEval：

```bash
for task in biology chemistry material physics; do
  singularity exec \
    --bind "$PROJECT_ROOT:$PROJECT_ROOT" \
    --bind "$STORAGE_ROOT:$STORAGE_ROOT" \
    "$IMAGE" \
    bash -lc "
      source '$CONTAINER_ENV/bin/activate'
      cd '$PROJECT_ROOT'
      python data/preprocess.py --data_source datasets/sciknoweval/$task
    "
done
```

检查：

```bash
find "$PROJECT_ROOT/datasets" -name '*.parquet' -ls
```

启动训练前，目标数据集目录必须至少包含：

```text
train.parquet
test.parquet
```

## 10. Hugging Face 和 W&B

模型缓存必须放到持久化目录，否则作业结束后可能重新下载：

```bash
export HF_HOME="$STORAGE_ROOT/huggingface"
export WANDB_DIR="$STORAGE_ROOT/wandb"
```

如果训练节点不能联网，应提前下载默认模型：

```bash
singularity exec \
  --bind "$STORAGE_ROOT:$STORAGE_ROOT" \
  --env HF_HOME="$HF_HOME" \
  "$IMAGE" \
  bash -lc '
    hf download Qwen/Qwen3-8B
  '
```

若 `hf` 命令不存在，可在容器 venv 中安装 `huggingface_hub`。私有或受限模型需要先执行 `hf auth login`，不要把 token 写进 Git 或 sbatch 文件。

首次 smoke test 建议使用离线 W&B：

```bash
export WANDB_MODE=offline
```

## 11. Slurm + Singularity smoke test 脚本

在项目根目录创建 `run_sdpo_singularity.sbatch`：

```bash
#!/bin/bash
#SBATCH --job-name=sdpo-smoke
#SBATCH --partition=gpu_chen
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=400G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
set -x

module purge
module load singularity/4.0.2

export PROJECT_ROOT="${PROJECT_ROOT:-/share/home/dengkn/SDPO_SVC}"
export STORAGE_ROOT="${STORAGE_ROOT:-$PROJECT_ROOT/.runtime}"
export IMAGE="${IMAGE:-$STORAGE_ROOT/containers/verl-vllm085.sif}"
export CONTAINER_ENV="${CONTAINER_ENV:-$STORAGE_ROOT/envs/sdpo-vllm085}"

export TASK="${TASK:-datasets/tooluse}"
export CONFIG_NAME="${CONFIG_NAME:-sdpo}"
export EXPERIMENT="${EXPERIMENT:-${SLURM_JOB_NAME}-${SLURM_JOB_ID}}"

export HF_HOME="$STORAGE_ROOT/huggingface"
export WANDB_DIR="$STORAGE_ROOT/wandb"
export WANDB_MODE="${WANDB_MODE:-offline}"

# Ray 和 Triton 临时文件应位于计算节点本地盘，不要放入共享目录。
JOB_TMP="${SLURM_TMPDIR:-/tmp/$USER/sdpo-$SLURM_JOB_ID}"
export RAY_TMPDIR="$JOB_TMP/ray"
export TRITON_CACHE_DIR="$JOB_TMP/triton"

mkdir -p \
  "$RAY_TMPDIR" \
  "$TRITON_CACHE_DIR" \
  "$STORAGE_ROOT/logs" \
  "$STORAGE_ROOT/checkpoints" \
  "$WANDB_DIR"

cleanup() {
  singularity exec \
    --bind "$CONTAINER_ENV:$CONTAINER_ENV" \
    "$IMAGE" \
    bash -lc "source '$CONTAINER_ENV/bin/activate'; ray stop -f" \
    || true
}
trap cleanup EXIT

singularity exec \
  --nv \
  --bind "$PROJECT_ROOT:$PROJECT_ROOT" \
  --bind "$STORAGE_ROOT:$STORAGE_ROOT" \
  --bind "$JOB_TMP:$JOB_TMP" \
  --env PROJECT_ROOT="$PROJECT_ROOT" \
  --env STORAGE_ROOT="$STORAGE_ROOT" \
  --env CONTAINER_ENV="$CONTAINER_ENV" \
  --env TASK="$TASK" \
  --env CONFIG_NAME="$CONFIG_NAME" \
  --env EXPERIMENT="$EXPERIMENT" \
  --env HF_HOME="$HF_HOME" \
  --env WANDB_DIR="$WANDB_DIR" \
  --env WANDB_MODE="$WANDB_MODE" \
  --env RAY_TMPDIR="$RAY_TMPDIR" \
  --env TRITON_CACHE_DIR="$TRITON_CACHE_DIR" \
  "$IMAGE" \
  bash -lc '
    set -euo pipefail

    source "$CONTAINER_ENV/bin/activate"
    cd "$PROJECT_ROOT"

    ray stop -f || true
    ray start \
      --head \
      --num-cpus="$SLURM_CPUS_PER_TASK" \
      --num-gpus=4 \
      --disable-usage-stats \
      --temp-dir="$RAY_TMPDIR"

    # 第一次只运行缩小后的 smoke test。
    python -m verl.trainer.main_ppo \
      --config-name "$CONFIG_NAME" \
      vars.dir="$PROJECT_ROOT" \
      vars.log_dir="$STORAGE_ROOT/logs" \
      vars.ckpt_dir="$STORAGE_ROOT/checkpoints/$TASK" \
      custom_reward_function.path="$PROJECT_ROOT/verl/utils/reward_score/feedback/__init__.py" \
      trainer.n_gpus_per_node=4 \
      trainer.nnodes=1 \
      trainer.total_epochs=1 \
      trainer.test_freq=1 \
      trainer.save_freq=0 \
      data.train_batch_size=8 \
      data.max_response_length=512 \
      actor_rollout_ref.rollout.n=2 \
      actor_rollout_ref.actor.ppo_mini_batch_size=8
  '
```

注意：仓库自带的 `experiments/verl_training.sbatch` 会激活 `.venv` 并申请 8 张 GPU，不适合直接用于当前 Singularity 方案。

## 12. 提交和查看 smoke test

```bash
cd /share/home/dengkn/SDPO_SVC
mkdir -p logs

sbatch \
  --export=ALL,PROJECT_ROOT="$PROJECT_ROOT",STORAGE_ROOT="$STORAGE_ROOT",IMAGE="$IMAGE",CONTAINER_ENV="$CONTAINER_ENV",TASK=datasets/tooluse,CONFIG_NAME=sdpo \
  run_sdpo_singularity.sbatch
```

查看状态：

```bash
squeue -u "$USER"
```

查看日志：

```bash
tail -f logs/sdpo-smoke-<JOB_ID>.out
tail -f logs/sdpo-smoke-<JOB_ID>.err
```

查看作业最终状态：

```bash
sacct -j <JOB_ID> --format=JobID,State,ExitCode,Elapsed,MaxRSS,NodeList
```

## 13. 切换到正式训练

Smoke test 成功后，再逐步恢复 `verl/trainer/config/sdpo.yaml` 中的正式参数。至少移除或修改：

```text
trainer.total_epochs=1
data.train_batch_size=8
data.max_response_length=512
actor_rollout_ref.rollout.n=2
actor_rollout_ref.actor.ppo_mini_batch_size=8
```

正式配置的主要参数为：

```text
trainer.total_epochs=30
data.train_batch_size=32
data.max_response_length=8192
actor_rollout_ref.rollout.n=8
actor_rollout_ref.actor.ppo_mini_batch_size=32
```

不要从 512 token smoke test 直接跳到全部正式参数。建议逐步增加响应长度和 rollout 数量，同时观察显存：

```bash
srun --jobid=<JOB_ID> --overlap nvidia-smi
```

## 14. 常见问题

### Docker daemon permission denied

```text
permission denied while trying to connect to /var/run/docker.sock
```

当前用户不属于 `docker` 组。不要使用 `sudo` 或修改 socket 权限；加载 `singularity/4.0.2` 并使用 SIF 镜像。

### xFormers 没有兼容 wheel

```text
No matching distribution found for xformers==0.0.29.post2
```

这是宿主 glibc 2.17 导致的。不要降级到 xFormers 0.0.27，也不要使用 `vllm --no-deps` 绕过。容器中已提供匹配的 xFormers。

### 容器中看不到 GPU

确认使用：

```bash
singularity exec --nv ...
```

并且作业申请了 GPU：

```bash
#SBATCH --gres=gpu:4
```

### 计算节点找不到 SIF

确认 SIF 位于共享文件系统：

```bash
srun --partition=gpu_chen --gres=gpu:1 --time=00:05:00 \
  ls -lh /share/home/dengkn/SDPO_SVC/.runtime/containers/verl-vllm085.sif
```

### `train.parquet` 不存在

运行 `data/preprocess.py`，并确认 `TASK` 指向包含 `train.parquet` 和 `test.parquet` 的目录。

### Hugging Face 模型重复下载

确认每次作业都设置并挂载：

```bash
HF_HOME=/share/home/dengkn/SDPO_SVC/.runtime/huggingface
```

### Ray 残留进程或目录冲突

每个作业使用独立路径：

```bash
/tmp/$USER/sdpo-$SLURM_JOB_ID
```

作业开始和结束都执行 `ray stop -f`。

## 15. 训练前检查清单

```text
[ ] singularity/4.0.2 可以加载
[ ] SIF 文件已下载且计算节点可读
[ ] singularity exec --nv 能识别 A800
[ ] torch、vLLM、xFormers、Ray 可以导入
[ ] 容器 venv 已创建
[ ] 当前 SDPO 源码以 editable 模式安装
[ ] train.parquet 和 test.parquet 已生成
[ ] HF_HOME 指向持久化目录
[ ] 模型已下载，或计算节点能够访问 Hugging Face
[ ] logs、checkpoints、wandb 目录存在
[ ] Ray 和 Triton 临时目录位于 SLURM_TMPDIR 或 /tmp
[ ] Slurm GPU 数量与 trainer.n_gpus_per_node 一致
[ ] 正式训练前已通过 1 epoch smoke test
```

