# SDPO_SVC Conda + Slurm 环境安装问题复盘

本文总结在老版本 Linux 集群上，以 Conda、pip 和 Slurm 配置 SDPO_SVC 训练环境时遇到的问题及解决办法。本文不涉及 Docker、Singularity、Apptainer 或其他容器方案。

## 1. 集群环境与目标版本

本次安装环境：

```text
项目目录：/share/home/dengkn/SDPO_SVC
Conda 环境：/share/home/dengkn/miniconda3/envs/sdpo
Python：3.12
GPU：NVIDIA A800 80GB
GPU 架构：8.0
驱动：535.104.12
系统 glibc：2.17
系统默认 GCC：4.8.5
可用 GCC 模块：最高 gcc/9.1.0
可用 CUDA 模块：最高 cuda/12.3.0
PyTorch：2.6.0+cu124
vLLM：0.8.5
```

这次安装困难的根本原因不是单个软件包，而是以下限制叠加：

- 系统 glibc 2.17 较老，新包的 wheel 可能要求 glibc 2.28。
- 默认 GCC 4.8.5 太旧，无法编译现代 NumPy、SciPy 和 CUDA 扩展。
- Python 3.12 对部分旧版 CUDA 软件包的 wheel 覆盖不完整。
- PyTorch、CUDA、vLLM、xFormers 和 FlashAttention 之间存在严格的版本约束。
- pip 找不到兼容 wheel 时会自动下载源码包并尝试本地编译。
- 项目自身不同依赖文件中的 NumPy 约束不完全一致。

## 2. 没有提前检查 GCC，导致源码编译失败

### 现象

安装 SciPy、NumPy、xFormers 或其他包含扩展的包时，出现 Meson、Cython 或编译器错误。例如：

```text
gcc (GCC) 4.8.5
ERROR: Compiler cython cannot compile programs
NumPy requires GCC >= 10.3
```

### 原因

登录节点和计算节点默认使用 GCC 4.8.5。pip 没找到可用 wheel 后转为源码编译，而现代科学计算软件已经不支持该版本。

即使加载集群最高的 GCC 9.1.0，也仍然无法编译要求 GCC 10.3 以上的最新 NumPy。因此不能只升级到 GCC 9，还必须避免编译过新的软件包。

### 解决办法

进入计算节点后加载 GCC 9，并明确设置编译器：

```bash
module load gcc/9.1.0

export CC="$(command -v gcc)"
export CXX="$(command -v g++)"

gcc --version
g++ --version
```

创建环境之前应先运行：

```bash
command -v gcc
command -v g++
gcc --version
g++ --version
ldd --version | head -n 1
python --version
```

## 3. Python 3.12 导致部分指定版本没有 wheel

### 现象

安装 `vllm==0.8.5` 时，pip 无法满足它要求的：

```text
xformers==0.0.29.post2
```

索引中只能找到较旧版本：

```text
0.0.27
0.0.27.post1
0.0.27.post2
```

### 原因

当前软件索引没有提供适配 Python 3.12、当前平台和 CUDA 组合的 `xformers==0.0.29.post2` wheel。

### 解决办法

本次环境通过源码编译 xFormers 解决。若重新创建环境，更推荐使用 Python 3.11：

```bash
conda create -n sdpo python=3.11 -y
conda activate sdpo
```

Python 3.12 并非不能使用，但在 glibc 2.17 的老集群上，会增加旧版 CUDA 软件包没有兼容 wheel 的概率。

## 4. xFormers 源码版本显示成 0.0.30

### 现象

目标版本是：

```text
0.0.29.post2
```

但从源码目录导入时显示：

```text
0.0.30+1298453.d...
```

并提示 C++ 扩展没有正确加载。

### 原因

有两个可能因素：

1. xFormers 源码根据 Git 提交动态生成版本号。
2. 在 xFormers 源码目录中执行 Python 时，当前目录优先于 site-packages，导入的是源码目录，而非已安装的 wheel。

### 解决办法

构建前固定版本：

```bash
export BUILD_VERSION=0.0.29.post2
```

构建并安装 wheel 后，离开源码目录再验证：

```bash
cd /tmp

python - <<'PY'
import importlib.metadata
import inspect
import xformers

print("version:", importlib.metadata.version("xformers"))
print("path:", inspect.getfile(xformers))
PY
```

不要在待验证包的源码目录中执行导入测试。

## 5. FlashAttention 编译时找不到 CUDA Toolkit

### 现象

安装 FlashAttention 时出现：

```text
No CUDA runtime is found, using CUDA_HOME='.'
FileNotFoundError: ./bin/nvcc
```

系统默认 `nvcc` 又是 CUDA 11.6，而 PyTorch 是 `2.6.0+cu124`。

### 原因

PyTorch wheel 自带运行时库，但不等于系统安装了完整 CUDA Toolkit。编译 FlashAttention 等 CUDA 扩展仍然需要 `nvcc`、CUDA headers 和适当版本的 GCC。

### 解决办法

在 GPU 计算节点加载最接近 PyTorch CUDA 版本的工具链：

```bash
module load gcc/9.1.0
module load cuda/12.3.0

export CC="$(command -v gcc)"
export CXX="$(command -v g++)"
export CUDA_HOME="$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST="8.0"
export MAX_JOBS=8
```

A800 的计算能力是 8.0，因此设置：

```bash
export TORCH_CUDA_ARCH_LIST="8.0"
```

检查工具链：

```bash
nvcc --version

python - <<'PY'
import torch
print("PyTorch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
PY
```

然后再编译：

```bash
python -m pip install \
  flash-attn==2.7.4.post1 \
  --no-build-isolation \
  --no-cache-dir
```

## 6. 管理节点安装和计算节点编译混在一起

### 问题

管理节点通常没有分配 GPU，也可能没有加载正确的 CUDA Toolkit。直接在管理节点编译 CUDA 扩展会导致 CUDA 探测失败或构建环境不一致。

### 解决办法

纯 Python 包和已有二进制 wheel 可以在管理节点安装；xFormers、FlashAttention 和 GPU 冒烟测试应在分配的 GPU 计算节点进行。

申请交互式 GPU 节点：

```bash
srun \
  --partition=gpu_chen \
  --gres=gpu:1 \
  --cpus-per-task=16 \
  --mem=64G \
  --time=04:00:00 \
  --pty bash
```

真正申请 GPU 的关键参数是：

```text
--partition=gpu_chen
--gres=gpu:1
```

进入节点后重新初始化环境：

```bash
module load gcc/9.1.0
module load cuda/12.3.0

source /share/home/dengkn/miniconda3/etc/profile.d/conda.sh
conda activate sdpo
```

推荐分工：

- 管理节点：下载 wheel、安装纯 Python 包、准备数据。
- GPU 节点：编译 CUDA 扩展、验证 vLLM、执行训练。

## 7. pip 找不到 wheel 后自动源码编译

### 现象

安装日志出现：

```text
Downloading numpy-*.tar.gz
Downloading scipy-*.tar.gz
Downloading pyarrow-*.tar.gz
Downloading pandas-*.tar.gz
Installing build dependencies...
```

随后由于编译器太旧、缺少工具链或系统依赖而失败。

### 原因

`.tar.gz` 是源码包。pip 找不到与 Python、glibc 和 CPU 架构兼容的 wheel 后，会自动尝试本地源码构建。

本次遇到的典型情况：

- SciPy 源码构建进一步拉取最新版 NumPy。
- PyArrow 源码构建需要完整 Arrow C++ 构建环境。
- Pandas 3.0.5 源码构建拉取 NumPy 2.5.1。
- W&B 源码构建需要 Go 编译器。

### 解决办法

优先强制使用二进制包：

```bash
python -m pip install \
  PACKAGE==VERSION \
  --only-binary=:all: \
  --index-url https://pypi.org/simple
```

如果没有匹配版本，应依次考虑：

1. 选择稍旧但仍满足项目要求的版本。
2. 使用 conda-forge 的二进制构建。
3. 确认必须源码编译后，再准备编译器和构建工具链。

例如 Pandas 最终选择：

```bash
python -m pip install \
  pandas==2.3.2 \
  --only-binary=:all: \
  --index-url https://pypi.org/simple
```

## 8. glibc 2.17 与新 wheel 不兼容

### 现象

明明 PyPI 上存在某个版本，pip 仍然报告：

```text
No matching distribution found
```

### 原因

集群是 glibc 2.17，部分新包只提供 `manylinux_2_28` wheel。pip 会正确地忽略这些不能在当前系统运行的文件。

### 解决办法

优先选择以下 wheel：

```text
manylinux_2_17_x86_64
manylinux2014_x86_64
```

以下 wheel 不适用于当前系统：

```text
manylinux_2_28_x86_64
```

如果 PyPI 没有兼容 wheel，可以使用 conda-forge 的二进制包，例如 `tensordict` 和 `pyarrow`。

## 9. Conda 镜像源不可用、速度慢或 solver 配置错误

### 现象

部分镜像返回：

```text
404 NOT FOUND
403 FORBIDDEN
```

同时 Conda 报告：

```text
libmamba was not recognized
choose one of: classic
```

### 原因

- 镜像站不一定同步了完整的 conda-forge channel。
- 当前 Conda 版本只支持 classic solver，但配置文件中残留了 `solver: libmamba`。

### 解决办法

恢复 classic solver：

```bash
conda config --set solver classic
```

也可以对单次命令强制指定：

```bash
CONDA_SOLVER=classic conda install ...
```

使用官方 conda-forge channel：

```bash
-c https://conda.anaconda.org/conda-forge
```

例如：

```bash
CONDA_SOLVER=classic conda install \
  --override-channels \
  -c https://conda.anaconda.org/conda-forge \
  "tensordict=0.10.0" \
  --freeze-installed \
  -y
```

镜像站首页可以访问，并不代表对应的 conda-forge channel 路径一定存在，必须通过 `conda search` 或 `conda install --dry-run` 验证。

## 10. 项目中的 NumPy 约束不一致

### 现象

项目不同依赖声明中出现：

```text
setup.py：numpy<2.0.0
requirements.txt：numpy==2.1.0
```

安装其他软件包后，Conda 又将 NumPy 更新成 2.2.6，最终 `pip check` 报告：

```text
verl requires numpy<2.0.0
```

### 解决办法

固定到兼容 Python 3.12、Pandas、PyArrow 和 Verl 的版本：

```bash
python -m pip install \
  numpy==1.26.4 \
  --only-binary=:all: \
  --index-url https://pypi.org/simple
```

安装关键依赖后再次检查 NumPy，避免 Conda 或 pip 在后续解析中将它升级。

## 11. tensordict 实际版本与目标不一致

### 现象

目标版本是：

```text
tensordict==0.10.0
```

但依赖检查发现实际版本仍然是：

```text
tensordict==0.7.2
```

### 原因

之前的安装可能没有成功完成，或者后续依赖解析又替换了版本。执行过安装命令不等于最终环境仍然保持该版本。

### 解决办法

通过 conda-forge 安装并冻结已有环境：

```bash
CONDA_SOLVER=classic conda install \
  --override-channels \
  -c https://conda.anaconda.org/conda-forge \
  "tensordict=0.10.0" \
  --freeze-installed \
  -y
```

如果依赖解析会大范围修改 PyTorch，可以先检查方案，确认风险后使用：

```bash
CONDA_SOLVER=classic conda install \
  --override-channels \
  -c https://conda.anaconda.org/conda-forge \
  "tensordict=0.10.0" \
  --no-deps \
  -y
```

安装后验证：

```bash
python - <<'PY'
import importlib.metadata
print(importlib.metadata.version("tensordict"))
PY
```

## 12. W&B 源码构建要求 Go

### 现象

安装 `wandb==0.23.1` 时下载源码包，随后报错：

```text
Building wandb-core Go binary...
Did not find the 'go' binary
```

### 原因

当前平台没有匹配的新版本 wheel，pip 转而源码构建，而该版本的 W&B 核心组件需要 Go。

### 解决办法

选择仍提供 glibc 2.17 兼容 wheel 的版本：

```bash
python -m pip install \
  wandb==0.17.9 \
  --only-binary=:all: \
  --index-url https://pypi.org/simple
```

没有必要仅为了 W&B 再配置一套 Go 编译环境。

## 13. 本地 datasets 目录被误认为 Hugging Face datasets 已安装

### 现象

项目中存在：

```text
/share/home/dengkn/SDPO_SVC/datasets
```

因此 `importlib.util.find_spec("datasets")` 能返回结果，但：

```bash
python -m pip show datasets
```

仍然显示软件包没有安装。

### 原因

`find_spec()` 找到的是项目中的同名目录，而不是 Hugging Face 发布的 Python 包。

### 解决办法

离开项目目录，并通过包元数据和实际文件路径验证：

```bash
cd /tmp

python - <<'PY'
import importlib.metadata
import inspect
import datasets

print("version:", importlib.metadata.version("datasets"))
print("path:", inspect.getfile(datasets))
PY
```

判断一个发行包是否安装时，优先使用：

```bash
python -m pip show datasets
```

或者：

```python
importlib.metadata.version("datasets")
```

## 14. `pip install -e .` 重新解析并修改整个环境

### 问题

直接运行：

```bash
python -m pip install -e .
```

会根据项目元数据重新解析所有依赖，可能：

- 升级已经固定的 NumPy。
- 下载与 glibc 2.17 不兼容的新包。
- 触发不必要的源码编译。
- 修改已经配好的 PyTorch、vLLM 或 xFormers 组合。

### 解决办法

先手动安装并固定依赖，再只安装项目本身：

```bash
cd /share/home/dengkn/SDPO_SVC

python -m pip install \
  --no-deps \
  -e .
```

这种方式适合系统较老、CUDA 版本组合严格的集群环境。

## 15. Slurm 多行命令的反斜杠后存在空格

### 现象

命令中写成：

```bash
--partition=gpu_chen \ 
```

随后出现：

```text
execve(): No such file or directory
```

或者只粘贴了参数部分：

```bash
--gres=gpu:1 \
```

导致：

```text
bash: --gres=gpu:1: command not found
```

### 原因

反斜杠必须是当前行的最后一个字符。反斜杠后存在空格时，续行失效。只复制参数而漏掉开头的 `srun`，Shell 会把参数当成命令执行。

### 解决办法

先使用单行命令测试：

```bash
srun --partition=gpu_chen --gres=gpu:1 --time=00:10:00 nvidia-smi
```

确认成功后再改成多行形式，并确保每个 `\` 后面没有空格。

## 16. 复制命令时 Python 双下划线被转成 Markdown

### 现象

原本应当是：

```python
torch.__version__
```

复制后变成：

```python
torch.**version**
```

最终产生 Python 语法错误。

### 解决办法

验证版本时避免使用双下划线属性，统一使用：

```bash
python - <<'PY'
import importlib.metadata

print("torch:", importlib.metadata.version("torch"))
print("vllm:", importlib.metadata.version("vllm"))
print("xformers:", importlib.metadata.version("xformers"))
PY
```

## 17. `Successfully installed` 不代表环境整体兼容

### 现象

某个包安装结束时显示：

```text
Successfully installed ...
```

但上方同时出现：

```text
pip's dependency resolver does not currently take into account all the packages that are installed
```

### 原因

这只表示当前安装动作完成，不表示环境中所有已安装包的版本要求都满足。

### 解决办法

每批依赖安装后执行：

```bash
python -m pip check
```

并进行真实导入测试，而不是只看安装日志。

## 18. 最终环境验证

先离开项目目录，避免本地同名目录污染导入：

```bash
cd /tmp
```

检查关键包版本和路径：

```bash
python - <<'PY'
import importlib.metadata as metadata
import inspect
import torch
import vllm
import xformers
import pandas
import pyarrow
import datasets
import tensordict

packages = [
    "numpy",
    "pandas",
    "pyarrow",
    "datasets",
    "tensordict",
    "torch",
    "vllm",
    "xformers",
    "verl",
]

for package in packages:
    print(package, metadata.version(package))

print("datasets path:", inspect.getfile(datasets))
print("xformers path:", inspect.getfile(xformers))
print("PyTorch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY
```

检查依赖一致性：

```bash
python -m pip check
```

检查 Hugging Face Dataset 与 PyArrow：

```bash
python - <<'PY'
from datasets import Dataset

dataset = Dataset.from_dict({
    "text": ["one", "two"],
    "value": [1, 2],
})

print(dataset)
print(dataset[0])
PY
```

GPU 节点上检查 vLLM 的本地扩展：

```bash
python - <<'PY'
import importlib.metadata as metadata
import torch
import vllm
import vllm._C

print("torch:", metadata.version("torch"))
print("vllm:", metadata.version("vllm"))
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))
PY
```

## 19. PyTorch、vLLM 与 FlashAttention 的正确安装顺序

这三个核心组件不能随意颠倒安装。推荐的完整顺序是：

```text
基础编译环境
    ↓
PyTorch 2.6.0+cu124
    ↓
xFormers 0.0.29.post2
    ↓
vLLM 0.8.5
    ↓
FlashAttention 2.7.4.post1
    ↓
项目依赖与 verl editable 安装
```

其中 xFormers 虽然没有写在“PyTorch → vLLM → FlashAttention”这个简写中，但它是 vLLM 0.8.5 的固定依赖，不能省略。本集群没有适用于当前 Python 3.12 环境的 `xformers==0.0.29.post2` wheel，因此需要在安装 vLLM 前先完成源码构建。

### 第一步：进入 GPU 计算节点并准备编译环境

CUDA 扩展必须在加载了正确工具链的计算节点上构建：

```bash
srun \
  --partition=gpu_chen \
  --gres=gpu:1 \
  --cpus-per-task=16 \
  --mem=64G \
  --time=04:00:00 \
  --pty bash
```

进入节点后：

```bash
module load gcc/9.1.0
module load cuda/12.3.0

source /share/home/dengkn/miniconda3/etc/profile.d/conda.sh
conda activate sdpo

export CC="$(command -v gcc)"
export CXX="$(command -v g++)"
export CUDA_HOME="$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST="8.0"
export MAX_JOBS=8
```

先确认没有混用系统默认 GCC 4.8.5 或 CUDA 11.6：

```bash
gcc --version
g++ --version
nvcc --version
python --version
```

### 第二步：先安装并固定 PyTorch

PyTorch 是 vLLM、xFormers 和 FlashAttention 的编译及运行基础，必须最先安装：

```bash
python -m pip install \
  torch==2.6.0 \
  torchvision==0.21.0 \
  torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124
```

立即验证版本，不要继续安装直到此处正确：

```bash
python - <<'PY'
import importlib.metadata as metadata
import torch

print("torch:", metadata.version("torch"))
print("torchvision:", metadata.version("torchvision"))
print("torchaudio:", metadata.version("torchaudio"))
print("PyTorch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))
PY
```

预期关键结果：

```text
torch: 2.6.0
PyTorch CUDA: 12.4
CUDA available: True
GPU: NVIDIA A800 80GB PCIe
```

### 第三步：安装 vLLM 所需的 xFormers

vLLM 0.8.5 要求 `xformers==0.0.29.post2`。必须在 PyTorch 安装完成后构建，因为 xFormers 的 C++/CUDA 扩展需要链接当前环境中的 PyTorch。

构建时固定版本和 A800 架构：

```bash
export BUILD_VERSION=0.0.29.post2
export TORCH_CUDA_ARCH_LIST="8.0"
export MAX_JOBS=8
```

构建完成后应安装生成的 wheel，而不是长期从源码目录直接导入。安装后离开源码目录验证：

```bash
cd /tmp

python - <<'PY'
import importlib.metadata as metadata
import inspect
import xformers

print("xformers:", metadata.version("xformers"))
print("path:", inspect.getfile(xformers))
PY
```

预期版本：

```text
0.0.29.post2
```

### 第四步：安装 vLLM 0.8.5

确认 PyTorch 和 xFormers 正确后再安装 vLLM：

```bash
python -m pip install \
  vllm==0.8.5 \
  --index-url https://pypi.org/simple
```

安装日志中需要特别检查 pip 是否准备卸载或替换以下核心包：

```text
torch
torchvision
torchaudio
xformers
```

如果 pip 仍试图替换已经固定的版本，应停止安装，先手动解决缺少的普通依赖，再使用：

```bash
python -m pip install \
  vllm==0.8.5 \
  --no-deps \
  --index-url https://pypi.org/simple
```

`--no-deps` 只适用于已经手动补齐 vLLM 依赖的环境，不能把它当作忽略所有依赖问题的通用方案。

在 GPU 节点验证 vLLM Python 包和本地扩展：

```bash
cd /tmp

python - <<'PY'
import importlib.metadata as metadata
import torch
import vllm
import vllm._C

print("torch:", metadata.version("torch"))
print("vllm:", metadata.version("vllm"))
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))
PY
```

仅仅能够 `import vllm` 不足以证明安装完整；`import vllm._C` 成功才能说明其本地扩展可以被加载。

### 第五步：最后安装 FlashAttention

FlashAttention 应在 PyTorch、编译器和 CUDA Toolkit 全部固定后安装。把它放在 vLLM 后面，便于先确认 vLLM 的固定依赖已经稳定，也可避免后续依赖解析意外更换 PyTorch。

```bash
module load gcc/9.1.0
module load cuda/12.3.0

export CC="$(command -v gcc)"
export CXX="$(command -v g++)"
export CUDA_HOME="$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")"
export TORCH_CUDA_ARCH_LIST="8.0"
export MAX_JOBS=8

python -m pip install \
  flash-attn==2.7.4.post1 \
  --no-build-isolation \
  --no-cache-dir
```

使用 `--no-build-isolation` 是为了让构建过程直接使用当前环境中已经固定的 PyTorch 2.6.0，而不是在临时构建环境中重新拉取另一套 PyTorch。

验证：

```bash
cd /tmp

python - <<'PY'
import importlib.metadata as metadata
import flash_attn

print("flash-attn:", metadata.version("flash-attn"))
PY
```

### 第六步：最后安装项目本身

核心 CUDA 软件栈稳定后，再安装普通依赖，最后以不解析依赖的方式安装项目：

```bash
cd /share/home/dengkn/SDPO_SVC

python -m pip install \
  --no-deps \
  -e .
```

### 为什么不能颠倒顺序

- 先装 vLLM、后装 PyTorch：安装 PyTorch 时可能覆盖 vLLM 所依赖的 ABI 和 CUDA 组合。
- 先编译 xFormers 或 FlashAttention、后换 PyTorch：已经编译的扩展可能与新 PyTorch ABI 不匹配，需要重新编译。
- 让 vLLM 自动选择所有依赖：在 Python 3.12 和 glibc 2.17 环境中，可能因为 xFormers wheel 不存在而失败，或者触发源码构建。
- 最后重新执行带依赖解析的 `pip install -e .`：可能再次升级 NumPy、PyTorch 或其他已经固定的组件。

### 核心软件栈的最终联合检查

在 GPU 节点执行：

```bash
cd /tmp

python - <<'PY'
import importlib.metadata as metadata
import torch
import vllm
import vllm._C
import xformers
import flash_attn

for package in ["torch", "torchvision", "torchaudio", "xformers", "vllm", "flash-attn"]:
    print(package, metadata.version(package))

print("PyTorch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))
PY

python -m pip check
```

本次环境的目标组合为：

```text
torch         2.6.0+cu124
torchvision   0.21.0+cu124
torchaudio    2.6.0+cu124
xformers      0.0.29.post2
vllm          0.8.5
flash-attn    2.7.4.post1
```

如果未来必须更换 PyTorch 版本，应将 xFormers 和 FlashAttention 视为需要重新构建，而不是假定原来的扩展可以继续使用。

## 20. 推荐的后续安装原则

在同类老集群上重新配置环境时，建议遵循以下顺序：

1. 调查 Python、GCC、glibc、CUDA、驱动和 GPU 架构。
2. 优先选择 Python 3.10 或 3.11。
3. 先固定 PyTorch、torchvision 和 torchaudio。
4. 再处理 vLLM、xFormers 和 FlashAttention。
5. 科学计算包优先使用 `manylinux_2_17` 或 `manylinux2014` wheel。
6. 看到 `.tar.gz` 时先暂停，确认是否真的需要源码编译。
7. 对 glibc 不兼容的软件包使用 conda-forge 二进制构建。
8. 项目使用 `pip install --no-deps -e .`，依赖分批固定版本。
9. 管理节点安装普通依赖，GPU 节点编译 CUDA 扩展并测试。
10. 每一步都通过版本、导入路径和 `pip check` 验证。

本次配置最终采用的是“固定版本、优先 wheel、必要时源码编译、分批安装、逐步验证”的方案。对于 glibc 2.17 和 GCC 9.1 的老集群，这是比直接执行完整 `requirements.txt` 更可靠的方式。
