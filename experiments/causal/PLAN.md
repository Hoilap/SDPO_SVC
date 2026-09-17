# GRPO Tail Directions 是否导致其比 SDPO 更具泛化性

状态：**首轮代码已实现，尚未启动实验**。运行方式见 [README.md](README.md)。

核心问题：

> GRPO 在 Math 训练中保留了一些对当前 Math 性能贡献很小的 tail directions。这些方向是否
> 是 GRPO 相比 SDPO 具有更好 OOD 泛化和持续学习能力的原因？

首轮只运行 **1 个配对 seed**，用于确认实验能否运行以及结果方向是否符合假设。一个 seed
不能支持统计显著性或普遍性结论。实验全程不使用 SVC，也不重复验证 low-rank lock-in 或
parameter-direction overlap。

## 1. 使用什么数据集

### 1.1 训练顺序

```text
Task A：Math    → DAPO-Math-17k
Task B：Science → SciKnowEval train

顺序：Math → Science
```

GRPO 和 SDPO 使用同一个基础模型、同一份训练数据、相同数据顺序、训练步数、prompt 数、
rollout 数和生成 token budget。

**每个单任务实际参与训练的样本数最多为 3000。** Math 和 Science 都设置
`data.train_max_samples=3000`；即使底层数据集更大也不得放宽。若有效样本不足 3000，则使用
全部有效样本并记录实际数量。GRPO/SDPO 以及所有 continuation arms 必须使用同一批、同一
顺序的至多 3000 个样本。

### 1.2 评测数据

| 评测目的 | 数据集 | 什么时候评测 |
| --- | --- | --- |
| 当前 Math 能力 | AIME24、AIME25 | 训练完 Math；训练完 Science |
| Math OOD 泛化/保留 | MATH-500 | 训练完 Math；训练完 Science |
| Science 学习效果 | SciKnowEval test | 训练 Science 前；训练 Science 过程中及结束后 |
| Science OOD 泛化 | GPQA Diamond | 训练 Science 前；训练 Science 后 |
| 未训练 Tool 能力 | ToolUse test | 训练完 Math；训练完 Science |
| 未训练 Code 能力 | LiveCodeBench-v6 | 训练完 Math；训练完 Science |

这些评测对象必须分开解释：

- AIME24/25：Task-A Math 能力；
- MATH-500：Math 域内 OOD；
- SciKnowEval test：Task-B 是否正常学会；
- GPQA：Science OOD 泛化；
- ToolUse/LiveCodeBench：始终没有训练过的远领域泛化。

首轮固定使用 top-10%-rank cutoff，不根据上述 test 结果调整 cutoff。如果删除 tail 后 Math
能力明显下降，则说明该 cutoff 没有成功提取“低即时效用 tail”，该轮结果记为不支持/无法
检验假设，不能事后换 cutoff 再把结果当作预注册实验。

## 2. 实验具体思路

### 2.1 在 Math 上分别训练 GRPO 和 SDPO

两种方法从同一个基础模型 `W0` 开始训练 Math，得到：

```text
W_math_GRPO
W_math_SDPO
```

对应的参数更新为：

\[
\Delta W_{GRPO}=W_{math,GRPO}-W_0,
\qquad
\Delta W_{SDPO}=W_{math,SDPO}-W_0.
\]

### 2.2 把 Math update 分为 principal 和 tail

对 attention 和 MLP 中的每个二维权重矩阵分别做 SVD：

\[
\Delta W=P+T.
\]

- `P`：每层 top 10% rank，称为 principal update；
- `T`：剩余 bottom 90% rank，称为 tail update。

GRPO 和 SDPO 分别得到：

```text
GRPO: P_G + T_G
SDPO: P_S + T_S
```

非二维参数不参与切分，在同一方法的不同 checkpoint 中保持训练 Math 后的值。每层必须满足
相对重构误差 `< 1e-5`。

### 2.3 阶段 1：训练完 Math 后做 checkpoint 干预

构造以下 8 个 checkpoint；这些 checkpoint 不需要重新训练，只需重组 Math update：

| Checkpoint | 内容 | 目的 |
| --- | --- | --- |
| `GRPO-full` | `P_G + T_G` | 原始 GRPO |
| `GRPO-no-tail` | 只保留 `P_G` | 检验 GRPO tail 是否必要 |
| `GRPO-random-tail` | `P_G + 随机 tail` | 排除“仅仅多了一些 update norm”的解释 |
| `SDPO-full` | `P_S + T_S` | 原始 SDPO |
| `SDPO-no-tail` | 只保留 `P_S` | 测量 SDPO tail 的作用 |
| `SDPO-random-tail` | `P_S + 随机 tail` | SDPO 的随机方向对照 |
| `GRPO-principal + SDPO-tail` | `P_G + T_S` | 查看 SDPO tail 放到 GRPO host 上的效果 |
| `SDPO-principal + GRPO-tail` | `P_S + T_G` | 检验 GRPO tail 能否把泛化能力转移给 SDPO host |

随机 tail 逐层匹配真实 tail 的 Frobenius norm 和尽可能相同的奇异值谱，但使用随机方向。
跨方法移植时，将 donor tail 逐层缩放到 host 原 tail 的 norm，避免把尺度差异误当作方向作用。

对 8 个 checkpoint 评测所有数据集，但阶段 1 的检验对象如下：

| 检验对象 | 使用的数据集 | 关键比较 |
| --- | --- | --- |
| tail 对当前任务是否“几乎无用” | AIME24、AIME25 | `GRPO-full` vs `GRPO-no-tail` |
| Math OOD 泛化 | MATH-500 | 所有 8 个 checkpoint |
| 对未来 Science 的 forward generalization | SciKnowEval test、GPQA | 所有 8 个 checkpoint；此时尚未训练 Science |
| 远领域泛化 | ToolUse、LiveCodeBench | 所有 8 个 checkpoint |

阶段 1 的因果逻辑是：

1. `GRPO-full` 与 `GRPO-no-tail` 的 AIME 表现接近，说明 tail 对当前 Math 的即时效用很低；
2. 原始 `GRPO-full` 的 OOD 表现高于 `SDPO-full`，确认存在需要解释的方法差距；
3. 删除 `T_G` 后，GRPO 的 OOD 表现下降，说明 `T_G` 对 GRPO 泛化有贡献；
4. 比较“`GRPO-full - SDPO-full`”和“`GRPO-no-tail - SDPO-no-tail`”，若后者明显更小，
   说明 tail 差异能解释一部分方法差距；
5. `GRPO-random-tail` 无法恢复表现，说明有效的是 tail 方向，而不只是 update norm；
6. `SDPO-principal + GRPO-tail` 得到提升，说明 GRPO tail 具有一定可转移性/充分性。

### 2.4 阶段 2：从核心 checkpoint 继续训练 Science

为了控制训练成本，只继续训练以下 4 个 checkpoint：

```text
GRPO-full    → 使用 GRPO 训练 Science
GRPO-no-tail → 使用 GRPO 训练 Science
SDPO-full    → 使用 SDPO 训练 Science
SDPO-no-tail → 使用 SDPO 训练 Science
```

Science 阶段为所有 checkpoint 重置 optimizer 和 scheduler，使用相同 prompt 顺序、训练步数、
rollout/token budget 和配对随机设置。SDPO teacher 从对应 checkpoint 自身初始化。

在 Science 训练进度 `0%、25%、50%、75%、100%` 评测以下对象：

| 检验对象 | 数据集 | 要比较什么 |
| --- | --- | --- |
| Science 是否正常学会 | SciKnowEval test | 同一方法的 `full` 与 `no-tail` 是否达到相近的 Science 水平 |
| Math retention | AIME24、AIME25、MATH-500 | 删除 tail 是否造成更多 Math forgetting |
| Science OOD 泛化 | GPQA Diamond | GRPO 的 tail 是否比 SDPO tail 更有助于 GPQA |
| 未训练领域泛化 | ToolUse、LiveCodeBench | GRPO 的 tail 是否保存更多通用能力 |

对任意指标 `Y`，分别计算：

\[
TailEffect_{GRPO}(Y)=Y(GRPO\text{-full})-Y(GRPO\text{-no-tail}),
\]

\[
TailEffect_{SDPO}(Y)=Y(SDPO\text{-full})-Y(SDPO\text{-no-tail}).
\]

上式用于 GPQA、ToolUse、LiveCodeBench 等“越高越好”的最终分数。Math retention 使用每个
checkpoint 自己的 Science 训练前后差值：

\[
Forgetting_{Math}=Math_{before\ Science}-Math_{after\ Science},
\]

然后比较 `Forgetting(no-tail)-Forgetting(full)`；正值表示 tail 减少了遗忘。

核心比较是：

\[
TailEffect_{GRPO}(Y)-TailEffect_{SDPO}(Y).
\]

它回答 tail 对 GRPO 的贡献是否大于对 SDPO 的贡献。还要比较：

```text
原始方法差距：GRPO-full - SDPO-full
无 tail 方法差距：GRPO-no-tail - SDPO-no-tail
```

如果删除双方 tail 后方法差距明显缩小，且 GRPO 的 tail effect 大于 SDPO，tail explanation
才得到支持。

注意：阶段 2 没有继续训练 random-tail 和 transplant checkpoint。因此阶段 2 只检验
tail 的必要性和 GRPO/SDPO 差异；“真实方向而非 update norm”以及“能否移植”的证据来自
阶段 1。

### 2.5 首轮运行量

```text
Training seed:             1 个配对 seed
Per-task train samples:    ≤3000（硬上限）
Math training:             2 runs（GRPO、SDPO）
Math 后 checkpoint:        8 个（重组参数，不需要训练）
Math 后 evaluation:        8 checkpoints × 全部评测集
Science continuation:      4 runs（full/no-tail × GRPO/SDPO）
Science 过程及结束评测:    4 runs × 全部评测集
SVC:                       不使用
```

## 3. 预期结果及能够证明的结论

### 3.1 阶段 1：只训练完 Math

预期看到：

```text
AIME(GRPO-full) ≈ AIME(GRPO-no-tail)

OOD(GRPO-full) > OOD(SDPO-full)
OOD(GRPO-no-tail) < OOD(GRPO-full)
|OOD(GRPO-no-tail) - OOD(SDPO-no-tail)|
  < |OOD(GRPO-full) - OOD(SDPO-full)|

OOD(GRPO-random-tail) < OOD(GRPO-full)
OOD(SDPO-principal + GRPO-tail) > OOD(SDPO-full)
```

这里的 `OOD` 必须分别报告：

- MATH-500；
- Science-forward：SciKnowEval、GPQA；
- ToolUse；
- LiveCodeBench。

不能只用一个 macro 掩盖数据集间的异质性。

若以上模式成立，可以得到：

> GRPO 的真实 tail directions 对当前 Math 性能贡献很小，却对 Math OOD、未来 Science 或
> 远领域泛化有贡献；它们能够解释一部分 GRPO–SDPO 单任务泛化差距。

### 3.2 阶段 2：继续训练完 Science

首先要求：

```text
SciKnowEval(GRPO-full) ≈ SciKnowEval(GRPO-no-tail)
SciKnowEval(SDPO-full) ≈ SciKnowEval(SDPO-no-tail)
```

这用于确认 retention/OOD 差异不是因为某个 checkpoint 没有学会 Science。

随后预期看到：

```text
Math forgetting(GRPO-no-tail) > Math forgetting(GRPO-full)
GRPO 的 Math tail effect > SDPO 的 Math tail effect

GPQA(GRPO-full) > GPQA(SDPO-full)
删除双方 tail 后，GRPO–SDPO 的 GPQA 差距缩小

Tool/Code(GRPO-full) > Tool/Code(SDPO-full)
删除双方 tail 后，GRPO–SDPO 的 Tool/Code 差距缩小
```

不同结果对应不同结论：

| 观察结果 | 能够支持的结论 |
| --- | --- |
| 只在 Math retention 上成立 | GRPO tail 主要保护旧任务，支持 continual retention 解释 |
| 在 GPQA 上成立 | GRPO tail 有助于学习 Science 后的 Science OOD 泛化 |
| 在 ToolUse/LiveCodeBench 上成立 | GRPO tail 有助于保存更广泛的未训练领域能力 |
| random tail 无效且 GRPO tail 移植有效 | 作用来自特定 tail directions，而非参数范数 |
| 删除 tail 后 GRPO–SDPO 差距不缩小 | tail 不能解释两种方法的泛化差距，需寻找 principal 或其他算法因素 |
| 删除 tail 同时显著降低 Science 学习效果 | 存在 stability–plasticity trade-off，不能称为“无代价”的泛化优势 |

### 3.3 最强结论需要满足什么

若同时满足以下条件：

1. 删除 GRPO tail 基本不影响当前 Math；
2. GRPO 原本确实比 SDPO 泛化更好；
3. 删除双方 tail 后，GRPO–SDPO 的泛化差距明显缩小；
4. 随机 tail 不能恢复优势；
5. GRPO tail 移植到 SDPO principal 后能够提升泛化；
6. 训练 Science 后，GRPO tail 对 Math retention、GPQA 或 Tool/Code 的贡献大于 SDPO tail；

则单 seed pilot 可以表述为：

> 在该训练轨迹上，观察到与“GRPO 保留的特定 tail directions 导致其比 SDPO 更具泛化性”
> 一致的因果证据模式。

但一个 seed 仍不能支持“该机制普遍成立”的正式结论。正式结论需要增加独立 training seeds，
并至少在另一组 Task A → Task B 上复现。
