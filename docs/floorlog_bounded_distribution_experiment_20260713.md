# Floor-log 有界分布实验（2026-07-13）

## 1. 目的

本实验比较四类在训练 likelihood 内直接建模有限区间的分布，而不是在推理后对样本施加 clamp、linear shrink 或 tanh shrink：

- Beta mixture
- tanh-transformed Student-t
- bounded skew-normal
- logit-normal / mixture logit-normal

所有实验仅继承 `k8-b256-veldlm` 的网络、数据和优化超参数，均从随机初始化开始训练，不加载其 checkpoint 或 pretrained backbone。

## 2. 协议

- 训练集：ASAP，固定 score-span window split。
- 短跑预算：4 epochs，836 optimizer steps。
- 每个 family 独占一张 RTX 3090；family 内四个配置串行。
- 推理集合：`data/cheap15_score_sources.txt`。
- 汇总规模：15 个 score，38 个 ground-truth performance；deterministic 与 sampling 各 15 个预测 MIDI。
- 表中为 score-level aggregate Wasserstein，数值越低越好。
- Pedal 仍为 `binary_4 + BCE`，不是本次新分布的一部分。

Timing 的真实 floor-log deviation 先按以下固定支持映射到 `(0,1)`：

- zero-score-IOI：`[0.0, 5.0]`
- nonzero-score-IOI：`[-2.5, 1.5]`
- duration：`[-3.0, 2.0]`
- velocity：原始标签已经位于 `[0,1]`

因此分布的 location/scale/shape/mixture weight 由训练学习，location 不被强制为 0；但本轮支持区间本身是固定的，并没有实现逐音符可学习的 `L(x), U(x)` 或半径。这一点必须与“可学习支持宽度”实验区分。

## 3. Sampling PN Wasserstein

| Family | 配置 | IOI | Duration | Velocity | Pedal |
|---|---|---:|---:|---:|---:|
| Beta | beta-k2 | 48.89 | 187.67 | **18.37** | 0.465 |
| Beta | beta-k3 | 47.81 | 187.24 | 18.47 | 0.472 |
| Beta | beta-k5 | 46.56 | 177.78 | 18.84 | 0.479 |
| Beta | **beta-k8** | **44.86** | **173.92** | 18.71 | 0.470 |
| Tanh-t | tanh-smax2 | 142.21 | 549.60 | 38.42 | 0.474 |
| Tanh-t | tanh-smax5 | 142.21 | 549.60 | 38.44 | 0.486 |
| Tanh-t | tanh-smin1e-2 | 142.21 | 549.60 | 38.42 | 0.483 |
| Tanh-t | tanh-smin1e-3 | 142.21 | 549.60 | 38.42 | 0.484 |
| Bounded SN | **sn-smax2** | **127.85** | **256.53** | **25.98** | 0.475 |
| Bounded SN | sn-smax5 | 154.67 | 257.49 | 26.87 | **0.473** |
| Bounded SN | sn-smin1e-3 | 154.67 | 257.49 | 26.87 | 0.475 |
| Bounded SN | sn-smin1e-4 | 154.67 | 257.49 | 26.87 | 0.477 |
| Logit-normal | ln-k1 | 100.29 | 219.11 | **18.57** | **0.467** |
| Logit-normal | **ln-k2** | **48.74** | 180.35 | 19.45 | 0.471 |
| Logit-normal | ln-k3 | 57.75 | 184.13 | 18.98 | 0.467 |
| Logit-normal | ln-k5 | 53.70 | **177.58** | 18.97 | 0.469 |
| 16-epoch DLM reference | k8-b256-veldlm | 63.56 | 197.10 | 22.84 | 0.488 |

Beta-k8 是本轮 PN timing 最好配置。LN-k2 的 IOI 接近 Beta，但 duration 略差；LN-k5 的 duration 接近 Beta-k5/k8。单分量 LN、tanh-t 和 bounded SN 的 IOI sampling 都明显失控。

## 4. Sampling PP Wasserstein

| Family | 配置 | IOI | Duration | Velocity | Pedal |
|---|---|---:|---:|---:|---:|
| Beta | beta-k2 | 12.77 | 61.11 | **6.36** | 0.257 |
| Beta | beta-k3 | 12.61 | 62.25 | 6.39 | 0.238 |
| Beta | beta-k5 | 11.34 | 54.31 | 6.83 | 0.259 |
| Beta | **beta-k8** | **10.88** | **48.30** | 7.05 | **0.203** |
| Tanh-t | tanh-smax2 | 70.84 | 423.94 | 23.64 | 0.247 |
| Tanh-t | tanh-smax5 | 70.85 | 423.94 | 23.69 | 0.257 |
| Tanh-t | tanh-smin1e-2 | 70.84 | 423.94 | 23.64 | **0.230** |
| Tanh-t | tanh-smin1e-3 | 70.84 | 423.94 | 23.64 | 0.269 |
| Bounded SN | **sn-smax2** | **57.77** | **119.66** | **10.26** | 0.233 |
| Bounded SN | sn-smax5 | 79.82 | 119.97 | 17.30 | **0.197** |
| Bounded SN | sn-smin1e-3 | 79.82 | 119.97 | 17.30 | 0.222 |
| Bounded SN | sn-smin1e-4 | 79.82 | 119.97 | 17.30 | 0.198 |
| Logit-normal | ln-k1 | 38.36 | 77.34 | 6.10 | **0.214** |
| Logit-normal | **ln-k2** | **13.19** | **52.90** | 7.81 | 0.270 |
| Logit-normal | ln-k3 | 16.57 | 68.12 | **5.39** | 0.258 |
| Logit-normal | ln-k5 | 14.91 | 55.53 | 7.31 | 0.269 |
| 16-epoch DLM reference | k8-b256-veldlm | 32.46 | 73.48 | 14.50 | 0.255 |

Beta-k8 同时取得全局最佳 PP IOI、duration 和 pedal；LN-k3 的 velocity 最低，但 timing 明显弱于 Beta-k8。Beta-k2/k3 在 velocity 上优于 Beta-k8，说明增加 mixture component 改善 timing 与 pedal 时存在轻微 velocity trade-off。

## 5. Deterministic 与 sampling 的差异

| 配置 | Det PN IOI | Det PN Dur | Sampling PN IOI | Sampling PN Dur |
|---|---:|---:|---:|---:|
| beta-k8 | 28.10 | 121.09 | 44.86 | 173.92 |
| ln-k2 | 28.84 | 116.85 | 48.74 | 180.35 |
| tanh-smax2 | 45.75 | 116.18 | 142.21 | 549.60 |
| sn-smax2 | 45.75 | 116.18 | 127.85 | 256.53 |

四族的 sampling PN 均差于 deterministic PN。Beta/LN 的退化相对有限，而 tanh-t/SN 的退化非常大。这说明问题不只是输出是否越界：即使所有样本严格处于训练支持内，模型仍可能学习过宽的条件分布。

因此，之前 clamp/tanh shrink 能把 PN 压回约 `30/120`，其主要作用不是修复非法区间，而是削弱 sampling variance。仅将分布改成有界 family，并不会自动学到足够窄的随机尺度。

## 6. Family 内部分析

### 6.1 Beta

component 从 2 增加到 8 时，sampling timing 基本单调改善：

- PN IOI：48.89 → 44.86
- PN duration：187.67 → 173.92
- PP IOI：12.77 → 10.88
- PP duration：61.11 → 48.30

这表明 Beta family 确实受益于 mixture capacity，不是单纯依赖固定边界。`beta-k8` 是完整训练最有价值的候选。

### 6.2 Logit-normal

LN 对 component 数量不单调：

- k1 明显欠拟合或尺度过宽；
- k2 的 PN IOI 最好；
- k5 的 PN duration 最好；
- k3 的 velocity PP 最好，但 timing 退化。

不能仅依据 eval NLL 选模型：LN-k5 的 best eval loss 最低，但外部 PN/PP 并未全面最好。后续应以 score-level Wasserstein 作为筛选依据，首选 LN-k2，并保留 LN-k5 作为 duration 对照。

### 6.3 Tanh-transformed Student-t

四组结果几乎完全相同。所改的 `sigma_min/sigma_max` 没有约束到实际学习尺度，因而这些配置并不是有效的 family 内消融。Student-t 重尾再经 tanh 映射后，会把较多质量推向支持边缘，造成最严重的 sampling timing 退化。

本轮结果足以淘汰当前参数化，但不足以证明所有 tanh-transformed family 都无效。若继续，应固定或正则化 latent scale/df，而不是继续只改未触发的 clip 上下限。

### 6.4 Bounded skew-normal

`sigma_max=2` 明显优于其余 SN 配置，说明限制 latent scale 有效，但仍远逊于 Beta/LN。其余三组几乎相同，说明 `sigma_max=5`、`sigma_min=1e-3/1e-4` 都没有形成有效约束。

bounded SN 的 deterministic duration 尚可，但 sampling IOI 很差；这再次指向尺度而非 location 是主要问题。

## 7. 结论与下一步

1. **首选 Beta-k8。** 它是唯一同时在 PN timing 和 PP timing 上稳定领先的配置，并且 pedal PP 也最好。
2. **第二候选是 LN-k2。** 它验证了 bounded mixture logit-normal 可行，但 component 数量敏感且不单调。
3. **当前 tanh-t 与 bounded SN 不应直接进入 16-epoch 复跑。** 两者的 sampling variance 明显失控；增加训练轮数不等价于解决尺度问题。
4. **有限支持不等于适当的分布宽度。** 本轮没有复现 shrink 后约 `30/120` 的 PN 平台，说明需要显式的 scale/radius regularization 或条件支持宽度约束。
5. **不要宣称本轮验证了可学习半径。** 本轮学习的是固定全局支持内部的分布参数；`L/U` 本身没有学习。

建议下一阶段先做两个完整 16-epoch 复跑：

- `beta-k8`
- `ln-k2`

同时增加一个针对 sampling scale 的训练期正则消融，例如约束预测分布相对其 location 的条件标准差，而不是推理后 shrink。若仍希望研究可学习半径，应单独实现带覆盖约束的 `L(x),U(x)`，不能把本轮固定支持结果当成该方案的证据。

## 8. 产物

- 运行目录：`results/floorlog_bounded_families_4gpu/20260713_bounded_families_short4`
- 配置清单：`results/floorlog_bounded_families_4gpu/20260713_bounded_families_short4/configs/manifest.json`
- 启动脚本：`script/launch_floorlog_bounded_families_4gpu.sh`
- 每个配置目录内包含 `summary.json`、deterministic/sampling manifests、训练日志和 checkpoint。
