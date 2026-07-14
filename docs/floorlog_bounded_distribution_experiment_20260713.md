# Floor-log / Bounded Distribution 实验总记录（2026-07-13 至 2026-07-14）

本文整理 `results/` 内这一轮围绕 floor-log timing、bounded distribution、DLM support、tail mask、pedal 表示、musical feature 和 PianoCoRe pretrain 的实验。目标不是只记录最终数字，而是把采用过的方法、有效/无效技巧、踩坑经验、当前中断状态和后续计划都放在一个地方，方便继续接力。

## 0. 当前结论速览

1. **当前最强 ASAP-only 主线是 tight support + DLM-k1 bounded std + no IOI inflated + pedal binary4 BCE。**
   - run: `results/floorlog_tight_k1_binary4_bce_2gpu/20260714_tight_k1_binary4_bce_v1/dlm-k1-noinfl-ioi-binary4-bce-pedalw1`
   - sampling PN: IOI `35.63`, duration `126.66`, velocity `20.44`
   - sampling PP: IOI `11.26`, duration `52.81`, velocity `6.23`

2. **`IOI zero-inflated` 不应该继续用。**
   - inflated run 的 IOI/duration sampling 略好于 binary4 baseline，但 loss 曲线显示 eval 没有变好，并且 inflated 机制会把“score IOI=0 的和弦内部顺序错位”当成真实零点结构学习，方向不对。

3. **pedal 当前应回到 4 binary + BCE。**
   - pedal-start scalar/DLM 可以算 `pedal_start_wass`，但 start 一个 continuous value 太弱，而且 DLM/0-1 inflated 没有解决 pedal。
   - binary4 的 loss 分项应显示为 `pedal_0/25/50/75`；这是确认没跑错的快速信号。

4. **大部分极端 GT dev 很可能来自和弦内部 note order 错位。**
   - score 内同一和弦强制从低到高排序，但 perf MIDI 不一定同序。
   - 这会让相邻和弦音符的 IOI/duration feature 互换，产生 `log IOI dev < -2` 或 `> 1` 的伪异常。
   - 因此简单扩大分布尾部是在拟合脏标签，不是学习真实演奏不确定性。

5. **tight mask / support 很关键，但 mask 和 distribution head support 必须同步。**
   - tight mask: IOI `[-1, 1]`，duration `[-2, 1]`
   - 如果只 mask loss 而不改 DLM support，采样仍会从旧 `[-2.5, 1.5]` support 出来。
   - TF 可以 clamp/trunc，loss 可以 mask invalid note；二者作用不同。

6. **仅靠 NLL 学分布会偏宽，采样 PN 常明显差于 deterministic PN。**
   - 降 temperature / mean-centered trunc 能改善听感和 PN，但会损 PP、让分布往中心坍缩。
   - bounded family 解决“越界”，但不自动解决“太宽”。
   - 需要 scale/radius 约束、std supervision、或更干净的 note matching 数据。

7. **GPU 状态：2026-07-14 11:20 左右 GPU2 发生设备级错误。**
   - `nvidia-smi -L` 报：`Unable to determine the device handle for gpu 0000:67:00.0: Unknown Error`
   - 队列第 3 个任务中断，PianoCoRe 长训 base 完成但 ASAP adapt 因 CUDA/NVML 初始化失败未继续。

## 1. 实验背景和核心矛盾

最初矛盾是：

- 降低 sampling temperature、mean-centered trunc 或 shrink 可以明显改善听感和 PN Wasserstein；
- 但这些方法会让采样分布向中心坍缩，PP Wasserstein 或分布宽度会变差；
- 原始 DLM/NLL 学到的条件分布太宽，采样会出现破坏听感的 outlier；
- GT 中又确实存在很多极端 dev，但后续检查发现其中相当一部分是和弦内 note order 错位造成的伪异常。

因此这轮实验围绕三个方向展开：

1. **损失函数方向**：weighted NLL、raw-ms CRPS、tail loss、target-tail loss。
2. **分布/支持方向**：bounded family、DLM support 改窄、bounded sigmoid std、temperature。
3. **数据/表示方向**：tail mask、pedal 表示、IOI/pedal inflated、musical feature、PianoCoRe pretrain。

## 2. 关键实验与结果

### 2.1 Weighted NLL vs raw-ms CRPS

运行目录：

- `results/floorlog_weighted_nll_crps_2x2gpu/20260713_weighted_nll_vs_raw_ms_crps16ep/dlm-weighted-nll-a05`
- `results/floorlog_weighted_nll_crps_2x2gpu/20260713_weighted_nll_vs_raw_ms_crps16ep/dlm-raw-ms-crps-l1`

动机：

- 对长 IOI / duration 的 note，同样 dev 会造成更大毫秒偏差，听感破坏更明显。
- raw-ms CRPS 形式：

```text
y_k = t_score * exp(d_k)
CRPS(F, y) = E|X-y| - 1/2 E|X-X'|
```

预期：

- `E|X-y|` 惩罚远离真实演奏的采样；
- `-1/2 E|X-X'|` 防止分布直接坍缩成 mean；
- raw-ms 域天然给长音符更大权重，不必再手工乘 IOI/duration。

结果：

| run | sampling PN IOI | Dur | Vel | sampling PP IOI | Dur | Vel |
|---|---:|---:|---:|---:|---:|---:|
| weighted NLL a=0.5 | 53.94 | 201.58 | 20.40 | 21.62 | 109.50 | 6.83 |
| raw-ms CRPS l1 | 46.52 | 161.87 | 22.46 | 13.09 | 46.50 | 13.78 |

经验：

- weighted NLL 对 timing 没救回来，deterministic 甚至很差。
- CRPS 的确能降低 timing 采样 outlier，但 velocity 和 deterministic 不理想。
- CRPS 思路合理，但在当前脏 outlier/宽分布问题下不是最优主线。

### 2.2 GT 极端 outlier 诊断：和弦内顺序错位

代表问题：

- `Glinka, Mikhail - The Lark` 中 F#4 附近采样到极小 IOI；
- 进一步查 GT 中 `log IOI dev < -2` 或 `> 1` 的 note，很多位于和弦内部；
- score note 顺序固定低到高，但 performance MIDI 内部顺序不保证一致。

结论：

- 很多极端 IOI/duration dev 不是演奏风格，而是 note-feature 对齐错位。
- 仅仅交换 perf MIDI 的下键顺序不一定够，因为 feature 已经按序列相邻关系计算；需要在构造 label/feature 时按 chord group 内 pitch/staff 等做一致 matching。
- 短期可用 mask 排除极端 dev；长期要修 data pipeline。

采用过的短期规则：

- IOI tight mask: `|dev| <= 1`
- duration tight mask: `-2 <= dev <= 1`
- 之前讨论过 soft mask / loose mask，例如 IOI `|dev| <= 1.5`，duration `|dev| < 2`

避坑：

- **不进入 loss** 和 **teacher-forcing 输入如何处理** 是两件事。
- 对 invalid note：loss 可 mask 掉；TF/decoder feedback 建议 clamp/trunc，否则错误值仍会污染自回归条件。

### 2.3 tail mask + pedal-start DLM

主要目录：

- `results/floorlog_tailmask_pedalstart_dlm_2x2gpu/20260713_tailmask_pedalstart_dlm_support_v1`
- `results/floorlog_tailmask_pedalstart_dlm_2x2gpu/20260713_tailmask_pedalstart_dlm_v3`

关键改动：

- pedal 先改成只用 `start` 一个 continuous value；
- pedal start 也尝试用 b128 DLM；
- tight/soft mask；
- support 改窄：IOI `[-1,1]`，duration `[-2,1]`；
- 修正 DLM head support，否则 mask 后仍会从旧 support 采样。

结果：

| run | sampling PN IOI | Dur | Vel | sampling PP IOI | Dur | Vel |
|---|---:|---:|---:|---:|---:|---:|
| support_v1 soft | 46.89 | 174.78 | 21.85 | 14.76 | 71.94 | 10.66 |
| support_v1 tight | 42.24 | 164.21 | 20.87 | 11.89 | 65.32 | 9.31 |
| v3 soft | 43.71 | 171.06 | 20.31 | 12.10 | 71.88 | 6.82 |
| v3 tight | 42.03 | 159.60 | 20.96 | 11.52 | 58.32 | 8.50 |

经验：

- tight mask 稳定优于 soft mask。
- 仅改 mask 不够，必须同步改 support。
- pedal-start DLM 的 timing 有改善，但 pedal 本身不理想；start 一个 value 对 pedal 表示太弱。

### 2.4 DLM-k1 bounded std：当前最重要的 timing 改动

目录：

- `results/floorlog_tight_targettail_vs_k1std_2x2gpu/20260713_tight_targettail_vs_k1std_v1/dlm-k1-bounded-std-s012`
- 对照：`dlm-target-tail-r10-l1`

关键配置：

```json
{
  "dlm_components": 1,
  "dlm_timing_scale_parameterization": "bounded_sigmoid",
  "dlm_timing_scale_min": 0.002,
  "dlm_timing_scale_max": 0.12,
  "tail_mask_ioi_min": -1.0,
  "tail_mask_ioi_max": 1.0,
  "tail_mask_duration_min": -2.0,
  "tail_mask_duration_max": 1.0
}
```

结果：

| run | sampling PN IOI | Dur | Vel | sampling PP IOI | Dur | Vel |
|---|---:|---:|---:|---:|---:|---:|
| target-tail r10 l1 | 38.48 | 148.52 | 21.78 | 10.94 | 50.14 | 10.86 |
| k1 bounded std s012 | **33.95** | **131.29** | **19.64** | **10.58** | 58.90 | **4.37** |

经验：

- 对 DLM 来说，减少 mixture 数、约束 std，比再加 tail loss 更有效。
- `std_bound` 应与 support 范围成比例；后来倾向统一用 5% bounded sigmoid，而不是固定 min/max。
- 单纯 target-tail loss 不够，容易仍保留宽尾。

### 2.5 temperature sweep

目录：

- `results/floorlog_tight_targettail_vs_k1std_temp_sweep/20260713_k1_bounded_std_temp_sweep_v1`

结果：

| temp | sampling PN IOI | Dur | Vel | sampling PP IOI | Dur | Vel |
|---|---:|---:|---:|---:|---:|---:|
| 0.1 | **28.00** | **119.94** | **14.97** | 14.22 | 60.65 | 8.88 |
| 0.25 | 28.77 | 121.30 | 15.40 | 13.03 | 60.32 | 7.46 |
| 0.5 | 30.26 | 125.79 | 17.03 | 11.90 | 62.69 | 6.35 |
| 0.75 | 31.94 | 121.51 | 18.55 | **10.79** | **52.18** | **4.81** |

经验：

- 降 temperature 能显著改善 PN，听感更稳。
- 但 PP 会变差或改变分布宽度；这不能作为训练目标的最终答案。
- temperature 是诊断工具：如果低 temp 好很多，说明模型 distribution scale 仍偏宽。

### 2.6 bounded family 4-epoch 短跑

目录：

- `results/floorlog_bounded_families_4gpu/20260713_bounded_families_short4`
- 脚本：`script/launch_floorlog_bounded_families_4gpu.sh`

协议：

- ASAP，fixed score-span split；
- 4 epochs；
- 推理 `cheap15_score_sources`;
- 固定全局 support 映射到 `(0,1)`：
  - zero-score-IOI: `[0, 5]`
  - nonzero-score-IOI: `[-2.5, 1.5]`
  - duration: `[-3, 2]`
  - velocity: `[0,1]`

Sampling PN：

| Family | 配置 | IOI | Duration | Velocity | Pedal |
|---|---|---:|---:|---:|---:|
| Beta | beta-k2 | 48.89 | 187.67 | **18.37** | 0.465 |
| Beta | beta-k3 | 47.81 | 187.24 | 18.47 | 0.472 |
| Beta | beta-k5 | 46.56 | 177.78 | 18.84 | 0.479 |
| Beta | **beta-k8** | **44.86** | **173.92** | 18.71 | 0.470 |
| Tanh-t | tanh-* | 142.21 | 549.60 | 38.42 | ~0.47 |
| Bounded SN | sn-smax2 | 127.85 | 256.53 | 25.98 | 0.475 |
| Logit-normal | ln-k1 | 100.29 | 219.11 | 18.57 | 0.467 |
| Logit-normal | **ln-k2** | **48.74** | 180.35 | 19.45 | 0.471 |
| Logit-normal | ln-k5 | 53.70 | **177.58** | 18.97 | 0.469 |

Sampling PP：

| Family | 配置 | IOI | Duration | Velocity | Pedal |
|---|---|---:|---:|---:|---:|
| Beta | beta-k8 | **10.88** | **48.30** | 7.05 | **0.203** |
| Logit-normal | ln-k2 | 13.19 | 52.90 | 7.81 | 0.270 |
| Logit-normal | ln-k3 | 16.57 | 68.12 | **5.39** | 0.258 |

经验：

- Beta mixture 明显受益于 component 数，beta-k8 是 4ep 短跑最佳。
- LN component 数不单调，ln-k2 是主要候选。
- tanh-t 当前参数化基本失控，不应继续原样跑。
- bounded SN 有一定 scale 约束效果，但仍明显弱于 Beta/LN。
- bounded family 只解决“支持范围”，不保证“条件分布足够窄”。

### 2.7 Beta/LN 16-epoch + predictive variance

目录：

- `results/floorlog_beta5_ln2_variance_4gpu/20260713_beta5_ln2_var16ep`

Sampling PN：

| run | IOI | Duration | Velocity | Pedal |
|---|---:|---:|---:|---:|
| beta5-support | 47.34 | 182.22 | 20.99 | 0.474 |
| beta5-var-r005-l10 | 43.28 | 173.37 | **19.51** | 0.475 |
| ln2-support | 48.17 | 175.68 | 22.57 | 0.473 |
| ln2-var-r005-l10 | **41.85** | **153.22** | 20.94 | **0.458** |

经验：

- predictive variance/radius 约束有效，尤其 LN2。
- 但仍没有追上 tight DLM-k1 bounded std 的 PN 水平。
- Beta/LN 可作为 distribution family 候选，但 pedal 后来已改回 binary4 BCE；旧结果里 pedal distribution 不应直接和当前主线比较。

### 2.8 DLM constraints 16-epoch

目录：

- `results/floorlog_dlm_constraints_4gpu/20260713_dlm_constraints16ep_v2`

Sampling PN：

| run | IOI | Duration | Velocity | Pedal |
|---|---:|---:|---:|---:|
| dlm-base | 47.75 | 167.63 | 21.84 | 0.482 |
| dlm-scale-s001-s02 | 43.02 | 153.13 | 20.98 | 0.477 |
| dlm-tail-r005-l1 | 43.84 | 161.74 | 22.33 | 0.484 |
| dlm-scale-tail-r005-l1 | **42.67** | **148.17** | **20.33** | 0.480 |

经验：

- scale bound 比 tail loss 更可靠。
- scale + tail 叠加有收益，但不如后来的 tight support + k1 bounded std。
- 约束分布半径是主线，不是单纯换 family。

### 2.9 inflated 实验：为什么不继续

目录：

- `results/floorlog_tight_inflated_dlm_2gpu/20260714_tight_inflated_dlm_v1/dlm-k1-inflated-ioi-pedalw1`
- loss 曲线：`results/analysis/loss_curves_20260714_inflated_vs_old_k1`

配置：

- IOI zero-inflated
- pedal start DLM + 0/127 inflated
- pedal weight = 1

结果：

| run | sampling PN IOI | Dur | Vel | pedal_start | sampling PP IOI | Dur | Vel |
|---|---:|---:|---:|---:|---:|---:|---:|
| inflated IOI/pedal | 34.79 | 128.66 | 20.51 | 56.45 | 9.98 | 55.41 | 5.69 |

loss 曲线经验：

- 新 inflated 的 train IOI/pedal loss 更低，但 eval 不优于 old k1。
- old best eval loss 约 `17.7688`，inflated best eval loss 约 `17.8933`。
- 说明 inflated 在训练集上解释了某些特殊点，但没有泛化。

结论：

- **IOI 不要 inflated。**
- `score IOI=0` 里混有和弦错位，不是干净的零膨胀分布。
- pedal 也不继续用 0/1 inflated；先回 binary4 BCE。

### 2.10 当前主线：IOI no-inflated + pedal binary4 BCE

目录：

- `results/floorlog_tight_k1_binary4_bce_2gpu/20260714_tight_k1_binary4_bce_v1/dlm-k1-noinfl-ioi-binary4-bce-pedalw1`
- 脚本：`script/launch_floorlog_tight_k1_binary4_bce_2gpu.sh`

配置：

```json
{
  "epr_distribution": "dlm",
  "dlm_components": 1,
  "dlm_timing_scale_parameterization": "bounded_sigmoid",
  "dlm_timing_scale_min": 0.002,
  "dlm_timing_scale_max": 0.12,
  "dlm_ioi_zero_inflated": false,
  "pedal_representation": "binary_4",
  "pedal_distribution": "point",
  "pedal_output_activation": "linear",
  "dlm_pedal_zero_one_inflated": false,
  "loss_weights": {"ioi": 1, "duration": 1, "velocity": 1, "pedal": 1}
}
```

结果：

| protocol | PN IOI | PN Dur | PN Vel | PN pedal_start | PP IOI | PP Dur | PP Vel | PP pedal_start |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sampling | 35.63 | 126.66 | 20.44 | 63.40 | 11.26 | 52.81 | 6.23 | 36.10 |
| deterministic | 28.79 | 117.79 | 15.89 | 63.03 | 15.38 | 57.29 | 9.66 | 63.03 |

经验：

- binary4 pedal 回归后，timing 没有崩，整体接近 inflated。
- pedal binary 指标和 pedal_start 指标口径不同：binary pedal wass 是 0/1，pedal_start_wass 是 0-127 onset continuous 规则重算。
- binary4 BCE 下日志应出现 `train_loss_pedal_0/25/50/75`。

### 2.11 musical ablation 队列

目录：

- `results/floorlog_tight_binary4_ablation_queue_2gpu/20260714_tight_binary4_ablation_queue_v1`
- 脚本：`script/launch_floorlog_tight_binary4_ablation_queue_2gpu.sh`

原计划 12 个任务：

前 6 个 musical ablation，固定 DLM-k1：

1. `slot6-full-k1`
2. `slot6-onset-only-k1`
3. `slot6-annotation-only-k1`
4. `slot6-onset-annotation-k1`
5. `slot6-no-duration-k1`
6. `slot6-no-length-k1`

后 6 个 distribution ablation，全部 no-musical：

7. `slot5-nomus-k1`
8. `slot5-nomus-k4`
9. `slot5-nomus-k8`
10. `slot5-nomus-beta5`
11. `slot5-nomus-mln2`
12. `slot5-nomus-bsn`

已完成前两个：

| run | sampling PN IOI | Dur | Vel | sampling PP IOI | Dur | Vel |
|---|---:|---:|---:|---:|---:|---:|
| no-musical baseline | 35.63 | 126.66 | 20.44 | 11.26 | 52.81 | 6.23 |
| slot6-full-k1 | 34.91 | **123.46** | **20.11** | 10.44 | **48.69** | 5.82 |
| slot6-onset-only-k1 | **34.56** | 124.75 | 20.37 | **10.13** | 50.68 | **5.43** |

当前状态：

- 第 3 个 `slot6-annotation-only-k1` 在 step `468/1680` 处 CUDA launch failure。
- 队列已停止。
- GPU2 设备异常，需要 reset/reboot 后从第 3 个任务继续。

初步经验：

- musical full 和 onset-only 都比 no-musical baseline 略好。
- onset-only 的 IOI / velocity PP 最好，full 的 duration 最好。
- 这支持“onset type 信息对节奏有意义”的判断；duration/length musical 信息是否必要仍需后续任务完成后判断。

### 2.12 PianoCoRe pretrain -> ASAP adapt 长训

目录：

- `results/floorlog_tight_k1_binary4_bce_full_pianocore_2gpu/20260714_tight_k1_binary4_bce_full_v1/dlm-k1-noinfl-ioi-binary4-bce-pianocore4-asap8`
- 脚本：`script/launch_floorlog_tight_k1_binary4_bce_full_pianocore_2gpu.sh`

计划：

```text
PianoCoRe train 4ep -> ASAP adapt 8ep -> deterministic infer -> sampling infer -> summary/eval/statistics
```

为什么 base 4ep + adapt 8ep：

- PianoCoRe train examples 约 `782,045`，一轮已经非常大；
- 4ep 对应 `48,880` steps；
- ASAP-only 训练集很小，adapt 8ep 先看趋势，不够可 resume 追加。

PianoCoRe 数据状态：

- base `.pt` sidecar: `4094`
- ASAP `.ASAP.pt` sidecar: `1807`
- fixed split: `data/train_valid_asap3_nonasap05_v1_summary.json`

当前结果：

- PianoCoRe base 4ep 已完成。
- best checkpoint: `checkpoint-46000`
- best eval loss: `17.3211612701416`
- 最终 step: `48880`

中断点：

- 进入 ASAP adapt 后，CUDA/NVML 初始化失败：

```text
CUDA initialization: CUDA unknown error
ValueError: Your setup doesn't support bf16/gpu.
```

解释：

- 这不是模型或 loss 问题。
- 同一时间 GPU2 出现设备级错误，NVML 状态异常，导致后续 CUDA 初始化失败。
- 需要 GPU reset 或重启节点后，从 base `checkpoint-best` 继续 ASAP adapt。

## 3. 实现改动和配置约定

### 3.1 新增/修改过的能力

主要文件：

- `src/model/integrated_pianoformer.py`
- `src/train/train_inr.py`
- `src/evaluate/evaluate_inr_saved_midis.py`
- `src/inference/infer_inr_testset.py`
- `script/run_inr_epr_pipeline.sh`

能力：

- DLM timing support 可配置：
  - `dlm_ioi_nonzero_min/max`
  - `dlm_duration_min/max`
  - `dlm_ioi_zero_min/max`
- tail mask：
  - `tail_mask_enabled`
  - `tail_mask_tf_clamp`
  - `tail_mask_ioi_min/max`
  - `tail_mask_duration_min/max`
- DLM bounded sigmoid scale：
  - `dlm_timing_scale_parameterization=bounded_sigmoid`
  - `dlm_timing_scale_min/max`
- DLM inflated（已不推荐）：
  - `dlm_ioi_zero_inflated`
  - `dlm_pedal_zero_one_inflated`
- pedal representation：
  - `binary_4`
  - `start_valley`
  - `start`
- pedal_start_wass 评估：
  - onset pedal continuous，0-127 规则，PT official 和 ours 同口径重算。
- musical51 ablation modes：
  - `musical51_full`
  - `musical51_onset_only`
  - `musical51_annotation_only`
  - `musical51_onset_annotation`
  - `musical51_no_duration`
  - `musical51_no_length`
  - `musical51_no_duration_length`

### 3.2 当前推荐配置骨架

```json
{
  "epr_distribution": "dlm",
  "velocity_distribution": "dlm",
  "dlm_components": 1,
  "dlm_timing_scale_parameterization": "bounded_sigmoid",
  "dlm_timing_scale_min": 0.002,
  "dlm_timing_scale_max": 0.12,
  "tail_mask_enabled": true,
  "tail_mask_tf_clamp": true,
  "tail_mask_ioi_min": -1.0,
  "tail_mask_ioi_max": 1.0,
  "tail_mask_duration_min": -2.0,
  "tail_mask_duration_max": 1.0,
  "dlm_ioi_nonzero_min": -1.0,
  "dlm_ioi_nonzero_max": 1.0,
  "dlm_duration_min": -2.0,
  "dlm_duration_max": 1.0,
  "dlm_ioi_zero_inflated": false,
  "pedal_representation": "binary_4",
  "pedal_distribution": "point",
  "pedal_output_activation": "linear",
  "dlm_pedal_zero_one_inflated": false,
  "loss_weights": {
    "ioi": 1.0,
    "duration": 1.0,
    "velocity": 1.0,
    "pedal": 1.0
  }
}
```

### 3.3 维度避坑

binary4 EPR target:

- timing/velocity: 3
- pedal bits: 4
- `continuous_dim = output_continuous_dim = 7`

在 integrated floor-log no-musical schema 下：

- `input_continuous_dim = score_input_continuous_dim = decoder_input_continuous_dim = 12`

注意：

- `train_inr.py` 会根据 `pedal_representation` 和 musical mode 自动推断维度；
- 但 pipeline 顶层 `config.json` 最好也写对，否则 inference/eval 阶段可能读到旧维度。
- binary4 pedal head 必须输出 logits，因此 `pedal_output_activation=linear`；如果用 sigmoid 后再 BCE，会压坏动态范围。

## 4. 踩坑和经验清单

### 4.1 不要只改 mask，不改 support

症状：

- loss 里 invalid tail 被 mask 掉；
- 但采样图仍然超出 `[-1,1]`；
- 原因是 DLM head 仍使用旧 support。

规则：

- mask/support/head decode 三者必须同步。
- tight mask 的 IOI/duration 范围要同时写到：
  - `tail_mask_*`
  - `dlm_ioi_nonzero_*`
  - `dlm_duration_*`

### 4.2 不要把所有极端 dev 当成真实演奏分布

很多极端值来自：

- 和弦内部 score/perf note order 不一致；
- 同一 onset 多音符的 IOI feature 被相邻音互换；
- duration 也可能随 note 对齐错位。

这类点应该清理或 mask，不应靠更重尾分布拟合。

### 4.3 IOI inflated 容易学错对象

score IOI=0 不等于 label 应该有干净的 zero-inflated 结构。和弦错位会让 IOI=0 的 note 出现伪极端，inflated 机制可能正好记住这些伪结构。

### 4.4 Pedal DLM/start 表示不够好

只预测 pedal start 一个 continuous value 太弱；DLM/0-1 inflated 也没有解决 pedal。当前回到 binary4 BCE 更稳，也更接近旧 PT 评估口径。

### 4.5 Temperature 是诊断，不是最终训练答案

低 temp 可以把 PN 拉到 deterministic 附近，但会牺牲 PP。它说明分布 scale 过宽，不能替代训练期 scale 约束。

### 4.6 Eval loss 不总是等价于 PN/PP Wass

LN-k5 eval loss 可低，但 PN/PP 不一定最好。筛选分布实验时必须看 saved MIDI 的 score-level Wasserstein，而不是只看 eval NLL。

### 4.7 DDP/best checkpoint 小坑

一些 run 末尾出现：

```text
Could not locate the best model at checkpoint-xxx/pytorch_model.bin
```

但 pipeline 使用 `EVAL_CHECKPOINT_MODE=latest` 时仍可用 latest checkpoint 完成 infer/eval。需要区分 best checkpoint sync warning 和真正失败。

### 4.8 GPU/NVML 崩溃会连带影响其他任务

2026-07-14 11:20 后：

- GPU2 `nvidia-smi -L` 设备句柄错误；
- 队列 CUDA launch failure；
- 长训 adapt 由于 CUDA/NVML 初始化失败，落到 CPU 后 bf16 不支持。

这类不是代码/loss 问题，继续提交任务只会失败。应先 reset GPU 或重启节点。

## 5. 当前状态（截至 2026-07-14）

### 5.1 已完成

- bounded family 4ep 短跑；
- beta/LN variance 16ep；
- DLM constraints 16ep；
- weighted NLL vs raw-ms CRPS；
- tail mask + pedal-start DLM；
- tight DLM-k1 bounded std；
- temperature sweep；
- inflated IOI/pedal 实验；
- binary4 BCE baseline；
- musical queue 前两个任务：
  - `slot6-full-k1`
  - `slot6-onset-only-k1`
- PianoCoRe base 4ep。

### 5.2 中断

- `results/floorlog_tight_binary4_ablation_queue_2gpu/20260714_tight_binary4_ablation_queue_v1`
  - 第 3 个 `slot6-annotation-only-k1` 在 step 468/1680 CUDA failure；
  - 队列停止。

- `results/floorlog_tight_k1_binary4_bce_full_pianocore_2gpu/20260714_tight_k1_binary4_bce_full_v1`
  - PianoCoRe base 完成；
  - ASAP adapt 未开始成功；
  - 失败原因是 CUDA/NVML 初始化异常。

### 5.3 需要硬件处理

GPU2 当前异常：

```text
Unable to determine the device handle for gpu 0000:67:00.0: Unknown Error
```

建议先 reset GPU2 或重启节点，再恢复实验。

## 6. 恢复和未来计划

### 6.1 恢复队列

硬件恢复后，从第 3 个任务继续：

1. `slot6-annotation-only-k1`
2. `slot6-onset-annotation-k1`
3. `slot6-no-duration-k1`
4. `slot6-no-length-k1`
5. `slot5-nomus-k1`
6. `slot5-nomus-k4`
7. `slot5-nomus-k8`
8. `slot5-nomus-beta5`
9. `slot5-nomus-mln2`
10. `slot5-nomus-bsn`

注意：

- 前 6 个只比较 musical；
- 后 6 个只比较 distribution，必须 `musical_feature_mode=none`；
- 所有任务保持 binary4 BCE、no inflated。

### 6.2 恢复 PianoCoRe 长训

从：

```text
results/floorlog_tight_k1_binary4_bce_full_pianocore_2gpu/20260714_tight_k1_binary4_bce_full_v1/dlm-k1-noinfl-ioi-binary4-bce-pianocore4-asap8/training/floorlog_tight_dlm_k1_noinfl_ioi_binary4_bce_pedalw1/checkpoint-best
```

继续 ASAP adapt 8ep，然后 infer/eval/static。

### 6.3 数据清理优先级

最高优先级不是继续加重尾，而是修 note matching：

1. 识别 score chord group；
2. 在 group 内按 pitch/staff/onset proximity 对齐 perf note；
3. 重新计算 IOI/duration label；
4. 标记无法可靠 matching 的 note invalid；
5. 对 invalid note 不进 loss，TF 输入 clamp 到 support 内。

### 6.4 分布/损失后续方向

建议顺序：

1. 完成 musical/distribution 队列，看 musical onset 是否稳定有效。
2. 在 no-musical 条件下比较：
   - DLM-k1/k4/k8
   - beta5
   - mln2
   - bounded SN
3. 如果 DLM-k1 仍最强，集中研究 scale/radius：
   - support 5% bounded sigmoid；
   - per-feature proportional std bound；
   - target-radius loss；
   - raw-ms CRPS 小权重组合，而不是单独 CRPS。
4. 如果 beta/MLN 接近 DLM，做 16ep full ASAP 和 temp sweep。
5. 清理数据后重新评估是否还需要 heavy tail。

### 6.5 不建议继续的方向

- IOI zero-inflated；
- pedal 0/1 inflated；
- pedal start-only 作为最终表示；
- 只靠 lowering temperature 作为最终方案；
- 当前 tanh-t 参数化；
- 在未清理 chord order 前继续扩大 support/加重尾。

## 7. 重要产物索引

脚本：

- `script/launch_floorlog_weighted_nll_crps_2x2gpu.sh`
- `script/launch_floorlog_tailmask_pedalstart_dlm_2x2gpu.sh`
- `script/launch_floorlog_tight_targettail_vs_k1std_2x2gpu.sh`
- `script/launch_floorlog_dlm_constraints_4gpu.sh`
- `script/launch_floorlog_bounded_families_4gpu.sh`
- `script/launch_floorlog_beta5_ln2_variance_4gpu.sh`
- `script/launch_floorlog_tight_inflated_dlm_2gpu.sh`
- `script/launch_floorlog_tight_k1_binary4_bce_2gpu.sh`
- `script/launch_floorlog_tight_binary4_ablation_queue_2gpu.sh`
- `script/launch_floorlog_tight_k1_binary4_bce_full_pianocore_2gpu.sh`

分析图：

- `results/analysis/loss_curves_20260714_inflated_vs_old_k1/`
- `results/analysis/chord_gt_score_ioi_split_distribution.json`
- `results/analysis/chord_infer_distribution_plots/`
- `results/analysis/zero_nz_ioi_dev_scales/`

关键 run：

- `results/floorlog_tight_k1_binary4_bce_2gpu/20260714_tight_k1_binary4_bce_v1/dlm-k1-noinfl-ioi-binary4-bce-pedalw1`
- `results/floorlog_tight_binary4_ablation_queue_2gpu/20260714_tight_binary4_ablation_queue_v1`
- `results/floorlog_tight_k1_binary4_bce_full_pianocore_2gpu/20260714_tight_k1_binary4_bce_full_v1`

