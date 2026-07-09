# INR0624 最终 Raw-Log SN Chord 表示

本文档记录当前最终版相对原 INR note 表示的改动。当前目标配置是 ASAP-only、chord token、raw+log timing、skew-normal head、88 维 pitch multi-hot、无 musical channel。

## 表示改动

原 INR 是逐 note token；当前改为逐 chord token。chord 由 score MIDI 构建：

- continuation note 需要 `score IOI == 0` 且与前一个 note 的 `score duration` 相同。
- 如果候选 chord size `>= 3`，且所有 note 都有有效 staff feature，则不合并跨 staff chord。
- size 2 或 staff 缺失时不强制 staff 一致。
- chord base note 使用最高音。
- JSON 中 `pitch` 保存为升序 MIDI pitch list；模型输入使用 88 维 piano multi-hot。

构建脚本：[scripts/build_chord_asap_dataset.py](/home/sy/EPR/PianistTransformer/scripts/build_chord_asap_dataset.py:1)

sidecar schema：

```text
pianocore_chord_work_compact_v1
```

每个 work 仍是一份全曲 JSON/PT，window 只保存在 `windows` 字段中，不拆成多个文件。

## 字段

score 字段：

| 字段 | 含义 |
| --- | --- |
| `score.pitch` | `list[list[int]]`，每个 chord 的 MIDI pitch，升序 |
| `score.pitch_multihot` | `(num_chords, 88)`，A0-C8，A0 index 0，对应 MIDI 21 |
| `score.chord_size` | chord 内 note 数 |
| `score.score_raw` | `[ioi_ms_high, duration_ms_high, velocity_high]` |
| `score.score_offset_raw` | `[onset_ms_low_minus_high, duration_ms_low_minus_high, velocity_low_minus_high]` |
| `score.score_feature` | 最高音的原 score feature，仅兼容保留；本次训练不用 musical channel |

performance 字段：

| 字段 | 含义 |
| --- | --- |
| `label_shared_raw` | `[perf_ioi_ms_high, perf_duration_ms_high, velocity_high]` |
| `label_pedal4_raw` | chord span 内 0%, 25%, 50%, 75% pedal sample |
| `label_offset_raw` | `[onset_ms_low_minus_high, duration_ms_low_minus_high, velocity_low_minus_high]` |
| `interpolated` | chord 内是否包含插值 alignment note |

offset 方向固定为 `low - high`，可以为负。输入和 target 中：

- onset/duration offset 用秒：`ms / 1000`
- velocity offset 归一化：`velocity_delta / 127`

## 输入和 Target

当前不使用 musical 信息：

```json
"musical_feature_mode": "none",
"disable_musical_features": true
```

continuous input 维度是 22：

| block | dim | 内容 |
| --- | ---: | --- |
| score control | 5 | score IOI raw+log, score duration raw+log, score velocity |
| score offset | 3 | score onset/duration/velocity offset |
| performance control | 9 | perf IOI raw+log, perf duration raw+log, perf velocity, pedal4 |
| performance offset | 3 | perf onset/duration/velocity offset |
| masks | 2 | score/performance mask |

target 维度是 12：

| block | dim | 内容 |
| --- | ---: | --- |
| base target | 9 | `[log_ioi_dev, log_duration_dev, raw_ioi_dev_s, raw_duration_dev_s, velocity_norm, pedal4]` |
| chord offset target | 3 | `[onset_offset_s, duration_offset_s, velocity_offset_norm]` |

pitch 不再是单个 pitch id：

```json
"pitch_representation": "multihot",
"pitch_multihot_dim": 88
```

之前出现过的 128 维来自 MIDI pitch 全域 `0..127`。当前 sidecar 已改成 88 维；训练代码仍兼容旧 128 维 sidecar，会切 `[21:109]`。

## Raw-Log Timing

timing control 使用双表示：

```text
[raw_seconds, log1p(20 * seconds)]
```

因为当前 `timing_log_scale = 50 ms`，所以：

```text
logscale = log1p(time_ms / 50) = log1p(20 * seconds)
```

target dev 使用：

```text
log_ioi_dev      = logscale(perf_ioi_ms) - logscale(score_ioi_ms)
log_duration_dev = logscale(perf_duration_ms) - logscale(score_duration_ms)
raw_ioi_dev_s      = (perf_ioi_ms - score_ioi_ms) / 1000
raw_duration_dev_s = (perf_duration_ms - score_duration_ms) / 1000
```

旧的 `+0.5` 和 clamp 不用于当前 `raw_log_deviation` 路径。

## SN Head 和 Loss

当前使用：

```json
"epr_distribution": "skew_normal",
"raw_timing_loss_lambda": 0.5
```

IOI loss：

```text
loss_ioi = SN_NLL(log_ioi_dev) + 0.5 * SN_NLL(raw_ioi_dev_s)
```

Duration loss：

```text
loss_duration = SN_NLL(log_duration_dev) + 0.5 * SN_NLL(raw_duration_dev_s)
```

Velocity loss：

```text
loss_velocity = SN_NLL(velocity / 127)
```

Pedal loss：

```text
loss_pedal = BCEWithLogits(pedal4)
```

Offset loss：

```text
loss_offset = mean SN_NLL([onset_offset_s, duration_offset_s, velocity_offset_norm])
```

总 loss：

```text
loss =
  1.0 * loss_ioi
+ 1.0 * loss_duration
+ 1.0 * loss_velocity
+ 0.5 * loss_pedal
+ 1.0 * loss_offset
```

连续分布的 NLL 可以为负。这里是 `-log density`，不是离散概率的 `-log probability`；当 skew-normal 的 scale 较小且 target 落在高密度区域时，density 可以大于 1，因此 `NLL < 0` 是正常现象。

offset 没有独立大 head，而是挂在相关 head 的额外通道上：

| head | 输出 |
| --- | --- |
| IOI | log timing SN + raw timing SN + onset offset SN |
| duration | log timing SN + raw timing SN + duration offset SN |
| velocity | velocity SN + velocity offset SN |
| pedal | pedal4 logits |

因此 chord raw-log SN 输出宽度为：

```text
ioi 9 + duration 9 + velocity 6 + pedal 4 = 28
```

虽然 offset 的参数来自这些 head 的额外通道，但 logging 中仍单独汇总为 `train_loss_offset`。

## 数据统计

当前 ASAP chord sidecar：

| 项目 | 数值 |
| --- | ---: |
| works | 207 |
| performances | 969 |
| 原 note 数 | 570,623 |
| chord token 数 | 368,472 |
| notes/chord | 1.549 |
| work-level windows | 900 |
| window-performance examples | 4,944 |
| sidecar size | 约 282 MB |

chord size 分布：

| size | count | pct |
| ---: | ---: | ---: |
| 1 | 249,169 | 67.62% |
| 2 | 74,508 | 20.22% |
| 3 | 22,974 | 6.23% |
| 4 | 12,874 | 3.49% |
| 5 | 4,336 | 1.18% |
| 6 | 2,810 | 0.76% |
| 7 | 1,031 | 0.28% |
| 8 | 687 | 0.19% |
| 9 | 63 | 0.02% |
| 10 | 20 | 0.01% |

window：

```text
window_size = 512 chords
overlap = 64 chords
stride = 448 chords
overlap_ratio = 0.125
```

训练必须限制 ASAP-only，否则 `PianoCoRe/metadata.csv` 会混入 Aria-MIDI、ATEPP、PERiScoPe 等 performance，把 train examples 错误放大到 209,156。

正确 manifest：

| split | works | windows | performance refs | examples |
| --- | ---: | ---: | ---: | ---: |
| train | 188 | 795 | 896 | 4,510 |
| test | 19 | 105 | 82 | 551 |

当前 global batch：

```text
per_device_train_batch_size = 16
world_size = 2
gradient_accumulation_steps = 2
global_batch = 64
steps_per_epoch = ceil(4510 / 64) = 71
```

## 训练配置

最终配置：

| 项目 | 值 |
| --- | --- |
| epochs | 20 |
| eval/save | 每 71 steps，即每 1 epoch |
| train/eval dataset | ASAP only |
| input dim | 22 |
| target dim | 12 |
| pitch dim | 88 multi-hot |
| musical channel | disabled |
| timing target | raw-log deviation |
| distribution | skew-normal |
| raw timing loss weight | 0.5 |

配置文件：

- [configs/inr0624_chord_asap_sn_rawlog_multihot_nomus_sine.json](/home/sy/EPR/PianistTransformer/configs/inr0624_chord_asap_sn_rawlog_multihot_nomus_sine.json:1)
- [configs/inr0624_chord_asap_sn_rawlog_multihot_nomus_cine.json](/home/sy/EPR/PianistTransformer/configs/inr0624_chord_asap_sn_rawlog_multihot_nomus_cine.json:1)

启动脚本：

- [scripts/run_chord_multihot_nomus_train.sh](/home/sy/EPR/PianistTransformer/scripts/run_chord_multihot_nomus_train.sh:1)

两个任务并行：

| task | GPUs | note embedding |
| --- | --- | --- |
| sine | `0,1` | `sine` |
| cine | `2,3` | `cine` |

## 经验统计

用于支持 compact chord 表示的统计：

- score IOI=0 continuation 占所有 score note 的 47.57%。
- 在这些 continuation 中，74.50% 与前一个 note 有相同 score duration。
- score chord velocity 约 97.92% all-equal。
- performance chord onset 顺序不稳定：high-to-low 33.76%，low-to-high 23.46%，mixed 32.95%，全同时 9.83%。
- performance start stagger 中位数约 9.37 ms，p75 约 18.75 ms。
- performance duration 差异更大，中位数约 27.08 ms，p75 约 77.08 ms。
- same-duration continuation 中跨 staff 只占 0.85% unique，performance-weighted 后约 0.40%；其中 size 2 占 94% 以上。

因此最终只保留最高音 base 属性，并增加 3 个 `low - high` offset；不为 chord 内每个 note 重复一套完整 INR 字段。
