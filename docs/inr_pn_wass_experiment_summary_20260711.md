# INR cheap15 PN Wass 实验汇总

更新日期：2026-07-11

本文只汇报 `sampling` 协议下的 PN Wass，且只看非 log 指标：

- `ioi_wass`
- `duration_wass`
- `velocity_wass`
- `pedal_wass`

Wasserstein distance 越低越好。本文不使用 deterministic 结果，也不使用 `*_log50_wass`。

## 1. 最终对齐基线

最终比对对象以 PT 为主，采用当前确认的 cheap15 PN 口径：

| 编号 | 实验 | PN IOI | PN Duration | PN Velocity | PN Pedal | 说明 |
|---|---:|---:|---:|---:|---:|---|
| B0 | PT official | 39.44 | 144.93 | 13.57 | 0.331 | 最终主基线 |
| B1 | PT split->ASAP | 43.65 | 152.67 | 15.24 | 0.354 | 训练/切分更接近 INR 对照 |
| B2 | simple folded_abs | 61.10 | 195.76 | 18.89 | 0.492 | 当前 INR 最强代表配置 |

核心结论：

1. 目前最强 INR 族大多停在 PN IOI `60-65`、PN Duration `194-203`，明显落后 PT official 的 `39.44 / 144.93`。
2. `simple folded_abs` 相比 PT official：IOI 高 `+21.66`，约 `1.55x`；Duration 高 `+50.83`，约 `1.35x`；Velocity 高 `+5.32`；Pedal 高 `+0.161`。
3. 相比 PT split->ASAP，`simple folded_abs` 仍落后：IOI 高 `+17.45`，Duration 高 `+43.09`，Velocity 高 `+3.65`，Pedal 高 `+0.138`。
4. absolute timing 新实验的 Duration 没有完全崩，但 IOI 明显差：最好的 `C-absolute-log-overlap50` PN IOI 是 `132.42`，约为 `simple folded_abs` 的 `2.17x`，约为 PT official 的 `3.36x`。

## 2. 字段说明

主表中的配置缩写：

- `target`: timing target，例如 `raw_log_deviation` 或 `raw_log_absolute`
- `ov`: overlap ratio
- `ep`: epoch 数
- `embed`: note embedding mode
- `slot`: slot version
- `dim`: slot dim
- `in`: input continuous dim
- `tf`: `tf_embedding_mask_keep_prob`
- `decMask`: decoder timing-feature mask 是否启用
- `zeroEmb`: `zero_score_ioi_embedding`
- `zxf`: zero IOI transform
- `dual`: zero IOI dual distribution mode
- `legacyDual`: legacy dual timing head
- `pos`: zero IOI positive support
- `res`: zero IOI residual
- `rawL`: raw timing loss lambda
- `pedal`: pedal representation

## 3. 实验总表

排序方式：按 PN IOI 从低到高排序；同一 IOI 档再参考 Duration、Velocity、Pedal。表中只列 INR/变体实验，不把 PT official 混进排序。

| 编号 | 实验 | PN IOI | PN Duration | PN Velocity | PN Pedal | 关键配置 |
|---|---:|---:|---:|---:|---:|---|
| E01 | sine-control | 60.30 | 216.19 | 19.88 | 0.492 | target=raw_log_deviation; ov=0.125; ep=16; embed=sine; in=68; tf=0.5; decMask=True; legacyDual=True; rawL=0.25; pedal=binary_4 |
| E02 | simple-folded-abs-slot-zeroembed | 61.10 | 195.76 | 18.89 | 0.492 | target=raw_log_deviation; ov=0.125; ep=16; embed=slot_attribute; slot=slot5; dim=128; in=68; tf=0.5; decMask=True; zeroEmb=True; zxf=folded_abs; dual=none; legacyDual=True; pos=False; res=False; rawL=0.25; pedal=binary_4 |
| E03 | slot5-128-zero-ioi-folded-abs | 61.27 | 194.22 | 19.41 | 0.490 | target=raw_log_deviation; ov=0.125; ep=16; embed=slot_attribute; slot=slot5; dim=128; in=68; tf=0.5; decMask=True; zxf=folded_abs; legacyDual=True; pos=True; res=False; rawL=0.25; pedal=binary_4 |
| E04 | slot5-128-folded-abs-zero-residual-ioi-duration | 62.31 | 195.76 | 19.84 | 0.501 | target=raw_log_deviation; ov=0.125; ep=16; embed=slot_attribute; slot=slot5; dim=128; in=68; tf=0.5; decMask=True; zxf=folded_abs; legacyDual=True; pos=True; res=True; rawL=0.25; pedal=binary_4 |
| E05 | slot5-128-zeroembed-dual-zero-folded | 62.52 | 199.60 | 20.78 | 0.464 | target=raw_log_deviation; ov=0.125; ep=16; embed=slot_attribute; slot=slot5; dim=128; in=68; tf=0.5; decMask=True; zeroEmb=True; zxf=none; dual=zero_folded; legacyDual=True; pos=False; res=False; rawL=0.25; pedal=binary_4 |
| E06 | slot5-128-folded-abs-stable-v2 | 62.95 | 203.15 | 18.53 | 0.493 | target=raw_log_deviation; ov=0.125; ep=16; embed=slot_attribute; slot=slot5; dim=128; in=68; tf=0.5; decMask=True; zxf=folded_abs; legacyDual=True; pos=True; res=False; rawL=0.25; pedal=binary_4 |
| E07 | dual-zero-folded-no-zeroembed | 63.11 | 199.23 | 18.52 | 0.469 | target=raw_log_deviation; ov=0.125; ep=16; embed=slot_attribute; slot=slot5; dim=128; in=68; tf=0.5; decMask=True; zeroEmb=False; zxf=none; dual=zero_folded; legacyDual=True; pos=False; res=False; rawL=0.25; pedal=binary_4 |
| E08 | dual-zero-folded-timing-zeroembed | 64.23 | 200.77 | 19.28 | 0.469 | target=raw_log_deviation; ov=0.125; ep=16; embed=slot_attribute; slot=slot5; dim=128; in=68; tf=0.5; decMask=True; zeroEmb=False; zxf=none; dual=zero_folded; legacyDual=True; pos=False; res=False; rawL=0.25; pedal=binary_4 |
| E09 | sine | 64.65 | 209.02 | 18.57 | 0.487 | target=raw_log_deviation; ov=0.125; ep=16; embed=sine; in=68; tf=0.5; decMask=True; legacyDual=True; rawL=0.5; pedal=binary_4 |
| E10 | slot5-128-zeroembed-dual-sn | 64.65 | 198.18 | 20.13 | 0.476 | target=raw_log_deviation; ov=0.125; ep=16; embed=slot_attribute; slot=slot5; dim=128; in=68; tf=0.5; decMask=True; zeroEmb=True; zxf=none; dual=skew_normal; legacyDual=True; pos=False; res=False; rawL=0.25; pedal=binary_4 |
| E11 | dual-zero-folded-ioi-only | 65.02 | 199.74 | 20.31 | 0.472 | target=raw_log_deviation; ov=0.125; ep=16; embed=slot_attribute; slot=slot5; dim=128; in=68; tf=0.5; decMask=True; zeroEmb=True; zxf=none; dual=zero_folded; legacyDual=True; pos=False; res=False; rawL=0.25; pedal=binary_4 |
| E12 | slot5-128-zero-ioi-positive | 65.25 | 194.03 | 19.56 | 0.484 | target=raw_log_deviation; ov=0.125; ep=16; embed=slot_attribute; slot=slot5; dim=128; in=68; tf=0.5; decMask=True; legacyDual=True; pos=True; res=False; rawL=0.25; pedal=binary_4 |
| E13 | slot5-128-zero-ioi-squared | 65.46 | 192.80 | 21.06 | 0.501 | target=raw_log_deviation; ov=0.125; ep=16; embed=slot_attribute; slot=slot5; dim=128; in=68; tf=0.5; decMask=True; zxf=squared; legacyDual=True; pos=True; res=False; rawL=0.25; pedal=binary_4 |
| E14 | slot5-128-stable-dynamics | 65.87 | 199.39 | 19.73 | 0.485 | target=raw_log_deviation; ov=0.125; ep=16; embed=slot_attribute; slot=slot5; dim=128; in=68; tf=0.5; decMask=True; legacyDual=True; pos=False; res=False; rawL=0.25; pedal=binary_4 |
| E15 | slot5-128-whole-token-pad | 66.27 | 198.39 | 19.89 | 0.490 | target=raw_log_deviation; ov=0.125; ep=16; embed=slot_attribute; slot=slot5; dim=128; in=68; tf=0.5; decMask=True; legacyDual=True; rawL=0.25; pedal=binary_4 |
| E16 | slot5-128-zero-ioi-positive-residual | 67.78 | 206.60 | 19.03 | 0.494 | target=raw_log_deviation; ov=0.125; ep=16; embed=slot_attribute; slot=slot5; dim=128; in=68; tf=0.5; decMask=True; legacyDual=True; pos=True; res=True; rawL=0.25; pedal=binary_4 |
| E17 | slot6-128-mlp-decoder-musical-mask | 68.26 | 212.06 | 19.08 | 0.485 | target=raw_log_deviation; ov=0.125; ep=16; embed=slot_attribute; slot=slot6; dim=128; in=68; tf=0.5; decMask=True; legacyDual=True; pos=False; res=False; rawL=0.25; pedal=binary_4 |
| E18 | slot8-whole-token-mask | 68.42 | 212.33 | 18.75 | 0.493 | target=raw_log_deviation; ov=0.125; ep=16; embed=slot_attribute; slot=slot8; dim=128; in=68; tf=0.5; decMask=True; legacyDual=True; rawL=0.25; pedal=binary_4 |
| E19 | slot6-128-mlp | 68.54 | 204.39 | 18.95 | 0.499 | target=raw_log_deviation; ov=0.125; ep=16; embed=slot_attribute; slot=slot6; dim=128; in=68; tf=0.5; decMask=True; legacyDual=True; rawL=0.25; pedal=binary_4 |
| E20 | slot8-direct96-whole-token-pad | 68.72 | 237.88 | 19.34 | 0.491 | target=raw_log_deviation; ov=0.125; ep=16; embed=slot_attribute; slot=slot8; dim=96; in=68; tf=0.5; decMask=True; legacyDual=True; rawL=0.25; pedal=binary_4 |
| E21 | slot5-256-whole-token-pad | 68.98 | 203.28 | 18.45 | 0.488 | target=raw_log_deviation; ov=0.125; ep=16; embed=slot_attribute; slot=slot5; dim=256; in=68; tf=0.5; decMask=True; legacyDual=True; rawL=0.25; pedal=binary_4 |
| E22 | slot6-128-direct | 69.17 | 212.10 | 19.99 | 0.517 | target=raw_log_deviation; ov=0.125; ep=16; embed=slot_attribute; slot=slot6; dim=128; in=68; tf=0.5; decMask=True; legacyDual=True; rawL=0.25; pedal=binary_4 |
| E23 | slot5 | 69.88 | 223.69 | 21.55 | 0.490 | target=raw_log_deviation; ov=0.125; ep=16; embed=slot_attribute; slot=slot5; dim=128; in=68; tf=0.5; decMask=True; legacyDual=True; rawL=0.5; pedal=binary_4 |
| E24 | rawlog_nomus_retry | 70.47 | 241.19 | 18.40 | 0.480 | target=raw_log_deviation; ov=0.125; ep=8; embed=sine; in=16; tf=0.5; decMask=True; rawL=0.5; pedal=binary_4 |
| E25 | slot8-correlated-perf-pad50 | 72.98 | 260.77 | 17.75 | 0.486 | target=raw_log_deviation; ov=0.125; ep=16; embed=slot_attribute; slot=slot8; dim=128; in=68; tf=0.5; decMask=True; legacyDual=True; rawL=0.25; pedal=binary_4 |
| E26 | slot6-128-direct-gpt16-bs8acc8 | 75.95 | 242.46 | 21.39 | 0.491 | target=raw_log_deviation; ov=0.125; ep=16; embed=slot_attribute; slot=slot6; dim=128; in=68; tf=0.5; decMask=True; legacyDual=True; rawL=0.25; pedal=binary_4 |
| E27 | slot8-mixed-property-mask-stable | 76.95 | 262.81 | 18.30 | 0.498 | target=raw_log_deviation; ov=0.125; ep=16; embed=slot_attribute; slot=slot8; dim=128; in=68; tf=0.5; decMask=True; legacyDual=True; rawL=0.25; pedal=binary_4 |
| E28 | cine | 79.11 | 262.73 | 18.40 | 0.491 | target=raw_log_deviation; ov=0.125; ep=16; embed=cine; in=68; tf=0.5; decMask=True; legacyDual=True; rawL=0.5; pedal=binary_4 |
| E29 | slot8-stable-dynamic | 81.58 | 300.93 | 18.03 | 0.484 | target=raw_log_deviation; ov=0.125; ep=16; embed=slot_attribute; slot=slot8; dim=128; in=68; tf=0.5; decMask=True; legacyDual=True; rawL=0.25; pedal=binary_4 |
| E30 | slot8 | 81.87 | 286.03 | 18.45 | 0.490 | target=raw_log_deviation; ov=0.125; ep=16; embed=slot_attribute; slot=slot8; dim=128; in=68; tf=0.5; decMask=True; legacyDual=True; rawL=0.5; pedal=binary_4 |
| E31 | slot8-direct96-property-pad | 82.57 | 320.82 | 17.81 | 0.487 | target=raw_log_deviation; ov=0.125; ep=16; embed=slot_attribute; slot=slot8; dim=96; in=68; tf=0.5; decMask=True; legacyDual=True; rawL=0.25; pedal=binary_4 |
| E32 | slot8-fixed | 84.13 | 313.07 | 17.46 | 0.480 | target=raw_log_deviation; ov=0.125; ep=16; embed=slot_attribute; slot=slot8; dim=128; in=68; tf=0.5; decMask=True; legacyDual=True; rawL=0.25; pedal=binary_4 |
| E33 | slot6-128-direct-t5-6x6 | 90.82 | 333.72 | 23.97 | 0.485 | target=raw_log_deviation; ov=0.125; ep=16; embed=slot_attribute; slot=slot6; dim=128; in=68; tf=0.5; decMask=True; legacyDual=True; rawL=0.25; pedal=binary_4 |
| E34 | INR8-Dev-hybrid | 105.39 | 374.79 | 19.00 | 0.485 | target=raw_log_deviation; ov=0.125; ep=8; embed=slot_attribute; slot=slot8; dim=128; in=16; tf=1.0; decMask=False; rawL=0.25; pedal=binary_4 |
| E35 | C-absolute-log-overlap50 | 132.42 | 217.43 | 18.94 | 0.475 | target=raw_log_absolute; ov=0.5; ep=16; embed=slot_attribute; slot=slot5; dim=128; in=68; tf=0.5; decMask=True; zeroEmb=True; zxf=none; dual=none; legacyDual=False; pos=False; res=False; rawL=0.25; pedal=binary_4 |
| E36 | B-absolute-log-overlap125 | 144.82 | 229.89 | 19.71 | 0.488 | target=raw_log_absolute; ov=0.125; ep=16; embed=slot_attribute; slot=slot5; dim=128; in=68; tf=0.5; decMask=True; zeroEmb=True; zxf=none; dual=none; legacyDual=False; pos=False; res=False; rawL=0.25; pedal=binary_4 |

## 4. 重点实验族分析

### 4.1 当前最强区间：E01-E08

E01-E08 构成当前 INR 的第一梯队，PN IOI 在 `60.30-64.23`。这个区间的共同点是：大多仍使用 `raw_log_deviation`，overlap `12.5%`，16 epochs，`binary_4` pedal，decoder timing-feature mask 开启，且使用 68 维输入特征。

E01 `sine-control` 的 PN IOI 最低，为 `60.30`，但 Duration 是 `216.19`，明显比 E02/E03/E04 的 `194-196` 更差。因此如果只看 IOI，E01 排第一；如果把 IOI 和 Duration 同时作为 timing 主指标，E02/E03 更稳。

E02 `simple-folded-abs-slot-zeroembed` 是目前最值得作为代表的 INR 配置：IOI `61.10`、Duration `195.76`、Velocity `18.89`、Pedal `0.492`。它并非单项全最优，但 timing 两项均衡，且配置最干净：`zero_score_ioi_embedding=True`，`zero_ioi_transform=folded_abs`，没有 dual distribution，也没有 residual。

E03 `slot5-128-zero-ioi-folded-abs` 与 E02 几乎同档：IOI `61.27`、Duration `194.22`、Pedal `0.490`。它开启 `zero_ioi_positive_support=True`，Duration 略优于 E02，但 IOI、Velocity 没有更好。这个结果说明 positive support 不会伤害 Duration，但也没有突破 IOI 上限。

E05/E07/E08 的 dual-zero-folded 系列明显改善 pedal：E05 pedal `0.464` 是第一梯队里最好的 pedal；E07/E08 pedal 也在 `0.469`。代价是 IOI 和 Velocity 没有改善，尤其 E05 Velocity 到 `20.78`。也就是说 dual-zero-folded 更像 pedal/zero-IOI 形状修正，而不是主 timing 突破。

### 4.2 folded_abs / squared / positive support

folded_abs 是目前最稳定的 zero-IOI 处理方式。E02/E03/E04/E06 都在 IOI `61-63`、Duration `194-203`，整体优于普通 slot5、slot6、slot8 系列。

E13 `squared` 的 Duration 最低，为 `192.80`，但 IOI 升到 `65.46`，Velocity `21.06`，Pedal `0.501`。这说明 squared 对 Duration 有帮助，但会牺牲 IOI、Velocity 和 pedal，当前不适合作为主线。

residual 没带来收益。E04 开 residual 后 IOI `62.31`，E16 positive residual 更差到 IOI `67.78`、Duration `206.60`。从 PN Wass 看，zero-IOI residual 不是突破方向。

### 4.3 slot 结构与维度

slot5 是目前最稳定的 slot 版本。slot5 128 维的强配置集中在 IOI `61-66`；slot6 多数在 `68-76`；slot8 多数在 `68-84`。

slot5 加宽到 256 并没有收益。E21 `slot5-256-whole-token-pad` 的 IOI 是 `68.98`，比 E15 `slot5-128-whole-token-pad` 的 `66.27` 更差。当前问题不像是 slot hidden capacity 不够，更像是目标表示和生成分布仍未对齐 PT 的 timing prior。

slot8 的几个方向整体偏弱：`slot8-whole-token-mask` 是 `68.42`，`slot8-stable-dynamic` 是 `81.58`，`slot8-fixed` 是 `84.13`。这与早期 INR8-Dev-hybrid 的 `105.39` 一起说明，slot8/16维早期设置不是当前主线。

### 4.4 musical mask / slot6 / 架构变化

slot6 和 musical mask 没有带来 PN Wass 改善。E17 `slot6-128-mlp-decoder-musical-mask` 为 IOI `68.26`、Duration `212.06`；E19 `slot6-128-mlp` 为 IOI `68.54`、Duration `204.39`；E22 direct 更差到 pedal `0.517`。

E33 `slot6-128-direct-t5-6x6` 明显失败，IOI `90.82`、Duration `333.72`、Velocity `23.97`。更深/不同 decoder 结构没有自然转化为 sampling PN 分布改进。

### 4.5 absolute timing 目标

E35/E36 是最新 absolute timing 实验：

| 编号 | 实验 | target | overlap | PN IOI | PN Duration | PN Velocity | PN Pedal |
|---|---|---|---:|---:|---:|---:|---:|
| E35 | C-absolute-log-overlap50 | raw_log_absolute | 0.5 | 132.42 | 217.43 | 18.94 | 0.475 |
| E36 | B-absolute-log-overlap125 | raw_log_absolute | 0.125 | 144.82 | 229.89 | 19.71 | 0.488 |

overlap 50% 明显优于 overlap 12.5%：IOI 降低 `12.40`，Duration 降低 `12.46`，Velocity 和 Pedal 也略好。但 absolute timing 的 PN IOI 仍远高于 deviation 强配置：E35 的 IOI `132.42` 是 E02 的 `2.17x`。

更细的判断是：absolute timing 不是所有指标都崩。E35 的 Duration `217.43` 只比 E02 的 `195.76` 高约 `11%`，Velocity/Pedal 也没有大幅恶化；真正的问题集中在 IOI 分布。absolute target 直接预测绝对 IOI 时，模型更难学到演奏相对偏移的局部分布，导致 note-level IOI Wasserstein 拉大。

因此，absolute timing 目前不适合作为主线。它最多说明更大 overlap 能缓解 absolute target 的一些不稳定，但没有接近 deviation/folded_abs 族。

## 5. 与 PT 的差距

以 E02 `simple folded_abs` 作为当前 INR 代表：

| 对比 | PN IOI 差距 | PN Duration 差距 | PN Velocity 差距 | PN Pedal 差距 |
|---|---:|---:|---:|---:|
| E02 - PT official | +21.66 | +50.83 | +5.32 | +0.161 |
| E02 / PT official | 1.55x | 1.35x | 1.39x | 1.49x |
| E02 - PT split->ASAP | +17.45 | +43.09 | +3.65 | +0.138 |
| E02 / PT split->ASAP | 1.40x | 1.28x | 1.24x | 1.39x |

以 E35 `absolute overlap50` 对比：

| 对比 | PN IOI 差距 | PN Duration 差距 | PN Velocity 差距 | PN Pedal 差距 |
|---|---:|---:|---:|---:|
| E35 - PT official | +92.98 | +72.50 | +5.37 | +0.144 |
| E35 / PT official | 3.36x | 1.50x | 1.40x | 1.44x |
| E35 - E02 | +71.32 | +21.67 | +0.05 | -0.017 |
| E35 / E02 | 2.17x | 1.11x | 1.00x | 0.97x |

这里最关键的观察是：E35 的 Pedal 比 E02 略好，Velocity 几乎持平，但 IOI 大幅变差。所以 absolute target 的主要失败不是 pedal 或 velocity，而是 timing IOI 的 note-level 分布。

## 6. 结论与方向

当前应该把 `simple folded_abs` / `slot5 folded_abs` 作为 INR 主线，而不是 absolute timing。

最有价值的配置族：

1. E02 `simple-folded-abs-slot-zeroembed`
2. E03 `slot5-128-zero-ioi-folded-abs`
3. E05/E07 dual-zero-folded 作为 pedal 改善分支

不建议继续优先投入的方向：

1. absolute timing：IOI PN Wass 太高，overlap 50% 也没有接近 deviation 族。
2. slot8/stable-dynamic/property schedule：整体不如 slot5/folded_abs。
3. slot6 direct/T5 6x6：PN Wass 明显退化。
4. residual zero-IOI：当前没有看到收益。

如果下一轮目标是逼近 PT，优先级应当是：

1. 在 E02/E03 上做更接近 PT sampling prior 的改造，而不是换成 absolute target。
2. 单独研究 IOI PN Wass 缺口，因为 Duration 已经比 IOI 更接近 PT。
3. 保留 folded_abs/slot5/zero score IOI embedding 这一组强约束，再尝试改 sampling temperature、distribution calibration 或 score-conditioned timing prior。
4. pedal 可以借鉴 dual-zero-folded，但不要为了 pedal 牺牲 IOI 和 Velocity。

## 7. 结果来源路径

| 编号 | 结果目录 |
|---|---|
| E01 | `results/slot8_fixed_vs_sine_2gpu/20260710_slot8fix_2gpu_ddpfind/sine-control` |
| E02 | `results/zeroioi_loss_source_4gpu/20260711_loss_source/simple-folded-abs-slot-zeroembed` |
| E03 | `results/slot5_slot6_zeroioi_4gpu/slot5_zeroioi_transform_2x2gpu/20260711_abs_square/slot5-128-zero-ioi-folded-abs` |
| E04 | `results/slot5_folded_abs_next_2x2gpu/20260711_residual_stable/slot5-128-folded-abs-zero-residual-ioi-duration` |
| E05 | `results/slot5_zeroembed_dual_2x2gpu/20260711_dual_dist/slot5-128-zeroembed-dual-zero-folded` |
| E06 | `results/slot5_folded_abs_next_2x2gpu/20260711_residual_stable/slot5-128-folded-abs-stable-v2` |
| E07 | `results/zeroioi_loss_source_4gpu/20260711_loss_source/dual-zero-folded-no-zeroembed` |
| E08 | `results/zeroioi_loss_source_4gpu/20260711_loss_source/dual-zero-folded-timing-zeroembed` |
| E09 | `results/rawlog_30_80_repro_4gpu/20260710_173245/sine` |
| E10 | `results/slot5_zeroembed_dual_2x2gpu/20260711_dual_dist/slot5-128-zeroembed-dual-sn` |
| E11 | `results/zeroioi_loss_source_4gpu/20260711_loss_source/dual-zero-folded-ioi-only` |
| E12 | `results/slot5_slot6_zeroioi_4gpu/20260711_zeroioi_4gpu/slot5-128-zero-ioi-positive` |
| E13 | `results/slot5_slot6_zeroioi_4gpu/slot5_zeroioi_transform_2x2gpu/20260711_abs_square/slot5-128-zero-ioi-squared` |
| E14 | `results/slot5_slot6_zeroioi_4gpu/20260711_zeroioi_4gpu/slot5-128-stable-dynamics` |
| E15 | `results/slot5_width_2gpu/20260710_slot5width/slot5-128-whole-token-pad` |
| E16 | `results/slot5_slot6_zeroioi_4gpu/20260711_zeroioi_4gpu/slot5-128-zero-ioi-positive-residual` |
| E17 | `results/slot5_slot6_zeroioi_4gpu/20260711_zeroioi_4gpu/slot6-128-mlp-decoder-musical-mask` |
| E18 | `results/slot8_mask_stable_2gpu/20260710_slot8_mask_stable_v1/slot8-whole-token-mask` |
| E19 | `results/slot6_musical_4gpu/20260711_slot6_musical/slot6-128-mlp` |
| E20 | `results/slot8_direct96_2gpu/20260710_direct96/slot8-direct96-whole-token-pad` |
| E21 | `results/slot5_width_2gpu/20260710_slot5width/slot5-256-whole-token-pad` |
| E22 | `results/slot6_musical_4gpu/20260711_slot6_musical/slot6-128-direct` |
| E23 | `results/rawlog_30_80_repro_4gpu/20260710_173245/slot5` |
| E24 | `results/prechord_recovery_2x2gpu/20260710_160759/rawlog_nomus_retry` |
| E25 | `results/slot8_property_schedule_2gpu/20260710_slot8_property_schedule_v1/slot8-correlated-perf-pad50` |
| E26 | `results/slot6_musical_4gpu/20260711_slot6_musical/slot6-128-direct-gpt16-bs8acc8` |
| E27 | `results/slot8_property_schedule_2gpu/20260710_slot8_property_schedule_v1/slot8-mixed-property-mask-stable` |
| E28 | `results/rawlog_30_80_repro_4gpu/20260710_173245/cine` |
| E29 | `results/slot8_mask_stable_2gpu/20260710_slot8_mask_stable_v1/slot8-stable-dynamic` |
| E30 | `results/rawlog_30_80_repro_4gpu/20260710_173245/slot8` |
| E31 | `results/slot8_direct96_2gpu/20260710_direct96/slot8-direct96-property-pad` |
| E32 | `results/slot8_fixed_vs_sine_2gpu/20260710_slot8fix_2gpu_ddpfind/slot8-fixed` |
| E33 | `results/slot6_musical_4gpu/20260711_slot6_musical/slot6-128-direct-t5-6x6` |
| E34 | `results/prechord_recovery_2x2gpu/20260710_160759/INR8-Dev-hybrid` |
| E35 | `results/absolute_timing_overlap_2x2gpu/20260711_abs_timing/C-absolute-log-overlap50` |
| E36 | `results/absolute_timing_overlap_2x2gpu/20260711_abs_timing/B-absolute-log-overlap125` |

## 8. 备注

本文没有把本地 `results/pt_pipeline/*` 的 pedal 数值纳入主对比，因为那些文件中的 PN pedal 量纲与最终 PT 表不一致。最终 PT baseline 以本文第 1 节的确认表为准。

另外，早期 legacy metric 文件也存在可参考结果，例如 `INR8-Dev` cheap15 sampling PN 为 `112.19 / 415.19 / 19.49 / 0.476`，`INR8-Abs` 为 `300.64 / 509.45 / 20.34 / 0.489`。这些结果明显落后于当前 folded_abs 主线，因此不进入主排序表。
