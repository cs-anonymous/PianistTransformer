# baseline0807 validation HR analysis

Date: 2026-08-08

This note summarizes the corrected baseline0807 validation experiment for the
pedal-group+mask INR model. The corrected analysis includes pedal Wasserstein in
both validation HR and true test HR.

## Setup

The run is `results/baseline0807_epoch_hr_20260807_170343`.

- model: pedal-group+mask, binary pedal targets
- training: `bs=32`, `gbs=96`
- epochs saved and evaluated: 1 through 12
- validation file: `validation_window_rollout_hr_pedal.json`
- validation sweep: `m=32/64`, rollout `k=2/4/8/16/32`
- validation selection score: `mean(validation_pn_hr, validation_pp_hr)`
- true test metrics: `ep*/test/eval_pn_pp_metrics.json`

The current validation implementation uses the fixed valid windows from
`train_valid_asap3_nonasap05_m64_v1`: 68 eligible works, one valid window per
work, and each eligible work has at least two ASAP performances. That gives 68
possible validation windows in the current fixed split.

For comparison, the base train pool before the fixed valid-window restriction has
146 eligible works and 994 eligible work-window pairs. This experiment did not
sample from that larger pool; it used the fixed valid split.

## HR Definition

HR means human-relative Wasserstein distance. Lower is better.

For each prefix `pn` or `pp`:

```text
HR = mean(
  IOI_W1      / HUMAN_IOI_W1,
  DURATION_W1 / HUMAN_DURATION_W1,
  VELOCITY_W1 / HUMAN_VELOCITY_W1,
  PEDAL_W1    / HUMAN_PEDAL_W1
)
```

The denominators are fixed human reference Wasserstein values:

```text
PN:
  ioi       26.554209906777686
  duration 105.54706686568747
  velocity 11.716304664046339
  pedal    0.246551332481009

PP:
  ioi       7.952391192015138
  duration 25.9246360560533
  velocity 3.057647553002258
  pedal    0.07682493502888045
```

The validation script still emits `pn_hr3` and `pp_hr3` for the old
ioi/duration/velocity-only view, but the corrected selection and analysis use
the 4-feature HR above.

## Correlation

Pearson measures linear agreement across epochs. Spearman converts each sequence
to ranks first, then computes Pearson on those ranks. In other words, Spearman
asks whether validation orders checkpoints like the true test set.

### HR Pearson

```text
PN HR
        k=2    k=4    k=8    k=16   k=32
m=32   0.974  0.990  0.988  0.977  0.972
m=64   0.965  0.988  0.991  0.981  0.975

PP HR
        k=2    k=4    k=8    k=16   k=32
m=32   0.986  0.979  0.941  0.900  0.881
m=64   0.972  0.988  0.973  0.941  0.925

Mean HR
        k=2    k=4    k=8    k=16   k=32
m=32   0.985  0.984  0.955  0.921  0.905
m=64   0.972  0.989  0.978  0.950  0.938
```

### HR Spearman

```text
PN HR
        k=2    k=4    k=8    k=16   k=32
m=32   0.294  0.559  0.720  0.748  0.692
m=64   0.343  0.531  0.741  0.776  0.776

PP HR
        k=2    k=4    k=8    k=16   k=32
m=32   0.895  0.832  0.308  0.182  0.126
m=64   0.832  0.860  0.727  0.357  0.357

Mean HR
        k=2    k=4    k=8    k=16   k=32
m=32   0.902  0.769  0.301  0.189  0.133
m=64   0.832  0.839  0.608  0.343  0.301
```

### Raw Wass Correlation

The raw Wasserstein correlations are also high, especially for `m=64`.

```text
Mean Wass Pearson
        k=2    k=4    k=8    k=16   k=32
m=32   0.957  0.941  0.930  0.925  0.922
m=64   0.974  0.966  0.959  0.955  0.953

Mean Wass Spearman
        k=2    k=4    k=8    k=16   k=32
m=32   0.944  0.944  0.944  0.951  0.951
m=64   0.965  0.944  0.965  0.965  0.965
```

Raw Wass ranking is more stable than HR ranking because HR strongly weights
features with small human denominators, especially pedal. In this run the pedal
test values change very little across epochs, so small denominator effects can
move HR ranks without changing the broad quality trend.

Heatmaps were generated at:

- `results/baseline0807_epoch_hr_20260807_170343/validation_pedal_hr_correlation_heatmap.png`
- `results/baseline0807_epoch_hr_20260807_170343/validation_pedal_wass_correlation_heatmap.png`

## Recommended Validation

Use:

```text
m = 64
k = 4
selection score = mean(PN_HR, PP_HR)
```

Why this setting:

- best mean-HR Pearson: 0.989
- balanced PN/PP HR Pearson: PN 0.988, PP 0.988
- strong PP rank agreement: PP Spearman 0.860
- low rollout cost compared with `k=8/16/32`

`k=8` is already enough for PN Pearson, and `m=64,k=8` gives the best PN HR
Pearson at 0.991. But it is worse for the combined checkpoint-selection target:
mean-HR Pearson drops to 0.978 and mean-HR Spearman drops to 0.608. Larger
`k=16/32` further improves PN rank correlation but damages PP and mean-HR rank
correlation. So `k=4` is the best low-cost compromise.

## Epoch Selection

With `m=64,k=4`, validation selects epoch 12:

```text
epoch 12:
  validation PN_HR    1.644918
  validation PP_HR    4.116087
  validation mean_HR  2.880503

  true test PN_HR     1.521942
  true test PP_HR     2.272201
  true test mean_HR   1.897071
```

The true test oracle by mean HR is epoch 9:

```text
epoch 9:
  validation PN_HR    1.660780
  validation PP_HR    4.174709
  validation mean_HR  2.917745

  true test PN_HR     1.503703
  true test PP_HR     2.210840
  true test mean_HR   1.857272
```

The gap is small: epoch 12 is worse than epoch 9 by 0.0398 mean HR, and it
avoids the clearly bad epoch 5.

The late epochs are close:

```text
epoch 9   true mean_HR 1.857272
epoch 11  true mean_HR 1.889808
epoch 12  true mean_HR 1.897071
```

## Why Validation Is Higher Than Test

Validation HR is systematically higher than true test HR, especially PP HR. This
does not mean the correlation is wrong; it means the validation proxy has a
different scale.

Main causes:

- validation is local-window based, while test aggregates full saved-MIDI
  predictions;
- validation computes directly from model output distributions on grouped
  windows, while test computes Wasserstein after materialized MIDI generation;
- validation uses the fixed training-valid windows, not the same score
  distribution as the held-out test set;
- PP HR has small human denominators, so modest raw W1 differences become large
  HR values.

For checkpoint selection, the absolute value is less important than the
cross-epoch ordering and trend. On that criterion, `m=64,k=4` is strongly
correlated with true test HR.

## Pedal Detail

Pedal is included in the corrected HR. The true test pedal W1 values around the
best epochs are:

```text
epoch 9:
  PN pedal_W1 0.487289
  PP pedal_W1 0.207953

epoch 11:
  PN pedal_W1 0.487225
  PP pedal_W1 0.207573

epoch 12:
  PN pedal_W1 0.487074
  PP pedal_W1 0.208199
```

Pedal alone does not explain the epoch 9 vs epoch 12 difference; the pedal
values are nearly tied. The remaining difference comes from the combined
timing/duration/velocity/pedal HR on the full test distribution.

## Artifacts

Pedal-inclusive analysis artifacts:

- `results/baseline0807_epoch_hr_20260807_170343/epoch_hr_summary_pedal_full.json`
- `results/baseline0807_epoch_hr_20260807_170343/epoch_hr_summary_pedal_full.csv`
- `results/baseline0807_epoch_hr_20260807_170343/validation_pedal_correlations.csv`
- `results/baseline0807_epoch_hr_20260807_170343/validation_pedal_component_correlations.csv`

Conclusion: switch validation selection from teacher-forcing loss to
`m=64,k=4` 4-feature PN/PP HR. This is not an oracle for the exact best epoch,
but it is a low-cost, pedal-inclusive proxy that tracks true test performance
well across epochs.
