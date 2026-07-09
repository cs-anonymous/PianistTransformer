# INR DAgger-Lite Prefix Training

Last updated: 2026-07-08

This document summarizes the implemented DAgger-Lite prefix-training path used by:

```text
results/dagger_experiments_20260707/runs/
  dagger_tf_pred_rolloutval_k1_curriculum_noearly_asap_20260707
```

The corresponding diagnostic rollout experiments are under:

```text
results/stationary_rollout_20260708/
```

## Short Summary

The DAgger model does not change the INR EPR output representation or the loss target. It changes the training-time decoder feedback distribution.

Baseline teacher forcing:

```text
loss target:       GT target7
decoder feedback: GT target7
```

DAgger-Lite prefix training:

```text
loss target:       GT target7
decoder feedback: mix(GT target7, cached model-pred target7)
```

The cached prediction is inserted into the decoder input path only. The supervised label remains GT, so this is a prefix/feedback robustness method, not self-training on generated labels.

## Why It Was Added

Earlier INR EPR diagnostics showed a teacher-forcing/free-running mismatch:

- `k=1` self-feedback already degrades substantially from teacher forcing.
- Single-channel feedback pollution mostly damages the same channel.
- IOI and duration behave like a coupled timing subsystem.
- Velocity and pedal have more independent drift modes.

The main hypothesis was that the model sees mostly GT right-shifted performance rows during training, but generated rows during inference. DAgger-Lite exposes the decoder to model-like prefix rows while still optimizing against GT labels.

## Implemented Changes

### 1. Dataset-level prefix cache

`INRMapDataset` now keeps an in-memory DAgger cache:

```text
example_index -> pred_target7[seq_len, 7]
```

Relevant behavior:

- Every sample carries a stable `example_index`.
- `set_dagger_prefix_cache(cache)` installs cached predicted `target7` rows.
- `clear_dagger_prefix_cache()` clears cached predictions.
- `set_dagger_mask_cache(indices)` supports a mask-only ablation backend.
- When a cached index is loaded, the sample includes `dagger_prefix_continuous`.

The cache stores normalized target rows, not embeddings. The normal INR decoder-input builder still rebuilds embeddings from `decoder_feedback_continuous`, so future representation changes do not require changing the cache schema.

### 2. DAgger cache refresh in the trainer

`NodeSFTTrainer.refresh_dagger_prefix_cache()` materializes model predictions for a subset of training windows.

Supported cache backends:

| cache type | meaning | current model uses it |
|---|---|---:|
| `tf_pred` | one teacher-forced pass, then materialize predictions | yes |
| `k1_twopass` | TF prediction first, then a second pass using predicted feedback | no |
| `mask` | no prediction values; only mask selected feedback channels | no |

For `tf_pred`, the trainer performs:

1. Load selected train windows.
2. Run the model in eval mode with GT decoder feedback.
3. Materialize EPR predictions using `dagger_materialize_strategy`.
4. Store the resulting normalized `target7` rows in the dataset cache.

The current experiment used:

```text
dagger_cache_type: tf_pred
dagger_materialize_strategy: sample
dagger_cache_batch_size: 32
dagger_cache_num_workers: 4
```

### 3. Cache scope changed to next training interval

The current run did not cache a random global 50% of the dataset. It used:

```text
dagger_cache_scope: next_interval
dagger_cache_max_interval_fraction: 0.5
```

This means each refresh targets the examples expected in the next training/eval interval, then caps that interval selection to 50%. This is cheaper and more local than refreshing a large random subset of the whole train set.

### 4. Window curriculum

The current run enabled a linear window curriculum:

```text
dagger_window_curriculum: linear
dagger_window_curriculum_start: 0.0
dagger_window_curriculum_end: 1.0
```

At early steps, very few selected windows receive cached model feedback. As training progresses, the fraction ramps toward the full selected cache scope.

This curriculum controls how much of the selected cache scope is actually kept. It does not change the supervised target.

### 5. DDP cache sharding and gather

For distributed training, each rank refreshes a shard of the selected indices:

```text
local_indices = indices[rank::world_size]
```

Then all ranks gather the local cache shards and install the merged cache into their local dataset copy. This avoids every rank redundantly materializing all selected examples while keeping each rank's dataset cache logically complete.

### 6. Collator-level channel replacement

`NodeSFTDataCollator` creates `decoder_feedback_continuous` from GT labels, then selectively overwrites some channels with cached predictions.

The replacement mode is sampled per cached example:

| mode | default probability | replaced channels |
|---|---:|---|
| `full` | 0.30 | IOI + duration + velocity + pedal |
| `timing` | 0.20 | IOI + duration |
| `ioi` | 0.10 | IOI only |
| `duration` | 0.10 | duration only |
| `velocity` | 0.15 | velocity only |
| `pedal` | 0.15 | pedal4 only |

The current run used:

```text
dagger_apply_prob: 1.0
```

So if an example has a cached prefix, the collator always attempts one sampled replacement mode.

The replacement happens before the model constructs the right-shifted decoder input. Therefore cached `pred[t]` is used only as feedback for later positions, avoiding same-token label leakage.

### 7. Training dataloader worker behavior

When `dagger_prefix_training=true`, persistent dataloader workers are disabled:

```text
dataloader_persistent_workers: false
```

This prevents refreshed in-memory dataset caches from being trapped inside stale long-lived worker copies.

### 8. Rollout-aware eval loss

The model also enabled k-step rollout validation:

```text
rollout_eval_enabled: true
rollout_eval_k: 1
rollout_eval_feedback_strategy: sample
rollout_eval_materialize_strategy: sample
rollout_eval_weight: 1.0
metric_for_best_model: eval_loss
```

During evaluation, the trainer computes:

```text
eval_tf_loss = normal teacher-forced eval loss
eval_rollout_k1_loss = loss after one sampled feedback pass
eval_loss = eval_tf_loss + rollout_eval_weight * eval_rollout_k1_loss
```

Because `metric_for_best_model=eval_loss`, `checkpoint-best` is selected using this combined TF + k=1 rollout objective, not pure teacher-forced eval loss.

## Current Experiment Configuration

The stationary diagnostics used this checkpoint:

```text
results/dagger_experiments_20260707/runs/
  dagger_tf_pred_rolloutval_k1_curriculum_noearly_asap_20260707/
  training/dagger_tf_pred_rolloutval_k1_curriculum_noearly_asap_20260707/
  checkpoint-best
```

Important config values:

```text
train_performance_dataset: ASAP
num_train_epochs: 16.0
adapt_on_asap_after_train: true
adapt_num_train_epochs: 4

epr_distribution: mixture_logistic_normal
epr_mixture_components: 3
epr_timing_target: log_deviation

dagger_prefix_training: true
dagger_cache_type: tf_pred
dagger_cache_scope: next_interval
dagger_cache_max_interval_fraction: 0.5
dagger_window_curriculum: linear
dagger_window_curriculum_start: 0.0
dagger_window_curriculum_end: 1.0
dagger_materialize_strategy: sample
dagger_apply_prob: 1.0
dagger_refresh_at_train_start: true
dagger_refresh_on_eval: true

rollout_eval_enabled: true
rollout_eval_k: 1
rollout_eval_feedback_strategy: sample
rollout_eval_materialize_strategy: sample
rollout_eval_weight: 1.0
```

## What This Version Did Not Change

This DAgger-Lite version did not change:

- the EPR target schema;
- the timing target, still `log_deviation`;
- the probabilistic head, still `mixture_logistic_normal` with 3 components;
- the core supervised loss target, still GT `target7`;
- inference-time AR mechanics;
- any explicit long-horizon stationary-distribution regularizer.

## Current Finding

The 2026-07-08 cheap15 stationary diagnostics show that this DAgger model still does not learn a stable full-AR timing stationary distribution.

Observed behavior:

- Greedy rollout still collapses timing variance as k grows.
- Sampling rollout still drifts right and inflates IOI variance under full AR.
- Timing head stats show IOI predicted mean and std both increase under closed-loop history.
- Single-channel feedback ablation suggests IOI feedback, duration feedback, and cross-channel coupling all contribute.

Therefore, this DAgger-Lite model improved the training/inference feedback mismatch only locally. It is still not a full solution for long-horizon closed-loop timing stability.

## Likely Next Changes

The next version should target the closed-loop chain more directly:

- train with longer on-policy or approximate on-policy prefixes, not only `tf_pred`;
- increase coverage of timing-channel feedback, especially IOI + duration;
- include k-step rollout losses beyond `k=1`;
- add distribution matching terms for timing mean/std in log-deviation space;
- evaluate whether deterministic or low-temperature timing feedback plus sampled velocity/pedal is a useful inference mitigation.
