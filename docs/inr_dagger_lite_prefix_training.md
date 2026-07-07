# INR DAgger-Lite Prefix Training

## Motivation

Current INR EPR diagnostics show a clear teacher-forcing/free-running mismatch:

- `k=1` self-feedback already degrades substantially from teacher forcing.
- Single-channel pollution mostly hurts the same channel.
- Full-AR single-channel protection mostly rescues only the protected channel.
- IOI and duration form a coupled timing subsystem, but velocity and pedal drift largely independently.

This points to decoder-input regime mismatch rather than an output-head-only problem. The model is trained mostly with GT right-shifted performance feedback, while inference uses model-generated feedback.

## First Method: TF-Pred Prefix DAgger-Lite

The first implementation uses a cheap off-policy approximation:

1. Sample a fraction of train windows, e.g. `50%`.
2. Run one teacher-forced forward pass.
3. Materialize model outputs to normalized `target7` predictions.
4. Store `pred_target7[seq_len, 7]` in a prefix cache.
5. During training, keep GT labels as the loss target, but construct decoder input from a mixture of GT and cached predictions.

The model therefore learns:

```text
loss target:       GT target7
decoder feedback: mix(GT target7, TF-pred target7)
```

The cache stores targets, not embeddings. Embeddings are rebuilt by the existing INR decoder-input path, so this remains compatible with future representation changes.

## Replacement Policy

For cached windows, sample one replacement mode:

| mode | probability | replaced channels |
|---|---:|---|
| `full` | 0.30 | IOI + duration + velocity + pedal |
| `timing` | 0.20 | IOI + duration |
| `ioi` | 0.10 | IOI only |
| `duration` | 0.10 | duration only |
| `velocity` | 0.15 | velocity only |
| `pedal` | 0.15 | pedal4 only |

The replacement is applied before right-shifting. Thus cached `pred[t]` is only visible when predicting later positions, avoiding same-token leakage.

## Refresh Schedule

Recommended first experiment:

```text
cache_fraction = 0.50
cache_type = tf_pred
refresh = after each eval, roughly every 0.5 epoch
materialization = sample
```

The refresh cost is approximately one teacher-forced inference pass over the sampled train windows, not an AR rollout. This is much cheaper than full on-policy DAgger.

## Extension: K=1 Two-Pass Cache

A later cache backend can approximate `k=1` on-policy feedback without full sequential AR:

1. Pass A: teacher-forced prediction gives `pred_tf[t]`.
2. Pass B: feed `pred_tf[t-1]` as the previous token and predict again.

This costs roughly two TF passes and reuses the same `target7` cache schema.

Full on-policy AR cache is not the first priority because it is expensive and current diagnostics already implicate one-step self-prefix mismatch.

## Evaluation

Track:

- Teacher-forced `k=0` sample.
- `k=1` sample feedback.
- Full AR sample.

Initial success criteria:

```text
k=1 approaches TF:
IOI <= 22
Dur <= 30
Vel <= 1.2
Pedal <= 0.04

full AR moves toward <= 1.5x TF metrics
```

## Configuration Keys

Planned training keys:

```json
{
  "dagger_prefix_training": true,
  "dagger_cache_type": "tf_pred",
  "dagger_cache_fraction": 0.5,
  "dagger_refresh_on_eval": true,
  "dagger_refresh_at_train_start": true,
  "dagger_materialize_strategy": "sample",
  "dagger_apply_prob": 1.0,
  "dagger_replacement_weights": {
    "full": 0.30,
    "timing": 0.20,
    "ioi": 0.10,
    "duration": 0.10,
    "velocity": 0.15,
    "pedal": 0.15
  }
}
```

The default is disabled and should not change existing training.

## Implementation Notes

The current code path is intentionally in-memory:

- Dataset samples carry a stable `example_index`.
- The trainer stores `example_index -> pred_target7` in the train dataset.
- The collator samples the replacement mode and emits `decoder_feedback_continuous`.
- The model uses `decoder_feedback_continuous` only for decoder inputs; loss labels remain `labels_continuous`.

For DDP robustness, each rank refreshes the same selected index set on its own GPU and writes its own local dataset cache. This repeats work across ranks but avoids cache/sampler partition mismatch.

When `dagger_prefix_training=true`, training persistent workers are disabled so refreshed caches are not trapped inside long-lived dataloader worker copies. If dataloader workers are enabled, mid-epoch refreshes may still become visible at the next worker iterator boundary; `dataloader_num_workers=0` gives the most immediate cache updates for debugging.
