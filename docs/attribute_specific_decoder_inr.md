# Attribute-Specific Decoder INR

This note records the new ASD-INR direction for INR score-to-performance modeling.

## Goal

Test one structural question:

```text
Should the four performance attributes share one decoder dynamics, or use one decoder per attribute?
```

The new line keeps the score history and autoregressive note history unchanged. Only the decoder factorization changes.

## Main decision

Use a clean `10+2` backbone and disable the musical slot for this line.

Recommended settings:

```text
backbone: 10 encoder + 2 decoder
musical slot: off
musical_feature_mode = none
disable_musical_features = true
slot version: slot5
```

This gives a simpler 5-slot INR layout:

```text
pitch + ioi + duration + velocity + pedal
```

The musical slot can be revisited later, but it is not part of the main comparison.

## What to compare

For each family, keep only two directly comparable configs:

```text
baseline
ASD full decoder
```

That is the cleanest way to isolate decoder factorization.

### CINR line

Reference baseline already exists in submission:

```text
configs/inr_epr/cinr__default_dlm_k1_bounded5.json
```

Current submission metrics:

```text
PN-Wass:
  IOI      32.858689
  Duration 119.993948
  Velocity 16.663085
  Pedal    0.455257

PP-Wass:
  IOI      11.744861
  Duration 51.130663
  Velocity 6.376962
  Pedal    0.238378
```

Planned clean comparison:

```text
baseline: cinr__default_dlm_k1_bounded5 on 10+2, no musical
ASD:      cinr__asd_full4_dlm_k1_bounded5 on 10+2, no musical
```

### DINR line

Reference baseline already exists in submission:

```text
configs/inr_epr/dinr__default_no_coord.json
```

Current submission metrics:

```text
PN-Wass:
  IOI      42.845670
  Duration 150.791267
  Velocity 25.797259
  Pedal    0.396805

PP-Wass:
  IOI      24.707348
  Duration 71.138373
  Velocity 23.618771
  Pedal    0.208736
```

Planned clean comparison:

```text
baseline: dinr__default_no_coord on 10+2, no musical
ASD:      dinr__asd_full4_no_coord on 10+2, no musical
```

## Evaluation protocol

Keep the reporting simple:

```text
metrics: sampling PN-Wass, sampling PP-Wass
samples per score: 2-3
no deterministic main result
attributes: IOI, duration, velocity, pedal
```

## Parameter and cost snapshot

Measured single-decoder counts with `musical_feature_mode = none`:

| Family | Backbone | Total | Encoder | Decoder | Note emb. | Heads |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| CINR | 8+4 | 142.473M | 75.523M | 51.344M | 2.312M | 13.294M |
| CINR | 9+3 | 140.112M | 84.963M | 39.543M | 2.312M | 13.294M |
| CINR | 10+2 | 137.751M | 94.403M | 27.742M | 2.312M | 13.294M |
| DINR | 8+4 | 136.624M | 75.523M | 51.344M | 2.370M | 7.445M |
| DINR | 9+3 | 134.264M | 84.963M | 39.543M | 2.370M | 7.445M |
| DINR | 10+2 | 131.903M | 94.403M | 27.742M | 2.370M | 7.445M |

Estimated ASD full4 totals on the same no-musical setup:

| Family | Backbone | ASD full4 |
| --- | --- | ---: |
| CINR | 8+4 | 296.504M |
| CINR | 9+3 | 258.740M |
| CINR | 10+2 | 220.976M |
| DINR | 8+4 | 290.714M |
| DINR | 9+3 | 252.950M |
| DINR | 10+2 | 215.185M |

Short read:

```text
8+4 full4 is too heavy for the first clean pass.
9+3 is better.
10+2 is the best starting point now.
```

## Suggested mainline config names

```text
cinr__default_dlm_k1_bounded5_10x2_nomus
cinr__asd_full4_dlm_k1_bounded5_10x2_nomus
dinr__default_no_coord_10x2_nomus
dinr__asd_full4_no_coord_10x2_nomus
```

## Future work

After the 10+2 clean line is clear, the next useful variants are:

```text
8+4 or 9+3 efficiency comparisons
shared-lower / upper-specific decoder split
timing-expression grouped decoder
musical slot re-entry, only if it helps
```

The point of this version is clarity first: one backbone, one ablation axis, one sampling metric set.
