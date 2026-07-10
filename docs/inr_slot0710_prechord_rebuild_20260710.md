# INR Slot 0710 Pre-Chord Rebuild

Date: 2026-07-10

## Branch History

- Pre-chord base: `cc2320d rawlog rep`
- Chord-era branch backup: `backup/main-before-prechord-slot-rebuild-20260710`
- Chord-era bundle:
  `results/rebuild_inr_slot0710_pre_chord/main-before-prechord-slot-rebuild-20260710.bundle`
- Reapplied slot patches:
  - `0001-add-inr-slot-0709.patch`
  - `0002-add-inr0709.patch`
  - `0003-add-inr_slot0710.patch`

## Restored Slot Design

- PT uses shared score/performance IOI, duration, and velocity slots and predicts
  absolute timing.
- INR uses separate score/performance IOI, duration, and velocity slots and
  predicts relative timing.
- Pitch uses an embedding table.
- Musical onset, duration, and length use embedding tables in musical slot
  variants.
- Musical length has a separate no-value embedding for non-first nodes.
- Every slot has MASK and PAD embeddings.
- Decoder property dropout independently drops 50 percent of performance
  properties and routes dropped properties to the corresponding slot PAD
  embedding.

## Pre-Chord Cleanup

- Removed chord pitch multihot paths from note encoding.
- Removed chord offset targets, decoder rows, inference payloads, and loss
  components.
- Restored pure-note target layouts:
  - Legacy raw-log deviation: 9 dimensions.
- New INR raw-log deviation: 9 dimensions with log-deviation and raw-deviation
  timing targets.
  - Raw-log absolute: 9 dimensions.
  - Other absolute/deviation layouts: 7 dimensions.
- Legacy raw-log mode is inferred from an existing 9-dimensional output
  configuration; no extra user config key is required.
- Fixed skew-normal head packing for both legacy dual timing heads and new
  single timing heads.
- Split train loss logging into IOI, duration, velocity, pedal, per-pedal, and
  timing auxiliary details. Offset loss is no longer present.

## No-Musical Contract

Both recovery experiments use no musical input:

- `configs/inr0624_epr_sn_rawlog_sine_nomus_tfmask50.json`
- `configs/slot0710_inr8_dev.json`

Their integrated note input is 16 dimensions:

- score control: 5
- performance control: 9
- visibility masks: 2

No zero-filled 51-dimensional musical block is retained.

## Verification

Real ASAP prepared sidecars were used for forward smoke tests:

- Legacy raw-log: input 16, label 9, raw decoder output 19, finite loss.
- INR8-Dev uses skew-normal distribution loss for log-deviation timing and
  direct Huber regression loss for raw-deviation timing with weight 0.25.
- INR8-Dev: input 16, label 9, raw decoder output 15, materialized output 9,
  finite loss.
- INR8-Dev property dropout produced performance-property missing masks and
  routed them to an 8 by 128 slot PAD embedding table.
