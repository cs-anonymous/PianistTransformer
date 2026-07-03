# Data Process Pipeline

Integrated Note data processing is intentionally split into two stages.

## 1. Generate JSON With Paired MIDI

```bash
python src/data_process/generate_json_with_paired_midi.py --overwrite
```

This stage reads PianoCoRe-A metadata, refined score MIDI, refined performance
MIDI, and PianoCoRe alignment files. It writes one work-level JSON beside each
refined score MIDI, or mirrors the refined tree under `--output-dir`.

```text
data/pianocore/PianoCoRe/refined/**/*.json
```

The raw v3 output contains:

- `score.pitch`
- `score.score_raw`
- `performances[].label_shared_raw`
- `performances[].label_pedal4_raw`
- `performances[].label_pedal2_raw`
- `performances[].interpolated`

`label_shared_raw` stores `ioi_ms`, `duration_ms`, and `velocity`.
`label_pedal4_raw` stores the PT-style four sampled pedal values.
`label_pedal2_raw` stores the native start/control pedal representation
extracted directly from MIDI CC64 curves, not converted from `label_pedal4_raw`.

## 2. Update JSON Score Feature With XML

```bash
python src/data_process/update_json_score_feature_with_xml.py
```

This stage reads the existing `*.json` INR files and projects XML/MXL score
features onto the refined score notes. It updates each JSON in place to schema
`pianocore_integrated_node_work_v2`.

The output adds:

- `score.score_feature`
- `score.has_score_feature`
- `meta.xml_to_refined_score_alignment`

For the XML-derived score grid fields in `score.score_feature`:

- `mo` comes from `MIDI2ScoreTransformer` `offset` with raw range `[0, 6]`
- `md` comes from `MIDI2ScoreTransformer` `duration` with raw range `[0, 4]`
- `ml` comes from the measure-length form of `downbeat`, using raw range `[0, 6]`
- all three are quantized on a fixed `1/24` quarter-note grid before/after normalization

The normalized mapping used by the current schema is:

```python
mo_norm = clamp(mo / 6.0, 0.0, 1.0)
md_norm = clamp(md / 4.0, 0.0, 1.0)
ml_norm = clamp(ml / 6.0, 0.0, 1.0)
```

The inverse mapping for decode is:

```python
SCORE_GRID = 1.0 / 24.0
mo = round((mo_norm * 6.0) / SCORE_GRID) * SCORE_GRID
md = round((md_norm * 4.0) / SCORE_GRID) * SCORE_GRID
ml = round((ml_norm * 6.0) / SCORE_GRID) * SCORE_GRID
```

Storing normalized values in `[0, 1]` with 5 decimal places is sufficient for this
grid, because the smallest normalized step is `1/144 ≈ 0.00694444`, which is much
larger than `1e-5`.

Coverage summaries are written by this stage, so a separate audit entrypoint is
not part of the main pipeline anymore.

## Helper Modules

- `score_xml_alignment.py`: shared XML/MXL parsing and pitch-aware alignment
  helpers used by stage 2.

## Fixed Train/Valid Window Split

To create one shared, fixed `valid` split for all INR experiments, write a
window-level split scheme back into each processed work JSON:

```bash
python src/data_process/create_fixed_window_valid_split.py \
  --config <config.json> \
  --scheme-name train_valid_asap3_nonasap1_v1
```

This annotates each processed work JSON under:

- `meta.window_split_schemes[scheme_name]`

The scheme stores:

- `window_assignments`: every window labeled as `train` or `valid`
- aggregate valid counts for `ASAP` and `non-ASAP`
- the selection seed and target ratios used to build the split

After updating JSONs, rebuild shared INR sidecars so they carry the same fixed
metadata:

```bash
python src/data_process/prebuild_inr_work_pt.py \
  --config <config.json> \
  --split train
```

Training can then consume the fixed split by setting:

```json
{
  "fixed_window_split_scheme": "train_valid_asap3_nonasap1_v1",
  "fixed_window_base_split": "train",
  "fixed_window_train_split_name": "train",
  "fixed_window_eval_split_name": "valid"
}
```

Recommended stage usage:

- base `train`: use all performances with the fixed `valid` windows
- `adapt`: keep `train_performance_dataset = "ASAP"` and `eval_performance_dataset = "ASAP"`
  so the fixed `valid` windows are reused, but evaluation only keeps ASAP performances

## Legacy

`legacy_pt_cpt/` contains old PT/CPT preprocessing scripts for pretrain, Arrow,
and tokenizer-style SFT data. They are kept out of the active pipeline because
the current Integrated Note experiments train directly on PianoCoRe-A paired data
and do not use CPT data.
