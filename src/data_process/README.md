# Data Process Pipeline

`src/data_process` now exposes one canonical INR preprocessing chain:

`PianoCoRe metadata + refined MIDI + alignments -> paired work JSON -> XML score features -> fixed valid split -> prebuilt .pt/.ASAP.pt`

The training side is expected to stay read-only. All split metadata and all
prepared sidecars should be generated here before training starts.

## One-Command Entry Point

Use the shell pipeline below to build the current standard dataset:

```bash
bash script/build_pianocore_inr_sidecars.sh
```

By default it runs:

1. `generate_json_with_paired_midi.py`
2. `update_json_score_feature_with_xml.py`
3. `create_fixed_window_valid_split.py`
4. `prebuild_inr_work_pt.py --sidecar-tag NONE`
5. `prebuild_inr_work_pt.py --performance-dataset ASAP --sidecar-tag ASAP`

Default fixed split scheme:

- `train_valid_asap3_nonasap05_v1`
- ASAP valid target: `3%`
- non-ASAP valid target: `0.5%`

Important environment variables accepted by the shell entrypoint:

- `PIANOCORE_DIR`: source dataset directory, default `data/ASAP_processed`
- `PROCESSED_DIR`: output processed root, default `data/ASAP_processed`
- `RAW_MIDI_ZIP`: raw PianoCoRe zip used for XML/MXL lookup
- `WORKERS`: worker count shared by all stages, default `36`

After this pipeline finishes, training can read:

- base stage: `*.pt`
- ASAP adapt stage: `*.ASAP.pt`

No training-time sidecar writes should be necessary.

## Stage Details

### 1. Generate JSON With Paired MIDI

```bash
python src/data_process/generate_json_with_paired_midi.py \
  --pianocore-dir data/pianocore \
  --overwrite
```

This stage writes one work-level JSON beside each refined score MIDI, or under
`--output-dir` if a mirrored tree is requested.

Latest required raw fields include:

- `score.pitch`
- `score.score_raw`
- `performances[].label_shared_raw`
- `performances[].label_pedal4_raw`
- `performances[].interpolated`

### 2. Update JSON Score Feature With XML

```bash
python src/data_process/update_json_score_feature_with_xml.py \
  --pianocore-dir data/pianocore
```

This stage projects XML/MXL score annotations onto refined score notes and
updates each JSON in place. The output adds:

- `score.score_feature`
- `score.has_score_feature`
- `meta.xml_to_refined_score_alignment`

### 3. Create Shared Fixed Train/Valid Split

```bash
python src/data_process/create_fixed_window_valid_split.py \
  --metadata-path <metadata.csv> \
  --refined-dir <processed_dir> \
  --scheme-name train_valid_asap3_nonasap05_v1 \
  --asap-ratio 0.03 \
  --non-asap-ratio 0.005 \
  --skip-sidecars
```

This writes a fixed window-level split scheme into each processed work JSON:

- `meta.window_split_schemes[scheme_name]`

The canonical flow runs this step before sidecar prebuild, so `--skip-sidecars`
is intentional.

### 4. Prebuild Training Sidecars

Build the base all-performance sidecars:

```bash
python src/data_process/prebuild_inr_work_pt.py \
  --metadata-path <metadata.csv> \
  --refined-dir <processed_dir> \
  --split train \
  --sidecar-tag NONE
```

Build the ASAP-only adapt sidecars:

```bash
python src/data_process/prebuild_inr_work_pt.py \
  --metadata-path <metadata.csv> \
  --refined-dir <processed_dir> \
  --split train \
  --performance-dataset ASAP \
  --sidecar-tag ASAP
```

These sidecars contain the latest training-required payload, including:

- raw score timing rows
- score feature rows and masks
- raw per-performance labels
- fixed window split metadata copied from JSON

## Training Config Expectations

To consume the shared fixed valid set, configs should use:

```json
{
  "fixed_window_split_scheme": "train_valid_asap3_nonasap05_v1",
  "fixed_window_base_split": "train",
  "fixed_window_train_split_name": "train",
  "fixed_window_eval_split_name": "valid"
}
```

Recommended usage:

- base `train`: all performances, fixed `valid` windows
- `adapt`: `train_performance_dataset = "ASAP"` and `eval_performance_dataset = "ASAP"`

## Still Kept As Utilities

- `rebuild_single_processed_work.py`: rebuild one work JSON if a single score needs repair
- `score_xml_alignment.py`: shared XML/MXL alignment helpers
- `fixed_window_split.py`: fixed split lookup helpers used by training and sidecar prebuild

## Legacy

`legacy_pt_cpt/` contains old PT/CPT preprocessing scripts and is not part of
the active INR pipeline.
