# INSPIRE: Integrated Note-based Score-to-Performance Interpretation and Rendering

This repository contains the implementation materials for **INSPIRE: Integrated
Note-based Score-to-Performance Interpretation and Rendering for Expressive
Piano Performance**.

INSPIRE studies **expressive piano performance rendering (EPR)**: given a
symbolic piano score, the system generates a human-like performance by
predicting note-level timing, duration, velocity, and sustain-pedal behavior.
The central implementation idea is **Integrated Note Representation (INR)**.
Instead of expanding every note into multiple autoregressive MIDI tokens, INR
represents each aligned score note as one Transformer timestep with typed
internal slots.

The repository has been reorganized around the cleaned submission flow. Older
exploratory configs, launchers, diagnostics, logs, temporary files, and generated
results from the cleanup are archived under `backup/repo_cleanup_20260804/`.
The original Pianist Transformer (PT) code path is still kept for comparison and
reuse.

## Key Features

- **Note-level sequence modeling**: one aligned note corresponds to one
  Transformer timestep, reducing sequence length and autoregressive decoding
  cost compared with token-block MIDI decoders.
- **Typed note slots**: pitch, IOI, duration, velocity, pedal, and musical score
  descriptors are encoded with separate slot encoders before fusion.
- **CINR and DINR variants**: continuous INR predicts calibrated distributions
  for numerical controls; discrete INR uses typed value tables with optional
  numerical-coordinate augmentation.
- **Score-relative timing targets**: performance IOI and duration are modeled as
  log-scale deviations from score timing, which stabilizes long-tailed timing
  behavior.
- **Efficient sidecar loading**: prebuilt per-work `.pt` sidecars avoid repeated
  JSON parsing during training and inference.
- **Distributional evaluation**: generated MIDI is evaluated using PN/PP
  Wasserstein metrics over IOI, duration, velocity, and pedal.

## Repository Layout

| Path | Description |
| --- | --- |
| `src/` | Model, preprocessing, inference, and evaluation code. |
| `src/model/integrated_pianoformer.py` | INR/CINR/DINR model implementation and output heads. |
| `src/train/train_inr.py` | Main INR training entry point. |
| `src/inference/infer_inr_testset.py` | Autoregressive INR inference over a test score list. |
| `src/evaluate/evaluate_inr_saved_midis.py` | PN/PP Wasserstein evaluation for saved MIDI predictions. |
| `src/evaluate/summarize_inr_asap_pipeline.py` | Mainline ASAP rollout summary and distribution plots. |
| `src/evaluate/plot_floorlog_timing_stats.py` | Paper figure helper for floor-log timing/deviation statistics. |
| `src/evaluate/plot_sampling_matrix_humanrel.py` | Paper figure helper for sampling-sensitivity human-relative metrics. |
| `src/data_process/` | Dataset construction, MusicXML feature projection, fixed-window split, and sidecar builders. |
| `script/run_inr_epr_pipeline.sh` | Main train -> infer -> summarize pipeline used by the INR queue. |
| `script/build_pianocore_inr_sidecars.sh` | Main preprocessing and sidecar construction script. |
| `script/run_example_inference_eval.sh` | Minimal inference + evaluation command template for an existing checkpoint. |
| `script/run_pt_pipeline.sh` | Original Pianist Transformer pipeline entry point. |
| `configs/inr_epr/` | Mainline INR/EPR configs from the submission queue. |
| `configs/pt_*.json`, `configs/pretrain_*.json`, `configs/sft_*.json` | Retained PT/pretraining/SFT configs. |
| `data/ASAP_processed/` | Compact ASAP train/test subset with refined MIDI, alignments, metadata, and sidecars. |
| `submission/` | Original submission package and media supplement. |
| `backup/repo_cleanup_20260804/` | Files moved out during repository cleanup. |

Design notes:

- `docs/attribute_specific_decoder_inr.md`: new ASD-INR direction with CINR
  and DINR experiment plans.
- `docs/representation_metrics.md`: PN/PP Wasserstein interpretation and
  comparison protocol.
- `docs/inr_slot0710.md`: slot-attribute INR representation design.

## Method Summary

INSPIRE assumes note-aligned score and performance sequences. A score note
contains pitch, score IOI, score duration, score velocity, and musical
descriptors derived from MusicXML. A performance note contains the same pitch
plus realized IOI, duration, velocity, and four sustain-pedal snapshots.

INR uses six outer slots:

| Slot | Contents |
| --- | --- |
| Pitch | MIDI pitch category. |
| IOI | Score or performance inter-onset interval. |
| Duration | Score or performance note duration. |
| Velocity | Score or performance MIDI velocity. |
| Pedal | Four binary pedal states sampled at note-relative positions. |
| Musical | MusicXML-derived measure onset, measure duration, note length, and annotation flags. |

The musical slot follows a compact score descriptor layout:

| Feature | Meaning |
| --- | --- |
| `mo_q` | Measure onset position normalized by 24 quarter-length units. |
| `md_q` | Measure duration normalized by 24 quarter-length units. |
| `ml_q` | Local note length normalized by 24 quarter-length units. |
| `first` | Whether the note starts a local musical group. |
| `hand` | Hand/staff indicator. |
| `trill` | Trill flag. |
| `grace` | Grace-note flag. |
| `staccato` | Staccato flag. |
| `stem_code` | Encoded stem direction. |

Each slot is projected to a typed vector. The vectors are fused into a
768-dimensional note embedding. The encoder processes score-note embeddings
bidirectionally; the decoder uses shifted performance-note embeddings with
causal self-attention and cross-attention to the score encoder states.

## CINR and DINR

**CINR** encodes numerical controls continuously and predicts distribution
parameters for IOI, duration, and velocity. The mainline bounded CINR variants
use discretized logistic output heads. Timing targets are score-relative log
deviations:

```text
delta_ioi = log(max(performance_ioi, 1 ms)) - log(max(score_ioi, 1 ms))
delta_duration = log(max(performance_duration, 1 ms)) - log(max(score_duration, 1 ms))
```

The bounded variants restrict timing support during loss and sampling to reduce
the effect of long-tail alignment artifacts.

**DINR** represents timing and velocity as typed categorical values. It
optionally augments learned value-table entries with numerical coordinates so
neighboring values can share ordinal structure. DINR uses categorical heads for
IOI, duration, and velocity, and binary heads for pedal.

## Dataset Construction

The preprocessing pipeline builds training-ready aligned note data from
score-performance pairs:

1. Normalize score and performance MIDI into refined note sequences.
2. Align performance notes to score notes.
3. Project MusicXML score annotations onto refined score MIDI.
4. Build fixed 512-note training/evaluation windows.
5. Prebuild per-work sidecars for fast loading.

To regenerate processed INR sidecars from PianoCoRe/ASAP-style data:

```bash
bash script/build_pianocore_inr_sidecars.sh
```

The included compact ASAP subset is intended for code execution and data-format
demonstration. Full training data and trained checkpoints are not included.

## Environment

The packaged INR runs and supplement preparation used the following local
environment:

| Item | Value |
| --- | --- |
| Operating system | Linux 6.17.0-35-generic, x86_64, glibc 2.39 |
| GPU | 3 NVIDIA GeForce RTX 3090 GPUs, 24GB each |
| NVIDIA driver | 595.71.05 |
| Python | 3.13.9 |
| PyTorch | 2.12.0+cu130 |
| CUDA runtime | 13.0 |
| cuDNN | 9.2 |

Install dependencies with:

```bash
pip install -r requirements.txt
```

GPU execution is recommended for INR inference and required for practical
training. CPU execution is sufficient for reading the repository contents and
running lightweight smoke checks.

## Smoke Check

```bash
python -c "from pathlib import Path; import pandas as pd; df=pd.read_csv('data/ASAP_processed/metadata.generated_json.csv'); print('rows', len(df)); print(df.groupby('split').size().to_dict()); print('works', df.groupby('split')['refined_score_midi_path'].nunique().to_dict()); print('sidecars', len(list(Path('data/ASAP_processed').rglob('*.ASAP_MUSICAL51.pt'))))"
```

Expected output for the compact subset:

```text
rows 193
{'test': 82, 'train': 111}
works {'test': 19, 'train': 31}
sidecars 50
```

## Running Inference

This repository does not include trained INR checkpoints. Given a trained
checkpoint, run:

```bash
CONFIG=configs/inr_epr/cinr__default_dlm_k1_bounded5.json \
CHECKPOINT=/path/to/checkpoint-best \
OUT_DIR=results/example_inference \
bash script/run_example_inference_eval.sh
```

The script runs:

```bash
python src/inference/infer_inr_testset.py
python src/evaluate/evaluate_inr_saved_midis.py
```

and writes generated MIDI plus PN/PP Wasserstein metrics under `OUT_DIR`.

## Training Pipeline

The mainline INR/EPR configs point to the included metadata and sidecars:

```text
metadata_path = data/ASAP_processed/metadata.generated_json.csv
refined_dir = data/ASAP_processed
prepared_sidecar_tag = ASAP_MUSICAL51
```

For a short training smoke run on the included subset:

```bash
CONFIG=configs/inr_epr/cinr__default_dlm_k1_bounded5.json \
CUDA_VISIBLE_DEVICES=0 \
GLOBAL_BATCH_SIZE=32 \
BASE_NUM_TRAIN_EPOCHS=1 \
ADAPT_NUM_TRAIN_EPOCHS=0 \
bash script/run_inr_epr_pipeline.sh
```

The main full training/inference/summarization pipeline is:

```bash
CONFIG=configs/inr_epr/cinr__default_dlm_k1_bounded5.json \
CUDA_VISIBLE_DEVICES=0,1,2 \
GLOBAL_BATCH_SIZE=96 \
bash script/run_inr_epr_pipeline.sh
```

## Mainline Configurations

The current mainline INR/EPR queue lives in:

```text
configs/inr_epr/
```

The queue contains CINR variants, DINR variants, and representation/feature
ablations. The canonical baseline is:

```text
configs/inr_epr/cinr__default_dlm_k1_bounded5.json
```

The compact result table from the submission package is available at:

```text
submission/Code/results/mainline_rollout_metrics_26_configs.csv
```

## Evaluation Metrics

The code evaluates generated MIDI as distributions, because EPR is one-to-many:
the same score can have many plausible human performances.

- **PP Wasserstein** compares piece-level predicted and human distributions
  after pooling note values within each piece.
- **PN Wasserstein** compares generated and human values per aligned score note,
  then averages over notes and pieces.
- Metrics are reported for IOI, duration, velocity, and pedal.

Lower Wasserstein distance is better. Human-relative aggregates used in the
paper normalize these distances by leave-one-out human variation.

## Original Pianist Transformer Path

The original PT implementation is retained for comparison and compatibility:

```bash
python src/train/pretrain.py
python src/train/sft.py --config configs/sft_config_pianocore.json
python src/inference/inference.py
```

Related configs remain at the top level of `configs/`, and the PT shell pipeline
is kept as:

```bash
bash script/run_pt_pipeline.sh
```

## Cleanup Archive

The cleanup archive is:

```text
backup/repo_cleanup_20260804/
```

It contains older exploratory configs, previous launch scripts, diagnostics,
plotting helpers, logs, temporary files, and generated results. Core INR and PT
implementation files were kept in place.
