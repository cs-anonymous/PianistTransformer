# Script Directory Guide

`script/` keeps the current runnable shell entry points. Batch launchers,
diagnostics, one-off plotting scripts, and older experiment helpers were moved
to `backup/repo_cleanup_20260804/scripts/`.

## INR / EPR

Preprocess PianoCoRe/ASAP-style aligned data into INR JSON and sidecars:

```bash
bash script/build_pianocore_inr_sidecars.sh
```

Run the main score-to-performance training, inference, and evaluation pipeline:

```bash
CONFIG=configs/inr_epr/cinr__default_dlm_k1_bounded5.json \
CUDA_VISIBLE_DEVICES=0,1,2 \
GLOBAL_BATCH_SIZE=96 \
bash script/run_inr_epr_pipeline.sh
```

Run inference and PN/PP evaluation from an existing checkpoint:

```bash
CONFIG=configs/inr_epr/cinr__default_dlm_k1_bounded5.json \
CHECKPOINT=/path/to/checkpoint-best \
OUT_DIR=results/example_inference \
bash script/run_example_inference_eval.sh
```

## Pianist Transformer

The PT flow is still kept:

```bash
bash script/run_pt_pipeline.sh
```

Quick official-checkpoint evaluation helpers are also retained:

```bash
bash script/run_pt_official_cheap15_2gpu.sh
bash script/run_pt_official_redownload_cheap15.sh
```

## CINR serial-attribute ablation

The experimental `attribute_4n` decoder predicts IOI, duration, velocity, and
pedal in four autoregressive steps per note while retaining CINR's DLM output
parameterization. It is currently implemented for the T5 CINR DLM path.
Because the decoder sequence is four times longer, the launcher defaults to a
per-device microbatch of 4 and an effective global batch of 64:

```bash
CUDA_VISIBLE_DEVICES=0 bash script/run_cinr_serial_attribute_4n.sh
```

The matching note-level configuration remains
`configs/inr_epr/cinr__default_dlm_k1_bounded5.json`.
