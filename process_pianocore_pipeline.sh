#!/bin/bash
# Complete pipeline for processing PianoCoRe dataset
# This script handles both CPT and SFT data generation

set -e  # Exit on error

echo "=== PianoCoRe Data Processing Pipeline ==="
echo ""

# Check if PianoCoRe dataset exists
if [ ! -d "data/pianocore" ]; then
    echo "Error: PianoCoRe dataset not found at data/pianocore"
    echo "Please run: export http_proxy=http://127.0.0.1:7890 && export https_proxy=http://127.0.0.1:7890 && python download_pianocore_v2.py"
    exit 1
fi

# Stage 1: Generate pretrain data from PianoCoRe
echo "Stage 1: Generating pretrain data from PianoCoRe..."
python src/data_process/01_generate_pretrain_data_pianocore.py
echo "✓ Stage 1 complete"
echo ""

# Stage 2: Convert to arrow format (reusing existing script)
echo "Stage 2: Converting pretrain data to arrow format..."
python src/data_process/04_convert_to_arrow.py \
    --input_dir data/processed/pretrain/raw/pianocore \
    --output_dir data/processed/pretrain/arrow/pianocore
echo "✓ Stage 2 complete"
echo ""

# Stage 3: Generate SFT data from PianoCoRe
echo "Stage 3: Generating SFT data from PianoCoRe..."
python src/data_process/06_generate_sft_data_pianocore.py
echo "✓ Stage 3 complete"
echo ""

# Stage 4: Convert SFT data to arrow format
echo "Stage 4: Converting SFT data to arrow format..."
python src/data_process/04_convert_to_arrow.py \
    --input_dir data/processed/sft \
    --output_dir data/processed/sft/arrow \
    --sft_mode
echo "✓ Stage 4 complete"
echo ""

echo "=== Pipeline complete ==="
echo "Pretrain data: data/processed/pretrain/arrow/pianocore"
echo "SFT data: data/processed/sft/arrow"
