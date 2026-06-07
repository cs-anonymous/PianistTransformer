export PYTHONPATH="$(pwd)"

python src/train/sft.py --config configs/sft_config_pianocore.json
