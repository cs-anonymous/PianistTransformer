"""
Generate pretrain data from PianoCoRe dataset
Uses PianoCoRe performance MIDI files for self-supervised pre-training
"""
import json
import os
from pathlib import Path

from miditoolkit import MidiFile
from tqdm import tqdm

from src.utils.midi import midi_to_ids
from src.model.pianoformer import PianoT5GemmaConfig


def process_pianocore_pretrain():
    """Process PianoCoRe dataset for pretraining"""
    config = PianoT5GemmaConfig()

    # PianoCoRe dataset directories
    pianocore_dir = "data/pianocore"
    midi_base_dir = Path(pianocore_dir) / "PianoCoRe-1.0" / "refined"

    # Check if metadata exists
    metadata_path = os.path.join(pianocore_dir, "metadata_S.csv")
    if not os.path.exists(metadata_path):
        print(f"Error: metadata_S.csv not found at {metadata_path}")
        return

    import pandas as pd
    df = pd.read_csv(metadata_path)
    print(f"Loaded {len(df)} rows from metadata_S.csv")

    # Create output directories
    output_dir = "data/processed/pretrain/raw/pianocore"
    os.makedirs(output_dir, exist_ok=True)

    output = []
    cnt = 0
    success_count = 0
    fail_count = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing MIDI files"):
        try:
            # Try refined performance MIDI first, then raw
            midi_path = None
            midi_type = "performance"
            if pd.notna(row.get('refined_performance_midi_path')):
                midi_path = os.path.join(midi_base_dir, row['refined_performance_midi_path'])
            elif pd.notna(row.get('performance_midi_path')):
                midi_path = os.path.join(pianocore_dir, "raw", row['performance_midi_path'])

            if midi_path is None or not os.path.exists(midi_path):
                fail_count += 1
                continue

            # Load MIDI file
            midi_obj = MidiFile(midi_path)

            # Convert to token IDs
            ids = midi_to_ids(config, midi_obj)

            # Get relative path for source tracking
            rel_path = os.path.relpath(midi_path, pianocore_dir)

            output.append({
                "input_ids": ids,
                "source": rel_path,
                "composer": row.get('composer', ''),
                "composition": row.get('composition', ''),
                "split": row.get('split', 'train')
            })

            success_count += 1

            # Save in batches of 1000
            if len(output) >= 1000:
                output_file = os.path.join(output_dir, f"{cnt}.jsonl")
                with open(output_file, "w") as f:
                    for item in output:
                        f.write(json.dumps(item) + "\n")
                cnt += 1
                output = []

        except Exception as e:
            fail_count += 1
            if fail_count <= 10:
                print(f"Error processing {row.get('performance_midi_path', 'unknown')}: {e}")

    # Save remaining data
    if output:
        output_file = os.path.join(output_dir, f"{cnt}.jsonl")
        with open(output_file, "w") as f:
            for item in output:
                f.write(json.dumps(item) + "\n")
        cnt += 1

    print(f"\nProcessing complete:")
    print(f"  - Successfully processed: {success_count} files")
    print(f"  - Failed: {fail_count} files")
    print(f"  - Output files: {cnt}")
    print(f"  - Output directory: {output_dir}")


if __name__ == "__main__":
    process_pianocore_pretrain()
