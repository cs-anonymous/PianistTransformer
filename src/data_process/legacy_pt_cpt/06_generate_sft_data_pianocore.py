"""
Generate SFT data from PianoCoRe dataset
PianoCoRe contains paired score-performance MIDI files
"""
import json
import os
import random
from pathlib import Path

from miditoolkit import MidiFile
import pandas as pd
from tqdm import tqdm

from src.utils.midi import align_score_and_performance
from src.model.pianoformer import PianoT5GemmaConfig


def process_pianocore_sft():
    """Process PianoCoRe dataset for supervised fine-tuning"""
    config = PianoT5GemmaConfig()

    # PianoCoRe dataset directories
    pianocore_dir = "data/pianocore"
    refined_dir = Path(pianocore_dir) / "PianoCoRe-1.0" / "refined"
    raw_dir = Path(pianocore_dir) / "raw"

    # Check if metadata exists
    metadata_path = os.path.join(pianocore_dir, "metadata_S.csv")
    if not os.path.exists(metadata_path):
        print(f"Error: metadata_S.csv not found at {metadata_path}")
        return

    df = pd.read_csv(metadata_path)
    print(f"Loaded {len(df)} rows from metadata_S.csv")

    # Create output directory
    output_dir = "data/processed/sft"
    os.makedirs(output_dir, exist_ok=True)

    # Create temp directory if needed
    if not os.path.exists("temp"):
        os.makedirs("temp")

    # Initialize output file
    output_file = os.path.join(output_dir, "sft_pianocore.jsonl")
    with open(output_file, "w") as f:
        pass

    # Use existing ASAP-based split logic
    # For PianoCoRe, we use unique score files for train/test split
    scores_set = set()
    for _, row in df.iterrows():
        if pd.notna(row.get('refined_score_midi_path')):
            scores_set.add(row['refined_score_midi_path'])
        elif pd.notna(row.get('score_midi_path')):
            scores_set.add(row['score_midi_path'])

    random.seed(42)
    scores_set = sorted(list(scores_set))
    random.shuffle(scores_set)
    test_set = set(scores_set[:int(0.1 * len(scores_set))])

    print(f"Found {len(scores_set)} unique score files")
    print(f"Test set: {len(test_set)} score files")

    # Process pairs
    data = []
    success_count = 0
    fail_count = 0

    for i, (_, row) in tqdm(enumerate(df.iterrows()), total=len(df), desc="Processing score-performance pairs"):
        try:
            # Get score MIDI path
            score_path = None
            if pd.notna(row.get('refined_score_midi_path')):
                score_path = os.path.join(refined_dir, row['refined_score_midi_path'])
            elif pd.notna(row.get('score_midi_path')):
                score_path = os.path.join(raw_dir, row['score_midi_path'])

            # Get performance MIDI path
            perf_path = None
            if pd.notna(row.get('refined_performance_midi_path')):
                perf_path = os.path.join(refined_dir, row['refined_performance_midi_path'])
            elif pd.notna(row.get('performance_midi_path')):
                perf_path = os.path.join(raw_dir, row['performance_midi_path'])

            if score_path is None or perf_path is None:
                fail_count += 1
                continue

            if not os.path.exists(score_path) or not os.path.exists(perf_path):
                fail_count += 1
                continue

            # Determine split based on score file
            split = "test" if row.get('refined_score_midi_path', row.get('score_midi_path')) in test_set else "train"

            # Load MIDI files
            score_midi_obj = MidiFile(score_path)
            performance_midi_obj = MidiFile(perf_path)

            # Align and generate training data
            xs, labels = align_score_and_performance(config, score_midi_obj, performance_midi_obj)

            for j in range(len(xs)):
                data_item = {
                    "x": xs[j],
                    "label": labels[j],
                    "score_source": os.path.relpath(score_path, pianocore_dir),
                    "performance_source": os.path.relpath(perf_path, pianocore_dir),
                    "cut": j,
                    "split": split
                }
                data.append(data_item)

                # Write to file
                with open(output_file, "a") as f:
                    f.write(json.dumps(data_item) + "\n")

            success_count += 1

        except Exception as e:
            fail_count += 1
            if fail_count <= 10:  # Print first 10 errors
                print(f"Error processing pair ({row.get('score_midi_path', 'unknown')}, {row.get('performance_midi_path', 'unknown')}): {e}")

    print(f"\nProcessing complete:")
    print(f"  - Successfully processed: {success_count} pairs")
    print(f"  - Failed: {fail_count} pairs")
    print(f"  - Total training samples: {len(data)}")
    print(f"  - Output file: {output_file}")


if __name__ == "__main__":
    process_pianocore_sft()
