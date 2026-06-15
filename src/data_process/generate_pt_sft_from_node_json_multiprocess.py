"""
Generate PT (PianistTransformer) SFT data from PianoCoRe node_a.json files (multi-process version).
"""

import argparse
import json
import os
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial

from tqdm import tqdm


def denormalize_time(value, max_time_ms):
    """Convert normalized time (0-1) to ticks (0-4990)."""
    ticks = value * max_time_ms
    return int(round(ticks))


def denormalize_velocity(value):
    """Convert normalized velocity (0-1) to MIDI velocity (0-127)."""
    return int(round(value * 127))


def denormalize_pedal(value):
    """Convert normalized pedal (0-1) to MIDI CC64 value (0-127)."""
    return int(round(value * 127))


def continuous_to_pt_tokens(pitch, ioi, duration, velocity, pedals, config):
    """Convert continuous values to PT's 8-token format."""
    pitch = min(config['valid_id_range'][0][1] - 1, max(config['valid_id_range'][0][0], pitch + config['pitch_start']))
    ioi = min(config['valid_id_range'][1][1] - 1, max(config['valid_id_range'][1][0], ioi + config['timing_start']))
    velocity = min(config['valid_id_range'][2][1] - 1, max(config['valid_id_range'][2][0], velocity + config['velocity_start']))
    duration = min(config['valid_id_range'][3][1] - 1, max(config['valid_id_range'][3][0], duration + config['timing_start']))

    tokens = [pitch, ioi, velocity, duration]

    for i, pedal_val in enumerate(pedals):
        pedal_token = min(config['valid_id_range'][4 + i][1] - 1,
                         max(config['valid_id_range'][4 + i][0], pedal_val + config['pedal_start']))
        tokens.append(pedal_token)

    return tokens


def process_node_json(json_path, max_time_ms, config):
    """Process a single node_a.json file."""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        score = data['score']
        score_pitch = score['pitch']
        score_continuous = score['score_continuous']
        score_source = score.get('score_source', '')
        note_count = score['note_count']

        results = []

        for perf in data['performances']:
            perf_id = perf['performance_id']
            perf_source = perf.get('performance_source', '')
            split = perf.get('split', 'train')
            label_continuous = perf['label_continuous']

            if len(label_continuous) != note_count:
                continue

            x_tokens = []
            label_tokens = []

            for note_idx in range(note_count):
                pitch = score_pitch[note_idx]
                score_ioi_norm, score_dur_norm, score_vel_norm = score_continuous[note_idx]

                score_ioi = denormalize_time(score_ioi_norm, max_time_ms)
                score_dur = denormalize_time(score_dur_norm, max_time_ms)
                score_vel = denormalize_velocity(score_vel_norm)

                x_pedals = [0, 0, 0, 0]
                x_note_tokens = continuous_to_pt_tokens(
                    pitch, score_ioi, score_dur, score_vel, x_pedals, config
                )
                x_tokens.extend(x_note_tokens)

                perf_ioi_norm, perf_dur_norm, perf_vel_norm = label_continuous[note_idx][:3]
                perf_pedals = label_continuous[note_idx][3:7]

                perf_ioi = denormalize_time(perf_ioi_norm, max_time_ms)
                perf_dur = denormalize_time(perf_dur_norm, max_time_ms)
                perf_vel = denormalize_velocity(perf_vel_norm)
                perf_pedals_denorm = [denormalize_pedal(p) for p in perf_pedals]

                label_note_tokens = continuous_to_pt_tokens(
                    pitch, perf_ioi, perf_dur, perf_vel, perf_pedals_denorm, config
                )
                label_tokens.extend(label_note_tokens)

            results.append({
                "x": x_tokens,
                "label": label_tokens,
                "score_source": score_source,
                "performance_source": perf_source,
                "cut": 0,
                "split": split
            })

        return results
    except Exception as e:
        print(f"Error processing {json_path}: {e}")
        return []


def main():
    parser = argparse.ArgumentParser(description="Generate PT SFT data from node_a.json files (multi-process)")
    parser.add_argument("--processed-dir", type=str, default="PianoCoRe/processed")
    parser.add_argument("--output-file", type=str, default="data/processed/sft/sft_pianocore_from_json.jsonl")
    parser.add_argument("--max-time-ms", type=float, default=10000.0)
    parser.add_argument("--num-workers", type=int, default=None, help="Number of worker processes (default: CPU count)")
    args = parser.parse_args()

    # Config
    config = {
        'pitch_start': 5,
        'velocity_start': 133,
        'timing_start': 261,
        'pedal_start': 5261,
        'valid_id_range': [
            (5, 133), (261, 5252), (133, 261), (261, 5261),
            (5261, 5389), (5261, 5389), (5261, 5389), (5261, 5389),
        ]
    }

    processed_dir = Path(args.processed_dir)
    json_files = list(processed_dir.rglob("*.node_a.json"))
    print(f"Found {len(json_files)} node_a.json files")

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    num_workers = args.num_workers or cpu_count()
    print(f"Using {num_workers} worker processes")

    # Process files in parallel
    process_func = partial(process_node_json, max_time_ms=args.max_time_ms, config=config)

    total_performances = 0
    total_notes = 0

    with open(output_path, 'w', encoding='utf-8') as out_f:
        with Pool(processes=num_workers) as pool:
            for results in tqdm(pool.imap(process_func, json_files), total=len(json_files), desc="Processing"):
                for item in results:
                    out_f.write(json.dumps(item) + "\n")
                    total_performances += 1
                    total_notes += len(item['x']) // 8

    print(f"\nProcessing complete:")
    print(f"  - Total performances: {total_performances}")
    print(f"  - Total notes: {total_notes}")
    print(f"  - Output file: {output_path}")


if __name__ == "__main__":
    main()
