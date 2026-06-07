"""
Split test set into ASAP and PianoCoRe-only subsets for separate evaluation.
"""

import argparse
import json
import sys
from pathlib import Path
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata', type=str, required=True, help='PianoCoRe metadata CSV')
    parser.add_argument('--output', type=str, default='results/test_split_info.json', help='Output JSON')
    args = parser.parse_args()

    # Load metadata
    df = pd.read_csv(args.metadata)

    # Filter test set
    test_df = df[df['split'] == 'test'].copy()

    print(f"Total test samples: {len(test_df)}")

    # Identify ASAP samples
    # ASAP sources: performance_dataset contains 'ASAP' or score_dataset contains 'ASAP'
    asap_mask = (
        test_df['performance_dataset'].str.contains('ASAP', na=False, case=False) |
        test_df['score_dataset'].str.contains('ASAP', na=False, case=False)
    )

    asap_df = test_df[asap_mask]
    pianocore_only_df = test_df[~asap_mask]

    print(f"\nASAP subset: {len(asap_df)} samples")
    print(f"PianoCoRe-only subset: {len(pianocore_only_df)} samples")

    # Get unique works
    asap_works = asap_df['id'].unique()
    pianocore_works = pianocore_only_df['id'].unique()

    print(f"\nASAP unique works: {len(asap_works)}")
    print(f"PianoCoRe-only unique works: {len(pianocore_works)}")

    # Save split info
    split_info = {
        'total_test_samples': int(len(test_df)),
        'asap_samples': int(len(asap_df)),
        'pianocore_only_samples': int(len(pianocore_only_df)),
        'asap_works': asap_works.tolist(),
        'pianocore_only_works': pianocore_works.tolist(),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(split_info, f, indent=2)

    print(f"\nSplit info saved to {output_path}")


if __name__ == '__main__':
    main()
