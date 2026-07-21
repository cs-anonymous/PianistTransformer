#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data_process.generate_json_with_paired_midi import (
    find_refined_dir,
    make_work_tasks,
    write_work_json,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-csv", required=True)
    parser.add_argument("--pianocore-dir", default="../PianoCoRe")
    parser.add_argument("--score-rel-path", required=True)
    parser.add_argument("--output-dir", default="data/ASAP_processed")
    parser.add_argument("--max-time-ms", type=float, default=10000.0)
    parser.add_argument("--float-precision", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    df = pd.read_csv(args.metadata_csv)
    df = df[df["refined_score_midi_path"] == args.score_rel_path].copy()
    if df.empty:
        raise ValueError(f"No rows found for score_rel_path={args.score_rel_path}")
    df = df[df["tier_a"].fillna(False).astype(bool)]
    df = df[df["refined_performance_midi_path"].notna()]
    df = df[df["refined_alignment_path"].notna()]
    if df.empty:
        raise ValueError(f"No usable rows remain for score_rel_path={args.score_rel_path}")

    refined_dir = find_refined_dir(Path(args.pianocore_dir))
    task_args = argparse.Namespace(
        limit_performances_per_work=None,
        limit_works=None,
        max_time_ms=args.max_time_ms,
        float_precision=args.float_precision,
        overwrite=args.overwrite,
    )
    tasks = make_work_tasks(df, refined_dir, Path(args.output_dir), task_args)
    if len(tasks) != 1:
        raise ValueError(f"Expected exactly 1 work task, got {len(tasks)}")
    result = write_work_json(tasks[0])
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
