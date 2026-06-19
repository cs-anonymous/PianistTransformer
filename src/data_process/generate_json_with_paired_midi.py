import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from miditoolkit import MidiFile
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.utils.inr_midi import RAW_CONTINUOUS_KEYS, midi_to_note_features, sorted_piano_notes


SCHEMA_VERSION = "pianocore_node_work_raw_v2"
JSON_SUFFIX = ".json"


REQUIRED_COLUMNS = [
    "id",
    "split",
    "tier_a",
    "tier_a_star",
    "refined_score_midi_path",
    "refined_performance_midi_path",
    "refined_alignment_path",
]

OPTIONAL_COLUMNS = [
    "composer",
    "composition",
    "movement",
    "performance_id",
    "score_dataset",
    "score_id",
    "performance_dataset",
    "performer",
    "capture_model",
    "refined_score_note_count",
    "refined_score_duration",
    "refined_performance_note_count",
    "refined_performance_interpolated_note_count",
    "refined_performance_duration",
]


def find_refined_dir(pianocore_dir):
    candidates = [
        Path(pianocore_dir) / "PianoCoRe" / "refined",
        Path(pianocore_dir) / "PianoCoRe-1.0" / "refined",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not find PianoCoRe refined directory. Checked: "
        + ", ".join(str(candidate) for candidate in candidates)
    )


def relpath_string(path):
    return str(path).replace(os.sep, "/")


def optional_value(record, key):
    value = record.get(key)
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    if isinstance(value, np.generic):
        return value.item()
    return value


def raw_rows_to_int(rows):
    output = []
    for row in rows:
        output.append(
            [
                max(0, int(round(float(row[0])))),
                max(0, int(round(float(row[1])))),
                min(max(int(round(float(row[2]))), 0), 127),
                *[min(max(int(round(float(value))), 0), 127) for value in row[3:7]],
            ]
        )
    return output


def output_path_for_score(refined_dir, output_dir, score_rel_path):
    relative_path = Path(score_rel_path)
    if output_dir is None:
        return (refined_dir / relative_path).with_suffix(JSON_SUFFIX)
    return (Path(output_dir) / relative_path).with_suffix(JSON_SUFFIX)


def load_metadata(metadata_path):
    header = pd.read_csv(metadata_path, nrows=0).columns.tolist()
    missing = [column for column in REQUIRED_COLUMNS if column not in header]
    if missing:
        raise KeyError(f"Missing required metadata columns: {missing}")

    usecols = [column for column in REQUIRED_COLUMNS + OPTIONAL_COLUMNS if column in header]
    return pd.read_csv(metadata_path, usecols=usecols)


def build_score_payload(score_midi, max_time_ms, float_precision):
    score_notes = sorted_piano_notes(score_midi)
    score_features = midi_to_note_features(
        score_midi,
        notes=score_notes,
        max_time_ms=max_time_ms,
        normalize=False,
    )

    pitch = score_features["pitch"]
    score_raw = score_features["continuous"]
    if len(pitch) != len(score_raw):
        raise ValueError("score_feature_length_mismatch")
    if pitch and (min(pitch) < 0 or max(pitch) > 127):
        raise ValueError("score_pitch_out_of_range")

    return {
        "pitch": pitch,
        "score_raw": raw_rows_to_int(score_raw),
        "note_count": len(pitch),
    }


def aligned_performance_notes(performance_midi, alignment_path, note_count):
    alignment = np.load(alignment_path)
    if "perf_idx" not in alignment or "interpolated" not in alignment:
        raise ValueError("missing_alignment_arrays")

    perf_idx = alignment["perf_idx"].astype(int)
    interpolated = alignment["interpolated"]
    if len(perf_idx) != note_count:
        raise ValueError("alignment_length_mismatch")
    if len(interpolated) != note_count:
        raise ValueError("interpolated_length_mismatch")

    performance_notes_sorted = sorted_piano_notes(performance_midi)
    if len(perf_idx):
        min_idx = int(perf_idx.min())
        max_idx = int(perf_idx.max())
        if min_idx < 0 or max_idx >= len(performance_notes_sorted):
            raise ValueError("perf_idx_out_of_range")

    performance_notes = [performance_notes_sorted[int(index)] for index in perf_idx]
    interpolated_int = [1 if bool(value) else 0 for value in interpolated]
    return performance_notes, interpolated_int


def build_performance_payload(row, refined_dir, score_pitch, max_time_ms, float_precision):
    performance_rel_path = row["refined_performance_midi_path"]
    alignment_rel_path = row["refined_alignment_path"]
    performance_path = refined_dir / performance_rel_path
    alignment_path = refined_dir / alignment_rel_path

    if not performance_path.exists():
        raise FileNotFoundError("missing_performance")
    if not alignment_path.exists():
        raise FileNotFoundError("missing_alignment")

    performance_midi = MidiFile(str(performance_path))
    performance_notes, interpolated = aligned_performance_notes(
        performance_midi,
        alignment_path,
        len(score_pitch),
    )
    performance_features = midi_to_note_features(
        performance_midi,
        notes=performance_notes,
        max_time_ms=max_time_ms,
        normalize=False,
        force_monotonic_starts=True,
    )

    if performance_features["pitch"] != score_pitch:
        raise ValueError("pitch_mismatch")
    if len(performance_features["continuous"]) != len(score_pitch):
        raise ValueError("performance_feature_length_mismatch")

    payload = {
        "id": optional_value(row, "id"),
        "performance_id": optional_value(row, "performance_id"),
        "performance_source": performance_rel_path,
        "alignment_source": alignment_rel_path,
        "split": optional_value(row, "split"),
        "tier_a_star": bool(optional_value(row, "tier_a_star")),
        "label_raw": raw_rows_to_int(performance_features["continuous"]),
        "interpolated": interpolated,
    }

    for key in (
        "performance_dataset",
        "performer",
        "capture_model",
        "refined_performance_note_count",
        "refined_performance_interpolated_note_count",
        "refined_performance_duration",
    ):
        value = optional_value(row, key)
        if value is not None:
            payload[key] = value

    return payload


def build_work_meta(first_row, score_rel_path, performance_count, max_time_ms, float_precision):
    meta = {
        "schema": SCHEMA_VERSION,
        "score_source": score_rel_path,
        "performance_count": performance_count,
        "raw_keys": list(RAW_CONTINUOUS_KEYS),
        "timing_unit": "ms",
        "velocity_range": [0, 127],
        "pedal_range": [0, 127],
        "float_precision": float_precision,
        "time_normalization": "raw_ms",
    }

    for key in (
        "composer",
        "composition",
        "movement",
        "score_dataset",
        "score_id",
        "split",
        "refined_score_note_count",
        "refined_score_duration",
    ):
        value = optional_value(first_row, key)
        if value is not None:
            meta[key] = value
    return meta


def write_work_json(task):
    refined_dir = Path(task["refined_dir"])
    output_dir = Path(task["output_dir"]) if task["output_dir"] else None
    score_rel_path = task["score_rel_path"]
    records = task["records"]
    max_time_ms = task["max_time_ms"]
    float_precision = task["float_precision"]
    overwrite = task["overwrite"]

    output_path = output_path_for_score(refined_dir, output_dir, score_rel_path)
    if output_path.exists() and not overwrite:
        return {
            "status": "skipped",
            "score_source": score_rel_path,
            "output_path": str(output_path),
            "performance_count": len(records),
        }

    tmp_path = output_path.with_name(output_path.name + ".tmp")
    try:
        score_path = refined_dir / score_rel_path
        if not score_path.exists():
            raise FileNotFoundError("missing_score")

        score_midi = MidiFile(str(score_path))
        score_payload = build_score_payload(score_midi, max_time_ms, float_precision)
        score_payload["score_source"] = score_rel_path

        meta = build_work_meta(
            records[0],
            score_rel_path,
            len(records),
            max_time_ms,
            float_precision,
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        success_performances = 0
        failed_performances = []
        total_interpolated = 0

        with open(tmp_path, "w", encoding="utf-8") as file:
            file.write('{"schema":')
            json.dump(SCHEMA_VERSION, file, separators=(",", ":"), ensure_ascii=False)
            file.write(',"meta":')
            json.dump(meta, file, separators=(",", ":"), ensure_ascii=False)
            file.write(',"score":')
            json.dump(score_payload, file, separators=(",", ":"), ensure_ascii=False)
            file.write(',"performances":[')

            first_performance = True
            for row in records:
                try:
                    performance_payload = build_performance_payload(
                        row,
                        refined_dir,
                        score_payload["pitch"],
                        max_time_ms,
                        float_precision,
                    )
                    if not first_performance:
                        file.write(",")
                    json.dump(performance_payload, file, separators=(",", ":"), ensure_ascii=False)
                    first_performance = False
                    success_performances += 1
                    total_interpolated += sum(performance_payload["interpolated"])
                except Exception as exc:
                    failed_performances.append(
                        {
                            "id": optional_value(row, "id"),
                            "performance_source": optional_value(row, "refined_performance_midi_path"),
                            "alignment_source": optional_value(row, "refined_alignment_path"),
                            "reason": type(exc).__name__ + ": " + str(exc),
                        }
                    )

            file.write('],"failed_performances":')
            json.dump(failed_performances, file, separators=(",", ":"), ensure_ascii=False)
            file.write("}\n")

        if success_performances == 0:
            tmp_path.unlink(missing_ok=True)
            return {
                "status": "failed",
                "score_source": score_rel_path,
                "output_path": str(output_path),
                "reason": "no_successful_performances",
                "failed_performances": len(failed_performances),
                "failed_examples": failed_performances[:5],
            }

        os.replace(tmp_path, output_path)
        return {
            "status": "ok",
            "score_source": score_rel_path,
            "output_path": str(output_path),
            "note_count": score_payload["note_count"],
            "success_performances": success_performances,
            "failed_performances": len(failed_performances),
            "total_interpolated": total_interpolated,
            "failed_examples": failed_performances[:5],
        }
    except Exception as exc:
        tmp_path.unlink(missing_ok=True)
        return {
            "status": "failed",
            "score_source": score_rel_path,
            "output_path": str(output_path),
            "reason": type(exc).__name__ + ": " + str(exc),
            "failed_performances": len(records),
            "failed_examples": [],
        }


def make_work_tasks(df, refined_dir, output_dir, args):
    sort_columns = ["refined_score_midi_path", "refined_performance_midi_path", "id"]
    existing_sort_columns = [column for column in sort_columns if column in df.columns]
    df = df.sort_values(existing_sort_columns, kind="stable")

    tasks = []
    for score_rel_path, group in df.groupby("refined_score_midi_path", sort=True):
        if args.limit_performances_per_work is not None:
            group = group.head(args.limit_performances_per_work)
        tasks.append(
            {
                "score_rel_path": score_rel_path,
                "records": group.to_dict("records"),
                "refined_dir": str(refined_dir),
                "output_dir": str(output_dir) if output_dir else "",
                "max_time_ms": args.max_time_ms,
                "float_precision": args.float_precision,
                "overwrite": args.overwrite,
            }
        )

    if args.limit_works is not None:
        tasks = tasks[: args.limit_works]
    return tasks


def update_reason_counts(reason_counts, result):
    if result["status"] == "failed":
        reason = result.get("reason", "unknown")
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
    for failed in result.get("failed_examples", []):
        reason = failed.get("reason", "unknown")
        reason_counts[reason] = reason_counts.get(reason, 0) + 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pianocore-dir", default="data/pianocore")
    parser.add_argument("--metadata", default="metadata.csv")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="If unset, write each JSON beside its refined score MIDI. If set, mirror the refined tree here.",
    )
    parser.add_argument("--summary-path", default=None)
    parser.add_argument("--num-proc", type=int, default=40)
    parser.add_argument("--max-time-ms", type=float, default=10000.0)
    parser.add_argument("--float-precision", type=int, default=5)
    parser.add_argument("--limit-works", type=int, default=None)
    parser.add_argument("--limit-performances-per-work", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    pianocore_dir = Path(args.pianocore_dir)
    metadata_path = pianocore_dir / args.metadata
    refined_dir = find_refined_dir(pianocore_dir)
    output_dir = Path(args.output_dir) if args.output_dir else None

    df = load_metadata(metadata_path)
    print(f"Loaded {len(df)} rows from {metadata_path}")

    df = df[df["tier_a"].fillna(False).astype(bool)]
    df = df[df["refined_score_midi_path"].notna()]
    df = df[df["refined_performance_midi_path"].notna()]
    df = df[df["refined_alignment_path"].notna()]

    tasks = make_work_tasks(df, refined_dir, output_dir, args)
    summary_path = (
        Path(args.summary_path)
        if args.summary_path
        else (output_dir / "summary.json" if output_dir else refined_dir / "pianocore_a_node_summary.json")
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Rows after PianoCoRe-A/refined filter: {len(df)}")
    print(f"Works to process: {len(tasks)}")
    print(f"Refined directory: {refined_dir}")
    if output_dir:
        print(f"Output mode: mirrored tree under {output_dir}")
    else:
        print(f"Output mode: in-place JSON beside each refined score MIDI")
    print(f"Summary path: {summary_path}")

    success_works = 0
    skipped_works = 0
    failed_works = 0
    success_performances = 0
    failed_performances = 0
    total_notes = 0
    total_interpolated = 0
    reason_counts = {}
    failed_examples = []

    with ProcessPoolExecutor(max_workers=args.num_proc) as executor:
        futures = [executor.submit(write_work_json, task) for task in tasks]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing PianoCoRe-A works"):
            result = future.result()
            status = result["status"]
            if status == "ok":
                success_works += 1
                success_performances += int(result.get("success_performances", 0))
                failed_performances += int(result.get("failed_performances", 0))
                total_notes += int(result.get("note_count", 0))
                total_interpolated += int(result.get("total_interpolated", 0))
            elif status == "skipped":
                skipped_works += 1
            else:
                failed_works += 1
                failed_performances += int(result.get("failed_performances", 0))
                if len(failed_examples) < 50:
                    failed_examples.append(result)
            update_reason_counts(reason_counts, result)

    summary = {
        "schema": SCHEMA_VERSION,
        "metadata_path": str(metadata_path),
        "refined_dir": str(refined_dir),
        "output_dir": str(output_dir) if output_dir else None,
        "output_mode": "mirrored_tree" if output_dir else "in_place",
        "summary_path": str(summary_path),
        "num_proc": args.num_proc,
        "max_time_ms": args.max_time_ms,
        "float_precision": args.float_precision,
        "works_total": len(tasks),
        "success_works": success_works,
        "skipped_works": skipped_works,
        "failed_works": failed_works,
        "success_performances": success_performances,
        "failed_performances": failed_performances,
        "total_score_notes_across_successful_works": total_notes,
        "total_interpolated_notes_across_successful_performances": total_interpolated,
        "reason_counts": reason_counts,
        "failed_examples": failed_examples,
    }
    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, sort_keys=True, ensure_ascii=False)

    print("\nProcessing complete:")
    print(f"  - Successful works: {success_works}")
    print(f"  - Skipped works: {skipped_works}")
    print(f"  - Failed works: {failed_works}")
    print(f"  - Successful performances: {success_performances}")
    print(f"  - Failed performances: {failed_performances}")
    print(f"  - Summary: {summary_path}")
    if reason_counts:
        print("  - Failure reasons:")
        for reason, count in sorted(reason_counts.items(), key=lambda item: (-item[1], item[0])):
            print(f"    {reason}: {count}")


if __name__ == "__main__":
    main()
