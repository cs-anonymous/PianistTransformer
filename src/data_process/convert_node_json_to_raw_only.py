import argparse
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.utils.inr_midi import RAW_CONTINUOUS_KEYS, old_continuous_rows_to_raw


SCHEMA_VERSION = "pianocore_node_work_raw_v2"
SCORE_GRID = 1.0 / 24.0


def quantize_score_grid(value):
    return round(float(value) / SCORE_GRID) * SCORE_GRID


def score_feature_to_raw_grid(rows, already_raw_grid=False):
    output = []
    for row in rows or []:
        values = list(row)
        if len(values) < 8:
            values.extend([0.0] * (8 - len(values)))
        if already_raw_grid:
            mo = quantize_score_grid(min(max(float(values[0]), 0.0), 6.0))
            md = quantize_score_grid(min(max(float(values[1]), 0.0), 4.0))
            ml = quantize_score_grid(min(max(float(values[2]), 0.0), 6.0))
        else:
            mo = quantize_score_grid(min(max(float(values[0]), 0.0), 1.0) * 6.0)
            md = quantize_score_grid(min(max(float(values[1]), 0.0), 1.0) * 4.0)
            ml = quantize_score_grid(min(max(float(values[2]), 0.0), 1.0) * 6.0)
        output.append(
            [
                mo,
                md,
                ml,
                int(round(float(values[3]))),
                int(round(float(values[4]))),
                int(round(float(values[5]))),
                int(round(float(values[6]))),
                int(round(float(values[7]))),
            ]
        )
    return output


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


def convert_one(task):
    src_path = Path(task["src_path"])
    dst_path = Path(task["dst_path"])
    overwrite = task["overwrite"]
    precision = task["float_precision"]
    max_time_ms = task["max_time_ms"]

    if dst_path.exists() and not overwrite:
        return {"status": "skipped", "src": str(src_path), "dst": str(dst_path)}

    try:
        with open(src_path, "r", encoding="utf-8") as file:
            work = json.load(file)

        meta = dict(work.get("meta", {}))
        source_max_time_ms = float(meta.get("max_time_ms", max_time_ms))
        score = dict(work["score"])
        if "score_raw" not in score:
            if "score_continuous" not in score:
                raise KeyError("missing_score_continuous_or_score_raw")
            score["score_raw"] = raw_rows_to_int(
                old_continuous_rows_to_raw(score["score_continuous"], max_time_ms=source_max_time_ms),
            )
        else:
            score["score_raw"] = raw_rows_to_int(score["score_raw"])
        score.pop("score_continuous", None)
        if "score_feature" in score:
            score["score_feature"] = score_feature_to_raw_grid(
                score["score_feature"],
                already_raw_grid=meta.get("score_feature_unit") == "quarter_length_raw_grid_1/24",
            )

        performances = []
        for perf in work.get("performances", []):
            perf = dict(perf)
            if "label_raw" not in perf:
                if "label_continuous" not in perf:
                    raise KeyError("missing_label_continuous_or_label_raw")
                perf["label_raw"] = raw_rows_to_int(
                    old_continuous_rows_to_raw(perf["label_continuous"], max_time_ms=source_max_time_ms),
                )
            else:
                perf["label_raw"] = raw_rows_to_int(perf["label_raw"])
            perf.pop("label_continuous", None)
            performances.append(perf)

        meta["schema"] = SCHEMA_VERSION
        meta["raw_keys"] = list(RAW_CONTINUOUS_KEYS)
        meta["timing_unit"] = "ms"
        meta["velocity_range"] = [0, 127]
        meta["pedal_range"] = [0, 127]
        meta["time_normalization"] = "raw_ms"
        meta["score_feature_unit"] = "quarter_length_raw_grid_1/24"
        meta.pop("continuous_keys", None)
        meta.pop("max_time_ms", None)

        converted = {
            "schema": SCHEMA_VERSION,
            "meta": meta,
            "score": score,
            "performances": performances,
            "failed_performances": work.get("failed_performances", []),
        }

        dst_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = dst_path.with_name(dst_path.name + ".tmp")
        with open(tmp_path, "w", encoding="utf-8") as file:
            json.dump(converted, file, separators=(",", ":"), ensure_ascii=False)
            file.write("\n")
        os.replace(tmp_path, dst_path)
        return {
            "status": "ok",
            "src": str(src_path),
            "dst": str(dst_path),
            "performances": len(performances),
            "notes": int(score.get("note_count", len(score.get("pitch", [])))),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "failed",
            "src": str(src_path),
            "dst": str(dst_path),
            "reason": type(exc).__name__ + ": " + str(exc),
        }


def main():
    parser = argparse.ArgumentParser(description="Convert INR node JSON files to raw-only timing/value payloads.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-proc", type=int, default=30)
    parser.add_argument("--max-time-ms", type=float, default=10000.0)
    parser.add_argument("--float-precision", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--summary-path", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    src_paths = sorted(args.input_dir.rglob("*.node_a.json"))
    if args.limit is not None:
        src_paths = src_paths[: args.limit]
    tasks = []
    for src_path in src_paths:
        rel = src_path.relative_to(args.input_dir)
        tasks.append(
            {
                "src_path": str(src_path),
                "dst_path": str(args.output_dir / rel),
                "overwrite": args.overwrite,
                "float_precision": args.float_precision,
                "max_time_ms": args.max_time_ms,
            }
        )

    summary = {
        "schema": SCHEMA_VERSION,
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "num_proc": args.num_proc,
        "total": len(tasks),
        "ok": 0,
        "skipped": 0,
        "failed": 0,
        "performances": 0,
        "notes": 0,
        "failed_examples": [],
    }

    with ProcessPoolExecutor(max_workers=args.num_proc) as executor:
        futures = [executor.submit(convert_one, task) for task in tasks]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Converting raw-only node JSON"):
            result = future.result()
            status = result["status"]
            summary[status] += 1
            if status == "ok":
                summary["performances"] += int(result.get("performances", 0))
                summary["notes"] += int(result.get("notes", 0))
            elif status == "failed" and len(summary["failed_examples"]) < 50:
                summary["failed_examples"].append(result)

    summary_path = args.summary_path or (args.output_dir / "pianocore_a_node_raw_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
