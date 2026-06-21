#!/usr/bin/env python3
"""Update PianoCoRe-A INR JSON files with XML-derived score features.

The existing ``*.json`` files contain INR note objects with score pitch, score continuous
features, and performance targets. This script adds the v2 score-side fields:

    score.score_feature       [mo, md, ml, first, staff, trill, grace, staccato]
    score.has_score_feature   1 if the refined score note maps to an XML note

The first three score feature values are stored in raw quarter-length units on
the 1/24 grid, not normalized to [0, 1].

Unmapped notes are kept in the sequence with zero score_feature rows and
has_score_feature=0.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data_process import score_xml_alignment


SCHEMA_VERSION = "pianocore_integrated_node_work_v2"
JSON_SUFFIX = ".json"
FEATURE_WIDTH = 8

WORKER_RAW_ZIP = None
WORKER_ARGS: argparse.Namespace | None = None


def find_refined_dir(pianocore_dir: Path) -> Path:
    candidates = [
        pianocore_dir / "PianoCoRe" / "refined",
        pianocore_dir / "PianoCoRe-1.0" / "refined",
        pianocore_dir / "refined",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not find PianoCoRe refined directory")


def node_json_path(json_dir: Path, refined_score_midi_path: str) -> Path:
    return (json_dir / refined_score_midi_path).with_suffix(JSON_SUFFIX)


def round_rows(rows: list[list[float]], precision: int | None) -> list[list[float]]:
    if precision is None:
        return rows
    return [[round(float(value), precision) for value in row] for row in rows]


def clean_json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: clean_json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean_json_value(item) for item in value]
    if isinstance(value, tuple):
        return [clean_json_value(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def load_score_rows(metadata_path: Path, subset: str) -> pd.DataFrame:
    df = pd.read_csv(metadata_path)
    if subset == "a":
        df = df[df["tier_a"].fillna(False).astype(bool)]
    elif subset == "a_star":
        df = df[df["tier_a_star"].fillna(False).astype(bool)]
    df = df[df["score_xml_path"].notna()]
    df = df[df["score_midi_path"].notna()]
    df = df[df["refined_score_midi_path"].notna()]
    if "is_refined" in df.columns:
        df = df[df["is_refined"].fillna(False).astype(bool)]
    keep = [
        "composer",
        "composition",
        "movement",
        "score_dataset",
        "score_id",
        "score_xml_path",
        "score_midi_path",
        "score_note_count",
        "refined_score_midi_path",
        "refined_score_note_count",
    ]
    keep = [column for column in keep if column in df.columns]
    return df[keep].drop_duplicates("refined_score_midi_path").reset_index(drop=True)


def init_worker(raw_midi_zip: str, args_dict: dict[str, Any]) -> None:
    global WORKER_RAW_ZIP, WORKER_ARGS
    warnings.filterwarnings("ignore", category=UserWarning, module="music21")
    warnings.filterwarnings("ignore", message="Could not import.*", module="music21")
    WORKER_RAW_ZIP = score_xml_alignment.ZipResolver(Path(raw_midi_zip))
    WORKER_ARGS = argparse.Namespace(**args_dict)


def compose_refined_to_xml(
    raw_pitches: list[int],
    refined_pitches: list[int],
    xml_pitches: list[int],
    args: argparse.Namespace,
) -> tuple[str, str, dict[int, int]]:
    raw_refined_relation, refined_to_raw = score_xml_alignment.build_raw_to_refined_map(
        raw_pitches,
        refined_pitches,
        args.max_sequence_matcher_notes,
        args.timeout_sec,
        not args.disable_sequence_matcher,
    )
    xml_raw_relation, raw_to_xml = score_xml_alignment.build_raw_to_xml_map(
        raw_pitches,
        xml_pitches,
        args.max_sequence_matcher_notes,
        args.timeout_sec,
        not args.disable_sequence_matcher,
    )
    refined_to_xml = {
        refined_idx: raw_to_xml[raw_idx]
        for refined_idx, raw_idx in refined_to_raw.items()
        if raw_idx in raw_to_xml
    }
    return raw_refined_relation, xml_raw_relation, refined_to_xml


def build_score_features(
    score_xml_path: str,
    score_midi_path: str,
    refined_pitches: list[int],
    args: argparse.Namespace,
) -> dict[str, Any]:
    if WORKER_RAW_ZIP is None:
        raise RuntimeError("raw zip resolver was not initialized")
    xml_features = score_xml_alignment.parse_xml_features(WORKER_RAW_ZIP, score_xml_path, args.timeout_sec)
    raw_pitches = score_xml_alignment.load_midi_pitches_from_zip(WORKER_RAW_ZIP, score_midi_path)
    raw_refined_relation, xml_raw_relation, refined_to_xml = compose_refined_to_xml(
        raw_pitches,
        refined_pitches,
        xml_features["pitch"],
        args,
    )

    xml_score_features = [
        list(structure) + list(annotation)
        for structure, annotation in zip(xml_features["score_structure"], xml_features["score_annotation"])
    ]
    note_count = len(refined_pitches)
    score_feature = [[0.0] * FEATURE_WIDTH for _ in range(note_count)]
    has_score_feature = [0] * note_count
    for refined_idx, xml_idx in refined_to_xml.items():
        if 0 <= refined_idx < note_count and 0 <= xml_idx < len(xml_score_features):
            score_feature[refined_idx] = xml_score_features[xml_idx]
            has_score_feature[refined_idx] = 1

    matched = int(sum(has_score_feature))
    return {
        "status": "ok" if matched == note_count else "partial" if matched else "failed",
        "raw_refined_relation": raw_refined_relation,
        "xml_raw_relation": xml_raw_relation,
        "xml_note_count": len(xml_features["pitch"]),
        "raw_note_count": len(raw_pitches),
        "refined_note_count": note_count,
        "matched": matched,
        "unmatched": note_count - matched,
        "coverage": float(matched / note_count) if note_count else 1.0,
        "score_feature": round_rows(score_feature, args.float_precision),
        "has_score_feature": has_score_feature,
        "unknown_staff_count": xml_features["unknown_staff_count"],
        "trill_count": xml_features["trill_count"],
        "grace_count": xml_features["grace_count"],
        "staccato_count": xml_features["staccato_count"],
    }


def normalize_existing_score_continuous(rows: list[list[float]], precision: int | None) -> list[list[float]]:
    normalized = []
    for row in rows:
        if len(row) < 3:
            raise ValueError("score_continuous_has_fewer_than_3_dims")
        normalized.append([float(row[0]), float(row[1]), float(row[2])])
    return round_rows(normalized, precision)


def update_one(task: dict[str, Any]) -> dict[str, Any]:
    if WORKER_ARGS is None:
        raise RuntimeError("worker args were not initialized")
    args = WORKER_ARGS
    started = time.time()
    json_dir = Path(task["json_dir"])
    refined_score_midi_path = task["refined_score_midi_path"]
    json_path = node_json_path(json_dir, refined_score_midi_path)
    result = {
        "refined_score_midi_path": refined_score_midi_path,
        "score_xml_path": task["score_xml_path"],
        "score_midi_path": task["score_midi_path"],
        "json_path": str(json_path),
    }

    if not json_path.exists():
        result.update({"status": "error", "error": "missing_node_json"})
        return result

    tmp_path = json_path.with_name(json_path.name + ".tmp")
    try:
        with json_path.open(encoding="utf-8") as file:
            payload = json.load(file)

        score = payload["score"]
        pitch = [int(value) for value in score["pitch"]]
        old_score_continuous_dim = len(score["score_continuous"][0]) if score.get("score_continuous") else 0
        if "score_continuous" in score:
            score["score_continuous"] = normalize_existing_score_continuous(
                score["score_continuous"],
                args.float_precision,
            )

        try:
            feature_payload = build_score_features(
                task["score_xml_path"],
                task["score_midi_path"],
                pitch,
                args,
            )
        except Exception as exc:  # noqa: BLE001
            feature_payload = {
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "raw_refined_relation": None,
                "xml_raw_relation": None,
                "xml_note_count": None,
                "raw_note_count": None,
                "refined_note_count": len(pitch),
                "matched": 0,
                "unmatched": len(pitch),
                "coverage": 0.0,
                "score_feature": [[0.0] * FEATURE_WIDTH for _ in pitch],
                "has_score_feature": [0] * len(pitch),
                "unknown_staff_count": 0,
                "trill_count": 0,
                "grace_count": 0,
                "staccato_count": 0,
            }

        score["score_feature"] = feature_payload.pop("score_feature")
        score["has_score_feature"] = feature_payload.pop("has_score_feature")
        score["note_count"] = len(pitch)

        meta = payload.setdefault("meta", {})
        meta["schema"] = SCHEMA_VERSION
        meta["score_xml_source"] = task["score_xml_path"]
        meta["score_midi_source"] = task["score_midi_path"]
        meta["old_score_continuous_dim"] = old_score_continuous_dim
        if "score_raw" in score:
            meta["score_raw_keys"] = ["ioi_ms", "duration_ms", "velocity", "pedal_0", "pedal_25", "pedal_50", "pedal_75"]
        if "score_continuous" in score:
            meta["score_continuous_keys"] = ["ioi", "duration", "velocity"]
        meta["score_feature_keys"] = ["mo", "md", "ml", "first", "staff", "trill", "grace", "staccato"]
        meta["score_feature_unit"] = "quarter_length_raw_grid_1/24"
        meta["note_type_keys"] = ["has_score_feature", "has_pedal_feature"]
        meta["xml_to_refined_score_alignment"] = {
            "method": "midi2scoretransformer_parse_mxl + pitch_aware_monotonic_alignment",
            **feature_payload,
        }
        payload["schema"] = SCHEMA_VERSION

        if not args.dry_run:
            with tmp_path.open("w", encoding="utf-8") as file:
                json.dump(clean_json_value(payload), file, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
                file.write("\n")
            os.replace(tmp_path, json_path)

        matched = int(meta["xml_to_refined_score_alignment"]["matched"])
        note_count = len(pitch)
        result.update(
            {
                "status": meta["xml_to_refined_score_alignment"]["status"],
                "note_count": note_count,
                "matched": matched,
                "unmatched": note_count - matched,
                "coverage": float(matched / note_count) if note_count else 1.0,
                "raw_refined_relation": meta["xml_to_refined_score_alignment"]["raw_refined_relation"],
                "xml_raw_relation": meta["xml_to_refined_score_alignment"]["xml_raw_relation"],
                "old_score_continuous_dim": old_score_continuous_dim,
                "elapsed_sec": round(time.time() - started, 3),
            }
        )
        if "error" in meta["xml_to_refined_score_alignment"]:
            result["error"] = meta["xml_to_refined_score_alignment"]["error"]
        return result
    except Exception as exc:  # noqa: BLE001
        if not args.dry_run:
            tmp_path.unlink(missing_ok=True)
        result.update({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
        return result


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    total_notes = sum(int(item.get("note_count") or 0) for item in results)
    matched_notes = sum(int(item.get("matched") or 0) for item in results)
    return {
        "total_scores": len(results),
        "status_counts": dict(Counter(item.get("status", "unknown") for item in results)),
        "raw_refined_relation_counts": dict(Counter(str(item.get("raw_refined_relation")) for item in results)),
        "xml_raw_relation_counts": dict(Counter(str(item.get("xml_raw_relation")) for item in results)),
        "total_refined_notes": total_notes,
        "matched_score_feature_notes": matched_notes,
        "unmatched_score_feature_notes": total_notes - matched_notes,
        "note_level_score_feature_coverage": float(matched_notes / total_notes) if total_notes else 0.0,
        "full_coverage_scores": sum(1 for item in results if item.get("note_count") and item.get("matched") == item.get("note_count")),
        "partial_coverage_scores": sum(
            1 for item in results if item.get("note_count") and 0 < int(item.get("matched") or 0) < int(item["note_count"])
        ),
        "zero_coverage_scores": sum(1 for item in results if item.get("note_count") and int(item.get("matched") or 0) == 0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--pianocore-dir", type=Path, default=Path("data/pianocore"))
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--raw-midi-zip", type=Path, default=None)
    parser.add_argument(
        "--json-dir",
        type=Path,
        default=None,
        help="Root directory containing mirrored *.json INR files. Defaults to the refined directory.",
    )
    parser.add_argument("--subset", choices=["a", "a_star", "all"], default="a")
    parser.add_argument("--num-proc", type=int, default=20)
    parser.add_argument("--limit-works", type=int, default=None)
    parser.add_argument("--timeout-sec", type=float, default=180.0)
    parser.add_argument("--max-sequence-matcher-notes", type=int, default=13000)
    parser.add_argument("--disable-sequence-matcher", action="store_true")
    parser.add_argument("--float-precision", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=Path("data/pianocore/PianoCoRe/refined/processed_score_feature_update_summary.json"),
    )
    parser.add_argument(
        "--details-path",
        type=Path,
        default=Path("data/pianocore/PianoCoRe/refined/processed_score_feature_update_details.jsonl"),
    )
    args = parser.parse_args()

    metadata_path = args.metadata or (args.pianocore_dir / "metadata.csv")
    raw_midi_zip = args.raw_midi_zip or (args.pianocore_dir / "PianoCoRe-1.0-raw-midi.zip")
    refined_dir = find_refined_dir(args.pianocore_dir)
    json_dir = args.json_dir or refined_dir

    rows = load_score_rows(metadata_path, args.subset)
    if args.limit_works is not None:
        rows = rows.head(args.limit_works).reset_index(drop=True)

    tasks = []
    for row in rows.to_dict("records"):
        tasks.append(
            {
                "json_dir": str(json_dir),
                "score_xml_path": row["score_xml_path"],
                "score_midi_path": row["score_midi_path"],
                "refined_score_midi_path": row["refined_score_midi_path"],
            }
        )

    args.summary_path.parent.mkdir(parents=True, exist_ok=True)
    args.details_path.parent.mkdir(parents=True, exist_ok=True)
    args_dict = vars(args).copy()
    args_dict["pianocore_dir"] = str(args.pianocore_dir)
    args_dict["metadata"] = str(metadata_path)
    args_dict["raw_midi_zip"] = str(raw_midi_zip)
    args_dict["summary_path"] = str(args.summary_path)
    args_dict["details_path"] = str(args.details_path)
    args_dict["json_dir"] = str(json_dir)

    results = []
    with args.details_path.open("w", encoding="utf-8") as details_file:
        if args.num_proc > 1:
            with ProcessPoolExecutor(
                max_workers=args.num_proc,
                initializer=init_worker,
                initargs=(str(raw_midi_zip), args_dict),
            ) as executor:
                futures = [executor.submit(update_one, task) for task in tasks]
                for future in tqdm(as_completed(futures), total=len(futures), desc="Updating score features"):
                    result = future.result()
                    results.append(result)
                    details_file.write(json.dumps(clean_json_value(result), ensure_ascii=False, allow_nan=False) + "\n")
                    details_file.flush()
        else:
            init_worker(str(raw_midi_zip), args_dict)
            for task in tqdm(tasks, desc="Updating score features"):
                result = update_one(task)
                results.append(result)
                details_file.write(json.dumps(clean_json_value(result), ensure_ascii=False, allow_nan=False) + "\n")
                details_file.flush()

    summary = summarize(results)
    summary["dry_run"] = args.dry_run
    summary["schema"] = SCHEMA_VERSION
    summary["metadata"] = str(metadata_path)
    summary["raw_midi_zip"] = str(raw_midi_zip)
    summary["refined_dir"] = str(refined_dir)
    summary["json_dir"] = str(json_dir)
    summary["num_proc"] = args.num_proc
    summary["timeout_sec"] = args.timeout_sec
    summary["max_sequence_matcher_notes"] = args.max_sequence_matcher_notes
    with args.summary_path.open("w", encoding="utf-8") as summary_file:
        json.dump(clean_json_value(summary), summary_file, ensure_ascii=False, indent=2, allow_nan=False)
        summary_file.write("\n")
    print(json.dumps(clean_json_value(summary), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
