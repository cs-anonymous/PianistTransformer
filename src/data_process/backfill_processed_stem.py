#!/usr/bin/env python3
"""Backfill XML stem direction into PianoCoRe processed INR JSON files."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
import time
import zipfile
import xml.etree.ElementTree as ET
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


SCHEMA_VERSION = "pianocore_integrated_node_work_v2_stem"
STEM_NONE = 0
STEM_UP = 1
STEM_DOWN = 2

WORKER_RAW_ZIP = None
WORKER_ARGS: argparse.Namespace | None = None


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
    keep = ["score_xml_path", "score_midi_path", "refined_score_midi_path"]
    return df[keep].drop_duplicates("refined_score_midi_path").reset_index(drop=True)


def node_json_path(json_dir: Path, refined_score_midi_path: str) -> Path:
    return (json_dir / refined_score_midi_path).with_suffix(".json")


def init_worker(raw_midi_zip: str, args_dict: dict[str, Any]) -> None:
    global WORKER_RAW_ZIP, WORKER_ARGS
    WORKER_RAW_ZIP = score_xml_alignment.ZipResolver(Path(raw_midi_zip))
    WORKER_ARGS = argparse.Namespace(**args_dict)


def musicxml_bytes_from_zip(raw_zip: score_xml_alignment.ZipResolver, relative_path: str) -> bytes:
    member = raw_zip.resolve(relative_path)
    data = raw_zip.zip_file.read(member)
    if not relative_path.lower().endswith(".mxl"):
        return data
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile:
        return data
    with archive:
        container = "META-INF/container.xml"
        if container in archive.namelist():
            root = ET.fromstring(archive.read(container))
            rootfile = root.find(".//{*}rootfile")
            if rootfile is not None and rootfile.get("full-path"):
                return archive.read(rootfile.get("full-path"))
        for name in archive.namelist():
            if name.lower().endswith((".xml", ".musicxml")) and not name.startswith("META-INF/"):
                return archive.read(name)
    raise FileNotFoundError(f"no MusicXML score found in {relative_path}")


def direct_child(element: ET.Element, name: str) -> ET.Element | None:
    return element.find(f"{{*}}{name}")


def direct_child_text(element: ET.Element, name: str, default: str | None = None) -> str | None:
    child = direct_child(element, name)
    if child is None or child.text is None:
        return default
    return child.text.strip()


def pitch_step_to_midi(step: str, octave: str, alter: str | None = None) -> int:
    semitone = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}[step.upper()]
    return (int(octave) + 1) * 12 + semitone + int(round(float(alter or 0)))


def stem_code(text: str | None) -> int:
    value = (text or "").strip().lower()
    if value == "up":
        return STEM_UP
    if value == "down":
        return STEM_DOWN
    return STEM_NONE


def parse_xml_pitch_stem(raw_zip: score_xml_alignment.ZipResolver, score_xml_path: str) -> tuple[list[int], list[int]]:
    root = ET.fromstring(musicxml_bytes_from_zip(raw_zip, score_xml_path))
    entries: list[dict[str, Any]] = []

    for part in root.findall("{*}part"):
        divisions = 1.0
        absolute_measure_offset = 0.0
        for measure in score_xml_alignment.direct_expand_repeats(part.findall("{*}measure")):
            cursor = 0.0
            measure_max = 0.0
            previous_onset = 0.0
            previous_stem = STEM_NONE
            for item in list(measure):
                local_name = item.tag.rsplit("}", 1)[-1]
                if local_name == "attributes":
                    divisions_text = direct_child_text(item, "divisions")
                    if divisions_text:
                        divisions = max(score_xml_alignment.safe_float(divisions_text, divisions), 1e-9)
                    continue
                if local_name == "backup":
                    duration = score_xml_alignment.safe_float(direct_child_text(item, "duration"), 0.0) / divisions
                    cursor = max(0.0, cursor - duration)
                    continue
                if local_name == "forward":
                    duration = score_xml_alignment.safe_float(direct_child_text(item, "duration"), 0.0) / divisions
                    cursor += duration
                    measure_max = max(measure_max, cursor)
                    continue
                if local_name != "note" or item.get("print-object") == "no":
                    continue

                is_chord = direct_child(item, "chord") is not None
                is_grace = direct_child(item, "grace") is not None
                onset = previous_onset if is_chord else cursor
                duration = 0.0 if is_grace else score_xml_alignment.safe_float(direct_child_text(item, "duration"), 0.0) / divisions
                current_stem = stem_code(direct_child_text(item, "stem"))
                if is_chord and current_stem == STEM_NONE:
                    current_stem = previous_stem

                pitch_node = direct_child(item, "pitch")
                if pitch_node is not None:
                    step = direct_child_text(pitch_node, "step")
                    octave = direct_child_text(pitch_node, "octave")
                    if step is not None and octave is not None:
                        with contextlib.suppress(Exception):
                            entries.append(
                                {
                                    "offset": absolute_measure_offset + onset,
                                    "duration": duration,
                                    "pitch": pitch_step_to_midi(step, octave, direct_child_text(pitch_node, "alter")),
                                    "grace": int(is_grace),
                                    "stem": current_stem,
                                }
                            )

                if not is_chord:
                    previous_onset = onset
                    previous_stem = current_stem
                    if not is_grace:
                        cursor += duration
                        measure_max = max(measure_max, cursor)
                elif not is_grace:
                    measure_max = max(measure_max, onset + duration)

            absolute_measure_offset += max(measure_max, 0.0)

    entries.sort(key=lambda item: (item["offset"], not bool(item["grace"]), item["pitch"], item["duration"]))
    deduped: list[dict[str, Any]] = []
    for entry in entries:
        if not deduped or entry["offset"] != deduped[-1]["offset"] or entry["pitch"] != deduped[-1]["pitch"]:
            deduped.append(entry)
        elif bool(deduped[-1]["grace"]) or entry["duration"] > deduped[-1]["duration"]:
            deduped[-1] = entry

    return [int(item["pitch"]) for item in deduped], [int(item["stem"]) for item in deduped]


def compose_refined_to_xml(raw_pitches: list[int], refined_pitches: list[int], xml_pitches: list[int], args: argparse.Namespace):
    _, refined_to_raw = score_xml_alignment.build_raw_to_refined_map(
        raw_pitches,
        refined_pitches,
        args.max_sequence_matcher_notes,
        args.timeout_sec,
        not args.disable_sequence_matcher,
    )
    _, raw_to_xml = score_xml_alignment.build_raw_to_xml_map(
        raw_pitches,
        xml_pitches,
        args.max_sequence_matcher_notes,
        args.timeout_sec,
        not args.disable_sequence_matcher,
    )
    return {
        refined_idx: raw_to_xml[raw_idx]
        for refined_idx, raw_idx in refined_to_raw.items()
        if raw_idx in raw_to_xml
    }


def update_one(task: dict[str, Any]) -> dict[str, Any]:
    if WORKER_ARGS is None or WORKER_RAW_ZIP is None:
        raise RuntimeError("worker was not initialized")
    args = WORKER_ARGS
    started = time.time()
    json_path = node_json_path(Path(task["json_dir"]), task["refined_score_midi_path"])
    result = {"json_path": str(json_path), "refined_score_midi_path": task["refined_score_midi_path"]}
    if not json_path.exists():
        result.update({"status": "error", "error": "missing_json"})
        return result

    tmp_path = json_path.with_name(json_path.name + ".tmp")
    try:
        with json_path.open(encoding="utf-8") as file:
            payload = json.load(file)
        score = payload["score"]
        refined_pitches = [int(value) for value in score["pitch"]]
        features = score.get("score_feature") or []
        if len(features) != len(refined_pitches):
            raise ValueError("score_feature_length_mismatch")

        keys = list(payload.setdefault("meta", {}).get("score_feature_keys") or [])
        stem_idx = keys.index("stem") if "stem" in keys else None
        if stem_idx is not None and all(len(row) > stem_idx for row in features):
            result.update({"status": "skipped", "note_count": len(refined_pitches), "matched": 0, "elapsed_sec": round(time.time() - started, 3)})
            return result

        xml_pitches, xml_stems = parse_xml_pitch_stem(WORKER_RAW_ZIP, task["score_xml_path"])
        raw_pitches = score_xml_alignment.load_midi_pitches_from_zip(WORKER_RAW_ZIP, task["score_midi_path"])
        refined_to_xml = compose_refined_to_xml(raw_pitches, refined_pitches, xml_pitches, args)

        stem_values = [STEM_NONE] * len(refined_pitches)
        matched = 0
        stem_counts = Counter()
        for refined_idx, xml_idx in refined_to_xml.items():
            if 0 <= refined_idx < len(stem_values) and 0 <= xml_idx < len(xml_stems):
                stem_values[refined_idx] = int(xml_stems[xml_idx])
                matched += 1
                stem_counts[int(xml_stems[xml_idx])] += 1

        updated_features = []
        for row, stem in zip(features, stem_values):
            base = list(row)
            if stem_idx is None:
                base.append(float(stem))
            else:
                while len(base) <= stem_idx:
                    base.append(0.0)
                base[stem_idx] = float(stem)
            updated_features.append(base)
        score["score_feature"] = updated_features

        meta = payload.setdefault("meta", {})
        if "stem" not in keys:
            keys.append("stem")
        meta["score_feature_keys"] = keys
        meta["stem_encoding"] = {"none": STEM_NONE, "up": STEM_UP, "down": STEM_DOWN}
        meta["stem_source"] = "MusicXML stem projected to refined score notes"
        meta["schema"] = SCHEMA_VERSION
        payload["schema"] = SCHEMA_VERSION

        align = meta.setdefault("xml_to_refined_score_alignment", {})
        align["stem_matched"] = matched
        align["stem_unmatched"] = len(refined_pitches) - matched
        align["stem_coverage"] = float(matched / len(refined_pitches)) if refined_pitches else 1.0
        align["stem_counts"] = {
            "none": int(stem_counts.get(STEM_NONE, 0)),
            "up": int(stem_counts.get(STEM_UP, 0)),
            "down": int(stem_counts.get(STEM_DOWN, 0)),
        }

        if not args.dry_run:
            with tmp_path.open("w", encoding="utf-8") as file:
                json.dump(clean_json_value(payload), file, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
                file.write("\n")
            os.replace(tmp_path, json_path)

        result.update(
            {
                "status": "ok" if matched == len(refined_pitches) else "partial" if matched else "failed",
                "note_count": len(refined_pitches),
                "matched": matched,
                "stem_none": int(stem_counts.get(STEM_NONE, 0)),
                "stem_up": int(stem_counts.get(STEM_UP, 0)),
                "stem_down": int(stem_counts.get(STEM_DOWN, 0)),
                "elapsed_sec": round(time.time() - started, 3),
            }
        )
        return result
    except Exception as exc:  # noqa: BLE001
        if not args.dry_run:
            tmp_path.unlink(missing_ok=True)
        result.update({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
        return result


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    total_notes = sum(int(item.get("note_count") or 0) for item in results)
    matched = sum(int(item.get("matched") or 0) for item in results)
    return {
        "total_scores": len(results),
        "status_counts": dict(Counter(item.get("status", "unknown") for item in results)),
        "total_notes": total_notes,
        "stem_matched_notes": matched,
        "stem_unmatched_notes": total_notes - matched,
        "stem_coverage": float(matched / total_notes) if total_notes else 0.0,
        "stem_counts": {
            "none": sum(int(item.get("stem_none") or 0) for item in results),
            "up": sum(int(item.get("stem_up") or 0) for item in results),
            "down": sum(int(item.get("stem_down") or 0) for item in results),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--pianocore-dir", type=Path, default=Path("../PianoCoRe"))
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--raw-midi-zip", type=Path, default=None)
    parser.add_argument("--json-dir", type=Path, default=Path("data/ASAP_processed"))
    parser.add_argument("--subset", choices=["a", "a_star", "all"], default="a")
    parser.add_argument("--num-proc", type=int, default=30)
    parser.add_argument("--limit-works", type=int, default=None)
    parser.add_argument("--timeout-sec", type=float, default=180.0)
    parser.add_argument("--max-sequence-matcher-notes", type=int, default=13000)
    parser.add_argument("--disable-sequence-matcher", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summary-path", type=Path, default=Path("results/data_process/processed_stem_backfill_summary.json"))
    parser.add_argument("--details-path", type=Path, default=Path("results/data_process/processed_stem_backfill_details.jsonl"))
    args = parser.parse_args()

    metadata_path = args.metadata or (args.pianocore_dir / "metadata.csv")
    raw_midi_zip = args.raw_midi_zip or (args.pianocore_dir / "PianoCoRe-1.0-raw-midi.zip")
    rows = load_score_rows(metadata_path, args.subset)
    if args.limit_works is not None:
        rows = rows.head(args.limit_works).reset_index(drop=True)

    tasks = [
        {
            "json_dir": str(args.json_dir),
            "score_xml_path": row["score_xml_path"],
            "score_midi_path": row["score_midi_path"],
            "refined_score_midi_path": row["refined_score_midi_path"],
        }
        for row in rows.to_dict("records")
    ]
    args.summary_path.parent.mkdir(parents=True, exist_ok=True)
    args.details_path.parent.mkdir(parents=True, exist_ok=True)
    args_dict = vars(args).copy()
    args_dict.update(
        {
            "pianocore_dir": str(args.pianocore_dir),
            "metadata": str(metadata_path),
            "raw_midi_zip": str(raw_midi_zip),
            "json_dir": str(args.json_dir),
            "summary_path": str(args.summary_path),
            "details_path": str(args.details_path),
        }
    )

    results = []
    with args.details_path.open("w", encoding="utf-8") as details_file:
        with ProcessPoolExecutor(
            max_workers=args.num_proc,
            initializer=init_worker,
            initargs=(str(raw_midi_zip), args_dict),
        ) as executor:
            futures = [executor.submit(update_one, task) for task in tasks]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Backfilling stem"):
                result = future.result()
                results.append(result)
                details_file.write(json.dumps(clean_json_value(result), ensure_ascii=False, allow_nan=False) + "\n")
                details_file.flush()

    summary = summarize(results)
    summary.update(
        {
            "dry_run": args.dry_run,
            "schema": SCHEMA_VERSION,
            "metadata": str(metadata_path),
            "raw_midi_zip": str(raw_midi_zip),
            "json_dir": str(args.json_dir),
            "num_proc": args.num_proc,
        }
    )
    with args.summary_path.open("w", encoding="utf-8") as summary_file:
        json.dump(clean_json_value(summary), summary_file, ensure_ascii=False, indent=2, allow_nan=False)
        summary_file.write("\n")
    print(json.dumps(clean_json_value(summary), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
