#!/usr/bin/env python3
"""Summarize XML -> refined MIDI coverage at score and note level."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build note-level coverage reports from XML/refined audit details.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--metadata", type=Path, default=Path("data/pianocore/metadata.csv"))
    parser.add_argument("--details", type=Path, default=Path("results/xml_refined_alignment_audit_details_best.jsonl"))
    parser.add_argument("--subset", choices=["a", "a_star", "all"], default="a")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/xml_refined_alignment_note_level_summary.json"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/xml_refined_alignment_score_level_coverage.csv"),
    )
    return parser.parse_args()


def load_metadata(path: Path, subset: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if subset == "a":
        df = df[df["tier_a"].astype(bool)]
    elif subset == "a_star":
        df = df[df["tier_a_star"].astype(bool)]
    df = df[df["refined_score_midi_path"].notna()]
    if "is_refined" in df.columns:
        df = df[df["is_refined"].astype(bool)]
    return df.drop_duplicates("refined_score_midi_path").reset_index(drop=True)


def load_details(path: Path) -> dict[str, dict[str, Any]]:
    details = {}
    with path.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            row = json.loads(line)
            details[row["refined_score_midi_path"]] = row
    return details


def add_group(group: dict[str, int], mapped: int, total: int) -> None:
    group["scores"] += 1
    group["mapped_notes"] += mapped
    group["total_notes"] += total
    group["unmapped_notes"] += total - mapped


def finalize_group(group: dict[str, int]) -> dict[str, Any]:
    total = group["total_notes"]
    return {
        **group,
        "coverage": float(group["mapped_notes"] / total) if total else 0.0,
    }


def coverage_bin(coverage: float) -> str:
    if coverage == 0.0:
        return "0"
    if coverage < 0.5:
        return "(0,0.5)"
    if coverage < 0.9:
        return "[0.5,0.9)"
    if coverage < 0.99:
        return "[0.9,0.99)"
    if coverage < 1.0:
        return "[0.99,1)"
    return "1"


def clean_json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: clean_json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean_json_value(item) for item in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def main() -> None:
    args = parse_args()
    metadata = load_metadata(args.metadata, args.subset)
    details = load_details(args.details)

    score_rows = []
    by_status = defaultdict(lambda: {"scores": 0, "mapped_notes": 0, "total_notes": 0, "unmapped_notes": 0})
    by_xml_raw_relation = defaultdict(lambda: {"scores": 0, "mapped_notes": 0, "total_notes": 0, "unmapped_notes": 0})
    by_raw_refined_relation = defaultdict(lambda: {"scores": 0, "mapped_notes": 0, "total_notes": 0, "unmapped_notes": 0})
    by_bin = defaultdict(lambda: {"scores": 0, "mapped_notes": 0, "total_notes": 0, "unmapped_notes": 0})

    for _, meta_row in metadata.iterrows():
        path = meta_row["refined_score_midi_path"]
        detail = details.get(path, {})
        total = detail.get("refined_note_count")
        if total is None:
            total = meta_row.get("refined_score_note_count", 0)
        total = int(total or 0)
        mapped = int(detail.get("mapped_refined_note_count") or 0)
        coverage = float(mapped / total) if total else 0.0
        status = detail.get("status", "missing")
        xml_raw_relation = detail.get("xml_raw_relation", "none")
        raw_refined_relation = detail.get("raw_refined_relation", "none")
        unmapped = total - mapped

        row = {
            "refined_score_midi_path": path,
            "score_xml_path": meta_row.get("score_xml_path"),
            "score_midi_path": meta_row.get("score_midi_path"),
            "composer": meta_row.get("composer"),
            "composition": meta_row.get("composition"),
            "movement": meta_row.get("movement"),
            "status": status,
            "xml_raw_relation": xml_raw_relation,
            "raw_refined_relation": raw_refined_relation,
            "refined_notes": total,
            "mapped_notes": mapped,
            "unmapped_notes": unmapped,
            "coverage": coverage,
            "error": detail.get("error"),
        }
        score_rows.append(row)
        add_group(by_status[status], mapped, total)
        add_group(by_xml_raw_relation[xml_raw_relation], mapped, total)
        add_group(by_raw_refined_relation[raw_refined_relation], mapped, total)
        add_group(by_bin[coverage_bin(coverage)], mapped, total)

    total_notes = sum(row["refined_notes"] for row in score_rows)
    mapped_notes = sum(row["mapped_notes"] for row in score_rows)
    summary = {
        "coverage_denominator": "all selected unique refined_score_midi_path notes; error/missing notes count as unmapped",
        "total_scores": len(score_rows),
        "total_refined_notes": total_notes,
        "mapped_refined_notes": mapped_notes,
        "unmapped_refined_notes": total_notes - mapped_notes,
        "note_level_coverage": float(mapped_notes / total_notes) if total_notes else 0.0,
        "score_level_full": sum(1 for row in score_rows if row["coverage"] >= 1.0),
        "score_level_partial": sum(1 for row in score_rows if 0.0 < row["coverage"] < 1.0),
        "score_level_zero": sum(1 for row in score_rows if row["coverage"] == 0.0),
        "by_status": {key: finalize_group(value) for key, value in sorted(by_status.items())},
        "by_xml_raw_relation": {key: finalize_group(value) for key, value in sorted(by_xml_raw_relation.items())},
        "by_raw_refined_relation": {key: finalize_group(value) for key, value in sorted(by_raw_refined_relation.items())},
        "by_score_coverage_bin": {key: finalize_group(value) for key, value in sorted(by_bin.items())},
        "top_unmapped_scores": sorted(score_rows, key=lambda row: row["unmapped_notes"], reverse=True)[:30],
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary = clean_json_value(summary)
    args.output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with args.output_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(score_rows[0].keys()))
        writer.writeheader()
        writer.writerows(score_rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
