#!/usr/bin/env python3
"""Build musical51 sidecars alongside the existing ASAP compact sidecars.

This keeps the JSON tree shared, while writing a second tagged sidecar per work:

- ``*.pt`` stays as the canonical compact 4slot sidecar.
- ``*.ASAP_MUSICAL51.pt`` stores the converted musical51-compatible score rows.
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from tqdm import tqdm


def _json_path_from_metadata_root(refined_dir: Path, score_rel_path: str) -> Path:
    score_path = refined_dir / score_rel_path
    return score_path.with_suffix(".json")


def _load_payload(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _convert_row(row: list[float]) -> list[float]:
    mo_idx = float(row[0]) if len(row) > 0 else 0.0
    md_idx = float(row[1]) if len(row) > 1 else 0.0
    ml_idx = float(row[2]) if len(row) > 2 else 0.0
    first = 1.0 if ml_idx > 0.0 else 0.0
    hand = 1.0 if len(row) > 3 and float(row[3]) >= 0.5 else 0.0
    trill = 1.0 if len(row) > 4 and float(row[4]) >= 0.5 else 0.0
    grace = 1.0 if len(row) > 5 and float(row[5]) >= 0.5 else 0.0
    staccato = 1.0 if len(row) > 6 and float(row[6]) >= 0.5 else 0.0
    stem_up = 1.0 if len(row) > 7 and float(row[7]) >= 0.5 else 0.0
    stem_down = 1.0 if len(row) > 8 and float(row[8]) >= 0.5 else 0.0
    stem_code = 1.0 if stem_up >= 0.5 else 2.0 if stem_down >= 0.5 else 0.0
    return [
        mo_idx / 24.0,
        md_idx / 24.0,
        (ml_idx / 24.0) if ml_idx > 0.0 else 0.0,
        first,
        hand,
        trill,
        grace,
        staccato,
        stem_code,
    ]


def _convert_payload(source_path: Path, target_path: Path) -> dict[str, Any]:
    payload = _load_payload(source_path)
    score = dict(payload.get("score") or {})
    score_feature = score.get("score_feature")
    has_score_feature = score.get("has_score_feature")
    pitch = score.get("pitch") or []

    if not isinstance(score_feature, list):
        raise ValueError(f"missing_score_feature: {source_path}")
    if not isinstance(has_score_feature, list):
        raise ValueError(f"missing_has_score_feature: {source_path}")
    if len(score_feature) != len(pitch):
        raise ValueError(f"score_feature_length_mismatch: {source_path}")
    if len(has_score_feature) != len(pitch):
        raise ValueError(f"has_score_feature_length_mismatch: {source_path}")

    converted = []
    matched = 0
    for row, has_feature in zip(score_feature, has_score_feature):
        if bool(has_feature):
            converted.append(_convert_row(row))
            matched += 1
        else:
            converted.append([0.0] * 9)

    score["score_feature"] = converted
    score["has_score_feature"] = [1 if bool(value) else 0 for value in has_score_feature]

    meta = dict(payload.get("meta") or {})
    meta["score_feature_layout"] = "musical51_from_compact_4slot"
    meta["score_feature_source_layout"] = "compact_4slot_v2"
    meta["score_feature_keys"] = [
        "mo_q",
        "md_q",
        "ml_q",
        "first",
        "hand",
        "trill",
        "grace",
        "staccato",
        "stem_code",
    ]
    meta["score_feature_unit"] = "quarter_length_raw"

    payload["score"] = score
    payload["meta"] = meta
    payload["_cache_signature"] = json.dumps(
        {"schema": 5, "kind": "inr_raw_sidecar"},
        sort_keys=True,
        separators=(",", ":"),
    )
    json_source = source_path.with_suffix(".json")
    if not payload.get("_source_identity") and json_source.exists():
        stat = json_source.stat()
        payload["_source_identity"] = {
            "path": str(json_source.resolve()),
            "mtime_ns": stat.st_mtime_ns,
            "size": stat.st_size,
        }

    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_name(f"{target_path.name}.{os.getpid()}.tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(target_path)
    return {
        "source": str(source_path),
        "target": str(target_path),
        "notes": int(len(pitch)),
        "matched": int(matched),
    }


def _work_item(metadata_root: Path, score_rel_path: str) -> tuple[Path, Path]:
    json_path = _json_path_from_metadata_root(metadata_root, score_rel_path)
    source_path = json_path.with_suffix(".pt")
    target_path = json_path.with_suffix(".ASAP_MUSICAL51.pt")
    return source_path, target_path


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--metadata", type=Path, default=Path("data/ASAP_processed/metadata.generated_json.csv"))
    parser.add_argument("--refined-dir", type=Path, default=Path("data/ASAP_processed"))
    parser.add_argument("--workers", type=int, default=40)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--summary-path", type=Path, default=Path("data/ASAP_processed/musical51_sidecar_summary.json"))
    args = parser.parse_args()

    df = pd.read_csv(args.metadata, usecols=["refined_score_midi_path"])
    score_paths = sorted(set(df["refined_score_midi_path"].dropna().astype(str)))

    tasks = []
    for score_rel_path in score_paths:
        source_path, target_path = _work_item(args.refined_dir, score_rel_path)
        if not source_path.exists():
            continue
        if target_path.exists() and not args.overwrite:
            continue
        tasks.append((source_path, target_path))

    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(_convert_payload, source, target) for source, target in tasks]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Building ASAP musical51 sidecars"):
            results.append(future.result())

    notes = sum(item["notes"] for item in results)
    matched = sum(item["matched"] for item in results)
    summary = {
        "works": len(results),
        "notes": notes,
        "matched": matched,
        "coverage": float(matched / notes) if notes else 1.0,
        "source_metadata": str(args.metadata),
        "refined_dir": str(args.refined_dir),
        "target_suffix": ".ASAP_MUSICAL51.pt",
        "details": sorted(results, key=lambda item: item["source"]),
    }
    args.summary_path.parent.mkdir(parents=True, exist_ok=True)
    args.summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"event": "musical51_sidecars_done", **{k: v for k, v in summary.items() if k != "details"}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
