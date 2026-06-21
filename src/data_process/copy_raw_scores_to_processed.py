#!/usr/bin/env python3
"""Copy raw score MusicXML/MXL files into mirrored processed work folders."""

from __future__ import annotations

import argparse
import json
import os
import zipfile
from collections import Counter
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def clean_value(value):
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    return value


def zip_member_for_score(zip_file: zipfile.ZipFile, score_xml_path: str) -> str | None:
    normalized = str(score_xml_path).replace("\\", "/")
    candidates = [
        normalized,
        f"PianoCoRe/raw/{normalized}",
        f"raw/{normalized}",
    ]
    names = set(zip_file.namelist())
    for candidate in candidates:
        if candidate in names:
            return candidate
    suffix = "/" + normalized
    matches = [name for name in zip_file.namelist() if name.endswith(suffix)]
    if len(matches) == 1:
        return matches[0]
    return None


def load_score_rows(metadata_path: Path, subset: str) -> pd.DataFrame:
    usecols = ["tier_a", "tier_a_star", "refined_score_midi_path", "score_xml_path"]
    df = pd.read_csv(metadata_path, usecols=[col for col in usecols if col in pd.read_csv(metadata_path, nrows=0).columns])
    if subset == "a":
        df = df[df["tier_a"].fillna(False).astype(bool)]
    elif subset == "a_star":
        df = df[df["tier_a_star"].fillna(False).astype(bool)]
    df = df[df["refined_score_midi_path"].notna()]
    df = df[df["score_xml_path"].notna()]
    return df.drop_duplicates("refined_score_midi_path").reset_index(drop=True)


def processed_dir_for_score(processed_root: Path, refined_score_midi_path: str) -> Path:
    return (processed_root / refined_score_midi_path).parent


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--raw-midi-zip", type=Path, required=True)
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--subset", choices=["a", "a_star", "all"], default="a")
    parser.add_argument("--summary-path", type=Path, required=True)
    parser.add_argument("--details-path", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    rows = load_score_rows(args.metadata, args.subset)
    args.summary_path.parent.mkdir(parents=True, exist_ok=True)
    args.details_path.parent.mkdir(parents=True, exist_ok=True)

    results = []
    with zipfile.ZipFile(args.raw_midi_zip) as archive, args.details_path.open("w", encoding="utf-8") as details:
        for row in tqdm(rows.to_dict("records"), desc="Copying raw scores"):
            refined_score_midi_path = str(row["refined_score_midi_path"]).replace("\\", "/")
            score_xml_path = str(row["score_xml_path"]).replace("\\", "/")
            out_dir = processed_dir_for_score(args.processed_root, refined_score_midi_path)
            out_path = out_dir / Path(score_xml_path).name
            result = {
                "refined_score_midi_path": refined_score_midi_path,
                "score_xml_path": score_xml_path,
                "output_path": str(out_path),
            }
            try:
                member = zip_member_for_score(archive, score_xml_path)
                if member is None:
                    result.update({"status": "error", "error": "missing_zip_member"})
                elif out_path.exists() and not args.overwrite:
                    result.update({"status": "skipped", "zip_member": member})
                else:
                    out_dir.mkdir(parents=True, exist_ok=True)
                    tmp_path = out_path.with_name(out_path.name + ".tmp")
                    with archive.open(member) as src, tmp_path.open("wb") as dst:
                        dst.write(src.read())
                    os.replace(tmp_path, out_path)
                    result.update({"status": "ok", "zip_member": member, "bytes": out_path.stat().st_size})
            except Exception as exc:  # noqa: BLE001
                result.update({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
            results.append(result)
            details.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")

    summary = {
        "metadata": str(args.metadata),
        "raw_midi_zip": str(args.raw_midi_zip),
        "processed_root": str(args.processed_root),
        "subset": args.subset,
        "total": len(results),
        "status_counts": dict(Counter(item["status"] for item in results)),
        "summary_path": str(args.summary_path),
        "details_path": str(args.details_path),
    }
    with args.summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
