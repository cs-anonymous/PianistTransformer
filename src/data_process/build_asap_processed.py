#!/usr/bin/env python3
"""Build a clean ASAP-only processed dataset.

Outputs under data/ASAP_processed:

- metadata.csv: ASAP-only metadata rows.
- **/score*.json: work JSON files with refined score/performance-aligned raw labels.
- **/score*.pt: raw sidecars with real compact 4-slot score features
  stored as [mo, md, ml, annotation6].
- **/score*.mxl and **/score*.musicxml: reconstructed scores.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data_process.sidecar_builder import build_sidecar_for_work
from src.train.train_inr import PianoCoReNodeSFTDataset, infer_input_feature_mode


DEFAULT_TAG = "ASAP_MUSICAL_V2"
FEATURE_KEYS = ["mo_idx", "md_idx", "ml_idx", "staff", "trill", "grace", "staccato", "stem_up", "stem_down"]


def run_command(command: list[str]) -> None:
    print(" ".join(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def write_asap_metadata(source: Path, output: Path) -> dict[str, Any]:
    df = pd.read_csv(source)
    mask = df["performance_dataset"].fillna("").astype(str).str.upper().eq("ASAP")
    mask &= df["is_refined"].fillna(False).astype(bool)
    mask &= df["tier_a"].fillna(False).astype(bool)
    mask &= df["refined_score_midi_path"].notna()
    mask &= df["refined_performance_midi_path"].notna()
    mask &= df["refined_alignment_path"].notna()
    mask &= df["score_xml_path"].notna()
    mask &= df["score_midi_path"].notna()
    asap = df[mask].copy()
    asap = asap.sort_values(
        ["split", "refined_score_midi_path", "refined_performance_midi_path", "id"],
        kind="stable",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    asap.to_csv(output, index=False)
    return {
        "rows": int(len(asap)),
        "works": int(asap["refined_score_midi_path"].nunique()),
        "splits": {str(k): int(v) for k, v in asap["split"].value_counts().sort_index().items()},
    }


def json_paths(json_root: Path) -> list[Path]:
    return sorted(path for path in json_root.rglob("score_*_refined.json") if path.is_file())


def write_metadata_for_existing_jsons(metadata_path: Path, json_root: Path, output_path: Path) -> dict[str, Any]:
    df = pd.read_csv(metadata_path)
    existing = set()
    for path in json_paths(json_root):
        rel = path.relative_to(json_root).with_suffix(".mid")
        existing.add(str(rel).replace("\\", "/"))
    filtered = df[df["refined_score_midi_path"].isin(existing)].copy()
    filtered.to_csv(output_path, index=False)
    return {"rows": int(len(filtered)), "works": int(filtered["refined_score_midi_path"].nunique())}


def validate_json_one(path: Path, min_coverage: float) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        payload = json.load(file)
    score = payload.get("score") or {}
    pitch = score.get("pitch") or []
    score_feature = score.get("score_feature")
    has_score_feature = score.get("has_score_feature")
    if not isinstance(score_feature, list):
        raise ValueError(f"missing_score_feature: {path}")
    if not isinstance(has_score_feature, list):
        raise ValueError(f"missing_has_score_feature: {path}")
    if len(score_feature) != len(pitch):
        raise ValueError(f"score_feature_length_mismatch: {path}")
    if len(has_score_feature) != len(pitch):
        raise ValueError(f"has_score_feature_length_mismatch: {path}")
    keys = (payload.get("meta") or {}).get("score_feature_keys") or []
    if keys[:9] != FEATURE_KEYS:
        raise ValueError(f"unexpected_score_feature_keys={keys}: {path}")
    for idx, (row, has_feature) in enumerate(zip(score_feature, has_score_feature)):
        if not bool(has_feature):
            continue
        if len(row) != 9:
            raise ValueError(f"score_feature_width_mismatch[{idx}]: {path}")
        mo_idx = int(round(float(row[0])))
        md_idx = int(round(float(row[1])))
        ml_idx = int(round(float(row[2])))
        if mo_idx < 0 or mo_idx > 144 or md_idx < 0 or md_idx > 144 or ml_idx < 0 or ml_idx > 144:
            raise ValueError(f"score_feature_idx_out_of_range[{idx}]: {path}")
        for value in row[3:9]:
            if float(value) not in (0.0, 1.0):
                raise ValueError(f"score_feature_binary_value_invalid[{idx}]: {path}")
    matched = sum(1 for value in has_score_feature if bool(value))
    coverage = matched / len(pitch) if pitch else 1.0
    total_abs = float(np.abs(np.asarray(score_feature, dtype=float)).sum()) if score_feature else 0.0
    if pitch and matched <= 0:
        raise ValueError(f"zero_score_feature_coverage: {path}")
    if pitch and total_abs <= 0.0:
        raise ValueError(f"all_zero_score_feature: {path}")
    if coverage < min_coverage:
        raise ValueError(f"low_score_feature_coverage={coverage:.6f}: {path}")
    align = (payload.get("meta") or {}).get("xml_to_refined_score_alignment") or {}
    return {
        "path": str(path),
        "notes": int(len(pitch)),
        "matched": int(matched),
        "coverage": float(coverage),
        "alignment_status": align.get("status"),
        "raw_refined_relation": align.get("raw_refined_relation"),
        "xml_raw_relation": align.get("xml_raw_relation"),
        "performances": int(len(payload.get("performances") or [])),
    }


def validate_json_tree(json_root: Path, min_coverage: float) -> dict[str, Any]:
    details = [validate_json_one(path, min_coverage) for path in json_paths(json_root)]
    notes = sum(item["notes"] for item in details)
    matched = sum(item["matched"] for item in details)
    return {
        "works": len(details),
        "notes": notes,
        "matched": matched,
        "coverage": matched / notes if notes else 1.0,
        "min_coverage": min((item["coverage"] for item in details), default=1.0),
        "details": details,
    }


def build_dataset(config: dict[str, Any], work_path: str):
    manifest = [{"path": work_path, "windows": [(0, 1)], "estimated_performances": 1}]
    return PianoCoReNodeSFTDataset(
        manifest,
        split="train",
        task_type=config.get("task_type", "epr"),
        input_feature_mode=infer_input_feature_mode(config),
        shuffle=False,
        seed=config.get("seed", 42),
        cache_size=1,
        musical_feature_mode=config.get("musical_feature_mode", "musical4slot"),
        disable_musical_features=False,
        epr_timing_target=config.get("epr_timing_target", "floor_log_deviation"),
        use_timing_scale_bit=False,
        timing_control_mode=config.get("timing_control_mode", "dinr_floor_log"),
        timing_log_scale=config.get("timing_log_scale", 50.0),
        pedal_representation=config.get("pedal_representation", "binary_4"),
        use_prepared_sidecar=False,
        prepared_sidecar_tag=config["sidecar_tag"],
    )


def build_sidecar_worker(task: dict[str, Any]) -> dict[str, Any]:
    path = task["path"]
    dataset = build_dataset(task["config"], path)
    sidecar = build_sidecar_for_work(
        dataset,
        path,
        selected_sources=None,
        performance_time_normalization=task["performance_time_normalization"],
    )
    clean_sidecar = Path(path).with_suffix(".pt")
    tagged_sidecar = Path(path).with_suffix(f".{task['config']['sidecar_tag']}.pt")
    if sidecar is None:
        sidecar = tagged_sidecar if tagged_sidecar.exists() else clean_sidecar
    sidecar = Path(sidecar)
    if sidecar != clean_sidecar:
        clean_sidecar.unlink(missing_ok=True)
        shutil.move(str(sidecar), str(clean_sidecar))
    return {"path": path, "sidecar": str(clean_sidecar)}


def build_sidecars(json_root: Path, workers: int, tag: str, performance_time_normalization: str) -> dict[str, Any]:
    paths = [str(path) for path in json_paths(json_root)]
    config = {
        "sidecar_tag": tag,
        "musical_feature_mode": "musical4slot",
        "pedal_representation": "binary_4",
        "epr_timing_target": "floor_log_deviation",
        "timing_control_mode": "dinr_floor_log",
        "timing_log_scale": 50.0,
        "input_feature_mode": "integrated",
        "task_type": "epr",
    }
    tasks = [
        {
            "path": path,
            "config": config,
            "performance_time_normalization": performance_time_normalization,
        }
        for path in paths
    ]
    results = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(build_sidecar_worker, task) for task in tasks]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Building ASAP sidecars"):
            results.append(future.result())
    return {"works": len(results), "details": sorted(results, key=lambda item: item["path"])}


def validate_sidecar_one(path: Path, tag: str | None = None) -> dict[str, Any]:
    sidecar = path.with_suffix(".pt")
    if not sidecar.exists():
        raise FileNotFoundError(f"missing_sidecar: {sidecar}")
    payload = torch.load(sidecar, map_location="cpu", weights_only=False)
    score = payload.get("score") or {}
    pitch = score.get("pitch") or []
    score_feature = score.get("score_feature")
    has_score_feature = score.get("has_score_feature")
    if not isinstance(score_feature, list) or len(score_feature) != len(pitch):
        raise ValueError(f"bad_sidecar_score_feature: {sidecar}")
    if not isinstance(has_score_feature, list) or len(has_score_feature) != len(pitch):
        raise ValueError(f"bad_sidecar_has_score_feature: {sidecar}")
    for idx, (row, has_feature) in enumerate(zip(score_feature, has_score_feature)):
        if not bool(has_feature):
            continue
        if len(row) != 9:
            raise ValueError(f"bad_sidecar_score_feature_width[{idx}]: {sidecar}")
        if not (
            0 <= int(round(float(row[0]))) <= 144
            and 0 <= int(round(float(row[1]))) <= 144
            and 0 <= int(round(float(row[2]))) <= 144
        ):
            raise ValueError(f"bad_sidecar_score_feature_idx[{idx}]: {sidecar}")
        if any(float(value) not in (0.0, 1.0) for value in row[3:9]):
            raise ValueError(f"bad_sidecar_score_feature_binary[{idx}]: {sidecar}")
    total_abs = float(np.abs(np.asarray(score_feature, dtype=float)).sum()) if score_feature else 0.0
    matched = sum(1 for value in has_score_feature if bool(value))
    if pitch and (matched <= 0 or total_abs <= 0.0):
        raise ValueError(f"empty_sidecar_score_feature: {sidecar}")
    return {
        "path": str(path),
        "sidecar": str(sidecar),
        "notes": int(len(pitch)),
        "matched": int(matched),
        "coverage": matched / len(pitch) if pitch else 1.0,
        "performances": int(len(payload.get("performances") or [])),
    }


def validate_sidecars(json_root: Path, tag: str | None = None) -> dict[str, Any]:
    details = [validate_sidecar_one(path, tag) for path in json_paths(json_root)]
    notes = sum(item["notes"] for item in details)
    matched = sum(item["matched"] for item in details)
    return {
        "works": len(details),
        "notes": notes,
        "matched": matched,
        "coverage": matched / notes if notes else 1.0,
        "min_coverage": min((item["coverage"] for item in details), default=1.0),
        "details": details,
    }


def extract_plain_musicxml(mxl_path: Path, musicxml_path: Path) -> None:
    with zipfile.ZipFile(mxl_path) as archive:
        member = "score.musicxml"
        if member not in archive.namelist():
            container = "META-INF/container.xml"
            if container in archive.namelist():
                import xml.etree.ElementTree as ET

                root = ET.fromstring(archive.read(container))
                rootfile = root.find(".//{*}rootfile")
                if rootfile is not None and rootfile.get("full-path"):
                    member = rootfile.get("full-path")
        musicxml_path.parent.mkdir(parents=True, exist_ok=True)
        musicxml_path.write_bytes(archive.read(member))


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--output-root", type=Path, default=Path("data/ASAP_processed"))
    parser.add_argument("--pianocore-root", type=Path, default=Path("data/ASAP_processed"))
    parser.add_argument("--metadata", type=Path, default=Path("data/ASAP_processed/metadata.generated_json.csv"))
    parser.add_argument("--raw-root", type=Path, default=Path("data/ASAP_processed"))
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--sidecar-tag", default=DEFAULT_TAG)
    parser.add_argument("--min-score-feature-coverage", type=float, default=0.95)
    parser.add_argument("--performance-time-normalization", choices=["none", "score_onset_span"], default="none")
    parser.add_argument("--limit-works", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    output_root = args.output_root
    metadata_out = output_root / "metadata.csv"
    work_root = output_root
    summary_path = output_root / "pipeline_summary.json"

    if args.clean and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "output_root": str(output_root),
        "workers": args.workers,
        "sidecar_tag": args.sidecar_tag,
        "performance_time_normalization": args.performance_time_normalization,
    }

    summary["metadata"] = write_asap_metadata(args.metadata, metadata_out)
    print(json.dumps({"event": "metadata_written", **summary["metadata"]}, ensure_ascii=False), flush=True)

    run_command(
        [
            sys.executable,
            "src/data_process/generate_json_with_paired_midi.py",
            "--pianocore-dir",
            str(args.pianocore_root),
            "--metadata",
            str(metadata_out.resolve()),
            "--output-dir",
            str(work_root),
            "--summary-path",
            str(output_root / "json_generation_summary.json"),
            "--num-proc",
            str(args.workers),
            "--overwrite",
            *(["--limit-works", str(args.limit_works)] if args.limit_works is not None else []),
        ]
    )

    feature_metadata = output_root / "metadata.generated_json.csv"
    summary["metadata_generated_json"] = write_metadata_for_existing_jsons(metadata_out, work_root, feature_metadata)

    run_command(
        [
            sys.executable,
            "src/data_process/update_json_score_feature_with_xml.py",
            "--pianocore-dir",
            str(args.pianocore_root),
            "--metadata",
            str(feature_metadata.resolve()),
            "--raw-midi-zip",
            str(args.raw_root),
            "--json-dir",
            str(work_root),
            "--summary-path",
            str(output_root / "score_feature_update_summary.json"),
            "--details-path",
            str(output_root / "score_feature_update_details.jsonl"),
            "--num-proc",
            str(args.workers),
        ]
    )

    summary["json_validation"] = validate_json_tree(work_root, args.min_score_feature_coverage)
    print(json.dumps({"event": "json_validated", **{k: v for k, v in summary["json_validation"].items() if k != "details"}}, ensure_ascii=False), flush=True)

    summary["sidecar_build"] = build_sidecars(
        work_root,
        args.workers,
        args.sidecar_tag,
        args.performance_time_normalization,
    )
    summary["sidecar_validation"] = validate_sidecars(work_root, args.sidecar_tag)
    print(json.dumps({"event": "sidecars_validated", **{k: v for k, v in summary["sidecar_validation"].items() if k != "details"}}, ensure_ascii=False), flush=True)

    run_command(
        [
            sys.executable,
            "src/data_process/note_features_to_score.py",
            "--json-root",
            str(work_root),
            "--refined-dir",
            str(args.pianocore_root / "refined"),
            "--summary-path",
            str(output_root / "musicxml_generation_summary.json"),
            "--details-path",
            str(output_root / "musicxml_generation_details.jsonl"),
            "--num-proc",
            str(args.workers),
            "--overwrite",
            "--output-mode",
            "basename",
        ]
    )

    generated_musicxml = sorted(work_root.rglob("score*.mxl"))
    extracted = 0
    for target in generated_musicxml:
        plain_target = target.with_suffix(".musicxml")
        extract_plain_musicxml(target, plain_target)
        extracted += 1
    summary["musicxml"] = {"works": len(generated_musicxml), "plain_musicxml": extracted, "root": str(work_root)}

    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2, allow_nan=False)
        file.write("\n")
    print(json.dumps({"event": "asap_pipeline_done", "summary": str(summary_path)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
