#!/usr/bin/env python3
"""Prepare median-label sidecars, held-out-work split, and 8-epoch configs."""

import copy
import json
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_process.work_manifest import make_windows, score_json_path


DATA = ROOT / "data/ASAP_processed"
TAG = "ASAP_MUSICAL51_MEDIAN"
SPLIT_NAME = "asap_train_valid_heldout_12works_median50_v1"
SPLIT_SPEC = ROOT / "INSPIRE2/configs/asap_valid_heldout_12works.json"
SUMMARY = ROOT / "results/outer_median_overfit_8ep/fixed_split_summary.json"
CONFIG_DIR = ROOT / "configs/inr_epr/median_overfit_8ep"

MODEL_CONFIGS = {
    "dinr_no_coord": ROOT / "configs/inr_epr/dinr_slot_coord_matrix_20261026/slot_sum__no_coord.json",
    "dinr_coord": ROOT / "configs/inr_epr/dinr_slot_coord_matrix_20261026/slot_sum__coord_all.json",
    "cinr": ROOT / "configs/inr_epr/cinr__dist_ln_k1.json",
}


def torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def prepare_sidecars():
    sources = sorted(DATA.rglob("score_ASAP_refined.ASAP_MUSICAL51.pt"))
    if not sources:
        raise FileNotFoundError("No outer ASAP_MUSICAL51 sidecars found")
    written = 0
    for musical_path in sources:
        median_path = musical_path.with_name("score_ASAP_refined.ASAP_GROUPED_MEDIAN.pt")
        if not median_path.exists():
            continue
        musical = torch_load(musical_path)
        median = torch_load(median_path)
        median_by_source = {
            perf.get("performance_source"): perf
            for perf in median.get("performances", [])
            if perf.get("performance_source") is not None
        }
        merged_performances = []
        for original in musical.get("performances", []):
            source = original.get("performance_source")
            normalized = median_by_source.get(source)
            if normalized is None:
                raise KeyError(f"No median labels for performance {source} in {median_path}")
            if len(original.get("label_shared_raw", [])) != len(normalized.get("label_shared_raw", [])):
                raise ValueError(f"Label length mismatch for {source} in {musical_path}")
            perf = dict(original)
            shared = [list(row) for row in normalized["label_shared_raw"]]
            grouped = normalized.get("grouped_ioi_raw")
            if grouped is None or len(grouped) != len(shared):
                raise ValueError(f"Invalid grouped IOI labels for {source} in {median_path}")
            # The outer EPR label builder reads the IOI field in label_shared_raw.
            # Store the grouped target there while retaining chord IOIs as provided
            # by the canonical grouped-median sidecar.
            for row, value in zip(shared, grouped):
                row[0] = float(value)
            perf["label_shared_raw"] = shared
            perf["label_pedal4_raw"] = normalized["label_pedal4_raw"]
            perf["grouped_ioi_raw"] = [float(value) for value in grouped]
            perf["global_timing_scale"] = normalized.get("global_timing_scale")
            merged_performances.append(perf)
        payload = dict(musical)
        payload["performances"] = merged_performances
        payload["performances_by_source"] = {
            perf["performance_source"]: perf for perf in merged_performances
        }
        payload["performance_time_normalization"] = "grouped_median_log_ratio"
        payload["median_sidecar_source"] = str(median_path.relative_to(ROOT))
        out_path = musical_path.with_name(f"score_ASAP_refined.{TAG}.pt")
        torch.save(payload, out_path)
        written += 1
    if written == 0:
        raise RuntimeError("No matched outer/median sidecar pairs were written")
    print(f"Prepared {written} median+musical51 sidecars tagged {TAG}")


def prepare_split():
    spec = json.loads(SPLIT_SPEC.read_text(encoding="utf-8"))
    valid_rel = {row["work_path"] for row in spec["valid_work_paths"]}
    metadata = pd.read_csv(DATA / "metadata.generated_json.csv")
    data = metadata[
        metadata["tier_a"].fillna(False).astype(bool)
        & (metadata["split"] == "train")
        & metadata["refined_score_midi_path"].notna()
        & metadata["refined_performance_midi_path"].notna()
        & metadata["refined_alignment_path"].notna()
        & (metadata["performance_dataset"].fillna("").astype(str) == "ASAP")
    ]
    rows = []
    valid_found = set()
    for score_rel, group in data.groupby("refined_score_midi_path", sort=True):
        path = score_json_path(DATA, score_rel)
        rel = path.parent.relative_to(DATA).as_posix()
        is_valid = rel in valid_rel
        if is_valid:
            valid_found.add(rel)
        note_count = int(float(group["refined_score_note_count"].iloc[0]))
        windows = make_windows(note_count, spec["window_notes"], spec["window_overlap_ratio"], 64)
        valid_windows = windows if is_valid else []
        rows.append({
            "path": str(path.resolve()),
            "work_path": rel,
            "valid_window": None,
            "valid_windows": valid_windows,
            "train_window_count": 0 if is_valid else len(windows),
            "valid_window_count": len(valid_windows),
            "canonical_window_count": len(windows),
        })
    missing = valid_rel - valid_found
    if missing:
        raise ValueError(f"Held-out works not found in outer ASAP train metadata: {sorted(missing)}")
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(json.dumps({
        "scheme_name": SPLIT_NAME,
        "description": "Fixed work-level validation; all windows from 12 held-out works are valid.",
        "window_notes": spec["window_notes"],
        "window_overlap_ratio": spec["window_overlap_ratio"],
        "valid_work_paths": sorted(valid_rel),
        "work_summaries": rows,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote split summary: {SUMMARY}; valid works={len(valid_found)}, train works={len(rows)-len(valid_found)}")


def prepare_configs():
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    for name, source in MODEL_CONFIGS.items():
        cfg = copy.deepcopy(json.loads(source.read_text(encoding="utf-8")))
        cfg.update({
            "overlap_ratio": 0.5,
            "num_train_epochs": 8.0,
            "max_train_epochs": 8.0,
            "adapt_num_train_epochs": 0,
            "prepared_sidecar_tag": TAG,
            "fixed_window_split_scheme": SPLIT_NAME,
            "fixed_window_split_summary_path": str(SUMMARY),
            "fixed_window_base_split": "train",
            "fixed_window_train_split_name": "train",
            "fixed_window_eval_split_name": "valid",
            "eval_split": "valid",
            "force_epoch_eval": True,
            "output_dir": str(ROOT / "results/outer_median_overfit_8ep" / name / "training"),
            "logging_dir": str(ROOT / "results/outer_median_overfit_8ep" / name / "tf-logs"),
            "run_name": f"median50_heldout12_{name}_8ep",
            "overwrite_output_dir": True,
            "resume_trainer_state": False,
            "load_best_model_at_end": False,
            "save_strategy": "epoch",
            "save_total_limit": 10,
            "logging_steps": 20,
            "early_stopping_patience": 0,
            "max_train_works": None,
            "max_eval_works": None,
        })
        out = CONFIG_DIR / f"{name}.json"
        out.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
        print(out)


if __name__ == "__main__":
    prepare_sidecars()
    prepare_split()
    prepare_configs()
