#!/usr/bin/env python3
"""Audit the full-data slot-sum experiment and persist its validation history."""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

STAGES = (
    ("pianocoreA_pretrain", "stage1", "pianocoreA_slot_sum_pretrain4_gpu01_global64", 4),
    ("asap_finetune", "stage2", "asap_slot_sum_finetune8_gpu01_global64", 8),
)
ANCHOR = "backup/repo_cleanup_20260804/results/external_eval_lengthnorm_20260724/human_leave_one_score_span_eval_binary.json"


def save_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temp.replace(path)


def preflight(run):
    import torch
    from src.data_process.work_manifest import build_work_manifest

    preparation = json.loads((run / "data_preparation_summary.json").read_text())
    details = [json.loads(line) for line in Path(preparation["details"]).read_text().splitlines() if line.strip()]
    assert len(details) == 1780 and not any("error" in item for item in details)
    prepared = {str(Path(item["path"]).resolve()): item for item in details}
    assert len(prepared) == 1780
    for item in details:
        assert Path(item["sidecar"]).is_file(), item
    availability = {}
    output = {"metadata_performances": preparation["metadata_performances"],
              "usable_performances": preparation["performances"],
              "excluded_performances": preparation["excluded_performances"],
              "annotation_fallback_works": preparation["annotation_fallback_works"],
              "zero_annotation_works": preparation["zero_annotation_works"],
              "validation_tf_hr_protocol": "conditional analytic/histogram W1 on one fixed held-out window per eligible work; raw training timing targets; diagnostic only",
              "final_test_protocol": "AR sampling; prediction and GT independently score-onset-span normalized; binary pedal; 3 samples per score; seed 42"}
    configs = ("pianocoreA_slot_sum_pretrain4.json", "asap_slot_sum_finetune8.json")
    for label, filename in zip(("pianocoreA", "asap"), configs):
        config = json.loads((run / "runtime_configs" / filename).read_text())
        assert config["slot_fusion"] == "sum" and config["slot_version"] == "slot6"
        assert config["per_device_train_batch_size"] * 2 * config["gradient_accumulation_steps"] == 64
        output[label] = {}
        for split in (("train", "valid", "test") if label == "asap" else ("train", "valid")):
            data_split = "test" if split == "test" else "train"
            manifest = build_work_manifest(
                config["metadata_path"], config["refined_dir"], split=data_split,
                block_notes=config["block_notes"], overlap_ratio=config["overlap_ratio"], min_notes=config["min_notes"],
                performance_dataset=config.get(f"{'eval' if split == 'valid' else 'train'}_performance_dataset"),
                include_all_performance_dataset=config.get("eval_include_all_performance_dataset") if split == "valid" else None,
                max_non_asap_performances_per_work=config.get("max_eval_non_asap_performances_per_work") if split == "valid" else None,
                selection_seed=config["seed"], prepared_sidecar_tag=config["prepared_sidecar_tag"],
                window_split_scheme=config["fixed_window_split_scheme"] if split != "test" else None,
                window_split_name=split if split != "test" else None,
                window_split_summary_path=config["fixed_window_split_summary_path"],
            )
            examples = performances = windows = eligible = usable_works = 0
            for item in manifest:
                key = str(Path(item["path"]).resolve())
                sources = set(item["selected_performance_sources"])
                if label == "pianocoreA":
                    audit = prepared[key]
                    assert audit["notes"] == item["note_count"], (key, audit["notes"], item["note_count"])
                    sources -= {row["source"] for row in audit.get("excluded_performances", [])}
                else:
                    if key not in availability:
                        sidecar = Path(item["path"]).with_suffix(f".{config['prepared_sidecar_tag']}.pt")
                        payload = torch.load(sidecar, map_location="cpu", weights_only=False)
                        stat = Path(key).stat()
                        assert payload["_source_identity"] == {"path": key, "mtime_ns": stat.st_mtime_ns, "size": stat.st_size}, key
                        assert len(payload["score"]["pitch"]) == item["note_count"], key
                        assert payload["meta"]["score_feature_layout"] == "musical51_from_compact_4slot", key
                        assert all(len(row) == 9 for row in payload["score"]["score_feature"]), key
                        if split == "test":
                            assert sum(row[0] for row in payload["score"]["score_raw"][1:]) > 0, key
                        availability[key] = {p["performance_source"] for p in payload["performances"] if p.get("split", data_split) == data_split}
                        del payload
                    if split == "test" and sources - availability[key]:
                        raise ValueError(f"Test metadata references missing from sidecar: {key}: {sources - availability[key]}")
                    sources &= availability[key]
                if not sources:
                    continue
                usable_works += 1
                eligible += int(len(sources) >= 2)
                performances += len(sources)
                windows += len(item["windows"])
                examples += len(sources) * len(item["windows"])
            assert examples > 0 and eligible > 0
            output[label][split] = {"works": usable_works, "performances": performances,
                                    "windows": windows, "examples": examples, "hr_eligible_works": eligible}
            if split == "test":
                assert usable_works == 19, usable_works
    save_json(run / "data_preflight.json", output)
    print(json.dumps(output, ensure_ascii=False), flush=True)


def summarize_validation(run, require_complete=False):
    rows = []
    for label, stage, name, epochs in STAGES:
        output = run / stage / "training" / name
        for epoch in range(1, epochs + 1):
            state_path = output / f"epoch-{epoch}" / "trainer_state.json"
            if not state_path.exists():
                if require_complete:
                    raise ValueError(f"Missing {state_path}")
                continue
            state = json.loads(state_path.read_text())
            candidates = [record for record in state.get("log_history", [])
                          if "eval_loss" in record and abs(float(record.get("epoch", -1)) - epoch) < .001]
            if not candidates:
                raise ValueError(f"Missing valid loss: {label}, epoch {epoch}")
            record = candidates[-1]
            row = {"stage": label, "epoch": epoch, "step": state["global_step"],
                   "valid_loss": float(record["eval_loss"]), "valid_tf_PN_HR": None,
                   "valid_tf_PP_HR": None, "valid_tf_loss": None, "valid_tf_works": None,
                   "checkpoint": str((output / f"epoch-{epoch}").resolve()), "tf_hr_json": None}
            hr_path = run / "validation_hr" / f"{label}_epoch-{epoch}.json"
            if hr_path.exists():
                payload = json.loads(hr_path.read_text())
                selected = next(iter(payload["m"].values()))
                metrics = selected["metrics"]["0"]
                row.update(valid_tf_PN_HR=float(metrics["pn_hr"]), valid_tf_PP_HR=float(metrics["pp_hr"]),
                           valid_tf_loss=float(metrics["tf_loss"]), valid_tf_works=len(selected["selected_windows"]),
                           tf_hr_json=str(hr_path.resolve()))
            elif require_complete:
                raise ValueError(f"Missing {hr_path}")
            for value in row.values():
                if isinstance(value, float) and not math.isfinite(value):
                    raise ValueError(f"Nonfinite validation metric: {row}")
            rows.append(row)
    if rows:
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        target = run / "validation_metrics_by_epoch.csv"
        temporary = target.with_suffix(".csv.tmp")
        temporary.write_text(buffer.getvalue())
        temporary.replace(target)
    save_json(run / "validation_metrics_by_epoch.json", {
        "tf_hr_protocol": "conditional analytic/histogram W1, raw training targets; separate from final score-span-normalized AR test HR",
        "epochs": rows,
    })
    print(json.dumps({"validation_epochs_saved": len(rows), "with_tf_hr": sum(r["valid_tf_PP_HR"] is not None for r in rows)}))


def audit_normalization(manifest_path):
    from src.evaluate.normalize_prediction_manifest_score_span import onset_span_ms

    manifest = json.loads(manifest_path.read_text())
    assert manifest["prediction_time_normalization"] == "score_onset_span"
    assert len(manifest["items"]) == 19 and manifest["num_samples"] == 3
    rows = []
    for item in manifest["items"]:
        assert item["eval_gt_time_normalization"] == "score_onset_span", item["score_source"]
        assert len(item["prediction_paths"]) == 3 and item["ground_truth_paths"]
        score_span = onset_span_ms(Path(item["score_midi"]))
        for kind, paths in (("prediction", item["prediction_paths"]), ("ground_truth", item["ground_truth_paths"])):
            for path in paths:
                span = onset_span_ms(Path(path))
                error = abs(span - score_span)
                if error > max(2., score_span * 2e-5):
                    raise ValueError(f"Score-span mismatch: {path}: {span} vs {score_span}")
                rows.append({"score_source": item["score_source"], "kind": kind, "path": path,
                             "onset_span_ms": span, "score_span_ms": score_span, "absolute_error_ms": error})
    save_json(manifest_path.parent / "score_span_normalization_audit.json", {
        "prediction_normalized": True, "ground_truth_normalized": True, "score_count": 19,
        "prediction_count": sum(row["kind"] == "prediction" for row in rows),
        "ground_truth_count": sum(row["kind"] == "ground_truth" for row in rows), "details": rows,
    })


def summarize_final(run):
    summarize_validation(run, require_complete=True)
    anchor = json.loads((ROOT / ANCHOR).read_text())
    features = ("ioi", "duration", "velocity", "pedal")
    for label, stage, name, _ in STAGES:
        test_stage = "pretrained" if stage == "stage1" else "finetuned"
        folder = run / "final_asap_test" / test_stage
        metrics = json.loads((folder / "score_span_metrics.json").read_text())
        norm = json.loads((folder / "sampling/score_span_normalization_audit.json").read_text())
        assert metrics["num_scores"] == 19 and metrics["num_samples"] == 3
        assert metrics["pedal_metric_support"] == anchor["pedal_metric_support"] == "binary_0_1"
        aggregate = metrics["aggregate"]
        hr = {f"{prefix.upper()}_HR": sum(aggregate[f"{prefix}_wass"][f"{f}_wass"] / anchor["aggregate"][f"{prefix}_wass"][f"{f}_wass"] for f in features) / len(features)
              for prefix in ("pn", "pp")}
        assert all(math.isfinite(value) for value in hr.values()), hr
        hr.update(stage=label, checkpoint=str((run / stage / "training" / name / "checkpoint-best").resolve()),
                  checkpoint_selection="lowest epoch validation loss within the stage",
                  num_scores=19, samples_per_score=3, sampling_seed=42,
                  num_ground_truth=norm["ground_truth_count"], normalization="score_onset_span",
                  prediction_time_normalization="score_onset_span", ground_truth_time_normalization="score_onset_span",
                  tempo_map_source="score/performance raw timing serialized at 120 BPM; each prediction/GT onset span matched to the score",
                  pedal_support=metrics["pedal_metric_support"], human_anchor_source=str((ROOT / ANCHOR).resolve()),
                  human_anchor=anchor["aggregate"], metrics=aggregate)
        save_json(folder / "score_span_hr.json", hr)
    timing = {}
    for _, stage, _, _ in STAGES:
        start, end = map(int, (run / f"{stage}_timing.csv").read_text().strip().split(","))
        timing[stage] = {"start_unix": start, "end_unix": end, "wall_seconds": end - start,
                         "training_gpu_hours_including_epoch_validation": 2 * (end - start) / 3600}
    save_json(run / "training_timing.json", timing)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--summarize-validation", action="store_true")
    parser.add_argument("--audit-normalization", type=Path)
    parser.add_argument("--final", action="store_true")
    args = parser.parse_args()
    if args.preflight:
        preflight(args.run_dir)
    if args.summarize_validation:
        summarize_validation(args.run_dir)
    if args.audit_normalization:
        audit_normalization(args.audit_normalization)
    if args.final:
        summarize_final(args.run_dir)


if __name__ == "__main__":
    main()
