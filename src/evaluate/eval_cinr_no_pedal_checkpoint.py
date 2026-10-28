#!/usr/bin/env python3
"""Evaluate a three-attribute CINR epoch with fixed TF/AR sampling protocol."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import subprocess
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data_process.work_manifest import build_work_manifest
from src.evaluate.eval_inr_window_rollout_hr import make_dataset
from src.evaluate.evaluate_inr_saved_midis import cached_note_arrays
from scipy.stats import wasserstein_distance
from src.evaluate.plot_sampling_matrix_humanrel import HUMAN_PN, HUMAN_PP
from src.inference.infer_inr_testset import (
    build_windows, continuation_window_predictions, load_config, load_model,
)
from src.model.integrated_pianoformer import _materialize_epr_prediction, _target7_to_raw7
from src.train.train_inr import NodeSFTDataCollator
from src.utils.inr_midi import note_features_to_midi

FEATURES = ("ioi", "duration", "velocity")
ANCHOR_SOURCE = "backup/repo_cleanup_20260804/results/external_eval_lengthnorm_20260724/human_leave_one_score_span_eval_binary.json"


def dump_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_manifest(config, split):
    fixed = config.get("fixed_window_split_scheme") if split == "valid" else None
    data_split = config.get("fixed_window_base_split", "train") if fixed else split
    manifest = build_work_manifest(
        metadata_path=config["metadata_path"], refined_dir=config["refined_dir"],
        split=data_split, block_notes=config["block_notes"],
        overlap_ratio=config["overlap_ratio"], min_notes=config["min_notes"],
        performance_dataset="ASAP", selection_seed=int(config.get("seed", 42)),
        skip_work_paths=config.get("skip_work_paths"),
        prepared_sidecar_tag=config.get("prepared_sidecar_tag"),
        window_split_scheme=fixed,
        window_split_name=config.get("fixed_window_eval_split_name", "valid") if fixed else None,
        window_split_summary_path=config.get("fixed_window_split_summary_path"),
    )
    return manifest, data_split


def normalized_midi(path, pitch, raw, score_raw):
    """Apply the paper score-onset-span protocol before fixed-120 serialization."""
    raw = np.asarray(raw, dtype=np.float64).copy()
    score_raw = np.asarray(score_raw, dtype=np.float64)
    if raw.shape != (len(pitch), 3) or not np.isfinite(raw).all():
        raise ValueError(f"Invalid no-pedal rows for {path}: {raw.shape}")
    score_span = float(score_raw[1:, 0].sum())
    perf_span = float(raw[1:, 0].sum())
    if score_span <= 0 or perf_span <= 0:
        raise ValueError(f"Invalid onset spans: score={score_span}, performance={perf_span}")
    scale = score_span / perf_span
    raw[:, :2] *= scale
    raw[0, 0] = 0.0  # Local origin, including held-out validation excerpts.
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    midi = note_features_to_midi(pitch, raw.tolist(), target_ticks_per_beat=500,
                                 target_tempo=120, normalized=False)
    if any(cc.number == 64 for ins in midi.instruments for cc in ins.control_changes):
        raise AssertionError("No-pedal experiment emitted CC64")
    if sum(len(ins.notes) for ins in midi.instruments) != len(pitch):
        raise AssertionError("MIDI note count mismatch")
    midi.dump(str(path))
    return {"scale": scale, "score_onset_span_ms": score_span,
            "performance_onset_span_ms_before": perf_span, "midi": str(path.resolve())}


def metric_set(prediction, references):
    # Exactly the saved-MIDI paper evaluator for the three retained features.
    # With one prediction, W1(delta_x, empirical_GT) = mean(abs(x-GT)).
    # Vectorize that identity to avoid evaluating unused log/pedal metrics.
    predicted = cached_note_arrays(str(prediction))
    ground_truth = [cached_note_arrays(str(path)) for path in references]
    out = {"pn": {}, "pp": {}}
    for feature in FEATURES:
        pred = np.asarray(predicted[feature], dtype=np.float64)
        gt = [np.asarray(ref[feature], dtype=np.float64) for ref in ground_truth]
        usable = min(map(len, [pred, *gt]))
        if usable != len(pred) or any(len(g) != len(pred) for g in gt):
            raise ValueError("Aligned MIDI note counts differ")
        if not np.isfinite(pred).all() or not all(np.isfinite(g).all() for g in gt):
            raise ValueError("Nonfinite MIDI features")
        out["pn"][f"{feature}_wass"] = float(np.abs(np.stack(gt)-pred[None, :]).mean())
        out["pp"][f"{feature}_wass"] = float(wasserstein_distance(pred, np.concatenate(gt)))
    for name, anchor in (("pn", HUMAN_PN), ("pp", HUMAN_PP)):
        values = out[name]
        values["hr"] = float(np.mean([values[f"{f}_wass"] / anchor[f"{f}_wass"] for f in FEATURES]))
        if not np.isfinite(list(values.values())).all():
            raise ValueError(f"Nonfinite metrics: {values}")
    return out


def teacher_forced(model, dataset, prepared, perf, windows, collator, device):
    chunks = []
    previous_end = None
    for index, (start, end) in enumerate(windows):
        example = dataset._sample_for_performance(prepared, perf, start, end, index)
        batch = collator([example])
        keys = ("pitch_ids", "continuous", "score_shared_raw", "labels_continuous",
                "attention_mask", "label_mask", "label_valid_mask",
                "decoder_feedback_continuous", "labels_epr_bins")
        batch = {k: batch[k].to(device) for k in keys if k in batch}
        outputs = model(**batch, continuous_sampling_strategy="sample")
        prediction = _materialize_epr_prediction(
            model.config, outputs.logits, sampling_strategy="sample",
            score_shared_raw=batch["score_shared_raw"],
        )[0].detach().float().cpu()
        overlap = max(0, previous_end - start) if previous_end is not None else 0
        chunks.append(prediction[overlap:])
        previous_end = end
    return torch.cat(chunks)


def evaluate_split(model, config, split, output, args, collator, device):
    manifest, dataset_split = make_manifest(config, split)
    if split == "test" and not args.max_works and len(manifest) != 19:
        raise ValueError(f"Expected all 19 test works, got {len(manifest)}")
    if args.max_works:
        manifest = manifest[:args.max_works]
    manifest = manifest[args.shard_index::args.num_shards]
    data_config = dict(config, validation_multi_perf_group_size=1,
                       validation_multi_perf_min_group_size=1, node_cache_size=1)
    dataset = make_dataset(data_config, manifest, dataset_split)
    rows = []
    start_time = time.monotonic()
    for work_index, item in enumerate(manifest):
        prepared = dataset._load_or_prepare_work(item["path"])
        performances = sorted(dataset._selected_performances(prepared, item),
                              key=lambda p: str(p.get("performance_source", p.get("performance_id"))))
        if not performances:
            raise ValueError(f"No references for {item['path']}")
        identity = str(item.get("score_source") or item["path"])
        digest = int(hashlib.sha256(identity.encode()).hexdigest()[:8], 16)
        work_seed = (args.seed + digest) % (2**31)
        prefix_perf = performances[work_seed % len(performances)]
        score = prepared["score"]
        pitch = score["pitch"]
        score_raw = [row[:3] for row in score["score_raw"]]
        continuous = dataset._score_input_rows(prepared)
        if split == "test":
            windows = build_windows(len(pitch), config["block_notes"], config["overlap_ratio"])
            groups = [windows]
        else:
            # Score each actual fixed held-out window, never an arbitrary subset.
            groups = [[tuple(w)] for w in item["windows"]]
        work_rows = []
        for group_index, windows in enumerate(groups):
            if args.max_notes:
                start, end = windows[0]
                windows = [(start, min(end, start + args.max_notes))]
            start, end = windows[0][0], windows[-1][1]
            local_pitch, local_score = pitch[start:end], score_raw[start:end]
            directory = output / split / f"score_{work_index:02d}" / f"window_{group_index:02d}"
            directory.mkdir(parents=True, exist_ok=True)
            references, normalization = [], {"gt": []}
            for ref_index, perf in enumerate(performances):
                raw = perf.get("label_shared_raw", perf.get("label_raw"))
                if raw is None:
                    target, _ = dataset._performance_labels(prepared, perf)
                    raw = _target7_to_raw7(torch.tensor(score_raw), torch.tensor(target),
                                          config=model.config).cpu().tolist()
                path = directory / f"gt_{ref_index:03d}.mid"
                norm = normalized_midi(path, local_pitch, np.asarray(raw)[start:end, :3], local_score)
                norm["performance_source"] = str(perf.get("performance_source", perf.get("performance_id")))
                normalization["gt"].append(norm)
                references.append(str(path.resolve()))
            seed_all((work_seed + group_index) % (2**31))
            tf_start = time.monotonic()
            tf = teacher_forced(model, dataset, prepared, prefix_perf, windows, collator, device)
            raw_tf = _target7_to_raw7(torch.tensor(local_score), tf, config=model.config).cpu().numpy()
            tf_path = directory / "tf_sample_0.mid"
            normalization["tf"] = normalized_midi(tf_path, local_pitch, raw_tf, local_score)
            entry = {"work": identity, "note_range": [start, end], "notes": end-start,
                     "reference_count": len(performances), "sampling_seed": (work_seed+group_index) % (2**31),
                     "tf_prefix_reference": str(prefix_perf.get("performance_source", prefix_perf.get("performance_id"))),
                     "tf": metric_set(tf_path.resolve(), references),
                     "tf_seconds": time.monotonic()-tf_start}
            targets = {"tf_target": tf.numpy()}
            if split == "test":
                seed_all((work_seed + group_index) % (2**31))
                ar_start = time.monotonic()
                local_windows = [(s-start, e-start) for s, e in windows]
                ar = continuation_window_predictions(
                    model, local_pitch, continuous[start:end], local_score, score,
                    local_windows, config["pitch_pad_id"], device,
                    sampling_strategy="sample", drop_ratio=0.0,
                    oracle_prefix_targets=None, oracle_style_targets=None,
                )
                raw_ar = _target7_to_raw7(torch.tensor(local_score), ar, config=model.config).cpu().numpy()
                ar_path = directory / "ar_sample_0.mid"
                normalization["ar"] = normalized_midi(ar_path, local_pitch, raw_ar, local_score)
                entry.update(ar=metric_set(ar_path.resolve(), references),
                             ar_seconds=time.monotonic()-ar_start)
                targets["ar_target"] = ar.numpy()
            np.savez_compressed(directory / "targets.npz", **targets)
            entry["normalization"] = normalization
            dump_json(directory / "metrics.json", entry)
            work_rows.append(entry)
        # Work-wise equal weighting, averaging multiple validation windows within work.
        work_result = {"work": identity, "windows": work_rows}
        for mode in (("tf", "ar") if split == "test" else ("tf",)):
            work_result[mode] = {
                metric: {key: float(np.mean([row[mode][metric][key] for row in work_rows]))
                         for key in (*[f"{f}_wass" for f in FEATURES], "hr")}
                for metric in ("pn", "pp")
            }
        rows.append(work_result)
        dump_json(output / f"{split}_progress.json", {"works": rows, "expected_works": len(manifest)})
        print(json.dumps({"event": "score_evaluated", "split": split,
                          "completed": len(rows), "total": len(manifest),
                          "tf_hr": work_result["tf"]["pn"]["hr"],
                          "ar_hr": work_result.get("ar", {}).get("pn", {}).get("hr")}), flush=True)
    if not rows:
        raise ValueError(f"Empty {split} evaluation")
    aggregate = {}
    for mode in (("tf", "ar") if split == "test" else ("tf",)):
        aggregate[mode] = {metric: {key: float(np.mean([r[mode][metric][key] for r in rows]))
                                  for key in (*[f"{f}_wass" for f in FEATURES], "hr")}
                           for metric in ("pn", "pp")}
    return {"score_count": len(rows), "window_count": sum(len(r["windows"]) for r in rows),
            "aggregate": aggregate, "works": rows, "seconds": time.monotonic()-start_time}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-works", type=int, default=None, help="Smoke test only")
    parser.add_argument("--max-notes", type=int, default=None, help="Smoke test only")
    parser.add_argument("--workers", type=int, default=8, help="Independent score shards, all on GPU2")
    parser.add_argument("--num-shards", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--shard-index", type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if str(args.device).startswith("cuda") and os.environ.get("CUDA_VISIBLE_DEVICES") != "2":
        raise RuntimeError("This experiment must run with CUDA_VISIBLE_DEVICES=2")
    config = load_config(args.config, str(args.checkpoint))
    if config.get("pedal_representation") != "none" or config["output_continuous_dim"] != 3:
        raise ValueError("Expected the three-attribute no-pedal configuration")
    if config.get("use_style_tokens", False):
        raise ValueError("This protocol requires the paper configuration without style tokens")
    if args.workers < 1 or args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Invalid worker/shard configuration")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    workers = args.workers
    if args.num_shards == 1 and workers > 1:
        workers = min(workers, args.max_works or 19)
        if str(args.device).startswith("cuda"):
            free_bytes, _ = torch.cuda.mem_get_info()
            workers = max(1, min(workers, int(free_bytes // (2.5 * 1024**3))))
        if workers > 1:
            def run_shard(index):
                directory = output / f"shard_{index:02d}"
                directory.mkdir(parents=True, exist_ok=True)
                command = [sys.executable, str(Path(__file__).resolve()),
                           "--config", str(args.config), "--checkpoint", str(args.checkpoint),
                           "--output-dir", str(directory), "--device", args.device,
                           "--seed", str(args.seed), "--workers", "1",
                           "--num-shards", str(workers), "--shard-index", str(index)]
                for flag, value in (("--max-works", args.max_works), ("--max-notes", args.max_notes)):
                    if value:
                        command.extend([flag, str(value)])
                with (directory / "worker.log").open("a") as log:
                    subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
                result = json.loads((directory / "metrics.json").read_text())
                print(json.dumps({"event": "shard_evaluated", "shard": index,
                                  "workers": workers, "test_scores": result["test"]["score_count"]}), flush=True)
                return result
            print(json.dumps({"event": "parallel_evaluation", "workers": workers,
                              "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                              "checkpoint": str(args.checkpoint)}), flush=True)
            with ThreadPoolExecutor(max_workers=workers) as pool:
                shards = list(pool.map(run_shard, range(workers)))
            def merge(split):
                rows = [row for shard in shards for row in shard[split]["works"]]
                rows.sort(key=lambda row: row["work"])
                if len({r["work"] for r in rows}) != len(rows):
                    raise ValueError("Duplicate works across evaluation shards")
                aggregate = {}
                for mode in (("tf", "ar") if split == "test" else ("tf",)):
                    aggregate[mode] = {
                        metric: {key: float(np.mean([r[mode][metric][key] for r in rows]))
                                 for key in (*[f"{f}_wass" for f in FEATURES], "hr")}
                        for metric in ("pn", "pp")
                    }
                return {"score_count": len(rows), "window_count": sum(len(r["windows"]) for r in rows),
                        "aggregate": aggregate, "works": rows,
                        "seconds": sum(s[split]["seconds"] for s in shards)}
            valid, test = merge("valid"), merge("test")
            if not args.max_works and test["score_count"] != 19:
                raise ValueError("Parallel evaluation did not cover all 19 test works")
            save_results(args, valid, test, time.monotonic()-start, workers)
            return
    seed_all(args.seed)
    torch.set_num_threads(1 if args.num_shards > 1 else 4)
    device = torch.device(args.device)
    model = load_model(config, device)
    if model.config.output_continuous_dim != 3:
        raise AssertionError("Model output is not three-dimensional")
    collator_keys = ("pitch_pad_id", "epr_timing_target", "tail_mask_enabled", "tail_mask_tf_clamp",
                     "tail_mask_ioi_min", "tail_mask_ioi_max", "tail_mask_duration_min", "tail_mask_duration_max")
    collator = NodeSFTDataCollator(**{key: config[key] for key in collator_keys if key in config})
    with torch.inference_mode():
        valid = evaluate_split(model, config, "valid", output, args, collator, device)
        test = evaluate_split(model, config, "test", output, args, collator, device)
    save_results(args, valid, test, time.monotonic()-start, 1)


def save_results(args, valid, test, elapsed, workers):
    payload = {
        "checkpoint": str(args.checkpoint.resolve()), "sampling_seed": args.seed,
        "smoke_test": bool(args.max_works or args.max_notes),
        "feature_set": list(FEATURES), "primary_hr": "PN",
        "valid_tf_hr": valid["aggregate"]["tf"]["pn"]["hr"],
        "test_tf_hr": test["aggregate"]["tf"]["pn"]["hr"],
        "test_ar_hr": test["aggregate"]["ar"]["pn"]["hr"],
        "valid_tf_pp_hr": valid["aggregate"]["tf"]["pp"]["hr"],
        "test_tf_pp_hr": test["aggregate"]["tf"]["pp"]["hr"],
        "test_ar_pp_hr": test["aggregate"]["ar"]["pp"]["hr"],
        "protocol": {"timing_normalization": "score_onset_span",
                     "prediction_normalized": True, "gt_normalized": True,
                     "tempo_map_source": "fixed 120 BPM, 500 ticks/beat after raw-ms normalization",
                     "pedal_support": "none; no pedal targets, decoder inputs, head, loss, CC64, or HR term",
                     "samples_per_score_per_mode": 1,
                     "tf_conditioning": "one fixed reference prefix per score; causal decoder, all references used as metric GT",
                     "tf_reference_selection": "sorted references[(seed + sha256(score)[:8]) % 2**31 % reference_count]",
                     "ar_conditioning": "generated prefix only; no GT prefix; overlap continuation drop_ratio=0",
                     "valid_windows": "all fixed held-out validation windows",
                     "test_windows": "full score, same overlap and window merge for TF and AR",
                     "hr_definition": "mean of IOI/duration/velocity Wasserstein divided by matching human anchors; lower is better",
                     "aggregation": "equal mean over works; validation windows averaged within work",
                     "human_anchor_source": ANCHOR_SOURCE,
                     "human_pn": {f"{f}_wass": HUMAN_PN[f"{f}_wass"] for f in FEATURES},
                     "human_pp": {f"{f}_wass": HUMAN_PP[f"{f}_wass"] for f in FEATURES}},
        "valid": valid, "test": test, "evaluation_seconds": elapsed,
        "parallel_workers": workers, "num_shards": args.num_shards, "shard_index": args.shard_index,
    }
    dump_json(args.output_dir.resolve() / "metrics.json", payload)
    print(json.dumps({k: payload[k] for k in ("valid_tf_hr", "test_tf_hr", "test_ar_hr", "evaluation_seconds")}), flush=True)


if __name__ == "__main__":
    main()
