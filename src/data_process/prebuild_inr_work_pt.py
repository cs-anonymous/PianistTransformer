#!/usr/bin/env python
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data_process.work_manifest import build_work_manifest
from src.data_process.sidecar_builder import build_sidecar_for_work
from src.train.train_inr import (
    PianoCoReNodeSFTDataset,
    infer_input_feature_mode,
)


def make_manifest(config, split, performance_dataset_override=None):
    is_train = split == "train"
    if performance_dataset_override == "ALL":
        performance_dataset = None
    elif performance_dataset_override is not None:
        performance_dataset = performance_dataset_override
    else:
        performance_dataset = config.get("train_performance_dataset" if is_train else "eval_performance_dataset")
    return build_work_manifest(
        metadata_path=config["metadata_path"],
        refined_dir=config["refined_dir"],
        split=split,
        block_notes=config["block_notes"],
        overlap_ratio=config["overlap_ratio"],
        min_notes=config["min_notes"],
        max_works=config.get("max_train_works" if is_train else "max_eval_works"),
        include_all_performance_dataset=None if is_train else config.get("eval_include_all_performance_dataset"),
        max_non_asap_performances_per_work=None if is_train else config.get("max_eval_non_asap_performances_per_work"),
        selection_seed=config.get("seed", 42),
        skip_work_paths=config.get("skip_work_paths"),
        performance_dataset=performance_dataset,
        exclude_performance_dataset=config.get("train_exclude_performance_dataset" if is_train else "eval_exclude_performance_dataset"),
    )


def unique_work_paths(manifest):
    seen = set()
    paths = []
    for item in manifest:
        path = str(Path(item["path"]))
        if path not in seen:
            seen.add(path)
            paths.append(path)
    return paths


def work_selected_sources(manifest):
    out = {}
    for item in manifest:
        path = str(Path(item["path"]))
        sources = item.get("selected_performance_sources")
        if sources is None:
            out[path] = None
            continue
        if out.get(path) is None and path in out:
            continue
        out.setdefault(path, set()).update(sources)
    return out


def build_dataset(config, manifest, split):
    return PianoCoReNodeSFTDataset(
        manifest,
        split=split,
        task_type=config.get("task_type", "epr"),
        input_feature_mode=infer_input_feature_mode(config),
        shuffle=False,
        seed=config.get("seed", 42),
        max_performances_per_work=None,
        max_windows_per_work=None,
        cache_size=max(1, int(config.get("node_cache_size", 8) or 8)),
        timing_normalization=config.get("timing_input_normalization", "scaled_log_5000_s10"),
        max_time_ms=config.get("max_time_ms", 10000.0),
        epr_timing_bins=config.get("epr_timing_bins", 5000),
        epr_value_bins=config.get("epr_value_bins", 128),
        pedal_representation=config.get("pedal_representation", "continuous_4"),
        musical_feature_mode=config.get(
            "musical_feature_mode",
            "continuous" if str(config.get("task_type", "epr")).lower() == "csr" else "categorical",
        ),
        epr_timing_target=config.get("epr_timing_target", "absolute"),
        use_timing_scale_bit=config.get("use_timing_scale_bit", True),
        timing_control_mode=config.get("timing_control_mode"),
        timing_log_scale=config.get("timing_log_scale", 50.0),
        split_zero_ioi_head=config.get("split_zero_ioi_head", False),
        ioi_nonzero_dev_scale=config.get("ioi_nonzero_dev_scale", 2.0),
        ioi_zero_dev_scale=config.get("ioi_zero_dev_scale", 4.0),
        use_prepared_sidecar=True,
        prepared_sidecar_tag=config.get("prepared_sidecar_tag"),
    )


def worker(args):
    config, split, work_path, selected_sources = args
    manifest = [
        {
            "path": work_path,
            "windows": [(0, 1)],
            "estimated_performances": 1,
        }
    ]
    dataset = build_dataset(config, manifest, split)
    sidecar = build_sidecar_for_work(dataset, work_path, selected_sources=selected_sources)
    if sidecar is None or not Path(sidecar).exists():
        raise RuntimeError(f"Failed to write sidecar for {work_path}: {sidecar}")
    return str(work_path), str(sidecar)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-path", required=True)
    parser.add_argument("--refined-dir", required=True)
    parser.add_argument("--split", default="train", choices=["train", "test", "valid"])
    parser.add_argument("--block-notes", type=int, default=512)
    parser.add_argument("--overlap-ratio", type=float, default=0.125)
    parser.add_argument("--min-notes", type=int, default=64)
    parser.add_argument("--task-type", default="epr")
    parser.add_argument("--input-feature-mode", default="integrated")
    parser.add_argument("--timing-input-normalization", default="log1p_t_over_50_5000")
    parser.add_argument("--max-time-ms", type=float, default=10000.0)
    parser.add_argument("--pedal-representation", default="start_ctrl")
    parser.add_argument("--musical-feature-mode", default="musical51")
    parser.add_argument("--epr-timing-target", default="absolute")
    parser.add_argument("--use-timing-scale-bit", type=int, default=1)
    parser.add_argument("--timing-control-mode", default=None)
    parser.add_argument("--timing-log-scale", type=float, default=50.0)
    parser.add_argument("--split-zero-ioi-head", type=int, default=0)
    parser.add_argument("--ioi-nonzero-dev-scale", type=float, default=2.0)
    parser.add_argument("--ioi-zero-dev-scale", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--node-cache-size", type=int, default=8)
    parser.add_argument("--fixed-window-split-scheme", default="train_valid_asap3_nonasap05_v1")
    parser.add_argument("--fixed-window-base-split", default="train")
    parser.add_argument("--fixed-window-eval-split-name", default="valid")
    parser.add_argument("--fixed-window-train-split-name", default="train")
    parser.add_argument(
        "--fixed-window-split-summary-path",
        default=str(ROOT_DIR / "data" / "train_valid_asap3_nonasap05_v1_summary.json"),
    )
    parser.add_argument(
        "--performance-dataset",
        default=None,
        help="Override manifest performance dataset. Use ALL to include all datasets.",
    )
    parser.add_argument("--workers", type=int, default=36)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sidecar-tag", default=None)
    args = parser.parse_args()

    config = {
        "metadata_path": args.metadata_path,
        "refined_dir": args.refined_dir,
        "block_notes": args.block_notes,
        "overlap_ratio": args.overlap_ratio,
        "min_notes": args.min_notes,
        "task_type": args.task_type,
        "input_feature_mode": args.input_feature_mode,
        "timing_input_normalization": args.timing_input_normalization,
        "max_time_ms": args.max_time_ms,
        "pedal_representation": args.pedal_representation,
        "musical_feature_mode": args.musical_feature_mode,
        "epr_timing_target": args.epr_timing_target,
        "use_timing_scale_bit": bool(args.use_timing_scale_bit),
        "timing_control_mode": args.timing_control_mode,
        "timing_log_scale": args.timing_log_scale,
        "split_zero_ioi_head": bool(args.split_zero_ioi_head),
        "ioi_nonzero_dev_scale": args.ioi_nonzero_dev_scale,
        "ioi_zero_dev_scale": args.ioi_zero_dev_scale,
        "seed": args.seed,
        "node_cache_size": args.node_cache_size,
        "fixed_window_split_scheme": args.fixed_window_split_scheme,
        "fixed_window_base_split": args.fixed_window_base_split,
        "fixed_window_eval_split_name": args.fixed_window_eval_split_name,
        "fixed_window_train_split_name": args.fixed_window_train_split_name,
        "fixed_window_split_summary_path": args.fixed_window_split_summary_path,
    }
    if args.sidecar_tag is not None:
        if str(args.sidecar_tag).upper() in {"NONE", "NULL", "NO", "OFF"}:
            config.pop("prepared_sidecar_tag", None)
        else:
            config["prepared_sidecar_tag"] = args.sidecar_tag
    split = args.split
    if split == "valid":
        split = config.get("fixed_window_base_split", "train")
        config["fixed_window_split_scheme"] = config.get("fixed_window_split_scheme") or "train_valid_asap3_nonasap05_v1"
        config["fixed_window_eval_split_name"] = "valid"
    manifest = make_manifest(config, split, performance_dataset_override=args.performance_dataset)
    paths = unique_work_paths(manifest)
    selected_by_path = work_selected_sources(manifest)
    if args.limit is not None:
        paths = paths[: args.limit]

    print(
        json.dumps(
            {
                "event": "inr_sidecar_prebuild_start",
                "metadata_path": args.metadata_path,
                "refined_dir": args.refined_dir,
                "split": args.split,
                "performance_dataset": args.performance_dataset or config.get(
                    "train_performance_dataset" if args.split == "train" else "eval_performance_dataset"
                ),
                "works": len(paths),
                "workers": args.workers,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )

    done = 0
    if args.workers <= 1:
        for path in paths:
            worker((config, args.split, path, selected_by_path.get(path)))
            done += 1
            if done % 10 == 0 or done == len(paths):
                print(json.dumps({"event": "inr_sidecar_prebuild_progress", "done": done, "works": len(paths)}), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [
                pool.submit(worker, (config, args.split, path, selected_by_path.get(path)))
                for path in paths
            ]
            for future in as_completed(futures):
                future.result()
                done += 1
                if done % 10 == 0 or done == len(paths):
                    print(json.dumps({"event": "inr_sidecar_prebuild_progress", "done": done, "works": len(paths)}), flush=True)

    print(
        json.dumps(
            {
                "event": "inr_sidecar_prebuild_done",
                "split": args.split,
                "works": len(paths),
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
