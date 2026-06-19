import argparse
import json
import random
import sys
from collections import defaultdict
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.inference.infer_inr_testset import (
    build_windows,
    continuation_window_predictions,
    load_score_from_node,
)
from src.train.train_inr import build_work_manifest, create_model, infer_input_feature_mode
from src.train.train_inr import NodeSFTDataCollator, PianoCoReNodeSFTDataset
from src.model.integrated_pianoformer import _materialize_epr_prediction


FEATURES = ("ioi", "duration", "velocity", "pedal_0", "pedal_25", "pedal_50", "pedal_75")


def empty_arrays():
    return {name: [] for name in FEATURES}


def extend_continuous(arrays, rows):
    cont = np.asarray(rows, dtype=np.float64)
    if cont.ndim != 2 or cont.shape[1] < 7 or cont.shape[0] == 0:
        return
    cont = cont[:, :7]
    for idx, name in enumerate(FEATURES):
        arrays[name].append(cont[:, idx])


def concat_arrays(arrays):
    return {
        name: np.concatenate(chunks).astype(np.float64, copy=False)
        if chunks
        else np.asarray([], dtype=np.float64)
        for name, chunks in arrays.items()
    }


def summarize_arrays(arrays):
    output = {}
    for name, values in arrays.items():
        if len(values) == 0:
            output[name] = {
                "count": 0,
                "mean": float("nan"),
                "std": float("nan"),
                "min": float("nan"),
                "p01": float("nan"),
                "p05": float("nan"),
                "p10": float("nan"),
                "p25": float("nan"),
                "median": float("nan"),
                "p75": float("nan"),
                "p90": float("nan"),
                "p95": float("nan"),
                "p99": float("nan"),
                "max": float("nan"),
            }
            continue
        qs = np.percentile(values, [1, 5, 10, 25, 50, 75, 90, 95, 99])
        output[name] = {
            "count": int(len(values)),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "p01": float(qs[0]),
            "p05": float(qs[1]),
            "p10": float(qs[2]),
            "p25": float(qs[3]),
            "median": float(qs[4]),
            "p75": float(qs[5]),
            "p90": float(qs[6]),
            "p95": float(qs[7]),
            "p99": float(qs[8]),
            "max": float(np.max(values)),
        }
    return output


def score_json_path(processed_dir, score_rel_path):
    score_path = Path(score_rel_path)
    return Path(processed_dir) / score_path.parent / f"{score_path.stem}.node_a.json"


def load_metadata_rows(metadata_path, split, subset):
    columns = [
        "tier_a",
        "split",
        "performance_dataset",
        "refined_score_midi_path",
        "refined_performance_midi_path",
        "refined_alignment_path",
    ]
    df = pd.read_csv(metadata_path, usecols=columns)
    df = df[df["tier_a"].fillna(False).astype(bool)]
    df = df[df["split"].eq(split)]
    df = df[df["refined_score_midi_path"].notna()]
    df = df[df["refined_performance_midi_path"].notna()]
    df = df[df["refined_alignment_path"].notna()]
    if subset == "asap":
        df = df[df["performance_dataset"].eq("ASAP")]
    elif subset == "non_asap":
        df = df[~df["performance_dataset"].eq("ASAP")]
    elif subset != "all":
        raise ValueError(f"Unsupported subset: {subset}")
    return df.sort_values(["refined_score_midi_path", "refined_performance_midi_path"], kind="stable")


def summarize_processed_labels(args):
    df = load_metadata_rows(args.metadata, args.split, args.subset)
    if args.max_performances is not None and len(df) > args.max_performances:
        df = df.sample(n=args.max_performances, random_state=args.seed)
    selected = defaultdict(set)
    for _, row in df.iterrows():
        selected[row["refined_score_midi_path"]].add(row["refined_performance_midi_path"])

    tasks = [
        (str(score_json_path(args.processed_dir, score_rel_path)), sorted(perf_sources))
        for score_rel_path, perf_sources in selected.items()
        if score_json_path(args.processed_dir, score_rel_path).exists()
    ]
    if args.num_workers > 1:
        ctx = get_context("spawn")
        with ctx.Pool(processes=args.num_workers) as pool:
            rows = list(
                tqdm(
                    pool.imap(read_processed_label_task, tasks, chunksize=4),
                    total=len(tasks),
                    desc="processed labels",
                )
            )
    else:
        rows = [
            read_processed_label_task(task)
            for task in tqdm(tasks, total=len(tasks), desc="processed labels")
        ]
    labels = empty_arrays()
    score_inputs = empty_arrays()
    for row_labels, row_score_inputs in rows:
        for name in FEATURES:
            if len(row_labels[name]):
                labels[name].append(row_labels[name])
            if len(row_score_inputs[name]):
                score_inputs[name].append(row_score_inputs[name])

    return {
        "label": concat_arrays(labels),
        "score_input": concat_arrays(score_inputs),
    }, {
        "mode": "processed",
        "metadata": str(args.metadata),
        "processed_dir": str(args.processed_dir),
        "split": args.split,
        "subset": args.subset,
        "performances": int(len(df)),
        "scores": int(df["refined_score_midi_path"].nunique()),
        "dataset_counts": df["performance_dataset"].value_counts(dropna=False).to_dict(),
    }


def read_processed_label_task(task):
    path, perf_sources = task
    perf_sources = set(perf_sources)
    labels = empty_arrays()
    score_inputs = empty_arrays()
    with Path(path).open("r", encoding="utf-8") as file:
        work = json.load(file)
    score_cont = work["score"].get("score_continuous", [])
    if score_cont:
        padded_score = []
        for row in score_cont:
            base = list(row[:3])
            padded_score.append(base + [0.0] * 4)
        extend_continuous(score_inputs, padded_score)
    for perf in work.get("performances", []):
        if perf.get("performance_source") in perf_sources:
            extend_continuous(labels, perf.get("label_continuous", []))
    return concat_arrays(labels), concat_arrays(score_inputs)


def select_device(device_arg):
    if device_arg:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_config(config_path, checkpoint):
    with open(config_path, "r", encoding="utf-8") as file:
        config = json.load(file)
    config["input_feature_mode"] = infer_input_feature_mode(config)
    if checkpoint:
        config["resume_path"] = checkpoint
    return config


def filter_manifest_by_subset(manifest, metadata_path, split, subset):
    if subset == "all":
        return manifest
    df = load_metadata_rows(metadata_path, split, subset)
    allowed = set(df["refined_score_midi_path"].unique())
    return [item for item in manifest if item["score_source"] in allowed]


def summarize_model_outputs(args):
    config = load_config(args.config, args.checkpoint)
    device = select_device(args.device)
    model = create_model(config).to(device)
    model.eval()

    manifest = build_work_manifest(
        metadata_path=config["metadata_path"],
        refined_dir=config["refined_dir"],
        split=args.split,
        block_notes=config["block_notes"],
        overlap_ratio=config["overlap_ratio"],
        min_notes=config["min_notes"],
        max_works=args.max_works,
    )
    manifest = filter_manifest_by_subset(manifest, config["metadata_path"], args.split, args.subset)
    if args.max_works is not None:
        manifest = manifest[: args.max_works]

    predictions = empty_arrays()
    score_inputs = empty_arrays()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    for work in tqdm(manifest, desc="model outputs"):
        pitch, continuous = load_score_from_node(Path(work["path"]), config["input_feature_mode"])
        score_cont = np.asarray(continuous, dtype=np.float64)
        if score_cont.ndim == 2:
            if score_cont.shape[1] >= 5:
                padded = np.concatenate(
                    [score_cont[:, 2:5], np.zeros((score_cont.shape[0], 4), dtype=np.float64)],
                    axis=-1,
                )
                extend_continuous(score_inputs, padded)
            elif score_cont.shape[1] >= 3:
                padded = np.concatenate(
                    [score_cont[:, :3], np.zeros((score_cont.shape[0], 4), dtype=np.float64)],
                    axis=-1,
                )
                extend_continuous(score_inputs, padded)

        windows = build_windows(len(pitch), config["block_notes"], config["overlap_ratio"])
        pred = continuation_window_predictions(
            model=model,
            pitch=pitch,
            continuous=continuous,
            windows=windows,
            pitch_pad_id=config["pitch_pad_id"],
            device=device,
            sampling_strategy="sample" if args.protocol == "sampling" else "mean",
            drop_ratio=args.continuation_drop_ratio,
        )
        extend_continuous(predictions, pred.detach().cpu().numpy())

    return {
        "prediction": concat_arrays(predictions),
        "score_input": concat_arrays(score_inputs),
    }, {
        "mode": "model",
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "device": str(device),
        "split": args.split,
        "subset": args.subset,
        "protocol": args.protocol,
        "continuation_drop_ratio": args.continuation_drop_ratio,
        "works": len(manifest),
    }


def restrict_manifest_performances(manifest, metadata_path, split, subset):
    if subset == "all":
        return manifest
    df = load_metadata_rows(metadata_path, split, subset)
    allowed_by_score = defaultdict(set)
    for _, row in df.iterrows():
        allowed_by_score[row["refined_score_midi_path"]].add(row["refined_performance_midi_path"])
    output = []
    for item in manifest:
        allowed = allowed_by_score.get(item["score_source"])
        if not allowed:
            continue
        copied = dict(item)
        copied["selected_performance_sources"] = [
            source for source in item["selected_performance_sources"] if source in allowed
        ]
        if copied["selected_performance_sources"]:
            copied["estimated_performances"] = len(copied["selected_performance_sources"])
            copied["estimated_examples"] = len(copied["windows"]) * len(copied["selected_performance_sources"])
            output.append(copied)
    return output


def summarize_teacher_forced_outputs(args):
    config = load_config(args.config, args.checkpoint)
    device = select_device(args.device)
    model = create_model(config).to(device)
    model.eval()

    manifest = build_work_manifest(
        metadata_path=config["metadata_path"],
        refined_dir=config["refined_dir"],
        split=args.split,
        block_notes=config["block_notes"],
        overlap_ratio=config["overlap_ratio"],
        min_notes=config["min_notes"],
        max_works=args.max_works,
    )
    manifest = restrict_manifest_performances(manifest, config["metadata_path"], args.split, args.subset)
    if args.max_works is not None:
        manifest = manifest[: args.max_works]

    dataset = PianoCoReNodeSFTDataset(
        manifest,
        split=args.split,
        task_type=config.get("task_type", "epr"),
        input_feature_mode=config["input_feature_mode"],
        shuffle=False,
        max_performances_per_work=None,
        max_windows_per_work=None,
        cache_size=4,
    )
    collator = NodeSFTDataCollator(config["pitch_pad_id"], task_type=config.get("task_type", "epr"))
    predictions = empty_arrays()
    labels = empty_arrays()

    for start in tqdm(range(0, len(dataset), args.batch_size), desc="teacher forced"):
        examples = [dataset[idx] for idx in range(start, min(start + args.batch_size, len(dataset)))]
        batch = collator(examples)
        batch = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
        with torch.no_grad():
            outputs = model(**batch)
            pred = _materialize_epr_prediction(
                model.config,
                outputs.logits,
                sampling_strategy="mean",
            )
        mask = batch["attention_mask"].bool().detach().cpu().numpy()
        pred_np = pred.detach().float().cpu().numpy()
        label_np = batch["labels_continuous"].detach().float().cpu().numpy()
        for row_idx in range(pred_np.shape[0]):
            extend_continuous(predictions, pred_np[row_idx][mask[row_idx]])
            extend_continuous(labels, label_np[row_idx][mask[row_idx]])

    return {
        "prediction": concat_arrays(predictions),
        "label": concat_arrays(labels),
    }, {
        "mode": "teacher_forced",
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "device": str(device),
        "split": args.split,
        "subset": args.subset,
        "works": len(manifest),
        "examples": len(dataset),
        "batch_size": args.batch_size,
    }


def write_outputs(groups, meta, output):
    payload = {"meta": meta, "groups": {name: summarize_arrays(arrays) for name, arrays in groups.items()}}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    rows = []
    for group, features in payload["groups"].items():
        for feature, stats in features.items():
            rows.append({"group": group, "feature": feature, **stats})
    pd.DataFrame(rows).to_csv(output.with_suffix(".csv"), index=False)
    print(f"Wrote {output}")
    print(f"Wrote {output.with_suffix('.csv')}")
    arrays_output = getattr(parse_args.cached_args, "arrays_output", None)
    if arrays_output:
        arrays_output = Path(arrays_output)
        arrays_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            arrays_output,
            **{
                f"{group}__{feature}": values
                for group, arrays in groups.items()
                for feature, values in arrays.items()
            },
        )
        print(f"Wrote {arrays_output}")


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize INR normalized [0,1] distributions.")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    processed = subparsers.add_parser("processed")
    processed.add_argument("--metadata", type=Path, default=Path("../PianoCoRe/metadata.csv"))
    processed.add_argument("--processed-dir", type=Path, default=Path("../PianoCoRe/processed"))
    processed.add_argument("--split", type=str, default="train")
    processed.add_argument("--subset", choices=["all", "asap", "non_asap"], default="all")
    processed.add_argument("--max-performances", type=int, default=None)
    processed.add_argument("--seed", type=int, default=42)
    processed.add_argument("--num-workers", type=int, default=10)
    processed.add_argument("--output", type=Path, required=True)
    processed.add_argument("--arrays-output", type=Path, default=None)

    model = subparsers.add_parser("model")
    model.add_argument("--config", type=Path, required=True)
    model.add_argument("--checkpoint", type=str, required=True)
    model.add_argument("--split", type=str, default="test")
    model.add_argument("--subset", choices=["all", "asap", "non_asap"], default="asap")
    model.add_argument("--protocol", choices=["deterministic", "sampling"], default="deterministic")
    model.add_argument("--continuation-drop-ratio", type=float, default=0.0)
    model.add_argument("--device", type=str, default=None)
    model.add_argument("--max-works", type=int, default=None)
    model.add_argument("--seed", type=int, default=42)
    model.add_argument("--output", type=Path, required=True)
    model.add_argument("--arrays-output", type=Path, default=None)

    teacher = subparsers.add_parser("teacher")
    teacher.add_argument("--config", type=Path, required=True)
    teacher.add_argument("--checkpoint", type=str, required=True)
    teacher.add_argument("--split", type=str, default="test")
    teacher.add_argument("--subset", choices=["all", "asap", "non_asap"], default="asap")
    teacher.add_argument("--device", type=str, default=None)
    teacher.add_argument("--max-works", type=int, default=None)
    teacher.add_argument("--batch-size", type=int, default=8)
    teacher.add_argument("--output", type=Path, required=True)
    teacher.add_argument("--arrays-output", type=Path, default=None)

    args = parser.parse_args()
    parse_args.cached_args = args
    return args


def main():
    args = parse_args()
    if args.mode == "processed":
        groups, meta = summarize_processed_labels(args)
    elif args.mode == "model":
        groups, meta = summarize_model_outputs(args)
    elif args.mode == "teacher":
        groups, meta = summarize_teacher_forced_outputs(args)
    else:
        raise ValueError(args.mode)
    write_outputs(groups, meta, args.output)


if __name__ == "__main__":
    main()
