import argparse
import json
import math
import random
import sys
from collections import defaultdict
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.evaluate.compute_saved_midi_mae_wass import extract_note_arrays


FEATURES = ("ioi", "duration", "velocity", "pedal")
PEDAL_KEYS = ("pedal_0", "pedal_25", "pedal_50", "pedal_75")


def denormalize_time_ms(values, max_time_ms=10000.0):
    values = np.asarray(values, dtype=np.float64)
    values = np.clip(values, 0.0, 1.0)
    return np.expm1(values * math.log1p(float(max_time_ms)))


def empty_arrays():
    return {name: [] for name in FEATURES}


def extend_from_continuous(arrays, rows, max_time_ms=10000.0):
    cont = np.asarray(rows, dtype=np.float64)
    if cont.ndim != 2 or cont.shape[1] < 7 or cont.shape[0] == 0:
        return
    arrays["ioi"].append(denormalize_time_ms(cont[:, 0], max_time_ms=max_time_ms))
    arrays["duration"].append(denormalize_time_ms(cont[:, 1], max_time_ms=max_time_ms))
    arrays["velocity"].append(np.clip(cont[:, 2], 0.0, 1.0) * 127.0)
    arrays["pedal"].append(np.clip(cont[:, 3:7], 0.0, 1.0).reshape(-1) * 127.0)


def concat_arrays(arrays):
    output = {}
    for name, chunks in arrays.items():
        if chunks:
            output[name] = np.concatenate(chunks).astype(np.float64, copy=False)
        else:
            output[name] = np.asarray([], dtype=np.float64)
    return output


def summarize_arrays(arrays):
    summary = {}
    for name in FEATURES:
        values = arrays[name]
        if len(values) == 0:
            summary[name] = {
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
        percentiles = np.percentile(values, [1, 5, 10, 25, 50, 75, 90, 95, 99])
        summary[name] = {
            "count": int(len(values)),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "p01": float(percentiles[0]),
            "p05": float(percentiles[1]),
            "p10": float(percentiles[2]),
            "p25": float(percentiles[3]),
            "median": float(percentiles[4]),
            "p75": float(percentiles[5]),
            "p90": float(percentiles[6]),
            "p95": float(percentiles[7]),
            "p99": float(percentiles[8]),
            "max": float(np.max(values)),
        }
    return summary


def read_node_perf_chunk(path, selected_sources, max_time_ms):
    path = Path(path)
    selected = set(selected_sources) if selected_sources is not None else None
    arrays = empty_arrays()
    per_dataset = defaultdict(empty_arrays)
    with path.open("r", encoding="utf-8") as file:
        work = json.load(file)
    for perf in work.get("performances", []):
        source = perf.get("performance_source")
        if selected is not None and source not in selected:
            continue
        labels = perf.get("label_continuous")
        if labels is None:
            continue
        dataset = str(perf.get("performance_dataset") or infer_dataset_from_source(source))
        extend_from_continuous(arrays, labels, max_time_ms=max_time_ms)
        extend_from_continuous(per_dataset[dataset], labels, max_time_ms=max_time_ms)
    output = {"__all__": concat_arrays(arrays)}
    for dataset, dataset_arrays in per_dataset.items():
        output[dataset] = concat_arrays(dataset_arrays)
    return output


def infer_dataset_from_source(source):
    if not source:
        return "unknown"
    name = Path(source).name
    if name.startswith("ASAP_"):
        return "ASAP"
    if name.startswith("Aria_"):
        return "Aria-MIDI"
    if name.startswith("ATEPP_"):
        return "ATEPP"
    if name.startswith("GiantMIDI"):
        return "GiantMIDI-Piano"
    if name.startswith("PERiScoPe_"):
        return "PERiScoPe"
    return "unknown"


def merge_group_chunks(chunks):
    groups = defaultdict(empty_arrays)
    for chunk in chunks:
        for group, arrays in chunk.items():
            for name in FEATURES:
                if len(arrays[name]):
                    groups[group][name].append(arrays[name])
    return {group: concat_arrays(arrays) for group, arrays in groups.items()}


def score_json_path(processed_dir, score_rel_path):
    score_path = Path(score_rel_path)
    return Path(processed_dir) / score_path.parent / f"{score_path.stem}.node_a.json"


def metadata_train_tasks(args):
    columns = [
        "tier_a",
        "split",
        "performance_dataset",
        "refined_score_midi_path",
        "refined_performance_midi_path",
        "refined_alignment_path",
    ]
    df = pd.read_csv(args.metadata, usecols=columns)
    df = df[df["tier_a"].fillna(False).astype(bool)]
    df = df[df["split"].eq(args.split)]
    df = df[df["refined_score_midi_path"].notna()]
    df = df[df["refined_performance_midi_path"].notna()]
    df = df[df["refined_alignment_path"].notna()]
    if args.max_performances is not None and len(df) > args.max_performances:
        if args.sample_by_dataset:
            pieces = []
            per_dataset = max(1, args.max_performances // max(1, df["performance_dataset"].nunique()))
            for _, group in df.groupby("performance_dataset", sort=True):
                n = min(len(group), per_dataset)
                pieces.append(group.sample(n=n, random_state=args.seed))
            sampled = pd.concat(pieces, axis=0)
            remaining = args.max_performances - len(sampled)
            if remaining > 0:
                rest = df.drop(index=sampled.index, errors="ignore")
                if len(rest) > 0:
                    sampled = pd.concat(
                        [sampled, rest.sample(n=min(remaining, len(rest)), random_state=args.seed)],
                        axis=0,
                    )
            df = sampled
        else:
            df = df.sample(n=args.max_performances, random_state=args.seed)
    df = df.sort_values(["refined_score_midi_path", "refined_performance_midi_path"], kind="stable")

    tasks = []
    for score_rel_path, group in df.groupby("refined_score_midi_path", sort=True):
        path = score_json_path(args.processed_dir, score_rel_path)
        if not path.exists():
            continue
        tasks.append((str(path), group["refined_performance_midi_path"].tolist(), args.max_time_ms))
    return tasks, df


def extract_midi_group(path):
    path = str(Path(path).resolve())
    arrays = extract_note_arrays(Path(path))
    return {
        "ioi": arrays["ioi"],
        "duration": arrays["duration"],
        "velocity": arrays["velocity"],
        "pedal": np.concatenate([arrays[key] for key in PEDAL_KEYS]).astype(np.float64, copy=False),
    }


def list_unique_paths_from_evaluate_list(path, side):
    data = json.loads(Path(path).read_text())
    if side == "both":
        paths = {item["pred"] for item in data} | {item["gt"] for item in data}
    else:
        paths = {item[side] for item in data}
    return sorted(paths)


def summarize_midi_paths(paths, num_workers):
    if num_workers > 1:
        ctx = get_context("spawn")
        with ctx.Pool(processes=num_workers) as pool:
            rows = list(
                tqdm(
                    pool.imap(extract_midi_group, paths, chunksize=8),
                    total=len(paths),
                    desc="MIDI distributions",
                )
            )
    else:
        rows = [
            extract_midi_group(path)
            for path in tqdm(paths, total=len(paths), desc="MIDI distributions")
        ]
    arrays = empty_arrays()
    for row in rows:
        for name in FEATURES:
            arrays[name].append(row[name])
    return concat_arrays(arrays)


def summarize_train(args):
    tasks, df = metadata_train_tasks(args)
    print(
        f"Reading {len(df)} {args.split} performances from {len(tasks)} node json files",
        flush=True,
    )
    if args.num_workers > 1:
        ctx = get_context("spawn")
        with ctx.Pool(processes=args.num_workers) as pool:
            chunks = list(
                tqdm(
                    pool.starmap(read_node_perf_chunk, tasks, chunksize=4),
                    total=len(tasks),
                    desc="Node distributions",
                )
            )
    else:
        chunks = [
            read_node_perf_chunk(*task)
            for task in tqdm(tasks, total=len(tasks), desc="Node distributions")
        ]
    groups = merge_group_chunks(chunks)
    meta = {
        "source": "metadata_node_labels",
        "metadata": str(args.metadata),
        "processed_dir": str(args.processed_dir),
        "split": args.split,
        "performances": int(len(df)),
        "scores": int(df["refined_score_midi_path"].nunique()),
        "sampled": args.max_performances is not None,
        "max_performances": args.max_performances,
        "dataset_counts": df["performance_dataset"].value_counts(dropna=False).to_dict(),
    }
    return groups, meta


def summarize_eval(args):
    paths = list_unique_paths_from_evaluate_list(args.evaluate_list, args.side)
    print(f"Reading {len(paths)} unique MIDI files from {args.evaluate_list} side={args.side}", flush=True)
    arrays = summarize_midi_paths(paths, args.num_workers)
    meta = {
        "source": "evaluate_list_midis",
        "evaluate_list": str(args.evaluate_list),
        "side": args.side,
        "files": len(paths),
    }
    return {"__all__": arrays}, meta


def write_outputs(groups, meta, output):
    payload = {"meta": meta, "groups": {name: summarize_arrays(arrays) for name, arrays in groups.items()}}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    csv_rows = []
    for group, feature_summary in payload["groups"].items():
        for feature, stats in feature_summary.items():
            csv_rows.append({"group": group, "feature": feature, **stats})
    csv_path = output.with_suffix(".csv")
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    print(f"Wrote {output}")
    print(f"Wrote {csv_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize performance feature distributions.")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    train = subparsers.add_parser("train", help="Summarize normalized node labels from metadata split.")
    train.add_argument("--metadata", type=Path, default=Path("../PianoCoRe/metadata.csv"))
    train.add_argument("--processed-dir", type=Path, default=Path("../PianoCoRe/processed"))
    train.add_argument("--split", type=str, default="train")
    train.add_argument("--max-time-ms", type=float, default=10000.0)
    train.add_argument("--max-performances", type=int, default=None)
    train.add_argument("--sample-by-dataset", action="store_true")
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--num-workers", type=int, default=10)
    train.add_argument("--output", type=Path, required=True)

    eval_parser = subparsers.add_parser("eval", help="Summarize MIDI files referenced by evaluate_list.")
    eval_parser.add_argument("--evaluate-list", type=Path, required=True)
    eval_parser.add_argument("--side", choices=["pred", "gt", "both"], default="pred")
    eval_parser.add_argument("--num-workers", type=int, default=10)
    eval_parser.add_argument("--output", type=Path, required=True)

    return parser.parse_args()


def main():
    args = parse_args()
    if args.mode == "train":
        groups, meta = summarize_train(args)
    elif args.mode == "eval":
        groups, meta = summarize_eval(args)
    else:
        raise ValueError(args.mode)
    write_outputs(groups, meta, args.output)


if __name__ == "__main__":
    main()
