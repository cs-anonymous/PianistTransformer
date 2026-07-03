#!/usr/bin/env python
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
import random
from pathlib import Path
import sys
import os

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data_process.work_manifest import build_work_manifest
import torch


def _candidate_entries(manifest, performance_dataset, selection_seed):
    entries = []
    for item in manifest:
        windows = list(item["windows"])
        if len(windows) <= 1:
            continue
        counts = item.get("performance_dataset_counts") or {}
        if performance_dataset == "non-ASAP":
            target_examples = int(sum(int(v) for k, v in counts.items() if str(k) != "ASAP"))
        else:
            target_examples = int(counts.get(str(performance_dataset), 0))
        total_examples = int(item.get("estimated_performances", 0))
        if target_examples <= 0 or total_examples <= 0:
            continue
        work_key = item.get("score_source", item["path"])
        rng = random.Random(f"{selection_seed}:{work_key}")
        eval_window = windows[rng.randrange(len(windows))]
        entries.append(
            {
                "path": str(item["path"]),
                "score_source": work_key,
                "eval_window": [int(eval_window[0]), int(eval_window[1])],
                "total_examples": total_examples,
                "target_examples": target_examples,
                "random_key": rng.random(),
            }
        )
    return entries


def _select_primary_asap(entries, target_examples):
    ordered = sorted(
        entries,
        key=lambda entry: (
            entry["total_examples"] / max(entry["target_examples"], 1),
            entry["total_examples"],
            -entry["target_examples"],
            entry["random_key"],
            entry["score_source"],
        ),
    )
    selected = {}
    current = 0
    for entry in ordered:
        if current >= target_examples:
            break
        selected[entry["path"]] = entry
        current += entry["target_examples"]
    return selected


def _augment_non_asap(entries, selected, target_examples):
    current = 0
    for entry in selected.values():
        current += entry["total_examples"] - entry["target_examples"]
    if current >= target_examples:
        return selected

    ordered = sorted(
        [
            entry
            for entry in entries
            if entry["path"] not in selected and entry["total_examples"] == entry["target_examples"]
        ],
        key=lambda entry: (
            entry["total_examples"] / max(entry["target_examples"], 1),
            entry["target_examples"],
            entry["total_examples"],
            entry["random_key"],
            entry["score_source"],
        ),
    )
    for entry in ordered:
        if current >= target_examples:
            break
        selected[entry["path"]] = entry
        current += entry["target_examples"]
    return selected


def build_fixed_scheme(manifest, asap_ratio, non_asap_ratio, selection_seed):
    asap_total_examples = 0
    non_asap_total_examples = 0
    for item in manifest:
        windows = list(item["windows"])
        counts = item.get("performance_dataset_counts") or {}
        asap_count = int(counts.get("ASAP", 0))
        non_asap_count = int(sum(int(v) for k, v in counts.items() if str(k) != "ASAP"))
        asap_total_examples += len(windows) * asap_count
        non_asap_total_examples += len(windows) * non_asap_count

    asap_target = max(1, int(math.ceil(asap_total_examples * float(asap_ratio))))
    non_asap_target = max(1, int(math.ceil(non_asap_total_examples * float(non_asap_ratio))))

    asap_entries = _candidate_entries(manifest, "ASAP", selection_seed)
    non_asap_entries = _candidate_entries(manifest, "non-ASAP", selection_seed)

    selected = _select_primary_asap(asap_entries, asap_target)
    selected = _augment_non_asap(non_asap_entries, selected, non_asap_target)

    by_path = {}
    summary_rows = []
    total_eval_examples = 0
    total_asap_eval_examples = 0
    total_non_asap_eval_examples = 0

    for item in manifest:
        path = str(item["path"])
        windows = [list(window) for window in item["windows"]]
        selected_entry = selected.get(path)
        eval_window = selected_entry["eval_window"] if selected_entry else None
        window_assignments = []
        for window in windows:
            split = "valid" if eval_window is not None and list(window) == list(eval_window) else "train"
            window_assignments.append({"window": window, "split": split})
        counts = item.get("performance_dataset_counts") or {}
        asap_count = int(counts.get("ASAP", 0))
        non_asap_count = int(sum(int(v) for k, v in counts.items() if str(k) != "ASAP"))
        eval_examples = int(item.get("estimated_performances", 0)) if eval_window is not None else 0
        eval_asap_examples = asap_count if eval_window is not None else 0
        eval_non_asap_examples = non_asap_count if eval_window is not None else 0
        total_eval_examples += eval_examples
        total_asap_eval_examples += eval_asap_examples
        total_non_asap_eval_examples += eval_non_asap_examples
        by_path[path] = {
            "window_assignments": window_assignments,
            "performance_dataset_counts": counts,
            "estimated_performances": int(item.get("estimated_performances", 0)),
            "valid_examples": eval_examples,
            "valid_asap_examples": eval_asap_examples,
            "valid_non_asap_examples": eval_non_asap_examples,
        }
        summary_rows.append(
            {
                "path": path,
                "score_source": item.get("score_source", path),
                "selected": eval_window is not None,
                "valid_window": eval_window,
                "estimated_performances": int(item.get("estimated_performances", 0)),
                "performance_dataset_counts": counts,
                "valid_examples": eval_examples,
                "valid_asap_examples": eval_asap_examples,
                "valid_non_asap_examples": eval_non_asap_examples,
            }
        )

    return {
        "scheme": {
            "scheme_type": "fixed_window_train_valid_v1",
            "base_split": "train",
            "train_split_name": "train",
            "valid_split_name": "valid",
            "selection_seed": int(selection_seed),
            "targets": {
                "ASAP": {
                    "ratio": float(asap_ratio),
                    "total_examples": int(asap_total_examples),
                    "valid_examples": int(total_asap_eval_examples),
                },
                "non-ASAP": {
                    "ratio": float(non_asap_ratio),
                    "total_examples": int(non_asap_total_examples),
                    "valid_examples": int(total_non_asap_eval_examples),
                },
            },
            "valid_examples": int(total_eval_examples),
            "selected_works": int(sum(1 for row in summary_rows if row["selected"])),
        },
        "assignments_by_path": by_path,
        "work_summaries": summary_rows,
    }


def update_json_with_scheme(json_path, scheme_name, scheme_payload):
    with open(json_path, "r", encoding="utf-8") as file:
        payload = json.load(file)
    meta = payload.setdefault("meta", {})
    schemes = meta.setdefault("window_split_schemes", {})
    schemes[str(scheme_name)] = scheme_payload
    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False)


def update_sidecars_with_scheme(json_path, scheme_name, scheme_payload):
    source = Path(json_path)
    sidecars = sorted(source.parent.glob(f"{source.stem}*.pt"))
    for sidecar_path in sidecars:
        try:
            payload = torch.load(sidecar_path, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(sidecar_path, map_location="cpu")
        meta = payload.setdefault("meta", {})
        schemes = meta.setdefault("window_split_schemes", {})
        schemes[str(scheme_name)] = scheme_payload
        tmp_path = sidecar_path.with_name(f"{sidecar_path.name}.{os.getpid()}.tmp")
        torch.save(payload, tmp_path)
        os.replace(tmp_path, sidecar_path)


def worker(args):
    path, scheme_name, payload, skip_sidecars = args
    update_json_with_scheme(path, scheme_name, payload)
    if not skip_sidecars:
        update_sidecars_with_scheme(path, scheme_name, payload)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--scheme-name", default="train_valid_asap3_nonasap1_v1")
    parser.add_argument("--base-split", default="train")
    parser.add_argument("--asap-ratio", type=float, default=0.03)
    parser.add_argument("--non-asap-ratio", type=float, default=0.01)
    parser.add_argument("--selection-seed", type=int, default=42)
    parser.add_argument("--output-summary", default=None)
    parser.add_argument("--skip-sidecars", action="store_true")
    parser.add_argument("--workers", type=int, default=36)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as file:
        config = json.load(file)

    manifest = build_work_manifest(
        metadata_path=config["metadata_path"],
        refined_dir=config["refined_dir"],
        split=args.base_split,
        block_notes=config["block_notes"],
        overlap_ratio=config["overlap_ratio"],
        min_notes=config["min_notes"],
        max_works=None,
        skip_work_paths=config.get("skip_work_paths"),
        performance_dataset=None,
        exclude_performance_dataset=None,
    )
    built = build_fixed_scheme(
        manifest,
        asap_ratio=args.asap_ratio,
        non_asap_ratio=args.non_asap_ratio,
        selection_seed=args.selection_seed,
    )
    scheme_payload = dict(built["scheme"])

    jobs = []
    total_paths = len(built["assignments_by_path"])
    for path, item in built["assignments_by_path"].items():
        payload = dict(scheme_payload)
        payload["window_assignments"] = item["window_assignments"]
        payload["performance_dataset_counts"] = item["performance_dataset_counts"]
        payload["estimated_performances"] = item["estimated_performances"]
        payload["valid_examples"] = item["valid_examples"]
        payload["valid_asap_examples"] = item["valid_asap_examples"]
        payload["valid_non_asap_examples"] = item["valid_non_asap_examples"]
        jobs.append((path, args.scheme_name, payload, bool(args.skip_sidecars)))

    done = 0
    if int(args.workers) <= 1:
        for job in jobs:
            worker(job)
            done += 1
            if done % 100 == 0 or done == total_paths:
                print(
                    json.dumps(
                        {
                            "event": "fixed_window_valid_split_progress",
                            "done": done,
                            "works": total_paths,
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    flush=True,
                )
    else:
        with ProcessPoolExecutor(max_workers=int(args.workers)) as pool:
            futures = [pool.submit(worker, job) for job in jobs]
            for future in as_completed(futures):
                future.result()
                done += 1
                if done % 100 == 0 or done == total_paths:
                    print(
                        json.dumps(
                            {
                                "event": "fixed_window_valid_split_progress",
                                "done": done,
                                "works": total_paths,
                            },
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                        flush=True,
                    )

    summary_path = (
        Path(args.output_summary)
        if args.output_summary
        else ROOT_DIR / "data" / f"{args.scheme_name}_summary.json"
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(
            {
                "scheme_name": args.scheme_name,
                "scheme": built["scheme"],
                "work_summaries": built["work_summaries"],
            },
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(
        json.dumps(
            {
                "event": "fixed_window_valid_split_created",
                "scheme_name": args.scheme_name,
                "summary_path": str(summary_path),
                "selected_works": built["scheme"]["selected_works"],
                "valid_examples": built["scheme"]["valid_examples"],
                "valid_asap_examples": built["scheme"]["targets"]["ASAP"]["valid_examples"],
                "valid_non_asap_examples": built["scheme"]["targets"]["non-ASAP"]["valid_examples"],
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
