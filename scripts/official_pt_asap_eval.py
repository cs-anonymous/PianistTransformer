#!/usr/bin/env python3
"""Render and evaluate the official PianistTransformer SFT checkpoint.

This script intentionally imports model/evaluation primitives from the fresh
`original/` checkout so that local experimental code does not leak into the
official checkpoint validation.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import miditoolkit
import numpy as np
import partitura as pt
import torch
from scipy.spatial.distance import jensenshannon


ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / "original"
if str(ORIGINAL) not in sys.path:
    sys.path.insert(0, str(ORIGINAL))

from src.model.generate import batch_performance_render, map_midi  # noqa: E402
from src.model.pianoformer import PianoT5Gemma, PianoT5GemmaConfig  # noqa: E402
from src.utils.midi import midi_to_ids, normalize_midi  # noqa: E402


PAPER_METRICS = {
    "velocity": {"js_distance": 0.1805, "intersection": 0.8517},
    "duration": {"js_distance": 0.1879, "intersection": 0.8303},
    "ioi": {"js_distance": 0.1740, "intersection": 0.8292},
    "pedal": {"js_distance": 0.1111, "intersection": 0.8893},
    "overall": {"js_distance": 0.1634, "intersection": 0.8501},
}


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def stem_id(path: Path) -> int:
    return int(path.stem.split("-")[0])


def count_humans_by_prefix(human_dir: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for path in sorted(human_dir.glob("*.mid")):
        prefix = path.stem.split("-")[0]
        counts[prefix] = counts.get(prefix, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: int(kv[0])))


def choose_prefixes(mode: str, explicit: str | None, human_dir: Path) -> list[int]:
    if explicit:
        return sorted({int(x) for x in explicit.split(",") if x.strip()})
    if mode == "official_104_contiguous":
        # The shipped official testset contains 166 human performances.
        # Prefixes 4..15 are the only contiguous score-id range totaling 104.
        return list(range(4, 16))
    if mode == "all_166":
        return sorted({stem_id(path) for path in human_dir.glob("*.mid")})
    if mode == "official_default_prefix0":
        return [0]
    raise ValueError(f"Unknown subset mode: {mode}")


def build_evaluate_list(human_dir: Path, pred_dir: Path, prefixes: list[int]) -> list[dict[str, str]]:
    prefix_set = {str(x) for x in prefixes}
    out = []
    for gt_file in sorted(human_dir.glob("*.mid"), key=lambda p: (int(p.stem.split("-")[0]), int(p.stem.split("-")[1]))):
        prefix, number = gt_file.stem.split("-")
        if prefix not in prefix_set:
            continue
        out.append({"gt": str(gt_file), "pred": str(pred_dir / f"{prefix}.mid")})
    return out


def normalize_testset(testset_dir: Path, norm_dir: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for subdir in ("score", "human", "performance"):
        src_dir = testset_dir / subdir
        dst_dir = norm_dir / subdir
        dst_dir.mkdir(parents=True, exist_ok=True)
        files = sorted(src_dir.glob("*.mid"))
        counts[subdir] = len(files)
        for src in files:
            dst = dst_dir / src.name
            if dst.exists():
                continue
            normalize_midi(miditoolkit.MidiFile(str(src))).dump(str(dst))
    return counts


def render_worker(
    worker_id: int,
    gpu_id: int,
    score_ids: list[int],
    model_dir: str,
    score_dir: str,
    out_dir: str,
    temperature: float,
    top_p: float,
    seed: int,
    max_context_length: int,
    overlap_ratio: float,
    overwrite: bool,
) -> dict[str, Any]:
    random.seed(seed + worker_id)
    np.random.seed(seed + worker_id)
    torch.manual_seed(seed + worker_id)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + worker_id)

    device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
    model = PianoT5Gemma.from_pretrained(model_dir, torch_dtype=torch.bfloat16)
    model.to(device)
    model.eval()

    rendered = []
    failures = []
    start_time = time.time()
    for score_id in score_ids:
        dst = Path(out_dir) / f"{score_id}.mid"
        if dst.exists() and not overwrite:
            rendered.append({"score_id": score_id, "status": "skipped_existing", "seconds": 0.0})
            continue
        t0 = time.time()
        try:
            score = miditoolkit.MidiFile(str(Path(score_dir) / f"{score_id}.mid"))
            with torch.inference_mode():
                result = batch_performance_render(
                    model,
                    [score],
                    max_context_length=max_context_length,
                    overlap_ratio=overlap_ratio,
                    temperature=temperature,
                    top_p=top_p,
                    device=device,
                )[0]
            mapped = map_midi(score, result)
            dst.parent.mkdir(parents=True, exist_ok=True)
            mapped.dump(str(dst))
            rendered.append({"score_id": score_id, "status": "rendered", "seconds": time.time() - t0})
        except Exception as exc:  # noqa: BLE001
            failures.append({"score_id": score_id, "error": repr(exc)})
    return {
        "worker_id": worker_id,
        "gpu_id": gpu_id,
        "score_ids": score_ids,
        "rendered": rendered,
        "failures": failures,
        "seconds": time.time() - start_time,
    }


def prob_from_hist(hist: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
    prob = hist.astype(np.float64)
    prob = prob / prob.sum()
    prob = prob + epsilon
    prob = prob / prob.sum()
    return prob


def metric_from_counts(gt_hist: np.ndarray, pred_hist: np.ndarray) -> dict[str, Any]:
    gt_prob = prob_from_hist(gt_hist)
    pred_prob = prob_from_hist(pred_hist)
    js_distance = float(jensenshannon(gt_prob, pred_prob, base=2))
    return {
        "js_distance": js_distance,
        "js_divergence_squared": float(js_distance * js_distance),
        "intersection": float(np.minimum(gt_prob, pred_prob).sum()),
        "gt_total": int(gt_hist.sum()),
        "pred_total": int(pred_hist.sum()),
    }


def midi_note_array(path: str, cache: dict[str, np.ndarray]) -> np.ndarray:
    if path in cache:
        return cache[path]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        perf = pt.load_performance_midi(path)
    cache[path] = perf.note_array()
    return cache[path]


def preload_note_arrays(evaluate_list: list[dict[str, str]]) -> dict[str, np.ndarray]:
    cache: dict[str, np.ndarray] = {}
    unique_paths = sorted({item["gt"] for item in evaluate_list} | {item["pred"] for item in evaluate_list})
    for path in unique_paths:
        midi_note_array(path, cache)
    return cache


def midi_pedal_patterns(path: str, config: PianoT5GemmaConfig, threshold: int, cache: dict[str, list[int]]) -> list[int]:
    if path in cache:
        return cache[path]
    midi = miditoolkit.MidiFile(path)
    tokens = midi_to_ids(config, midi)
    patterns = []
    for i in range(0, len(tokens), 8):
        chunk = tokens[i : i + 8]
        if len(chunk) < 8:
            continue
        values = []
        for token in chunk[4:8]:
            if config.pedal_start <= token < config.pedal_start + 128:
                values.append(1 if token - config.pedal_start >= threshold else 0)
        if len(values) == 4:
            patterns.append(values[0] * 8 + values[1] * 4 + values[2] * 2 + values[3])
    cache[path] = patterns
    return patterns


def preload_pedal_patterns(evaluate_list: list[dict[str, str]], threshold: int = 64) -> dict[str, list[int]]:
    config = PianoT5GemmaConfig()
    cache: dict[str, list[int]] = {}
    unique_paths = sorted({item["gt"] for item in evaluate_list} | {item["pred"] for item in evaluate_list})
    for path in unique_paths:
        midi_pedal_patterns(path, config, threshold, cache)
    return cache


def extract_velocity(evaluate_list: list[dict[str, str]], note_cache: dict[str, np.ndarray]) -> dict[str, Any]:
    gt_values: list[int] = []
    pred_values: list[int] = []
    skipped = []
    for item in evaluate_list:
        try:
            gt_arr = midi_note_array(item["gt"], note_cache)
            pred_arr = midi_note_array(item["pred"], note_cache)
            if gt_arr.size:
                gt_values.extend([int(x) for x in gt_arr["velocity"]])
            if pred_arr.size:
                pred_values.extend([int(x) for x in pred_arr["velocity"]])
        except Exception as exc:  # noqa: BLE001
            skipped.append({**item, "error": repr(exc)})
    bins = np.arange(0, 129)
    gt_hist, edges = np.histogram(gt_values, bins=bins)
    pred_hist, _ = np.histogram(pred_values, bins=bins)
    return {
        "metric": metric_from_counts(gt_hist, pred_hist),
        "bins": edges.astype(float).tolist(),
        "gt_hist": gt_hist.astype(int).tolist(),
        "pred_hist": pred_hist.astype(int).tolist(),
        "gt_minmax": [int(min(gt_values)), int(max(gt_values))] if gt_values else None,
        "pred_minmax": [int(min(pred_values)), int(max(pred_values))] if pred_values else None,
        "skipped": skipped,
    }


def extract_tick_feature(
    evaluate_list: list[dict[str, str]],
    feature: str,
    value_range: tuple[int, int],
    num_bins: int,
    note_cache: dict[str, np.ndarray],
) -> dict[str, Any]:
    gt_values = []
    pred_values = []
    skipped = []
    for item in evaluate_list:
        try:
            gt_arr = midi_note_array(item["gt"], note_cache)
            pred_arr = midi_note_array(item["pred"], note_cache)
            if feature == "duration":
                if gt_arr.size:
                    gt_values.extend(gt_arr["duration_tick"].astype(float).tolist())
                if pred_arr.size:
                    pred_values.extend(pred_arr["duration_tick"].astype(float).tolist())
            elif feature == "ioi":
                if gt_arr.size > 1:
                    gt_values.extend(np.diff(gt_arr["onset_tick"]).astype(float).tolist())
                if pred_arr.size > 1:
                    pred_values.extend(np.diff(pred_arr["onset_tick"]).astype(float).tolist())
            else:
                raise ValueError(feature)
        except Exception as exc:  # noqa: BLE001
            skipped.append({**item, "error": repr(exc)})

    gt_arr = np.asarray(gt_values, dtype=np.float64)
    pred_arr = np.asarray(pred_values, dtype=np.float64)
    gt_filtered = gt_arr[(gt_arr >= value_range[0]) & (gt_arr < value_range[1])]
    pred_filtered = pred_arr[(pred_arr >= value_range[0]) & (pred_arr < value_range[1])]
    bins = np.linspace(value_range[0], value_range[1], num_bins + 1)
    gt_hist, edges = np.histogram(gt_filtered, bins=bins)
    pred_hist, _ = np.histogram(pred_filtered, bins=bins)
    return {
        "metric": metric_from_counts(gt_hist, pred_hist),
        "range": list(value_range),
        "num_bins": num_bins,
        "bins": edges.astype(float).tolist(),
        "gt_hist": gt_hist.astype(int).tolist(),
        "pred_hist": pred_hist.astype(int).tolist(),
        "gt_raw_total": int(gt_arr.size),
        "pred_raw_total": int(pred_arr.size),
        "gt_filtered_total": int(gt_filtered.size),
        "pred_filtered_total": int(pred_filtered.size),
        "skipped": skipped,
    }


def extract_pedal(evaluate_list: list[dict[str, str]], pedal_cache: dict[str, list[int]], threshold: int = 64) -> dict[str, Any]:
    config = PianoT5GemmaConfig()
    gt_patterns = []
    pred_patterns = []
    skipped = []

    for item in evaluate_list:
        try:
            gt_patterns.extend(midi_pedal_patterns(item["gt"], config, threshold, pedal_cache))
            pred_patterns.extend(midi_pedal_patterns(item["pred"], config, threshold, pedal_cache))
        except Exception as exc:  # noqa: BLE001
            skipped.append({**item, "error": repr(exc)})
    bins = np.arange(17)
    gt_hist, edges = np.histogram(gt_patterns, bins=bins)
    pred_hist, _ = np.histogram(pred_patterns, bins=bins)
    return {
        "metric": metric_from_counts(gt_hist, pred_hist),
        "bins": edges.astype(int).tolist(),
        "gt_hist": gt_hist.astype(int).tolist(),
        "pred_hist": pred_hist.astype(int).tolist(),
        "threshold": threshold,
        "skipped": skipped,
    }


def evaluate_and_save(evaluate_list: list[dict[str, str]], out_dir: Path) -> dict[str, Any]:
    note_cache = preload_note_arrays(evaluate_list)
    pedal_cache = preload_pedal_patterns(evaluate_list)
    cache_summary = {
        "unique_note_arrays": len(note_cache),
        "unique_pedal_tokenizations": len(pedal_cache),
        "note_array_lengths": {path: int(array.size) for path, array in note_cache.items()},
        "pedal_pattern_lengths": {path: len(patterns) for path, patterns in pedal_cache.items()},
    }
    dump_json(out_dir / "midi_cache_summary.json", cache_summary)
    distributions = {
        "velocity": extract_velocity(evaluate_list, note_cache),
        "duration": extract_tick_feature(evaluate_list, "duration", (0, 500), 250, note_cache),
        "ioi": extract_tick_feature(evaluate_list, "ioi", (0, 200), 200, note_cache),
        "pedal": extract_pedal(evaluate_list, pedal_cache),
    }
    metrics = {name: payload["metric"] for name, payload in distributions.items()}
    metrics["overall"] = {
        key: float(np.mean([metrics[name][key] for name in ("velocity", "duration", "ioi", "pedal")]))
        for key in ("js_distance", "js_divergence_squared", "intersection")
    }
    metrics["paper_reference"] = PAPER_METRICS
    metrics["delta_vs_paper"] = {
        name: {
            key: float(metrics[name][key] - PAPER_METRICS[name][key])
            for key in ("js_distance", "intersection")
        }
        for name in PAPER_METRICS
    }
    dump_json(out_dir / "feature_distributions.json", distributions)
    dump_json(out_dir / "metrics.json", metrics)
    return metrics


def partition(items: list[int], workers: int) -> list[list[int]]:
    return [items[i::workers] for i in range(workers)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-dir", type=Path, default=ORIGINAL)
    parser.add_argument("--model-dir", type=Path, default=ORIGINAL / "models/sft")
    parser.add_argument("--subset-mode", choices=["official_104_contiguous", "all_166", "official_default_prefix0"], default="official_104_contiguous")
    parser.add_argument("--prefixes", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpus", type=str, default="0,1,2,3")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20260615)
    parser.add_argument("--max-context-length", type=int, default=4096)
    parser.add_argument("--overlap-ratio", type=float, default=0.5)
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--use-normalized-gt", action="store_true", default=True)
    args = parser.parse_args()

    out_dir = args.output_dir.resolve()
    pred_dir = out_dir / "pred_midis"
    pred_dir.mkdir(parents=True, exist_ok=True)

    testset_dir = args.original_dir / "data/midis/testset"
    norm_dir = args.original_dir / "data/midis/testset-norm"
    normalize_counts = normalize_testset(testset_dir, norm_dir)
    eval_base = norm_dir if args.use_normalized_gt else testset_dir
    score_dir = eval_base / "score"
    human_dir = eval_base / "human"

    prefixes = choose_prefixes(args.subset_mode, args.prefixes, human_dir)
    score_ids = sorted(prefixes)
    evaluate_list = build_evaluate_list(human_dir, pred_dir, prefixes)
    human_counts = count_humans_by_prefix(human_dir)
    config = {
        "original_dir": str(args.original_dir.resolve()),
        "model_dir": str(args.model_dir.resolve()),
        "subset_mode": args.subset_mode,
        "explicit_prefixes": args.prefixes,
        "prefixes": prefixes,
        "score_ids": score_ids,
        "evaluate_pairs": len(evaluate_list),
        "human_counts_by_prefix": human_counts,
        "human_total_all_prefixes": sum(human_counts.values()),
        "normalize_counts": normalize_counts,
        "score_dir": str(score_dir),
        "human_dir": str(human_dir),
        "pred_dir": str(pred_dir),
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": args.seed,
        "max_context_length": args.max_context_length,
        "overlap_ratio": args.overlap_ratio,
        "gpus": args.gpus,
        "paper_reference": PAPER_METRICS,
    }
    dump_json(out_dir / "run_config.json", config)
    dump_json(out_dir / "evaluate_list.json", evaluate_list)

    if not args.skip_generation:
        gpus = [int(x) for x in args.gpus.split(",") if x.strip()]
        shards = [shard for shard in partition(score_ids, len(gpus)) if shard]
        start = time.time()
        worker_results = []
        with ProcessPoolExecutor(max_workers=len(shards), mp_context=get_context("spawn")) as executor:
            futures = []
            for worker_id, shard in enumerate(shards):
                futures.append(
                    executor.submit(
                        render_worker,
                        worker_id,
                        gpus[worker_id % len(gpus)],
                        shard,
                        str(args.model_dir.resolve()),
                        str(score_dir.resolve()),
                        str(pred_dir.resolve()),
                        args.temperature,
                        args.top_p,
                        args.seed,
                        args.max_context_length,
                        args.overlap_ratio,
                        args.overwrite,
                    )
                )
            for future in as_completed(futures):
                worker_results.append(future.result())
                dump_json(out_dir / "render_worker_results.partial.json", worker_results)
        render_summary = {
            "seconds": time.time() - start,
            "workers": worker_results,
            "failures": [failure for worker in worker_results for failure in worker["failures"]],
        }
        dump_json(out_dir / "render_summary.json", render_summary)
    else:
        render_summary_path = out_dir / "render_summary.json"
        render_summary = load_json(render_summary_path) if render_summary_path.exists() else {"skipped": True}

    missing_preds = [item["pred"] for item in evaluate_list if not Path(item["pred"]).exists()]
    if missing_preds:
        dump_json(out_dir / "missing_predictions.json", missing_preds)
        raise FileNotFoundError(f"Missing {len(missing_preds)} prediction files; see {out_dir / 'missing_predictions.json'}")

    metrics = evaluate_and_save(evaluate_list, out_dir)
    dump_json(out_dir / "summary.json", {"config": config, "render_summary": render_summary, "metrics": metrics})

    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
