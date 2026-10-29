"""Compute fixed-bin KL and Jensen-Shannon divergences for EPR outputs.

Timing is compared in the score-relative log-deviation space used by INSPIRE.
The script reports PP divergences (pooled within each score) and PN divergences
(averaged over aligned notes), using score-span-normalized references supplied
by each prediction manifest.
"""

from __future__ import annotations

import argparse
import json
import math
from multiprocessing import get_context
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BACKUP_ROOT = ROOT / "backup" / "repo_cleanup_20260804"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluate.compute_saved_midi_mae_wass import extract_note_arrays


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        action="append",
        required=True,
        metavar="NAME=MANIFEST",
        help="Prediction manifest, repeated once per method.",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--timing-bins", type=int, default=256)
    parser.add_argument("--velocity-bins", type=int, default=128)
    parser.add_argument("--timing-lo", type=float, default=-2.0)
    parser.add_argument("--timing-hi", type=float, default=1.0)
    parser.add_argument("--ioi-zero-lo", type=float, default=0.0)
    parser.add_argument("--ioi-zero-hi", type=float, default=5.0)
    parser.add_argument("--velocity-lo", type=float, default=-0.5)
    parser.add_argument("--velocity-hi", type=float, default=127.5)
    parser.add_argument("--pseudocount", type=float, default=1e-8)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument(
        "--include-pn",
        action="store_true",
        help="Also compute aligned-note (PN) divergences; PP is the default fast report.",
    )
    return parser.parse_args()


def resolve_path(raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if path.exists():
        return path
    raw = str(raw_path)
    marker = "/PianistTransformer/"
    if marker in raw:
        relative = Path(raw.split(marker, 1)[1])
        candidates = [ROOT / relative, BACKUP_ROOT / relative]
        parts = relative.parts
        if len(parts) > 2 and parts[0] == "PianoCoRe" and parts[1] == "refined":
            candidates.extend([
                ROOT / "data" / "ASAP_processed" / Path(*parts[2:]),
                ROOT / "data" / "ASAP_processed_pedal_group" / Path(*parts[2:]),
            ])
        for candidate in candidates:
            if candidate.exists():
                return candidate
    raise FileNotFoundError(raw_path)


def resolve_score_path(score_source: str) -> Path:
    relative = Path(score_source)
    candidates = [
        ROOT / "data" / "ASAP_processed" / relative,
        ROOT / "data" / "ASAP_processed_pedal_group" / relative,
        BACKUP_ROOT / "data" / "ASAP_processed" / relative,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"score MIDI for {score_source}")


def load_manifest(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload["items"]
    result = {}
    for item in items:
        score_source = item["score_source"]
        result[score_source] = {
            "score": resolve_score_path(item.get("original_score_source", score_source)),
            "predictions": [resolve_path(path) for path in item["prediction_paths"]],
            "ground_truth": [resolve_path(path) for path in item["ground_truth_paths"]],
        }
    return result


def timing_deviation(values: np.ndarray, score_values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    score_values = np.asarray(score_values, dtype=np.float64)
    return np.log(np.maximum(values, 1.0)) - np.log(np.maximum(score_values, 1.0))


def feature_values(note_arrays, score_arrays, feature: str, limit: int | None = None, group_mask=None):
    if feature in {"ioi", "duration"}:
        n = min(len(note_arrays[feature]), len(score_arrays[feature]))
        values = timing_deviation(
            np.asarray(note_arrays[feature])[:n],
            np.asarray(score_arrays[feature])[:n],
        )
        if group_mask is not None:
            values = values[np.asarray(group_mask)[:n]]
    else:
        values = np.asarray(note_arrays[feature], dtype=np.float64)
        if limit is not None:
            values = values[:limit]
        if group_mask is not None:
            values = values[np.asarray(group_mask)[: len(values)]]
    if limit is not None:
        values = values[:limit]
    return values[np.isfinite(values)]


def probability(values: np.ndarray, edges: np.ndarray, pseudocount: float):
    clipped = np.clip(np.asarray(values, dtype=np.float64), edges[0], edges[-1])
    counts, _ = np.histogram(clipped, bins=edges)
    probs = counts.astype(np.float64) + float(pseudocount)
    return probs / probs.sum(), int(np.sum(clipped != np.asarray(values, dtype=np.float64)))


def divergences(p_values, q_values, edges, pseudocount):
    p, p_clipped = probability(p_values, edges, pseudocount)
    q, q_clipped = probability(q_values, edges, pseudocount)
    midpoint = 0.5 * (p + q)
    kl = float(np.sum(p * np.log(p / q)))
    js = float(0.5 * np.sum(p * np.log(p / midpoint)) + 0.5 * np.sum(q * np.log(q / midpoint)))
    return {"kl_nats": kl, "js_nats": js, "pred_values_clipped": p_clipped, "gt_values_clipped": q_clipped}


def matrix_probabilities(values, edges, pseudocount):
    """Return one smoothed histogram per note for a [samples, notes] matrix."""
    values = np.asarray(values, dtype=np.float64)
    values = np.clip(values, edges[0], edges[-1])
    indices = np.searchsorted(edges, values, side="right") - 1
    indices = np.clip(indices, 0, len(edges) - 2)
    counts = np.zeros((values.shape[1], len(edges) - 1), dtype=np.float64)
    note_indices = np.arange(values.shape[1])
    for row in indices:
        np.add.at(counts, (note_indices, row), 1.0)
    counts += float(pseudocount)
    return counts / counts.sum(axis=1, keepdims=True)


def matrix_divergences(pred_values, gt_values, edges, pseudocount):
    pred_values = np.asarray(pred_values, dtype=np.float64)
    gt_values = np.asarray(gt_values, dtype=np.float64)
    pred_clipped = np.clip(pred_values, edges[0], edges[-1]) != pred_values
    gt_clipped = np.clip(gt_values, edges[0], edges[-1]) != gt_values
    p = matrix_probabilities(pred_values, edges, pseudocount)
    q = matrix_probabilities(gt_values, edges, pseudocount)
    midpoint = 0.5 * (p + q)
    kl = np.sum(p * np.log(p / q), axis=1)
    js = 0.5 * np.sum(p * np.log(p / midpoint), axis=1) + 0.5 * np.sum(q * np.log(q / midpoint), axis=1)
    return {
        "kl_nats": float(np.mean(kl)),
        "js_nats": float(np.mean(js)),
        "pred_values_clipped": int(np.sum(pred_clipped)),
        "gt_values_clipped": int(np.sum(gt_clipped)),
    }


def _extract_path_worker(path: str):
    return path, extract_note_arrays(Path(path))


def summarize_method(
    method_items,
    score_sources,
    edges_by_feature,
    pseudocount,
    feature_cache,
    ioi_zero_lo,
    ioi_zero_hi,
    include_pn=False,
):
    pp_rows = {feature: [] for feature in edges_by_feature}
    pn_rows = {feature: [] for feature in edges_by_feature}
    clipping = {"pp": {feature: {"pred": 0, "gt": 0} for feature in edges_by_feature},
                "pn": {feature: {"pred": 0, "gt": 0} for feature in edges_by_feature}}

    for score_source in score_sources:
        item = method_items[score_source]
        score_arrays = feature_cache[str(item["score"])]
        pred_arrays = [feature_cache[str(path)] for path in item["predictions"]]
        gt_arrays = [feature_cache[str(path)] for path in item["ground_truth"]]
        max_notes = min(
            [len(score_arrays["ioi"])]
            + [len(arr["ioi"]) for arr in pred_arrays]
            + [len(arr["ioi"]) for arr in gt_arrays]
        )

        score_row = {"score_source": score_source, "pp": {}, "pn": {}}
        for feature, edges in edges_by_feature.items():
            groups = [None]
            if feature == "ioi":
                score_ioi = np.asarray(score_arrays["ioi"], dtype=np.float64)
                groups = [score_ioi <= 1.0, score_ioi > 1.0]
            group_rows = []
            group_weights = []
            for group_mask in groups:
                group_edges = edges
                if feature == "ioi" and group_mask is groups[0]:
                    group_edges = np.linspace(ioi_zero_lo, ioi_zero_hi, len(edges))
                pred_pool = np.concatenate([
                    feature_values(arr, score_arrays, feature, max_notes, group_mask) for arr in pred_arrays
                ])
                gt_pool = np.concatenate([
                    feature_values(arr, score_arrays, feature, max_notes, group_mask) for arr in gt_arrays
                ])
                if len(pred_pool) == 0 or len(gt_pool) == 0:
                    continue
                group_rows.append(divergences(pred_pool, gt_pool, group_edges, pseudocount))
                group_weights.append(len(gt_pool))
            pp = {
                key: float(np.average([row[key] for row in group_rows], weights=group_weights))
                for key in ("kl_nats", "js_nats")
            }
            score_row["pp"][feature] = pp
            pp_rows[feature].append(pp)
            clipping["pp"][feature]["pred"] += int(sum(row["pred_values_clipped"] for row in group_rows))
            clipping["pp"][feature]["gt"] += int(sum(row["gt_values_clipped"] for row in group_rows))

            if include_pn:
                pred_values_by_array = [feature_values(arr, score_arrays, feature, max_notes) for arr in pred_arrays]
                gt_values_by_array = [feature_values(arr, score_arrays, feature, max_notes) for arr in gt_arrays]
                pred_matrix = np.stack(pred_values_by_array, axis=0)
                gt_matrix = np.stack(gt_values_by_array, axis=0)
                pn_rows_for_groups = []
                pn_weights = []
                pn_clip_pred = 0
                pn_clip_gt = 0
                pn_groups = [None]
                if feature == "ioi":
                    score_ioi = np.asarray(score_arrays["ioi"], dtype=np.float64)[:max_notes]
                    pn_groups = [score_ioi <= 1.0, score_ioi > 1.0]
                for group_mask in pn_groups:
                    group_edges = edges
                    if feature == "ioi" and group_mask is pn_groups[0]:
                        group_edges = np.linspace(ioi_zero_lo, ioi_zero_hi, len(edges))
                    mask = np.ones(max_notes, dtype=bool) if group_mask is None else np.asarray(group_mask)[:max_notes]
                    if not np.any(mask):
                        continue
                    group_result = matrix_divergences(
                        pred_matrix[:, mask], gt_matrix[:, mask], group_edges, pseudocount
                    )
                    weight = int(np.sum(mask))
                    pn_rows_for_groups.append(group_result)
                    pn_weights.append(weight)
                    pn_clip_pred += group_result["pred_values_clipped"]
                    pn_clip_gt += group_result["gt_values_clipped"]
                pn = {
                    key: float(np.average([row[key] for row in pn_rows_for_groups], weights=pn_weights))
                    for key in ("kl_nats", "js_nats")
                }
                pn_rows[feature].append(pn)
                score_row["pn"][feature] = pn
                clipping["pn"][feature]["pred"] += pn_clip_pred
                clipping["pn"][feature]["gt"] += pn_clip_gt

    aggregate = {
        "pp": {
            feature: {
                key: float(np.mean([row[key] for row in rows]))
                for key in ("kl_nats", "js_nats")
            }
            for feature, rows in pp_rows.items()
        },
        "pn": {
            feature: {
                key: float(np.mean([row[key] for row in rows]))
                for key in ("kl_nats", "js_nats")
            }
            for feature, rows in pn_rows.items()
            if rows
        },
    }
    return {"aggregate": aggregate, "scores": [
        {"score_source": source, "pp": {}, "pn": {}}
        for source in score_sources
    ], "clipping": clipping}


def main():
    args = parse_args()
    methods = {}
    for spec in args.method:
        name, manifest = spec.split("=", 1)
        methods[name] = load_manifest(resolve_path(manifest))
    score_sources = sorted(set.intersection(*(set(items) for items in methods.values())))
    if not score_sources:
        raise ValueError("No common score sources across manifests")

    edges_by_feature = {
        "ioi": np.linspace(args.timing_lo, args.timing_hi, args.timing_bins + 1),
        "duration": np.linspace(args.timing_lo, args.timing_hi, args.timing_bins + 1),
        "velocity": np.linspace(args.velocity_lo, args.velocity_hi, args.velocity_bins + 1),
    }
    unique_paths = set()
    for items in methods.values():
        for source in score_sources:
            item = items[source]
            unique_paths.add(str(item["score"]))
            unique_paths.update(str(path) for path in item["predictions"])
            unique_paths.update(str(path) for path in item["ground_truth"])
    print(f"Extracting {len(unique_paths)} MIDI files with {args.workers} workers", flush=True)
    ctx = get_context("fork")
    with ctx.Pool(processes=max(1, int(args.workers))) as pool:
        cached_rows = pool.map(
            _extract_path_worker,
            sorted(unique_paths),
            chunksize=1,
        )
    feature_cache = dict(cached_rows)
    output = {
        "protocol": {
            "num_scores": len(score_sources),
            "score_sources": score_sources,
            "timing_transform": "log(max(performance_ms, 1)) - log(max(score_ms, 1))",
            "timing_support": {
                "ioi_nonzero_score": [args.timing_lo, args.timing_hi],
                "ioi_zero_score": [args.ioi_zero_lo, args.ioi_zero_hi],
                "duration": [args.timing_lo, args.timing_hi],
            },
            "timing_bins": args.timing_bins,
            "velocity_support": [args.velocity_lo, args.velocity_hi],
            "velocity_bins": args.velocity_bins,
            "pseudocount": args.pseudocount,
            "log_base": "e",
            "out_of_support": "clipped into the endpoint bins",
            "aggregation": "macro-average over scores; PN additionally averages over aligned notes",
        },
        "methods": {},
    }
    for name, items in methods.items():
        output["methods"][name] = summarize_method(
            items,
            score_sources,
            edges_by_feature,
            args.pseudocount,
            feature_cache,
            args.ioi_zero_lo,
            args.ioi_zero_hi,
            args.include_pn,
        )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2), encoding="utf-8")
    for name, payload in output["methods"].items():
        print(name, json.dumps(payload["aggregate"], sort_keys=True))


if __name__ == "__main__":
    main()
