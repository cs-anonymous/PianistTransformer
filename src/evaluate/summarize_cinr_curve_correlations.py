#!/usr/bin/env python3
"""Compute beat/measure tempo and velocity curve correlations per CINR epoch."""
import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.evaluate.compute_curve_correlations import _curve, _finite_corr
from src.evaluate.compute_strict_curve_correlations import measure_curve
from miditoolkit import MidiFile

ATTRS = ("tempo", "velocity")
MODES = ("valid_tf", "test_tf", "test_ar")


@lru_cache(maxsize=None)
def score_info(source):
    rel = Path(source)
    work = ROOT / "data" / "ASAP_processed" / rel.with_suffix(".json")
    midi_path = ROOT / "data" / "ASAP_processed" / rel
    payload = json.loads(work.read_text())
    midi = MidiFile(str(midi_path))
    notes = sorted(
        [n for ins in midi.instruments if not ins.is_drum for n in ins.notes],
        key=lambda n: (n.start, n.pitch, n.end, n.velocity),
    )
    score_raw = np.asarray(payload["score"]["score_raw"], dtype=float)
    if len(notes) != len(score_raw):
        raise ValueError(f"Score note mismatch for {source}: {len(notes)} vs {len(score_raw)}")
    onset = np.asarray([n.start / float(midi.ticks_per_beat) for n in notes], dtype=float)
    return midi, onset, score_raw


@lru_cache(maxsize=8192)
def note_arrays(path):
    midi = MidiFile(path)
    notes = sorted(
        [n for ins in midi.instruments if not ins.is_drum for n in ins.notes],
        key=lambda n: (n.start, n.pitch, n.end, n.velocity),
    )
    tick_to_ms = np.asarray(midi.get_tick_to_time_mapping(), dtype=float) * 1000.0
    starts_tick = np.asarray([n.start for n in notes], dtype=int)
    ends_tick = np.asarray([n.end for n in notes], dtype=int)
    starts = tick_to_ms[np.minimum(starts_tick, len(tick_to_ms) - 1)]
    ends = tick_to_ms[np.minimum(ends_tick, len(tick_to_ms) - 1)]
    previous = np.r_[0.0, starts[:-1]]
    return {
        "ioi": np.maximum(starts - previous, 0.0),
        "duration": np.maximum(ends - starts, 0.0),
        "velocity": np.asarray([n.velocity for n in notes], dtype=float),
    }


@lru_cache(maxsize=8192)
def reference_curve(source, start, end, path, curve_mode):
    midi, onset, score_raw = score_info(source)
    onset, score_raw = onset[start:end], score_raw[start:end]
    a = note_arrays(path)
    rows = np.column_stack([a["ioi"], a["duration"], a["velocity"]])
    return curve(rows, onset, score_raw, midi, curve_mode)


_GT_CURVE_CACHE = {}


def cached_reference_curve(source, start, end, gt, curve_mode):
    # The generated GT copies are identical across checkpoints.  Key by the
    # source performance instead of the checkpoint-local copied MIDI path.
    key = (source, start, end, gt.get("performance_source", gt["midi"]), curve_mode)
    if key not in _GT_CURVE_CACHE:
        _GT_CURVE_CACHE[key] = reference_curve(
            source, start, end, gt["midi"], curve_mode
        )
    return _GT_CURVE_CACHE[key]


def curve(rows, onset, score_raw, measure, mode):
    if mode == "beat":
        return _curve(onset, score_raw, rows, bin_beats=1.0, scale=1.0)
    return measure_curve(onset, rows, measure, sorted(measure.time_signature_changes, key=lambda x: x.time))


def align_curves(curves):
    curves = [c for c in curves if len(c[0]) >= 3]
    if not curves:
        return None
    common = set(int(x) for x in curves[0][0].tolist())
    for c in curves[1:]:
        common &= set(int(x) for x in c[0].tolist())
    common = sorted(common)
    if len(common) < 3:
        return None
    aligned = []
    for c in curves:
        lookup = {int(k): (float(t), float(v)) for k, t, v in zip(c[0], c[1], c[2])}
        aligned.append(np.asarray([lookup[k] for k in common], dtype=float))
    return np.asarray(aligned), common


def one_window(item, mode_name, method, curve_mode):
    midi, onset, score_raw = score_info(item["work"])
    window = item["windows"][0]
    start, end = map(int, window["note_range"])
    onset = onset[start:end]
    score_raw = score_raw[start:end]
    paths = item["windows"][0]["normalization"]
    pred_path = Path(paths["ar" if mode_name == "test_ar" else "tf"]["midi"])
    pred_arrays = note_arrays(str(pred_path))
    pred_rows = np.column_stack([
        pred_arrays["ioi"], pred_arrays["duration"], pred_arrays["velocity"]
    ])
    if len(pred_rows) != len(onset):
        n = min(len(pred_rows), len(onset), len(score_raw))
        pred_rows, onset, score_raw = pred_rows[:n], onset[:n], score_raw[:n]
        gt_rows = [x[:n] for x in gt_rows]
    pred_curve = curve(pred_rows, onset, score_raw, midi, curve_mode)
    gt_curves = [
        cached_reference_curve(item["work"], start, end, gt, curve_mode)
        for gt in paths["gt"]
    ]
    aligned_gt = align_curves(gt_curves)
    if aligned_gt is None:
        return None
    gt, common = aligned_gt
    lookup = {int(k): (float(t), float(v)) for k, t, v in zip(pred_curve[0], pred_curve[1], pred_curve[2])}
    pred_common = [k for k in common if k in lookup]
    if len(pred_common) < 3:
        return None
    ref = np.median(gt, axis=0)
    ref_lookup = {k: ref[idx] for idx, k in enumerate(common)}
    pred = np.asarray([lookup[k] for k in pred_common], dtype=float)
    target = np.asarray([ref_lookup[k] for k in pred_common], dtype=float)
    out = {}
    for j, attr in enumerate(ATTRS):
        x, y = pred[:, j], target[:, j]
        if np.std(x) < 1e-12 or np.std(y) < 1e-12 or len(x) < 3:
            out[attr] = None
        else:
            fn = pearsonr if method == "pearson" else spearmanr
            out[attr] = float(fn(x, y).statistic)
    vals = [out[x] for x in ATTRS if out[x] is not None and np.isfinite(out[x])]
    out["mean"] = float(np.mean(vals)) if vals else None
    out["n_curve_points"] = len(pred_common)
    return out


def aggregate_epoch(path, method, curve_mode):
    payload = json.loads(Path(path).read_text())
    output = {"epoch": int(payload["checkpoint"].split("epoch-")[-1].split("/")[0]), "scores": {}}
    for split, name_modes in (("valid", ("valid_tf",)), ("test", ("test_tf", "test_ar"))):
        for name in name_modes:
            rows = []
            for work in payload[split]["works"]:
                item = {"work": work["work"], "windows": work["windows"]}
                value = one_window(item, name, method, curve_mode)
                if value is not None:
                    rows.append(value)
            if not rows:
                output["scores"][name] = {x: None for x in (*ATTRS, "mean")}
                output["scores"][name]["n_scores"] = 0
            else:
                output["scores"][name] = {
                    x: float(np.nanmean([r[x] for r in rows if r[x] is not None]))
                    if any(r[x] is not None for r in rows) else None
                    for x in (*ATTRS, "mean")
                }
                output["scores"][name]["n_scores"] = len(rows)
                output["scores"][name]["mean_curve_points"] = float(np.mean([r["n_curve_points"] for r in rows]))
    return output


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-root", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--method", choices=("pearson", "spearman"), default="pearson")
    ap.add_argument("--curve", choices=("beat", "measure"), default="beat")
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()
    run = args.run_root.resolve()
    paths = [run / f"epoch-{epoch}" / "metrics.json"
             for epoch in range(1, args.epochs + 1)
             if (run / f"epoch-{epoch}" / "metrics.json").exists()]
    if args.workers > 1 and len(paths) > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            rows = list(pool.map(
                aggregate_epoch,
                paths,
                [args.method] * len(paths),
                [args.curve] * len(paths),
            ))
        rows.sort(key=lambda x: x["epoch"])
    else:
        rows = [aggregate_epoch(path, args.method, args.curve) for path in paths]
    out = {"curve": args.curve, "method": args.method, "rows": rows,
           "definition": "equal mean over works; each work uses median human curve and one generated sample"}
    out_path = run / f"curve_correlations_{args.curve}_{args.method}.json"
    out_path.write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")
    csv_path = run / f"curve_correlations_{args.curve}_{args.method}.csv"
    fields = ["epoch"] + [f"{m}_{a}" for m in MODES for a in (*ATTRS, "mean")]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            flat = {"epoch": row["epoch"]}
            flat.update({f"{m}_{a}": row["scores"].get(m, {}).get(a) for m in MODES for a in (*ATTRS, "mean")})
            writer.writerow(flat)
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True, constrained_layout=True)
    colors = {"valid_tf": "#2ca02c", "test_tf": "#9467bd", "test_ar": "#d62728"}
    epochs = [x["epoch"] for x in rows]
    for name in MODES:
        axes[0].plot(epochs, [x["scores"].get(name, {}).get("tempo") for x in rows], marker="o", label=f"{name} tempo", color=colors[name])
        axes[1].plot(epochs, [x["scores"].get(name, {}).get("velocity") for x in rows], marker="o", label=f"{name} velocity", color=colors[name])
    axes[0].set_ylabel(f"{args.method.title()} correlation")
    axes[1].set_ylabel(f"{args.method.title()} correlation")
    axes[1].set_xlabel("Epoch")
    axes[0].set_title(f"{args.curve.title()}-level tempo curve correlation")
    axes[1].set_title(f"{args.curve.title()}-level velocity curve correlation")
    for ax in axes:
        ax.set_ylim(-1.05, 1.05)
        ax.set_xticks(range(1, args.epochs + 1))
        ax.grid(alpha=0.25)
        ax.legend()
    fig.savefig(run / f"curve_correlations_{args.curve}_{args.method}.png", dpi=180)
    plt.close(fig)
    print(json.dumps({"curve": args.curve, "method": args.method, "epochs": len(rows),
                      "output": str(out_path)}))


if __name__ == "__main__":
    main()
