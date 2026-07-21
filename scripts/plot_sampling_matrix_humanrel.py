#!/usr/bin/env python3
import argparse
import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HUMAN_PN = {
    "ioi_wass": 26.554209906777686,
    "duration_wass": 105.54706686568747,
    "velocity_wass": 11.716304664046339,
    "pedal_wass": 0.246551332481009,
}
HUMAN_PP = {
    "ioi_wass": 7.952391192015138,
    "duration_wass": 25.9246360560533,
    "velocity_wass": 3.057647553002258,
    "pedal_wass": 0.07682493502888045,
}


def parse_value(text):
    return float(text.replace("p", "."))


def parse_run_name(name):
    match = re.fullmatch(r"t(?P<temp>[0-9p]+)_(?P<kind>k|p)(?P<value>[0-9p]+)", name)
    if not match:
        return None
    temp = parse_value(match.group("temp"))
    kind = "top-k" if match.group("kind") == "k" else "top-p"
    raw_value = match.group("value")
    value = int(raw_value) if kind == "top-k" else parse_value(raw_value)
    return temp, kind, value


def human_rel(metrics, human):
    keys = ["ioi_wass", "duration_wass", "velocity_wass", "pedal_wass"]
    return sum(metrics[k] / human[k] for k in keys) / len(keys)


def load_points(root):
    points = []
    for summary in sorted(root.glob("*/summary.json")):
        parsed = parse_run_name(summary.parent.name)
        if parsed is None:
            continue
        with summary.open() as f:
            data = json.load(f)
        aggregate = data["metrics"]["sampling"]["aggregate"]
        pn_rel = human_rel(aggregate["pn_wass"], HUMAN_PN)
        pp_rel = human_rel(aggregate["pp_wass"], HUMAN_PP)
        temp, kind, value = parsed
        points.append(
            {
                "temp": temp,
                "kind": kind,
                "value": value,
                "pn_rel": pn_rel,
                "pp_rel": pp_rel,
            }
        )
    return points


def matrix(points, kind, metric):
    subset = [p for p in points if p["kind"] == kind and p["temp"] >= 0.6 - 1e-9]
    temps = sorted({p["temp"] for p in subset})
    values = sorted({p["value"] for p in subset})
    arr = np.full((len(temps), len(values)), np.nan)
    index_t = {v: i for i, v in enumerate(temps)}
    index_v = {v: i for i, v in enumerate(values)}
    for p in subset:
        arr[index_t[p["temp"]], index_v[p["value"]]] = p[metric]
    return temps, values, arr


def annotate_cells(ax, arr, row_offset, vmin, vmax):
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if math.isnan(arr[i, j]):
                continue
            normalized = 0.0 if vmax <= vmin else (arr[i, j] - vmin) / (vmax - vmin)
            text_color = "white" if normalized > 0.55 else "black"
            ax.text(
                j,
                i + row_offset,
                f"{arr[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=8.5,
                color=text_color,
            )


def combined_matrix(left, right):
    left_temps, left_values, left_arr = left
    right_temps, right_values, right_arr = right
    if left_temps != right_temps:
        raise ValueError(f"Mismatched temperature axes: {left_temps} vs {right_temps}")
    arr = np.concatenate([left_arr, right_arr], axis=1)
    return left_temps, left_values, right_values, arr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--top-k-root", type=Path)
    parser.add_argument("--top-p-root", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    points = []
    if args.top_p_root or args.top_k_root:
        top_p_root = args.top_p_root or args.root
        top_k_root = args.top_k_root or args.root
        points.extend(p for p in load_points(top_p_root) if p["kind"] == "top-p")
        points.extend(p for p in load_points(top_k_root) if p["kind"] == "top-k")
    else:
        points = load_points(args.root)
    if not points:
        raise SystemExit(f"No sampling matrix summaries found under {args.root}")

    matrices = {}
    for kind in ("top-k", "top-p"):
        for metric in ("pn_rel", "pp_rel"):
            matrices[(kind, metric)] = matrix(points, kind, metric)

    pn_arrays = [matrices[(kind, "pn_rel")][2] for kind in ("top-k", "top-p")]
    pp_arrays = [matrices[(kind, "pp_rel")][2] for kind in ("top-k", "top-p")]
    pn_vmin = min(float(np.nanmin(arr)) for arr in pn_arrays if np.isfinite(arr).any())
    pn_vmax = max(float(np.nanmax(arr)) for arr in pn_arrays if np.isfinite(arr).any())
    pp_vmin = min(float(np.nanmin(arr)) for arr in pp_arrays if np.isfinite(arr).any())
    pp_vmax = max(float(np.nanmax(arr)) for arr in pp_arrays if np.isfinite(arr).any())

    pn_temps, k_values, p_values, pn_arr = combined_matrix(
        matrices[("top-k", "pn_rel")],
        matrices[("top-p", "pn_rel")],
    )
    pp_temps, _, _, pp_arr = combined_matrix(
        matrices[("top-k", "pp_rel")],
        matrices[("top-p", "pp_rel")],
    )
    if pn_temps != pp_temps:
        raise ValueError(f"Mismatched PN/PP temperature axes: {pn_temps} vs {pp_temps}")

    fig, ax = plt.subplots(figsize=(3.55, 2.85), constrained_layout=True)
    pn_canvas = np.full((6, 6), np.nan)
    pp_canvas = np.full((6, 6), np.nan)
    pn_canvas[:3, :] = pn_arr
    pp_canvas[3:, :] = pp_arr

    pn_image = ax.imshow(np.ma.masked_invalid(pn_canvas), cmap="Blues", vmin=pn_vmin, vmax=pn_vmax)
    pp_image = ax.imshow(np.ma.masked_invalid(pp_canvas), cmap="Greens", vmin=pp_vmin, vmax=pp_vmax)

    annotate_cells(ax, pn_arr, 0, pn_vmin, pn_vmax)
    annotate_cells(ax, pp_arr, 3, pp_vmin, pp_vmax)

    ax.axvline(2.5, color="black", linewidth=1.2)
    ax.axhline(2.5, color="black", linewidth=1.2)
    ax.set_xticks(np.arange(6))
    ax.set_xticklabels(
        [str(v) for v in k_values] + [f"{v:g}" for v in p_values],
        fontsize=8,
    )
    ax.set_yticks(np.arange(6))
    ax.set_yticklabels([f"{v:g}" for v in pn_temps] + [f"{v:g}" for v in pp_temps], fontsize=8)
    ax.set_ylabel("Temp.", fontsize=9)
    ax.tick_params(length=0)
    ax.text(1.0, -0.82, "top-k", ha="center", va="center", fontsize=9)
    ax.text(4.0, -0.82, "top-p", ha="center", va="center", fontsize=9)
    ax.set_title("Sampling Human-Rel. (temp. >= 0.6)", fontsize=10, pad=18)

    pn_cbar = fig.colorbar(pn_image, ax=ax, shrink=0.52, pad=0.02, location="right")
    pp_cbar = fig.colorbar(pp_image, ax=ax, shrink=0.52, pad=0.10, location="right")
    pn_cbar.ax.set_title("PN", fontsize=7, pad=2)
    pp_cbar.ax.set_title("PP", fontsize=7, pad=2)
    pn_cbar.ax.tick_params(labelsize=7)
    pp_cbar.ax.tick_params(labelsize=7)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=300)


if __name__ == "__main__":
    main()
