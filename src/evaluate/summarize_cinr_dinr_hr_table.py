#!/usr/bin/env python3
"""Recompute the CINR/DINR HR comparison table from archived raw summaries."""

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.evaluate.plot_sampling_matrix_humanrel import HUMAN_PN, HUMAN_PP


FEATURES = ("ioi", "duration", "velocity", "pedal")
ARCHIVE = Path("backup/repo_cleanup_20260804/results/inr_epr_pipeline/asap_processed_musical51_nolossnorm16_slot_effective_20260723")
DINR_ROOT = Path("results/inr_epr_pipeline/dinr_slot_coord_matrix_20261026")

RUNS = [
    ("CINR", "canonical slot128_mlp", ARCHIVE / "cinr/default_dlm_k1_bounded5/summary.json"),
    ("CINR", "sine", ARCHIVE / "extra/rep_sine/summary.json"),
    ("CINR", "slot256_mlp", ARCHIVE / "extra/rep_slot256_mlp/summary.json"),
    ("CINR", "slot_direct", ARCHIVE / "extra/rep_slot_direct/summary.json"),
    ("CINR", "slot_sum", ARCHIVE / "extra/rep_slot_sum/summary.json"),
    ("DINR", "slot128_mlp + coord", DINR_ROOT / "slot128_mlp__coord_all/summary.json"),
    ("DINR", "slot128_mlp + no_coord", DINR_ROOT / "slot128_mlp__no_coord/summary.json"),
    ("DINR", "slot256_mlp + coord", DINR_ROOT / "slot256_mlp__coord_all/summary.json"),
    ("DINR", "slot256_mlp + no_coord", DINR_ROOT / "slot256_mlp__no_coord/summary.json"),
    ("DINR", "slot_direct + coord", DINR_ROOT / "slot_direct__coord_all/summary.json"),
    ("DINR", "slot_direct + no_coord", DINR_ROOT / "slot_direct__no_coord/summary.json"),
    ("DINR", "slot_sum + coord", DINR_ROOT / "slot_sum__coord_all/summary.json"),
    ("DINR", "slot_sum + no_coord", DINR_ROOT / "slot_sum__no_coord/summary.json"),
]


def aggregate(summary):
    sampling = summary["metrics"]["sampling"]
    if sampling.get("protocol") != "sampling":
        raise ValueError("Expected sampling-protocol metrics")
    return sampling, sampling["aggregate"]


def make_row(model, configuration, path):
    summary = json.loads(path.read_text(encoding="utf-8"))
    sampling, metrics = aggregate(summary)
    row = {
        "model": model,
        "configuration": configuration,
        "PP_HR": sum(metrics["pp_wass"][f"{f}_wass"] / HUMAN_PP[f"{f}_wass"] for f in FEATURES) / len(FEATURES),
        "PN_HR": sum(metrics["pn_wass"][f"{f}_wass"] / HUMAN_PN[f"{f}_wass"] for f in FEATURES) / len(FEATURES),
    }
    for feature in FEATURES:
        row[f"PP_{feature}"] = metrics["pp_wass"][f"{feature}_wass"]
        row[f"PN_{feature}"] = metrics["pn_wass"][f"{feature}_wass"]
    row["num_scores"] = int(sampling["num_scores"])
    row["samples_per_score"] = int(sampling["num_samples"])
    row["source"] = path.as_posix()
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    rows = [make_row(model, configuration, path) for model, configuration, path in RUNS]
    output = {
        "human_reference_pp": HUMAN_PP,
        "human_reference_pn": HUMAN_PN,
        "hr_definition": "HR is the mean of four ratios (IOI, duration, velocity, pedal), each divided by the matching score-span Human leave-one-out Wasserstein reference.",
        "human_reference_source": "backup/repo_cleanup_20260804/results/external_eval_lengthnorm_20260724/human_leave_one_score_span_eval_binary.json",
        "rows": rows,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "full_cinr_dinr_hr_pn_table.json"
    csv_path = args.output_dir / "full_cinr_dinr_hr_pn_table.csv"
    md_path = args.output_dir / "full_cinr_dinr_hr_pn_table.md"
    json_path.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    fieldnames = list(rows[0])
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    columns = ["model", "configuration", "PP_HR", "PN_HR"] + [f"PP_{f}" for f in FEATURES] + [f"PN_{f}" for f in FEATURES]
    labels = ["Model", "Configuration", "PP HR", "PN HR"] + [f"PP {f.title()}" for f in FEATURES] + [f"PN {f.title()}" for f in FEATURES]
    lines = [
        "| " + " | ".join(labels) + " |",
        "|" + "|".join(["---"] * 2 + ["---:"] * (len(labels) - 2)) + "|",
    ]
    for row in rows:
        values = [row["model"], row["configuration"]]
        values += [f"{row[key]:.3f}" for key in columns[2:4]]
        values += [f"{row[key]:.2f}" for key in columns[4:]]
        lines.append("| " + " | ".join(values) + " |")
    lines += ["", "HR uses the score-span-normalized Human leave-one-out anchors from `human_leave_one_score_span_eval_binary.json`. Raw Wasserstein values are averaged over 19 test scores and 3 generated samples per score."]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(rows), "json": str(json_path), "csv": str(csv_path), "markdown": str(md_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
