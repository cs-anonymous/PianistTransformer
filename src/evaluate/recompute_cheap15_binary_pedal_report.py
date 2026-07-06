import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluate.evaluate_inr_saved_midis import (
    aggregate_score_metrics,
    evaluate_manifest,
    load_manifest_and_config,
    load_score_source_filter,
)

DEFAULT_SCORE_LIST = ROOT / "results/psr_oracle/window_style_prefix_enc_add/cheap15_score_sources.txt"


CURATED_SUBSET_MANIFESTS = [
    ("base_tf_sample", ROOT / "results/asap_full_compare/prefix_enc_add/tf_test_sample_best/prediction_manifest.json"),
    ("base_ar", ROOT / "results/asap_full_compare/prefix_enc_add/ar_test_sample_best/prediction_manifest.json"),
    ("gt_overlap_0125", ROOT / "results/asap_full_compare/prefix_enc_add/ar_gt_overlap_decoder_0125/prediction_manifest.json"),
    ("gt_overlap_050", ROOT / "results/asap_full_compare/prefix_enc_add/ar_gt_overlap_decoder_050/prediction_manifest.json"),
    ("gt_overlap_style_0125", ROOT / "results/asap_full_compare/prefix_enc_add/ar_gt_overlap_decoder_style_0125/prediction_manifest.json"),
    ("gt_overlap_style_050", ROOT / "results/asap_full_compare/prefix_enc_add/ar_gt_overlap_decoder_style_050/prediction_manifest.json"),
    ("psr_tf_sample", ROOT / "results/psr_oracle/window_style_prefix_enc_add/tf_test_sample_best/prediction_manifest.json"),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Recompute cheap15 experiments with binary-pedal PN/PP Wass.")
    parser.add_argument("--score-source-list", type=Path, default=DEFAULT_SCORE_LIST)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/analysis/cheap15_binary_pedal_recompute",
    )
    parser.add_argument("--num-workers", type=int, default=12)
    return parser.parse_args()


def discover_direct_cheap15_manifests():
    manifests = []
    for manifest_path in sorted(ROOT.glob("results/**/prediction_manifest.json")):
        text = str(manifest_path)
        if "cheap15" not in text:
            continue
        if any(token in text for token in ("diagnostics", "smoke")):
            continue
        label = manifest_path.parent.relative_to(ROOT / "results").as_posix().replace("/", "__")
        manifests.append((label, manifest_path))
    return manifests


def build_manifest_table():
    seen = set()
    rows = []
    for label, manifest_path in CURATED_SUBSET_MANIFESTS + discover_direct_cheap15_manifests():
        if not manifest_path.exists():
            continue
        if manifest_path in seen:
            continue
        seen.add(manifest_path)
        rows.append((label, manifest_path))
    return rows


def summarize_row(label, manifest_path, result, pedal_support):
    aggregate = result["aggregate"]
    pp = aggregate["pp_wass"]
    pn = aggregate["pn_wass"]
    return {
        "experiment": label,
        "manifest": str(manifest_path.relative_to(ROOT)),
        "pedal_metric_support": pedal_support,
        "num_scores": result["num_scores"],
        "pp_ioi_wass": pp.get("ioi_wass"),
        "pp_duration_wass": pp.get("duration_wass"),
        "pp_velocity_wass": pp.get("velocity_wass"),
        "pp_pedal_wass": pp.get("pedal_wass"),
        "pn_ioi_wass": pn.get("ioi_wass"),
        "pn_duration_wass": pn.get("duration_wass"),
        "pn_velocity_wass": pn.get("velocity_wass"),
        "pn_pedal_wass": pn.get("pedal_wass"),
    }


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    score_source_list = load_score_source_filter(args.score_source_list)
    manifest_rows = build_manifest_table()
    summary_rows = []
    failures = []

    for label, manifest_path in manifest_rows:
        try:
            manifest, config = load_manifest_and_config(manifest_path, score_source_list=score_source_list)
            pedal_binary_support = str(config.get("pedal_representation", "")).lower() == "binary_4"
            evaluated = evaluate_manifest(
                manifest,
                max_gt_per_score=None,
                num_workers=args.num_workers,
                pedal_binary_support=pedal_binary_support,
                pedal_binary_threshold=float(config.get("pedal_binary_threshold", 64.0)),
            )
            score_rows = evaluated["score_rows"]
            result = {
                "prediction_manifest": str(manifest_path.resolve()),
                "num_scores": len(score_rows),
                "pedal_metric_support": "binary_0_1" if pedal_binary_support else "raw_0_127",
                "aggregate": {
                    "pn_wass": aggregate_score_metrics(score_rows, "pn_wass"),
                    "pp_wass": aggregate_score_metrics(score_rows, "pp_wass"),
                },
                "scores": score_rows,
            }
            output_json = args.output_dir / f"{label}.json"
            output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
            summary_rows.append(summarize_row(label, manifest_path, result, result["pedal_metric_support"]))
        except Exception as exc:  # noqa: BLE001
            failures.append({"experiment": label, "manifest": str(manifest_path), "error": repr(exc)})

    summary_rows.sort(key=lambda row: row["experiment"])
    csv_path = args.output_dir / "summary.csv"
    if summary_rows:
        with csv_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
    json_path = args.output_dir / "summary.json"
    json_path.write_text(
        json.dumps({"rows": summary_rows, "failures": failures}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps({"rows": summary_rows, "failures": failures}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
