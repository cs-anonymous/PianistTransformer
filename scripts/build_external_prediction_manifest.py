#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-manifest", type=Path, required=True)
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--protocol", default="external")
    parser.add_argument("--model-name", default="external")
    return parser.parse_args()


def safe_stem(score_source):
    path = Path(score_source)
    parts = list(path.parts[:-1])
    score_stem = path.stem
    return "__".join(parts + [score_stem]).replace("/", "__")


def main():
    args = parse_args()
    reference = json.loads(args.reference_manifest.read_text(encoding="utf-8"))
    items = []
    missing = []
    for item in reference["items"]:
        stem = safe_stem(item["score_source"])
        candidates = sorted(args.prediction_dir.glob(f"{stem}*.mid"))
        if not candidates:
            missing.append(item["score_source"])
            continue
        items.append(
            {
                "score_source": item["score_source"],
                "score_midi": item["score_midi"],
                "prediction_paths": [str(path.resolve()) for path in candidates],
                "ground_truth_paths": item["ground_truth_paths"],
                "original_ground_truth_paths": item.get("original_ground_truth_paths", item["ground_truth_paths"]),
                "external_model": args.model_name,
            }
        )
    if missing:
        raise SystemExit(f"Missing {len(missing)} predictions, first: {missing[0]}")
    output = {
        "config": None,
        "task_type": "epr",
        "protocol": args.protocol,
        "num_samples": len(items[0]["prediction_paths"]) if items else 0,
        "items": items,
    }
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Wrote {args.output_manifest} with {len(items)} items")


if __name__ == "__main__":
    main()
