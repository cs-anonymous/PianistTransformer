"""Build manifests that point to the original, unscaled prediction MIDIs."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


def build(name: str, source_path: Path, output_root: Path) -> Path:
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    payload = copy.deepcopy(payload)
    for item in payload["items"]:
        records = item.get("length_normalized_predictions")
        if records:
            item["prediction_paths"] = [record["source"] for record in records]
        item["prediction_time_normalization"] = "none"
    payload["prediction_time_normalization"] = "none"
    payload["unscaled_source_manifest"] = str(source_path.resolve())
    output = output_root / name / "prediction_manifest.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(name, len(payload["items"]), "scores ->", output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", action="append", required=True, metavar="NAME=MANIFEST")
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    for spec in args.method:
        name, source = spec.split("=", 1)
        build(name, Path(source), args.output_root)


if __name__ == "__main__":
    main()
