#!/usr/bin/env python3
"""Run the public SyMuPe/PianoFlow model on the repository's 19-score test set.

The generated MIDI files are kept separate from the INR pipeline, then a
PianistTransformer-compatible prediction manifest is written for the existing
PN/PP evaluator.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
from symupe import AutoGenerator


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_MANIFEST = ROOT / "results/cinr_overlap50_24ep_20261026_200500/cinr_slot_sum/sampling/prediction_manifest.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--model", default="SyMuPe/PianoFlow-base")
    p.add_argument("--num-samples", type=int, default=3)
    p.add_argument("--seed", type=int, default=23)
    p.add_argument("--worker", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=1)
    p.add_argument("--score-list", type=Path, default=None)
    return p.parse_args()


def load_items(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload["items"]
    if len(items) != 19:
        raise ValueError(f"Expected the 19-score test manifest, got {len(items)} items")
    return items


def safe_name(score_source: str) -> str:
    return score_source.replace("/", "__").replace(":", "_")


def main() -> None:
    args = parse_args()
    items = load_items(args.source_manifest)
    if args.score_list is not None:
        wanted = {
            line.strip()
            for line in args.score_list.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
        items = [item for item in items if item["score_source"] in wanted]

    selected = [item for idx, item in enumerate(items) if idx % args.num_workers == args.worker]
    if not selected:
        print(f"worker={args.worker}: no assigned scores", flush=True)
        return

    out_root = args.output_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"worker={args.worker} device={device} loading {args.model}", flush=True)
    generator = AutoGenerator.from_pretrained(args.model, device=device)

    for item in selected:
        score_source = item["score_source"]
        score_path = ROOT / "data/ASAP_processed" / score_source
        if not score_path.exists():
            raise FileNotFoundError(score_path)
        score_dir = out_root / "midis" / safe_name(score_source)
        score_dir.mkdir(parents=True, exist_ok=True)
        marker = score_dir / "done.json"
        if marker.exists():
            print(f"skip completed {score_source}", flush=True)
            continue
        start = time.perf_counter()
        results = generator.perform_score(
            str(score_path),
            use_score_context=True,
            num_samples=args.num_samples,
            seed=args.seed,
            show_progress=True,
        )
        prediction_paths = []
        for sample_idx, result in enumerate(results):
            path = score_dir / f"sample_{sample_idx:03d}.mid"
            result.perf_midi.dump_midi(str(path))
            prediction_paths.append(str(path.resolve()))
        marker.write_text(
            json.dumps(
                {
                    "score_source": score_source,
                    "score_midi": str(score_path.resolve()),
                    "prediction_paths": prediction_paths,
                    "elapsed_seconds": time.perf_counter() - start,
                    "num_samples": len(prediction_paths),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"finished {score_source} elapsed={time.perf_counter() - start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
