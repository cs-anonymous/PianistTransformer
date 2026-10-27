#!/usr/bin/env python3
"""Summarize training logs and run manifests for the PN cached-W1 matrix."""
import argparse
import json
import re
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def last_json_event(path):
    last = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "step" in payload or "train_runtime" in payload:
            last = payload
    return last or {}


def main():
    args = parse_args()
    root = args.run_root
    rows = []
    for log_path in sorted((root / "logs").glob("*_seed_*.log")):
        match = re.match(r"(.+)_seed_(\d+)\.log$", log_path.name)
        if not match:
            continue
        condition, seed = match.group(1), int(match.group(2))
        payload = last_json_event(log_path)
        rows.append({
            "condition": condition,
            "seed": seed,
            "status": "failed" if "Traceback" in log_path.read_text(encoding="utf-8", errors="replace") else "done",
            "step": payload.get("step"),
            "train_loss": payload.get("train_loss"),
            "pn_cached_velocity_cdf_w1": payload.get("train_loss_pn_cached_velocity_cdf_w1"),
            "velocity_dlm_nll": payload.get("train_loss_velocity_dlm_nll"),
            "train_runtime": payload.get("train_runtime"),
        })
    output = {"run_root": str(root), "rows": rows}
    text = json.dumps(output, indent=2, ensure_ascii=False) + "\n"
    if args.output is None:
        print(text, end="")
    else:
        args.output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
