#!/usr/bin/env python3
"""Refresh the live training plot when logs or validation checkpoints change."""
import argparse
import subprocess
import sys
import time
from pathlib import Path


def signature(paths):
    return tuple((path.stat().st_size, path.stat().st_mtime_ns) if path.exists() else None for path in paths)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--interval", type=int, default=60)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    script = Path(__file__).with_name("plot_pianocoreA_training_curves.py")
    log = run / "logs/stage1_train.log"
    validations = [run / f"validation_hr/pianocoreA_pretrain_epoch-{epoch}.json" for epoch in range(1, 5)]
    paths = [log, *validations]
    previous = None
    while True:
        current = signature(paths)
        if current != previous and log.exists():
            result = subprocess.run([sys.executable, str(script), "--run-dir", str(run)], check=False)
            if result.returncode:
                raise SystemExit(result.returncode)
            previous = current
        finished = (run / "stage1_training_complete").exists() and all(path.exists() for path in validations)
        if finished or (run / "FAILED").exists() or (run / "COMPLETE").exists():
            break
        time.sleep(max(10, args.interval))


if __name__ == "__main__":
    main()
