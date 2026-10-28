#!/usr/bin/env python3
"""Train CINR without pedal, then evaluate every epoch on one GPU."""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--epochs", type=int, default=24)
    args = parser.parse_args()
    run = args.run_root.resolve()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="2", PYTHONUNBUFFERED="1",
               OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    config = run / "config.json"
    logs = run / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    status = {"gpu": 2, "config": str(config), "epochs": args.epochs,
              "pid": os.getpid(), "started_at": dt.datetime.now().isoformat()}
    def event(stage, **extra):
        status.update(stage=stage, updated_at=dt.datetime.now().isoformat(), **extra)
        tmp = run / "status.tmp"
        tmp.write_text(json.dumps(status, indent=2) + "\n")
        tmp.replace(run / "status.json")
        with (run / "runs.jsonl").open("a") as f:
            f.write(json.dumps(status) + "\n")
        print(json.dumps(status), flush=True)
    def execute(command, logfile):
        start = time.monotonic()
        with (logs / logfile).open("a") as f:
            f.write("\nCOMMAND: " + " ".join(map(str, command)) + "\n")
            f.flush()
            subprocess.run(list(map(str, command)), cwd=ROOT, env=env,
                           stdout=f, stderr=subprocess.STDOUT, check=True)
        return time.monotonic() - start
    try:
        if not args.skip_train:
            event("training")
            seconds = execute([sys.executable, "src/train/train_inr.py", "--config", config],
                              "train.log")
            event("trained", train_seconds=seconds)
        train_dir = run / "training" / "cinr_slot_sum_no_pedal"
        for epoch in range(1, args.epochs + 1):
            checkpoint = train_dir / f"epoch-{epoch}"
            state = json.loads((checkpoint / "trainer_state.json").read_text())
            if abs(float(state["epoch"]) - epoch) > 1e-5:
                raise ValueError(f"Checkpoint {checkpoint} is at epoch {state['epoch']}")
            result = run / f"epoch-{epoch}" / "metrics.json"
            if result.exists():
                event("evaluation_reused", epoch=epoch, checkpoint=str(checkpoint))
                continue
            event("evaluating", epoch=epoch, checkpoint=str(checkpoint), sampling_seed=42)
            seconds = execute([
                sys.executable, "src/evaluate/eval_cinr_no_pedal_checkpoint.py",
                "--config", config, "--checkpoint", checkpoint,
                "--output-dir", run / f"epoch-{epoch}",
            ], f"eval_epoch_{epoch:02d}.log")
            event("evaluated", epoch=epoch, last_eval_seconds=seconds)
            execute([sys.executable, "src/evaluate/summarize_cinr_no_pedal_epochs.py",
                     "--run-root", run, "--epochs", args.epochs], "summary.log")
        execute([sys.executable, "src/evaluate/summarize_cinr_no_pedal_epochs.py",
                 "--run-root", run, "--epochs", args.epochs, "--require-complete"], "summary.log")
        event("completed")
    except Exception as exc:
        event("failed", error=str(exc))
        raise

if __name__ == "__main__":
    main()
