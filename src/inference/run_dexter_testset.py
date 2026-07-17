import argparse
import json
import multiprocessing as mp
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--metadata", type=Path, default=Path("PianoCoRe/metadata.csv"))
    p.add_argument("--midi-root", type=Path, default=Path("PianoCoRe"))
    p.add_argument("--score-source-list", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--generated-xml-root", type=Path, default=Path("results/external_eval_20260717/generated_musicxml"))
    p.add_argument("--dexter-root", type=Path, default=Path("external/DExter"))
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--gpus", default="0,1,2")
    p.add_argument("--workers-per-gpu", type=int, default=8)
    p.add_argument("--num-samples", type=int, default=2)
    return p.parse_args()


def collect_items(args):
    allowed = {
        line.strip()
        for line in args.score_source_list.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    df = pd.read_csv(
        args.metadata,
        usecols=[
            "tier_a",
            "split",
            "refined_score_midi_path",
            "refined_performance_midi_path",
            "performance_dataset",
        ],
    )
    df = df[
        df["tier_a"].fillna(False).astype(bool)
        & df["split"].eq("test")
        & df["performance_dataset"].fillna("").astype(str).eq("ASAP")
        & df["refined_score_midi_path"].isin(allowed)
    ]
    items = []
    for score_source in sorted(allowed):
        rows = df[df["refined_score_midi_path"].eq(score_source)]
        if rows.empty:
            raise FileNotFoundError(f"Missing ASAP metadata row for {score_source}")
        rel = Path(score_source)
        xml_path = args.midi_root / "raw" / rel.parent / "score.musicxml"
        xml_source = "raw"
        if not xml_path.exists():
            xml_path = args.generated_xml_root / rel.with_suffix(".musicxml")
            xml_source = "generated_from_refined_score_midi"
        if not xml_path.exists():
            raise FileNotFoundError(f"Missing MusicXML and generated fallback: {score_source}")
        gt_paths = [
            str((args.midi_root / "refined" / p).resolve())
            for p in sorted(rows["refined_performance_midi_path"].dropna().unique())
        ]
        items.append(
            {
                "score_source": score_source,
                "xml_path": str(xml_path.resolve()),
                "xml_source": xml_source,
                "score_midi": str((args.midi_root / "refined" / score_source).resolve()),
                "ground_truth_paths": gt_paths,
            }
        )
    return items


def hydra_str(key, value):
    return f"{key}='{value}'"


def run_one(args, item, score_idx, sample_idx, gpu):
    out = args.output_dir / "midis" / f"{score_idx:02d}__sample_{sample_idx:03d}.mid"
    log = args.output_dir / "logs" / f"{score_idx:02d}__sample_{sample_idx:03d}.log"
    out.parent.mkdir(parents=True, exist_ok=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "inference.py",
        hydra_str("score_path", item["xml_path"]),
        hydra_str("pretrained_path", str(args.checkpoint.resolve())),
        hydra_str("output_path", str(out.resolve())),
        f"gpus=[{gpu}]",
    ]
    env = os.environ.copy()
    env["WANDB_MODE"] = "disabled"
    start = time.perf_counter()
    with log.open("w", encoding="utf-8") as f:
        proc = subprocess.run(
            cmd,
            cwd=str(args.dexter_root),
            stdout=f,
            stderr=subprocess.STDOUT,
            env=env,
        )
    elapsed = time.perf_counter() - start
    ok = out.exists() and out.stat().st_size > 0
    if proc.returncode != 0 and not ok:
        raise RuntimeError(f"DExter failed for {item['score_source']} sample {sample_idx}; see {log}")
    return str(out.resolve()), elapsed, proc.returncode, str(log.resolve())


def worker(worker_idx, gpu, args, jobs, results):
    while True:
        job = jobs.get()
        if job is None:
            return
        score_idx, item = job
        predictions = []
        timings = []
        returncodes = []
        logs = []
        for sample_idx in range(args.num_samples):
            pred, elapsed, returncode, log = run_one(args, item, score_idx, sample_idx, gpu)
            predictions.append(pred)
            timings.append(elapsed)
            returncodes.append(returncode)
            logs.append(log)
        results.put(
            (
                score_idx,
                {
                    **item,
                    "prediction_paths": predictions,
                    "inference_seconds": timings,
                    "returncodes": returncodes,
                    "logs": logs,
                    "gpu": gpu,
                    "worker_idx": worker_idx,
                },
            )
        )


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    items = collect_items(args)
    gpus = [int(x) for x in args.gpus.split(",") if x.strip()]
    ctx = mp.get_context("spawn")
    jobs, results = ctx.Queue(), ctx.Queue()
    workers = []
    for gpu in gpus:
        for _ in range(args.workers_per_gpu):
            worker_idx = len(workers)
            proc = ctx.Process(target=worker, args=(worker_idx, gpu, args, jobs, results))
            proc.start()
            workers.append(proc)
    wall_start = time.perf_counter()
    for i, item in enumerate(items):
        jobs.put((i, item))
    for _ in workers:
        jobs.put(None)
    collected = {}
    for n in range(len(items)):
        idx, item = results.get()
        collected[idx] = item
        print(f"completed {n + 1}/{len(items)} score={idx:02d} gpu={item['gpu']}", flush=True)
    for proc in workers:
        proc.join()
        if proc.exitcode:
            raise RuntimeError(f"worker {proc.pid} exited with {proc.exitcode}")
    wall_elapsed = time.perf_counter() - wall_start
    manifest = {
        "model": "DExter",
        "protocol": "dexter",
        "checkpoint": str(args.checkpoint.resolve()),
        "score_source_list": str(args.score_source_list.resolve()),
        "split": "test",
        "gt_filter": "performance_dataset=ASAP",
        "num_samples": args.num_samples,
        "gpus": gpus,
        "workers_per_gpu": args.workers_per_gpu,
        "generated_xml_root": str(args.generated_xml_root.resolve()),
        "items": [collected[i] for i in range(len(items))],
        "total_inference_seconds": sum(sum(x["inference_seconds"]) for x in collected.values()),
        "wall_inference_seconds": wall_elapsed,
    }
    (args.output_dir / "prediction_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
