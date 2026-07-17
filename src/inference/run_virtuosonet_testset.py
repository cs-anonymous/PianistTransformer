import argparse
import json
import multiprocessing as mp
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
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--num-workers", type=int, default=1)
    return p.parse_args()


def composer_from_source(source):
    return source.split("/", 1)[0].split(",", 1)[0].replace("_", " ")


def collect_items(args):
    allowed = {
        line.strip()
        for line in args.score_source_list.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    df = pd.read_csv(args.metadata, usecols=[
        "tier_a", "split", "refined_score_midi_path",
        "refined_performance_midi_path", "performance_dataset",
    ])
    df = df[
        df["tier_a"].fillna(False).astype(bool)
        & df["split"].eq("test")
        & df["performance_dataset"].fillna("").astype(str).eq("ASAP")
        & df["refined_score_midi_path"].isin(allowed)
    ]
    items = []
    for score_source in sorted(allowed):
        row = df[df["refined_score_midi_path"].eq(score_source)]
        if row.empty:
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
            for p in sorted(row["refined_performance_midi_path"].dropna().unique())
        ]
        items.append({
            "score_source": score_source,
            "xml_path": str(xml_path.resolve()),
            "xml_source": xml_source,
            "composer": composer_from_source(score_source),
            "ground_truth_paths": gt_paths,
        })
    return items


def worker(worker_idx, args, jobs, results):
    import torch
    from virtuoso.inference import InferenceModel

    torch.set_num_threads(1)
    device = args.device
    model = InferenceModel(
        str(args.checkpoint),
        device,
        args.output_dir,
        {"bool_pedal": True},
    )
    while True:
        job = jobs.get()
        if job is None:
            return
        idx, item = job
        predictions = []
        timings = []
        for sample_idx in range(2):
            # InferenceModel.infer_xml is fixed to initial_z='zero'; use the
            # underlying model for the second stochastic style sample.
            start = time.perf_counter()
            xml_path = Path(item["xml_path"])
            out = args.output_dir / "midis" / f"{idx:02d}__sample_{sample_idx:03d}.mid"
            score, inputs, edges, locations = model.get_input_from_xml(
                xml_path, item["composer"], None
            )
            initial_z = "zero"
            if sample_idx == 1:
                initial_z = model.model.sample_style_vector_from_normal_distribution(inputs.shape[0])
            with torch.no_grad():
                outputs, _, _, _ = model.model(
                    inputs, None, edges, locations, initial_z=initial_z
                )
            outputs = model.scale_model_prediction_to_original(outputs)
            features = model.model_prediction_to_feature(outputs)
            model.midi_decoder(score, locations, features, out)
            predictions.append(str(out.resolve()))
            timings.append(time.perf_counter() - start)
        results.put((idx, {
            "score_source": item["score_source"],
            "xml_path": item["xml_path"],
            "xml_source": item["xml_source"],
            "score_midi": str((args.midi_root / "refined" / item["score_source"]).resolve()),
            "prediction_paths": predictions,
            "ground_truth_paths": item["ground_truth_paths"],
            "inference_seconds": timings,
        }))


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    items = collect_items(args)
    ctx = mp.get_context("spawn")
    jobs, results = ctx.Queue(), ctx.Queue()
    workers = [
        ctx.Process(target=worker, args=(i, args, jobs, results))
        for i in range(args.num_workers)
    ]
    for w in workers:
        w.start()
    for i, item in enumerate(items):
        jobs.put((i, item))
    for _ in workers:
        jobs.put(None)
    collected = {}
    for _ in items:
        idx, item = results.get()
        collected[idx] = item
        print(f"completed {idx + 1}/{len(items)}", flush=True)
    for w in workers:
        w.join()
        if w.exitcode:
            raise RuntimeError(f"worker {w.pid} exited with {w.exitcode}")
    manifest = {
        "model": "VirtuosoNet HAN+GRU",
        "protocol": "virtuosonet_han_gru",
        "checkpoint": str(args.checkpoint.resolve()),
        "score_source_list": str(args.score_source_list.resolve()),
        "split": "test",
        "gt_filter": "performance_dataset=ASAP",
        "num_samples": 2,
        "sample_initial_z": ["zero", "normal"],
        "generated_xml_root": str(args.generated_xml_root.resolve()),
        "num_workers": args.num_workers,
        "items": [collected[i] for i in range(len(items))],
        "total_inference_seconds": sum(
            sum(x["inference_seconds"]) for x in collected.values()
        ),
    }
    (args.output_dir / "prediction_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
