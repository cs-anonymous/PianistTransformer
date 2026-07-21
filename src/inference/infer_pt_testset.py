import argparse
import hashlib
import json
import multiprocessing as mp
import random
import sys
from pathlib import Path

import pandas as pd
import torch
from miditoolkit import MidiFile
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.model.generate import batch_performance_render, map_midi
from src.model.pianoformer import PianoT5Gemma
from src.utils.midi import midi_to_ids


def parse_args():
    parser = argparse.ArgumentParser(description="Run PT inference on PianoCoRe test scores.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, default=Path("data/ASAP_processed/metadata.generated_json.csv"))
    parser.add_argument("--midi-root", type=Path, default=Path("/home/sy/EPR/PianoCoRe/refined"))
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-works", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--protocol", choices=["deterministic", "sampling"], default="sampling")
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--overlap-ratio", type=float, default=0.125)
    parser.add_argument("--max-context-length", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--performance-dataset",
        type=str,
        default=None,
        help="Optional performance_dataset filter, e.g. ASAP. Restricts scores and GT refs.",
    )
    parser.add_argument(
        "--score-source-list",
        type=Path,
        default=None,
        help="Optional newline-delimited refined_score_midi_path list to restrict evaluated scores.",
    )
    return parser.parse_args()


def select_device(device_arg):
    if device_arg:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def collect_test_items(metadata_path, midi_root, split, performance_dataset=None, score_source_list=None):
    columns = [
        "tier_a",
        "split",
        "performance_dataset",
        "refined_score_midi_path",
        "refined_performance_midi_path",
    ]
    df = pd.read_csv(metadata_path, usecols=columns)
    df = df[df["tier_a"].fillna(False).astype(bool)]
    df = df[df["split"] == split]
    df = df[df["refined_score_midi_path"].notna()]
    df = df[df["refined_performance_midi_path"].notna()]
    if performance_dataset is not None:
        df = df[df["performance_dataset"].fillna("").astype(str) == str(performance_dataset)]
    if score_source_list is not None:
        allowed = [
            line.strip()
            for line in score_source_list.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        df = df[df["refined_score_midi_path"].isin(set(allowed))]
    df = df.sort_values(["refined_score_midi_path", "refined_performance_midi_path"], kind="stable")

    items = []
    for score_source, group in df.groupby("refined_score_midi_path", sort=True):
        score_path = midi_root / score_source
        gt_paths = [(midi_root / path) for path in sorted(group["refined_performance_midi_path"].unique())]
        if not score_path.exists():
            raise FileNotFoundError(f"Missing score MIDI: {score_path}")
        missing_gt = [path for path in gt_paths if not path.exists()]
        if missing_gt:
            raise FileNotFoundError(f"Missing GT MIDI for {score_source}: {missing_gt[0]}")
        items.append(
            {
                "score_source": score_source,
                "score_path": score_path,
                "gt_paths": gt_paths,
            }
        )
    return items


def safe_name(score_source):
    return Path(score_source).with_suffix("").as_posix().replace("/", "__")


def stable_seed(base_seed, *parts):
    payload = "::".join([str(base_seed), *[str(part) for part in parts]])
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def load_model(model_path, device):
    print(f"Loading model from {model_path}", flush=True)
    model = PianoT5Gemma.from_pretrained(
        str(model_path),
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
    )
    model.to(device)
    model.eval()
    return model


def render_one_entry(model, device, entry, args, midi_dir):
    score_midi = MidiFile(str(entry["score_path"]))
    prediction_paths = []
    for sample_idx in range(args.num_samples):
        sample_seed = stable_seed(args.seed, entry["score_source"], sample_idx)
        random.seed(sample_seed)
        torch.manual_seed(sample_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(sample_seed)

        rendered = batch_performance_render(
            model,
            [score_midi],
            max_context_length=args.max_context_length,
            overlap_ratio=args.overlap_ratio,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=args.protocol == "sampling",
            device=str(device),
        )[0]
        mapped = map_midi(score_midi, rendered)
        pred_path = midi_dir / f"{safe_name(entry['score_source'])}__sample_{sample_idx:03d}.mid"
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        mapped.dump(str(pred_path))
        prediction_paths.append(str(pred_path.resolve()))

    gt_paths = [str(path.resolve()) for path in entry["gt_paths"]]
    manifest_item = {
        "score_source": entry["score_source"],
        "score_midi": str(entry["score_path"].resolve()),
        "prediction_paths": prediction_paths,
        "ground_truth_paths": gt_paths,
    }
    pair_rows = [
        {"pred": pred_path, "gt": gt_path}
        for pred_path in prediction_paths
        for gt_path in gt_paths
    ]
    return manifest_item, pair_rows


def worker_loop(worker_idx, args, job_queue, result_queue):
    random.seed(args.seed + worker_idx)
    torch.manual_seed(args.seed + worker_idx)
    device = select_device(args.device)
    print(f"Worker {worker_idx} using device: {device}", flush=True)
    model = load_model(args.model_path, device)
    midi_dir = args.output_dir / "midis"
    midi_dir.mkdir(parents=True, exist_ok=True)

    while True:
        job = job_queue.get()
        if job is None:
            break
        job_idx, entry = job
        try:
            manifest_item, pair_rows = render_one_entry(model, device, entry, args, midi_dir)
            result_queue.put((job_idx, manifest_item, pair_rows, None))
        except Exception as exc:  # noqa: BLE001 - preserve worker traceback in parent log.
            result_queue.put((job_idx, None, None, repr(exc)))


def run_dynamic_pool(args, items):
    ctx = mp.get_context("spawn")
    job_queue = ctx.Queue()
    result_queue = ctx.Queue()
    workers = [
        ctx.Process(target=worker_loop, args=(idx, args, job_queue, result_queue))
        for idx in range(args.num_workers)
    ]
    for worker in workers:
        worker.start()
    for job_idx, item in enumerate(items):
        job_queue.put((job_idx, item))
    for _ in workers:
        job_queue.put(None)

    manifest_by_idx = {}
    pair_list = []
    with tqdm(total=len(items), desc="PT test inference pool") as progress:
        for _ in range(len(items)):
            job_idx, manifest_item, pair_rows, error = result_queue.get()
            if error is not None:
                for worker in workers:
                    worker.terminate()
                raise RuntimeError(f"Worker failed on job {job_idx}: {error}")
            manifest_by_idx[job_idx] = manifest_item
            pair_list.extend(pair_rows)
            progress.update(1)

    for worker in workers:
        worker.join()
        if worker.exitcode != 0:
            raise RuntimeError(f"Worker {worker.pid} exited with code {worker.exitcode}")

    return [manifest_by_idx[idx] for idx in range(len(items))], pair_list


def run_single_process(args, items, model):
    midi_dir = args.output_dir / "midis"
    midi_dir.mkdir(parents=True, exist_ok=True)

    work_entries = []
    for item in tqdm(items, desc="Loading score MIDIs"):
        score_midi = MidiFile(str(item["score_path"]))
        work_entries.append(
            {
                **item,
                "score_midi": score_midi,
                "token_len": len(midi_to_ids(model.config, score_midi)),
            }
        )
    work_entries.sort(key=lambda entry: entry["token_len"])

    manifest_by_score = {}
    pair_list = []
    device = select_device(args.device)
    for batch_start in tqdm(range(0, len(work_entries), args.batch_size), desc="PT test inference batches"):
        batch = work_entries[batch_start : batch_start + args.batch_size]
        score_midis = [entry["score_midi"] for entry in batch]
        batch_predictions = [[] for _ in batch]

        for sample_idx in range(args.num_samples):
            rendered_batch = batch_performance_render(
                model,
                score_midis,
                max_context_length=args.max_context_length,
                overlap_ratio=args.overlap_ratio,
                temperature=args.temperature,
                top_p=args.top_p,
                do_sample=args.protocol == "sampling",
                device=str(device),
            )
            for entry_idx, (entry, rendered) in enumerate(zip(batch, rendered_batch)):
                mapped = map_midi(entry["score_midi"], rendered)
                pred_path = midi_dir / f"{safe_name(entry['score_source'])}__sample_{sample_idx:03d}.mid"
                pred_path.parent.mkdir(parents=True, exist_ok=True)
                mapped.dump(str(pred_path))
                batch_predictions[entry_idx].append(str(pred_path.resolve()))

        for entry, prediction_paths in zip(batch, batch_predictions):
            gt_paths = [str(path.resolve()) for path in entry["gt_paths"]]
            manifest_by_score[entry["score_source"]] = {
                "score_source": entry["score_source"],
                "score_midi": str(entry["score_path"].resolve()),
                "prediction_paths": prediction_paths,
                "ground_truth_paths": gt_paths,
                "token_len": entry["token_len"],
            }
            for pred_path in prediction_paths:
                for gt_path in gt_paths:
                    pair_list.append({"pred": pred_path, "gt": gt_path})
    return [manifest_by_score[item["score_source"]] for item in items], pair_list


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.num_workers < 1:
        raise ValueError("--num-workers must be >= 1")

    items = collect_test_items(
        metadata_path=args.metadata,
        midi_root=args.midi_root,
        split=args.split,
        performance_dataset=args.performance_dataset,
        score_source_list=args.score_source_list,
    )
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must satisfy 0 <= shard-index < num-shards")
    total_items = len(items)
    items = items[args.shard_index :: args.num_shards]
    if args.max_works is not None:
        items = items[: args.max_works]
    print(f"PT inference works: {len(items):,} / {total_items:,} total")
    print(f"Shard: {args.shard_index}/{args.num_shards}")
    print(f"GT pairs: {sum(len(item['gt_paths']) for item in items):,}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.num_workers > 1:
        manifest_items, pair_list = run_dynamic_pool(args, items)
    else:
        device = select_device(args.device)
        print(f"Using device: {device}")
        model = load_model(args.model_path, device)
        manifest_items, pair_list = run_single_process(args, items, model)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "model_path": str(args.model_path.resolve()),
        "metadata": str(args.metadata.resolve()),
        "midi_root": str(args.midi_root.resolve()),
        "protocol": f"pt_{args.protocol}",
        "split": args.split,
        "num_samples": args.num_samples,
        "num_workers": args.num_workers,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "overlap_ratio": args.overlap_ratio,
        "max_context_length": args.max_context_length,
        "score_source_list": str(args.score_source_list.resolve()) if args.score_source_list else None,
        "items": manifest_items,
    }
    manifest_path = args.output_dir / "prediction_manifest.json"
    pair_list_path = args.output_dir / "evaluate_list.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    pair_list_path.write_text(json.dumps(pair_list, indent=2, ensure_ascii=False))
    print(f"Saved prediction manifest to {manifest_path}")
    print(f"Saved pair list to {pair_list_path}")


if __name__ == "__main__":
    main()
