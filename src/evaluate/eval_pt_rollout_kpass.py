import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from miditoolkit import MidiFile
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.inference.infer_pt_testset import collect_test_items, safe_name
from src.model.generate import map_midi
from src.model.pianoformer import PianoT5Gemma
from src.utils.inr_midi import midi_to_note_features, sorted_piano_notes
from src.utils.midi import ids_to_midi, midi_to_ids


def parse_args():
    parser = argparse.ArgumentParser(description="PT k-pass rollout evaluation with GT/predicted token feedback.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, default=Path("../PianoCoRe/metadata.csv"))
    parser.add_argument("--midi-root", type=Path, default=Path("../PianoCoRe/refined"))
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--performance-dataset", type=str, default="ASAP")
    parser.add_argument("--score-source-list", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rollout-ks", type=str, default="0,1,4")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-context-length", type=int, default=4096)
    parser.add_argument("--overlap-ratio", type=float, default=0.125)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--do-sample", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def stable_seed(base_seed, *parts):
    payload = "::".join([str(base_seed), *[str(part) for part in parts]])
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def parse_ks(text):
    out = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    if not out:
        raise ValueError("No rollout k values provided")
    return sorted(set(out))


def load_score_source_filter(path):
    if path is None:
        return None
    values = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            values.append(line)
    return set(values)


def clamp_token(value, valid_range):
    lo, hi = valid_range
    return min(hi - 1, max(lo, int(round(float(value)))))


def load_alignment_lookup(metadata_path, midi_root, split, performance_dataset):
    columns = [
        "tier_a",
        "split",
        "performance_dataset",
        "refined_score_midi_path",
        "refined_performance_midi_path",
        "refined_alignment_path",
    ]
    df = pd.read_csv(metadata_path, usecols=columns)
    df = df[df["tier_a"].fillna(False).astype(bool)]
    df = df[df["split"] == split]
    df = df[df["refined_score_midi_path"].notna()]
    df = df[df["refined_performance_midi_path"].notna()]
    df = df[df["refined_alignment_path"].notna()]
    if performance_dataset is not None:
        df = df[df["performance_dataset"].fillna("").astype(str) == str(performance_dataset)]

    lookup = {}
    for _, row in df.iterrows():
        perf_path = (midi_root / row["refined_performance_midi_path"]).resolve()
        lookup[str(perf_path)] = (midi_root / row["refined_alignment_path"]).resolve()
    return lookup


def aligned_performance_ids(config, score_ids, performance_midi, alignment_path):
    alignment = np.load(alignment_path)
    if "perf_idx" not in alignment:
        raise ValueError(f"Missing perf_idx in {alignment_path}")
    perf_idx = alignment["perf_idx"].astype(int)
    note_count = len(score_ids) // 8
    if len(perf_idx) != note_count:
        raise ValueError(f"Alignment length mismatch for {alignment_path}: {len(perf_idx)} != {note_count}")

    performance_notes_sorted = sorted_piano_notes(performance_midi)
    if len(perf_idx):
        min_idx = int(perf_idx.min())
        max_idx = int(perf_idx.max())
        if min_idx < 0 or max_idx >= len(performance_notes_sorted):
            raise ValueError(f"perf_idx out of range for {alignment_path}")
    performance_notes = [performance_notes_sorted[int(index)] for index in perf_idx]
    performance_features = midi_to_note_features(
        performance_midi,
        notes=performance_notes,
        normalize=False,
        force_monotonic_starts=True,
    )

    ids = []
    for note_idx, row in enumerate(performance_features["continuous"]):
        pitch_token = int(score_ids[note_idx * 8])
        velocity = config.velocity_start + row[2]
        ioi = config.timing_start + row[0]
        duration = config.timing_start + row[1]
        pedals = [config.pedal_start + value for value in row[3:7]]
        ids.extend(
            [
                clamp_token(pitch_token, config.valid_id_range[0]),
                clamp_token(ioi, config.valid_id_range[1]),
                clamp_token(velocity, config.valid_id_range[2]),
                clamp_token(duration, config.valid_id_range[3]),
                *[
                    clamp_token(value, config.valid_id_range[4 + pedal_idx])
                    for pedal_idx, value in enumerate(pedals)
                ],
            ]
        )
    return ids


def slide_window(total_len, window_len, overlap_ratio):
    if total_len <= window_len:
        return [(0, total_len)]
    window_len = window_len // 8 * 8
    out = []
    start = 0
    while start + window_len <= total_len:
        out.append((start, start + window_len))
        start += int(window_len * (1 - overlap_ratio)) // 8 * 8
    if out[-1][1] != total_len:
        out.append((start, total_len))
    return out


def mask_pt_logits(logits, score_ids, valid_id_range):
    logits = logits.clone()
    seq_len = logits.shape[1]
    for pos in range(seq_len):
        lo, hi = valid_id_range[pos % 8]
        logits[:, pos, :lo] = -float("inf")
        logits[:, pos, hi:] = -float("inf")
        if pos % 8 == 0:
            logits[:, pos, :] = -float("inf")
            logits[:, pos, score_ids[:, pos]] = 0.0
    return logits


def sample_top_p(logits, temperature=1.0, top_p=0.95):
    if temperature <= 0:
        return logits.argmax(dim=-1)
    logits = logits / float(temperature)
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    remove = cumulative > float(top_p)
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False
    sorted_logits = sorted_logits.masked_fill(remove, -float("inf"))
    probs = F.softmax(sorted_logits, dim=-1)
    sampled_sorted = torch.multinomial(probs.reshape(-1, probs.shape[-1]), 1).reshape(probs.shape[:-1])
    return sorted_indices.gather(-1, sampled_sorted.unsqueeze(-1)).squeeze(-1)


@torch.no_grad()
def predict_pass(model, score_ids_full, feedback_ids_full, args):
    device = next(model.parameters()).device
    score_ids_full = score_ids_full.to(device)
    feedback_ids_full = feedback_ids_full.to(device)
    total_len = int(score_ids_full.shape[1])
    windows = slide_window(total_len, args.max_context_length, args.overlap_ratio)
    chunks = []
    for idx, (start, end) in enumerate(windows):
        score_window = score_ids_full[:, start:end]
        feedback_window = feedback_ids_full[:, start:end]
        bos = torch.full(
            (score_window.shape[0], 1),
            int(model.config.bos_token_id),
            dtype=torch.long,
            device=device,
        )
        decoder_input_ids = torch.cat([bos, feedback_window[:, :-1]], dim=1)
        outputs = model(input_ids=score_window, decoder_input_ids=decoder_input_ids)
        logits = mask_pt_logits(outputs.logits, score_window, model.config.valid_id_range)
        if args.do_sample:
            pred_window = sample_top_p(logits, temperature=args.temperature, top_p=args.top_p)
        else:
            pred_window = logits.argmax(dim=-1)
        if idx == 0:
            chunks.append(pred_window)
        else:
            last_start, last_end = windows[idx - 1]
            overlap_len = max(0, last_end - start)
            new_token_count = end - start - overlap_len
            chunks.append(pred_window[:, -new_token_count:])
    pred = torch.cat(chunks, dim=1)
    return pred[:, :total_len].detach().cpu()


def ids_to_mapped_midi(config, score_midi, score_ids, pred_ids):
    rendered = ids_to_midi(config, pred_ids, ref=score_ids)
    # k-pass token feedback can occasionally place trailing pedal CCs beyond the
    # final note region that PT's score-time mapper expects. The notes are still
    # usable, so trim only those tail CCs before mapping.
    if rendered.instruments and rendered.instruments[0].notes:
        max_note_start = max(note.start for note in rendered.instruments[0].notes)
        rendered.instruments[0].control_changes = [
            cc for cc in rendered.instruments[0].control_changes if cc.time < max_note_start + 4990
        ]
    return map_midi(score_midi, rendered)


def main():
    args = parse_args()
    ks = parse_ks(args.rollout_ks)
    max_k = max(ks)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    items = collect_test_items(args.metadata, args.midi_root, args.split, args.performance_dataset)
    selected = load_score_source_filter(args.score_source_list)
    if selected is not None:
        items = [item for item in items if item["score_source"] in selected]
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must satisfy 0 <= shard-index < num-shards")
    total_items = len(items)
    items = items[args.shard_index :: args.num_shards]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    alignment_lookup = load_alignment_lookup(
        args.metadata,
        args.midi_root,
        args.split,
        args.performance_dataset,
    )

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model = PianoT5Gemma.from_pretrained(
        str(args.model_path),
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
    )
    model.to(device)
    model.eval()

    manifest_by_k = {
        str(k): {
            "model_path": str(args.model_path.resolve()),
            "protocol": f"pt_kpass_k{k}",
            "feedback": "aligned_performance_by_refined_alignment",
            "split": args.split,
            "performance_dataset": args.performance_dataset,
            "num_samples": 1,
            "num_shards": args.num_shards,
            "shard_index": args.shard_index,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "overlap_ratio": args.overlap_ratio,
            "max_context_length": args.max_context_length,
            "items": [],
        }
        for k in ks
    }
    pairs_by_k = {str(k): [] for k in ks}

    for item in tqdm(items, desc="PT k-pass scores"):
        score_midi = MidiFile(str(item["score_path"]))
        score_ids_list = midi_to_ids(model.config, score_midi)
        score_ids = torch.tensor([score_ids_list], dtype=torch.long)
        pred_paths_by_k = {str(k): [] for k in ks}

        for gt_path in item["gt_paths"]:
            perf_midi = MidiFile(str(gt_path))
            align_path = alignment_lookup.get(str(gt_path.resolve()))
            if align_path is None:
                raise FileNotFoundError(f"Missing refined alignment for {gt_path}")
            perf_ids_list = aligned_performance_ids(model.config, score_ids_list, perf_midi, align_path)
            usable = min(len(score_ids_list), len(perf_ids_list)) // 8 * 8
            if usable <= 0:
                continue
            score_slice = score_ids[:, :usable]
            feedback = torch.tensor([perf_ids_list[:usable]], dtype=torch.long)
            preds = {}
            for k in range(max_k + 1):
                seed = stable_seed(args.seed, item["score_source"], gt_path, k)
                random.seed(seed)
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
                pred = predict_pass(model, score_slice, feedback, args)
                if k in ks:
                    preds[str(k)] = pred[0].tolist()
                feedback = pred

            gt_stem = Path(gt_path).with_suffix("").name
            for k_text, pred_ids in preds.items():
                midi_dir = args.output_dir / f"k{k_text}" / "midis"
                midi_dir.mkdir(parents=True, exist_ok=True)
                pred_midi = ids_to_mapped_midi(model.config, score_midi, score_ids_list[:usable], pred_ids)
                pred_path = midi_dir / f"{safe_name(item['score_source'])}__{gt_stem}__sample_000.mid"
                pred_path.parent.mkdir(parents=True, exist_ok=True)
                pred_midi.dump(str(pred_path))
                pred_paths_by_k[k_text].append(str(pred_path.resolve()))

        gt_paths = [str(path.resolve()) for path in item["gt_paths"]]
        for k_text in pred_paths_by_k:
            manifest_item = {
                "score_source": item["score_source"],
                "score_midi": str(item["score_path"].resolve()),
                "prediction_paths": pred_paths_by_k[k_text],
                "ground_truth_paths": gt_paths,
            }
            manifest_by_k[k_text]["items"].append(manifest_item)
            for pred_path in pred_paths_by_k[k_text]:
                for gt_path in gt_paths:
                    pairs_by_k[k_text].append({"pred": pred_path, "gt": gt_path})

    for k_text, manifest in manifest_by_k.items():
        out_dir = args.output_dir / f"k{k_text}"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "prediction_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        (out_dir / "evaluate_list.json").write_text(
            json.dumps(pairs_by_k[k_text], indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    summary = {
        "rollout_ks": [str(k) for k in ks],
        "num_scores": len(items),
        "total_scores_before_shard": total_items,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "output_dirs": {str(k): str((args.output_dir / f"k{k}").resolve()) for k in ks},
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
