import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import wasserstein_distance
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data_process.work_manifest import build_work_manifest
from src.evaluate.evaluate_inr_saved_midis import aggregate_score_metrics, score_level_metrics
from src.model.integrated_pianoformer import _target5_to_raw7
from src.train.train_inr import performance_dev_velocity_pedal4_binary_rows
from src.utils.inr_midi import note_features_to_midi


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a work/note lookup MLN3+BCE upper-bound model and evaluate rendered MIDI."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--performance-dataset", type=str, default="ASAP")
    parser.add_argument("--max-works", type=int, default=8)
    parser.add_argument("--components", type=int, default=3)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=32768)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--fit-space-eps", type=float, default=1e-4)
    parser.add_argument("--sigma-min", type=float, default=0.05)
    parser.add_argument("--sigma-max", type=float, default=3.0)
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260705)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def safe_stem(score_source):
    return Path(score_source).with_suffix("").as_posix().replace("/", "__")


def stable_seed(base_seed, *parts):
    import hashlib

    payload = "::".join([str(base_seed), *[str(part) for part in parts]])
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def logit_np(values, eps):
    values = np.clip(np.asarray(values, dtype=np.float64), eps, 1.0 - eps)
    return np.log(values) - np.log1p(-values)


def finite_wass(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    return float(wasserstein_distance(a, b))


def selected_perfs(work, item, split):
    all_perfs = [
        perf
        for perf in work.get("performances", [])
        if perf.get("label_shared_raw") is not None and perf.get("label_pedal4_raw") is not None
    ]
    by_source = {perf.get("performance_source"): perf for perf in all_perfs}
    selected_sources = item.get("selected_performance_sources")
    if selected_sources is None:
        selected = all_perfs
    else:
        selected = [by_source[source] for source in selected_sources if source in by_source]
    return [perf for perf in selected if perf.get("split", split) == split]


def refined_midi_dir_from_config(config):
    return Path(config["refined_dir"]).parent / "refined"


def load_tiny_dataset(config, args):
    manifest = build_work_manifest(
        metadata_path=config["metadata_path"],
        refined_dir=config["refined_dir"],
        split=args.split,
        block_notes=config["block_notes"],
        overlap_ratio=config["overlap_ratio"],
        min_notes=config["min_notes"],
        max_works=args.max_works,
        skip_work_paths=config.get("skip_work_paths"),
        performance_dataset=args.performance_dataset,
    )
    refined_midi_dir = refined_midi_dir_from_config(config)
    note_ids = []
    targets = []
    note_to_values = []
    works = []
    next_note_id = 0

    for work_idx, item in enumerate(manifest):
        work = json.loads(Path(item["path"]).read_text(encoding="utf-8"))
        score = work["score"]
        pitch = [int(value) for value in score["pitch"]]
        score_shared_raw = score["score_raw"]
        perfs = selected_perfs(work, item, args.split)
        if not perfs:
            continue
        start_note_id = next_note_id
        work_note_ids = np.arange(start_note_id, start_note_id + len(pitch), dtype=np.int64)
        next_note_id += len(pitch)
        note_to_values.extend([] for _ in pitch)

        gt_paths = []
        work_target_arrays = []
        for perf in perfs:
            rows = performance_dev_velocity_pedal4_binary_rows(
                perf,
                score_shared_raw,
                epr_timing_target=config.get("epr_timing_target", "deviation"),
                log_scale=float(config.get("timing_log_scale", 50.0)),
                split_zero_ioi_head=bool(config.get("split_zero_ioi_head", False)),
                ioi_nonzero_dev_scale=float(config.get("ioi_nonzero_dev_scale", 2.0)),
                ioi_zero_dev_scale=float(config.get("ioi_zero_dev_scale", 4.0)),
                pedal_binary_threshold=float(config.get("pedal_binary_threshold", 64.0)),
            )
            if rows is None:
                continue
            rows = np.asarray(rows, dtype=np.float32)
            if rows.shape[0] != len(pitch):
                raise ValueError(f"Label length mismatch for {perf.get('performance_source')}")
            note_ids.append(np.repeat(work_note_ids, 1))
            targets.append(rows)
            work_target_arrays.append(rows)
            for local_idx, row in enumerate(rows):
                note_to_values[start_note_id + local_idx].append(row)
            gt_path = refined_midi_dir / perf.get("performance_source")
            if gt_path.exists():
                gt_paths.append(str(gt_path.resolve()))

        works.append(
            {
                "work_idx": int(work_idx),
                "score_source": item["score_source"],
                "score_midi": str((refined_midi_dir / item["score_source"]).resolve()),
                "pitch": pitch,
                "score_shared_raw": score_shared_raw,
                "note_ids": work_note_ids,
                "target_arrays": work_target_arrays,
                "ground_truth_paths": gt_paths,
                "num_performances": int(len(gt_paths)),
            }
        )

    if not note_ids:
        raise RuntimeError("No training rows were loaded.")
    flat_note_ids = np.concatenate(note_ids, axis=0)
    flat_targets = np.concatenate(targets, axis=0)
    return {
        "note_ids": flat_note_ids.astype(np.int64),
        "targets": flat_targets.astype(np.float32),
        "note_to_values": note_to_values,
        "works": works,
        "num_notes": int(next_note_id),
    }


class LookupMln(torch.nn.Module):
    def __init__(self, num_notes, components):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.zeros(num_notes, 3, components))
        self.mu = torch.nn.Parameter(torch.zeros(num_notes, 3, components))
        self.log_sigma = torch.nn.Parameter(torch.zeros(num_notes, 3, components))
        self.pedal_logits = torch.nn.Parameter(torch.zeros(num_notes, 4))

    def continuous_params(self, note_ids):
        return self.logits[note_ids], self.mu[note_ids], self.log_sigma[note_ids]

    def pedal_params(self, note_ids):
        return self.pedal_logits[note_ids]


def initialize_from_empirical(model, note_to_values, eps, sigma_min, sigma_max):
    with torch.no_grad():
        k = model.logits.shape[-1]
        for note_idx, rows in enumerate(note_to_values):
            rows = np.asarray(rows, dtype=np.float64)
            if rows.size == 0:
                model.mu[note_idx].zero_()
                model.log_sigma[note_idx].fill_(math.log(sigma_min))
                model.pedal_logits[note_idx].zero_()
                continue
            for feature_idx in range(3):
                z = logit_np(rows[:, feature_idx], eps=eps)
                if len(z) == 1:
                    centers = np.repeat(z[0], k)
                    sigma = sigma_min
                else:
                    qs = np.linspace(0.0, 1.0, k + 2)[1:-1]
                    centers = np.quantile(z, qs)
                    sigma = float(np.std(z))
                    sigma = min(max(sigma if sigma > 0 else sigma_min, sigma_min), sigma_max)
                model.mu[note_idx, feature_idx] = torch.tensor(centers, dtype=model.mu.dtype)
                model.log_sigma[note_idx, feature_idx].fill_(math.log(sigma))
            pedal_prob = np.clip(np.mean(rows[:, 3:7], axis=0), eps, 1.0 - eps)
            model.pedal_logits[note_idx] = torch.tensor(np.log(pedal_prob) - np.log1p(-pedal_prob), dtype=model.pedal_logits.dtype)


def mln_log_prob(logits, mu, log_sigma, target, eps, sigma_min, sigma_max):
    target = target.float().clamp(float(eps), 1.0 - float(eps))
    z = torch.logit(target, eps=float(eps)).unsqueeze(-1)
    log_min = math.log(float(sigma_min))
    log_max = math.log(float(sigma_max))
    sigma = torch.exp(log_sigma.float().clamp(min=log_min, max=log_max))
    log_pi = F.log_softmax(logits.float(), dim=-1)
    log_normal = torch.distributions.Normal(mu.float(), sigma).log_prob(z)
    log_jacobian = -torch.log(target).unsqueeze(-1) - torch.log1p(-target).unsqueeze(-1)
    return torch.logsumexp(log_pi + log_normal + log_jacobian, dim=-1)


def loss_for_batch(model, note_ids, targets, args):
    logits, mu, log_sigma = model.continuous_params(note_ids)
    log_prob = mln_log_prob(
        logits,
        mu,
        log_sigma,
        targets[:, :3],
        eps=args.fit_space_eps,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
    )
    continuous_loss = -log_prob.mean()
    pedal_loss = F.binary_cross_entropy_with_logits(model.pedal_params(note_ids), targets[:, 3:7])
    return continuous_loss + pedal_loss, continuous_loss.detach(), pedal_loss.detach()


def train_model(dataset, args):
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    model = LookupMln(dataset["num_notes"], args.components).to(device)
    initialize_from_empirical(model, dataset["note_to_values"], args.fit_space_eps, args.sigma_min, args.sigma_max)
    model.to(device)
    note_ids = torch.tensor(dataset["note_ids"], dtype=torch.long, device=device)
    targets = torch.tensor(dataset["targets"], dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    history = []
    n = note_ids.shape[0]
    full_batch = args.batch_size <= 0 or args.batch_size >= n
    progress = tqdm(range(1, args.steps + 1), desc="train lookup MLN")
    for step in progress:
        if full_batch:
            batch_idx = torch.arange(n, device=device)
        else:
            batch_idx = torch.randint(0, n, (args.batch_size,), device=device)
        optimizer.zero_grad(set_to_none=True)
        loss, cont_loss, pedal_loss = loss_for_batch(model, note_ids[batch_idx], targets[batch_idx], args)
        loss.backward()
        optimizer.step()
        if step == 1 or step % max(1, args.steps // 20) == 0 or step == args.steps:
            with torch.no_grad():
                eval_loss, eval_cont, eval_pedal = loss_for_batch(model, note_ids, targets, args)
            row = {
                "step": int(step),
                "loss": float(eval_loss.cpu()),
                "continuous_loss": float(eval_cont.cpu()),
                "pedal_loss": float(eval_pedal.cpu()),
            }
            history.append(row)
            progress.set_postfix(loss=row["loss"], cont=row["continuous_loss"], pedal=row["pedal_loss"])
    return model.cpu(), history


def sample_targets_for_note_ids(model, note_ids, args, seed, strategy):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    note_ids_t = torch.tensor(note_ids, dtype=torch.long)
    with torch.no_grad():
        logits, mu, log_sigma = model.continuous_params(note_ids_t)
        probs = torch.softmax(logits.float(), dim=-1)
        sigma = torch.exp(log_sigma.float().clamp(min=math.log(args.sigma_min), max=math.log(args.sigma_max)))
        if strategy == "mean":
            continuous = torch.sum(probs * torch.sigmoid(mu.float()), dim=-1)
        elif strategy == "sample":
            flat_probs = probs.reshape(-1, probs.shape[-1])
            comp = torch.multinomial(flat_probs, num_samples=1, replacement=True, generator=generator).reshape(*probs.shape[:-1], 1)
            sampled_mu = mu.float().gather(dim=-1, index=comp).squeeze(-1)
            sampled_sigma = sigma.gather(dim=-1, index=comp).squeeze(-1)
            noise = torch.randn(sampled_mu.shape, generator=generator)
            continuous = torch.sigmoid(sampled_mu + sampled_sigma * noise)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        pedal_prob = torch.sigmoid(model.pedal_params(note_ids_t).float())
        if strategy == "mean":
            pedal = (pedal_prob >= 0.5).float()
        else:
            pedal = torch.bernoulli(pedal_prob, generator=generator)
    return torch.cat([continuous, pedal], dim=-1)


def raw_arrays_from_rows(rows):
    rows = np.asarray(rows, dtype=np.float64)
    return {
        "ioi": rows[:, 0],
        "duration": rows[:, 1],
        "velocity": rows[:, 2],
        "pedal": rows[:, 3:7].reshape(-1),
    }


def direct_pairwise_metrics(model, dataset, args, config, strategy):
    rows = []
    for work in dataset["works"]:
        pred_target = sample_targets_for_note_ids(
            model,
            work["note_ids"],
            args,
            stable_seed(args.seed, work["score_source"], strategy, "direct"),
            strategy,
        )
        pred_raw = _target5_to_raw7(
            torch.tensor(work["score_shared_raw"], dtype=torch.float32),
            pred_target.float(),
            config=config,
        ).cpu().numpy()
        pred_arrays = raw_arrays_from_rows(pred_raw)
        for target in work["target_arrays"]:
            target_raw = _target5_to_raw7(
                torch.tensor(work["score_shared_raw"], dtype=torch.float32),
                torch.tensor(target, dtype=torch.float32),
                config=config,
            ).cpu().numpy()
            target_arrays = raw_arrays_from_rows(target_raw)
            rows.append({f"{key}_wass": finite_wass(pred_arrays[key], target_arrays[key]) for key in pred_arrays})
    keys = ["ioi_wass", "duration_wass", "velocity_wass", "pedal_wass"]
    output = {"num_rows": int(len(rows))}
    for key in keys:
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        values = values[np.isfinite(values)]
        output[key] = float(np.mean(values)) if len(values) else float("nan")
    return output


def render_and_eval(model, dataset, args, config, strategy):
    output_dir = args.output_dir / strategy
    midi_dir = output_dir / "midis"
    items = []
    for work in tqdm(dataset["works"], desc=f"render {strategy}"):
        pred_paths = []
        for sample_idx in range(args.num_samples if strategy == "sample" else 1):
            target = sample_targets_for_note_ids(
                model,
                work["note_ids"],
                args,
                stable_seed(args.seed, work["score_source"], strategy, sample_idx),
                strategy,
            )
            raw = _target5_to_raw7(
                torch.tensor(work["score_shared_raw"], dtype=torch.float32),
                target.float(),
                config=config,
            ).cpu().numpy()
            midi = note_features_to_midi(
                pitch=work["pitch"],
                continuous=raw.tolist(),
                target_ticks_per_beat=500,
                target_tempo=120,
                max_time_ms=float(config.get("max_time_ms", 10000.0)),
                normalized=False,
            )
            pred_path = midi_dir / f"{safe_stem(work['score_source'])}__{strategy}_{sample_idx:03d}.mid"
            pred_path.parent.mkdir(parents=True, exist_ok=True)
            midi.dump(str(pred_path))
            pred_paths.append(str(pred_path.resolve()))
        items.append(
            {
                "score_source": work["score_source"],
                "score_midi": work["score_midi"],
                "prediction_paths": pred_paths,
                "ground_truth_paths": work["ground_truth_paths"],
            }
        )
    manifest = {"protocol": f"lookup_mln_{strategy}", "num_samples": len(items[0]["prediction_paths"]) if items else 0, "items": items}
    manifest_path = output_dir / "prediction_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    score_rows = [score_level_metrics(item) for item in tqdm(items, desc=f"eval {strategy}")]
    eval_output = {
        "prediction_manifest": str(manifest_path.resolve()),
        "protocol": manifest["protocol"],
        "num_samples": manifest["num_samples"],
        "num_scores": len(score_rows),
        "aggregate": {
            "pn_wass": aggregate_score_metrics(score_rows, "pn_wass"),
            "pp_wass": aggregate_score_metrics(score_rows, "pp_wass"),
        },
        "scores": score_rows,
    }
    eval_path = output_dir / "eval.json"
    eval_path.write_text(json.dumps(eval_output, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "manifest": str(manifest_path.resolve()),
        "eval": str(eval_path.resolve()),
        "pp_wass": eval_output["aggregate"]["pp_wass"],
        "pn_wass": eval_output["aggregate"]["pn_wass"],
        "direct_pairwise": direct_pairwise_metrics(model, dataset, args, config, strategy),
    }


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    dataset = load_tiny_dataset(config, args)
    model, history = train_model(dataset, args)
    torch.save(model.state_dict(), args.output_dir / "lookup_mln_state.pt")
    results = {
        "sample": render_and_eval(model, dataset, args, config, "sample"),
        "mean": render_and_eval(model, dataset, args, config, "mean"),
    }
    summary = {
        "config": str(args.config.resolve()),
        "split": args.split,
        "performance_dataset": args.performance_dataset,
        "max_works": args.max_works,
        "num_notes": dataset["num_notes"],
        "num_rows": int(len(dataset["targets"])),
        "components": args.components,
        "steps": args.steps,
        "lr": args.lr,
        "fit_space_eps": args.fit_space_eps,
        "sigma_min": args.sigma_min,
        "sigma_max": args.sigma_max,
        "num_samples": args.num_samples,
        "history": history,
        "results": results,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in {"history"}}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
