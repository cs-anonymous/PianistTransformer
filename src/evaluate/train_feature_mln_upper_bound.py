import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import wasserstein_distance
from torch import nn
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data_process.work_manifest import build_work_manifest
from src.evaluate.evaluate_inr_saved_midis import aggregate_score_metrics, score_level_metrics
from src.model.integrated_pianoformer import _target5_to_raw7
from src.train.train_inr import (
    STYLE_STAT_DIM,
    build_epr_score_input_rows,
    build_perf_style_prefix_cache,
    build_style_vocabs,
    performance_dev_velocity_pedal4_binary_rows,
    perf_style_stats_from_cache,
    score_style_stats,
)
from src.utils.inr_midi import note_features_to_midi


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train a score-feature-only MLN3+BCE upper-bound model. "
            "No work_id/note_id lookup is used; optional style conditioning uses creator/source/score/perf-prefix stats."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--eval-split", type=str, default=None)
    parser.add_argument("--performance-dataset", type=str, default="ASAP")
    parser.add_argument("--max-works", type=int, default=None)
    parser.add_argument("--use-style", action="store_true")
    parser.add_argument("--components", type=int, default=3)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--style-size", type=int, default=96)
    parser.add_argument("--layers", type=int, default=10)
    parser.add_argument("--kernel-size", type=int, default=7)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--fit-space-eps", type=float, default=1e-4)
    parser.add_argument("--sigma-min", type=float, default=0.05)
    parser.add_argument("--sigma-max", type=float, default=3.0)
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--render-style-performances", action=argparse.BooleanOptionalAction, default=True)
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


def logit_np(values, eps):
    values = np.clip(np.asarray(values, dtype=np.float64), eps, 1.0 - eps)
    return np.log(values) - np.log1p(-values)


def score_note_features(score, config):
    rows = build_epr_score_input_rows(
        score,
        use_timing_scale_bit=bool(config.get("use_timing_scale_bit", True)),
        timing_control_mode=config.get("timing_control_mode"),
        log_scale=float(config.get("timing_log_scale", 50.0)),
        musical_feature_mode=config.get("musical_feature_mode", "continuous"),
    )
    rows = np.asarray(rows, dtype=np.float32)
    pitch = np.asarray(score.get("pitch", []), dtype=np.float32)
    pitch_norm = np.clip(pitch, 0.0, 127.0)[:, None] / 127.0
    octave_norm = np.clip(np.floor(np.clip(pitch, 0.0, 127.0) / 12.0), 0.0, 10.0)[:, None] / 10.0
    pitch_class = np.zeros((len(pitch), 12), dtype=np.float32)
    for idx, value in enumerate(pitch.astype(np.int64)):
        if 0 <= value < 128:
            pitch_class[idx, value % 12] = 1.0
    return np.concatenate([pitch_norm, octave_norm, pitch_class, rows], axis=1).astype(np.float32)


@dataclass
class SequenceExample:
    work_idx: int
    perf_idx: int
    start: int
    end: int
    score_source: str
    performance_source: str
    score_features: np.ndarray
    targets: np.ndarray
    creator_id: int
    source_id: int
    score_stats: np.ndarray
    perf_prefix_stats: np.ndarray


def load_feature_dataset(config, args, split):
    manifest = build_work_manifest(
        metadata_path=config["metadata_path"],
        refined_dir=config["refined_dir"],
        split=split,
        block_notes=config["block_notes"],
        overlap_ratio=config["overlap_ratio"],
        min_notes=config["min_notes"],
        max_works=args.max_works,
        skip_work_paths=config.get("skip_work_paths"),
        performance_dataset=args.performance_dataset,
    )
    refined_midi_dir = refined_midi_dir_from_config(config)
    composer_vocab, source_vocab = build_style_vocabs(config["metadata_path"])
    unknown_composer_id = int(composer_vocab.get("<unk>", 0))
    unknown_source_id = int(source_vocab.get("<unk>", 0))

    examples = []
    works = []
    all_targets = []
    feature_dim = None

    for work_idx, item in enumerate(manifest):
        work = json.loads(Path(item["path"]).read_text(encoding="utf-8"))
        score = work["score"]
        score_shared_raw = score["score_raw"]
        score_features = score_note_features(score, config)
        feature_dim = int(score_features.shape[1])
        n_notes = int(score_features.shape[0])
        meta = work.get("meta", {})
        composer = str(meta.get("composer") or "")
        creator_id = int(composer_vocab.get(composer, unknown_composer_id))

        perfs = selected_perfs(work, item, split)
        gt_paths = []
        target_arrays = []
        example_indices = []
        per_perf_example_indices = []
        performance_sources = []
        for local_perf_idx, perf in enumerate(perfs):
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
            targets = np.asarray(rows, dtype=np.float32)
            if targets.shape[0] != n_notes:
                raise ValueError(f"Label length mismatch for {perf.get('performance_source')}")
            perf_cache = build_perf_style_prefix_cache(targets)
            performance_dataset = str(perf.get("performance_dataset") or args.performance_dataset or "")
            source_id = int(source_vocab.get(performance_dataset, unknown_source_id))
            window_example_indices = []
            for start, end in item["windows"]:
                start = int(start)
                end = int(end)
                if end <= start or start < 0 or end > n_notes:
                    continue
                example_idx = len(examples)
                score_stats = np.asarray(score_style_stats(score, start, end), dtype=np.float32)
                perf_prefix_stats = np.asarray(perf_style_stats_from_cache(perf_cache, start), dtype=np.float32)
                example_indices.append(example_idx)
                window_example_indices.append(example_idx)
                examples.append(
                    SequenceExample(
                        work_idx=work_idx,
                        perf_idx=local_perf_idx,
                        start=start,
                        end=end,
                        score_source=item["score_source"],
                        performance_source=perf.get("performance_source", ""),
                        score_features=score_features[start:end],
                        targets=targets[start:end],
                        creator_id=creator_id,
                        source_id=source_id,
                        score_stats=score_stats,
                        perf_prefix_stats=perf_prefix_stats,
                    )
                )
                all_targets.append(targets[start:end])
            if not window_example_indices:
                continue
            per_perf_example_indices.append(window_example_indices)
            performance_sources.append(perf.get("performance_source", ""))
            target_arrays.append(targets)
            gt_path = refined_midi_dir / perf.get("performance_source", "")
            if gt_path.exists():
                gt_paths.append(str(gt_path.resolve()))

        if example_indices:
            works.append(
                {
                    "work_idx": int(work_idx),
                    "score_source": item["score_source"],
                    "score_midi": str((refined_midi_dir / item["score_source"]).resolve()),
                    "pitch": [int(value) for value in score["pitch"]],
                    "score_shared_raw": score_shared_raw,
                    "ground_truth_paths": gt_paths,
                    "target_arrays": target_arrays,
                    "example_indices": example_indices,
                    "per_perf_example_indices": per_perf_example_indices,
                    "performance_sources": performance_sources,
                    "num_performances": int(len(per_perf_example_indices)),
                    "num_notes": int(n_notes),
                }
            )

    if not examples:
        raise RuntimeError("No training sequences were loaded.")
    targets_concat = np.concatenate(all_targets, axis=0).astype(np.float32)
    return {
        "examples": examples,
        "works": works,
        "feature_dim": int(feature_dim),
        "targets": targets_concat,
        "num_creators": max(composer_vocab.values(), default=0) + 1,
        "num_sources": max(source_vocab.values(), default=0) + 1,
        "composer_vocab_size": len(composer_vocab),
        "source_vocab_size": len(source_vocab),
    }


def pad_batch(examples, device, use_style):
    batch = len(examples)
    lengths = [example.score_features.shape[0] for example in examples]
    max_len = max(lengths)
    feature_dim = examples[0].score_features.shape[1]
    score_features = torch.zeros(batch, max_len, feature_dim, dtype=torch.float32)
    targets = torch.zeros(batch, max_len, 7, dtype=torch.float32)
    mask = torch.zeros(batch, max_len, dtype=torch.float32)
    creator_ids = torch.zeros(batch, dtype=torch.long)
    source_ids = torch.zeros(batch, dtype=torch.long)
    score_stats = torch.zeros(batch, STYLE_STAT_DIM, dtype=torch.float32)
    perf_stats = torch.zeros(batch, STYLE_STAT_DIM, dtype=torch.float32)
    for row, example in enumerate(examples):
        n = lengths[row]
        score_features[row, :n] = torch.from_numpy(example.score_features)
        targets[row, :n] = torch.from_numpy(example.targets)
        mask[row, :n] = 1.0
        if use_style:
            creator_ids[row] = int(example.creator_id)
            source_ids[row] = int(example.source_id)
            score_stats[row] = torch.from_numpy(example.score_stats)
            perf_stats[row] = torch.from_numpy(example.perf_prefix_stats)
    return {
        "score_features": score_features.to(device),
        "targets": targets.to(device),
        "mask": mask.to(device),
        "creator_ids": creator_ids.to(device),
        "source_ids": source_ids.to(device),
        "score_stats": score_stats.to(device),
        "perf_stats": perf_stats.to(device),
    }


class ResidualConvBlock(nn.Module):
    def __init__(self, hidden_size, kernel_size, dilation, dropout):
        super().__init__()
        padding = (int(kernel_size) // 2) * int(dilation)
        self.conv = nn.Conv1d(hidden_size, hidden_size * 2, kernel_size, padding=padding, dilation=dilation)
        self.proj = nn.Conv1d(hidden_size, hidden_size, 1)
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        y = x.transpose(1, 2)
        y = self.conv(y)
        y = F.glu(y, dim=1)
        y = self.proj(y).transpose(1, 2)
        y = self.dropout(y)
        return self.norm(residual + y)


class FeatureMlnModel(nn.Module):
    def __init__(
        self,
        feature_dim,
        components,
        hidden_size,
        layers,
        kernel_size,
        dropout,
        use_style,
        num_creators,
        num_sources,
        style_size,
    ):
        super().__init__()
        self.components = int(components)
        self.use_style = bool(use_style)
        self.input = nn.Sequential(
            nn.Linear(feature_dim, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
        )
        dilations = [2 ** (idx % 5) for idx in range(int(layers))]
        self.blocks = nn.ModuleList(
            [ResidualConvBlock(hidden_size, kernel_size, dilation, dropout) for dilation in dilations]
        )
        if self.use_style:
            self.creator_embed = nn.Embedding(max(1, int(num_creators)), style_size)
            self.source_embed = nn.Embedding(max(1, int(num_sources)), style_size)
            self.score_style = nn.Sequential(
                nn.Linear(STYLE_STAT_DIM, style_size),
                nn.GELU(),
                nn.Linear(style_size, style_size),
                nn.GELU(),
            )
            self.perf_style = nn.Sequential(
                nn.Linear(STYLE_STAT_DIM, style_size),
                nn.GELU(),
                nn.Linear(style_size, style_size),
                nn.GELU(),
            )
            self.style_proj = nn.Sequential(
                nn.Linear(style_size * 4, hidden_size),
                nn.GELU(),
                nn.LayerNorm(hidden_size),
            )
        out_dim = 3 * components * 3 + 4
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, out_dim),
        )

    def forward(self, batch):
        x = self.input(batch["score_features"])
        for block in self.blocks:
            x = block(x)
        if self.use_style:
            batch_size, seq_len = x.shape[:2]
            creator = self.creator_embed(batch["creator_ids"]).unsqueeze(1).expand(batch_size, seq_len, -1)
            source = self.source_embed(batch["source_ids"]).unsqueeze(1).expand(batch_size, seq_len, -1)
            score = self.score_style(batch["score_stats"]).unsqueeze(1).expand(batch_size, seq_len, -1)
            perf = self.perf_style(batch["perf_stats"]).unsqueeze(1).expand(batch_size, seq_len, -1)
            x = x + self.style_proj(torch.cat([creator, source, score, perf], dim=-1))
        out = self.head(x)
        k = self.components
        offset = 0
        logits = out[:, :, offset : offset + 3 * k].reshape(out.shape[0], out.shape[1], 3, k)
        offset += 3 * k
        mu = out[:, :, offset : offset + 3 * k].reshape(out.shape[0], out.shape[1], 3, k)
        offset += 3 * k
        log_sigma = out[:, :, offset : offset + 3 * k].reshape(out.shape[0], out.shape[1], 3, k)
        offset += 3 * k
        pedal_logits = out[:, :, offset : offset + 4]
        return logits, mu, log_sigma, pedal_logits


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


def loss_for_batch(model, batch, args):
    logits, mu, log_sigma, pedal_logits = model(batch)
    mask = batch["mask"].float()
    targets = batch["targets"].float()
    log_prob = mln_log_prob(
        logits,
        mu,
        log_sigma,
        targets[:, :, :3],
        eps=args.fit_space_eps,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
    )
    denom = mask.sum().clamp_min(1.0)
    continuous_loss = -(log_prob * mask.unsqueeze(-1)).sum() / (denom * 3.0)
    pedal_loss_raw = F.binary_cross_entropy_with_logits(pedal_logits.float(), targets[:, :, 3:7], reduction="none")
    pedal_loss = (pedal_loss_raw * mask.unsqueeze(-1)).sum() / (denom * 4.0)
    return continuous_loss + pedal_loss, continuous_loss.detach(), pedal_loss.detach()


def initialize_global_bias(model, targets, args):
    targets = np.asarray(targets, dtype=np.float64)
    with torch.no_grad():
        final = model.head[-1]
        if not isinstance(final, nn.Linear):
            return
        final.weight.zero_()
        final.bias.zero_()
        k = int(args.components)
        for feature_idx in range(3):
            z = logit_np(targets[:, feature_idx], args.fit_space_eps)
            mean = float(np.mean(z))
            std = float(np.std(z))
            std = min(max(std if std > 0 else args.sigma_min, args.sigma_min), args.sigma_max)
            mu_start = 3 * k + feature_idx * k
            sigma_start = 6 * k + feature_idx * k
            final.bias[mu_start : mu_start + k].fill_(mean)
            final.bias[sigma_start : sigma_start + k].fill_(math.log(std))
        pedal_prob = np.clip(np.mean(targets[:, 3:7], axis=0), args.fit_space_eps, 1.0 - args.fit_space_eps)
        pedal_logit = np.log(pedal_prob) - np.log1p(-pedal_prob)
        final.bias[9 * k : 9 * k + 4] = torch.tensor(pedal_logit, dtype=final.bias.dtype, device=final.bias.device)


def train_model(dataset, args):
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    model = FeatureMlnModel(
        feature_dim=dataset["feature_dim"],
        components=args.components,
        hidden_size=args.hidden_size,
        layers=args.layers,
        kernel_size=args.kernel_size,
        dropout=args.dropout,
        use_style=args.use_style,
        num_creators=dataset["num_creators"],
        num_sources=dataset["num_sources"],
        style_size=args.style_size,
    ).to(device)
    initialize_global_bias(model, dataset["targets"], args)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    examples = list(dataset["examples"])
    history = []
    progress = tqdm(range(1, args.steps + 1), desc=f"train feature MLN style={args.use_style}")
    rng = random.Random(args.seed)
    for step in progress:
        batch_examples = [examples[rng.randrange(len(examples))] for _ in range(args.batch_size)]
        batch = pad_batch(batch_examples, device, args.use_style)
        optimizer.zero_grad(set_to_none=True)
        loss, continuous_loss, pedal_loss = loss_for_batch(model, batch, args)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % max(1, args.steps // 25) == 0 or step == args.steps:
            model.eval()
            with torch.no_grad():
                eval_losses = []
                eval_cont = []
                eval_pedal = []
                for start in range(0, len(examples), max(1, args.batch_size)):
                    eval_batch = pad_batch(examples[start : start + max(1, args.batch_size)], device, args.use_style)
                    eval_loss, eval_c, eval_p = loss_for_batch(model, eval_batch, args)
                    eval_losses.append(float(eval_loss.cpu()))
                    eval_cont.append(float(eval_c.cpu()))
                    eval_pedal.append(float(eval_p.cpu()))
            model.train()
            row = {
                "step": int(step),
                "loss": float(np.mean(eval_losses)),
                "continuous_loss": float(np.mean(eval_cont)),
                "pedal_loss": float(np.mean(eval_pedal)),
            }
            history.append(row)
            progress.set_postfix(loss=row["loss"], cont=row["continuous_loss"], pedal=row["pedal_loss"])
    return model.cpu(), history


def sample_targets(model, example, args, seed, strategy):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    model.eval()
    batch = pad_batch([example], torch.device("cpu"), args.use_style)
    with torch.no_grad():
        logits, mu, log_sigma, pedal_logits = model(batch)
        logits = logits[0, : example.targets.shape[0]]
        mu = mu[0, : example.targets.shape[0]]
        log_sigma = log_sigma[0, : example.targets.shape[0]]
        pedal_logits = pedal_logits[0, : example.targets.shape[0]]
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
        pedal_prob = torch.sigmoid(pedal_logits.float())
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


def predict_full_target(model, dataset, work, example_indices, args, strategy, sample_idx):
    n_notes = int(work["num_notes"])
    full = torch.zeros(n_notes, 7, dtype=torch.float32)
    counts = torch.zeros(n_notes, 1, dtype=torch.float32)
    for example_idx in example_indices:
        example = dataset["examples"][example_idx]
        target = sample_targets(
            model,
            example,
            args,
            stable_seed(args.seed, work["score_source"], example_idx, strategy, sample_idx),
            strategy,
        ).float()
        full[example.start : example.end] += target
        counts[example.start : example.end] += 1.0
    missing = counts.squeeze(-1) <= 0
    if missing.any() and work["target_arrays"]:
        full[missing] = torch.tensor(work["target_arrays"][0], dtype=torch.float32)[missing]
        counts[missing] = 1.0
    full = full / counts.clamp_min(1.0)
    full[:, 3:7] = (full[:, 3:7] >= 0.5).float()
    return full


def render_groups_for_work(work, args):
    if args.use_style and args.render_style_performances:
        return list(enumerate(work["per_perf_example_indices"]))
    repeat_group = work["per_perf_example_indices"][0]
    return [(0, repeat_group)]


def direct_pairwise_metrics(model, dataset, args, config, strategy):
    rows = []
    for work in dataset["works"]:
        for perf_idx, example_indices in render_groups_for_work(work, args):
            pred_target = predict_full_target(
                model,
                dataset,
                work,
                example_indices,
                args,
                strategy,
                f"direct_{perf_idx}",
            )
            pred_raw = _target5_to_raw7(
                torch.tensor(work["score_shared_raw"], dtype=torch.float32),
                pred_target.float(),
                config=config,
            ).cpu().numpy()
            pred_arrays = raw_arrays_from_rows(pred_raw)
            targets_to_compare = [work["target_arrays"][perf_idx]] if args.use_style else work["target_arrays"]
            for target in targets_to_compare:
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
        for render_pos, example_indices in render_groups_for_work(work, args):
            repeat = args.num_samples if strategy == "sample" and not (args.use_style and args.render_style_performances) else 1
            for sample_idx in range(repeat):
                target = predict_full_target(
                    model,
                    dataset,
                    work,
                    example_indices,
                    args,
                    strategy,
                    sample_idx,
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
                pred_path = midi_dir / (
                    f"{safe_stem(work['score_source'])}__perf{render_pos:03d}__{strategy}_{sample_idx:03d}.mid"
                )
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
    manifest = {
        "protocol": f"feature_mln_style{int(args.use_style)}_{strategy}",
        "num_samples": len(items[0]["prediction_paths"]) if items else 0,
        "items": items,
    }
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
    eval_split = args.eval_split or args.split
    train_dataset = load_feature_dataset(config, args, args.split)
    eval_dataset = train_dataset if eval_split == args.split else load_feature_dataset(config, args, eval_split)
    model, history = train_model(train_dataset, args)
    torch.save(model.state_dict(), args.output_dir / "feature_mln_state.pt")
    results = {
        "sample": render_and_eval(model, eval_dataset, args, config, "sample"),
        "mean": render_and_eval(model, eval_dataset, args, config, "mean"),
    }
    summary = {
        "config": str(args.config.resolve()),
        "split": args.split,
        "train_split": args.split,
        "eval_split": eval_split,
        "performance_dataset": args.performance_dataset,
        "max_works": args.max_works,
        "use_style": args.use_style,
        "num_examples": len(train_dataset["examples"]),
        "num_train_examples": len(train_dataset["examples"]),
        "num_eval_examples": len(eval_dataset["examples"]),
        "num_scores": len(eval_dataset["works"]),
        "num_train_scores": len(train_dataset["works"]),
        "num_eval_scores": len(eval_dataset["works"]),
        "feature_dim": train_dataset["feature_dim"],
        "composer_vocab_size": train_dataset["composer_vocab_size"],
        "source_vocab_size": train_dataset["source_vocab_size"],
        "components": args.components,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "hidden_size": args.hidden_size,
        "style_size": args.style_size,
        "layers": args.layers,
        "kernel_size": args.kernel_size,
        "fit_space_eps": args.fit_space_eps,
        "sigma_min": args.sigma_min,
        "sigma_max": args.sigma_max,
        "num_samples": args.num_samples,
        "render_style_performances": args.render_style_performances,
        "history": history,
        "results": results,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "history"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
