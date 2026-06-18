import argparse
import bisect
import datetime
import json
import os
import random
import shutil
import re
import warnings
from collections import OrderedDict
from pathlib import Path

import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import SequentialSampler
from transformers import Trainer, TrainingArguments

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT_DIR))

from src.model.integrated_pianoformer import (
    IntegratedPianoT5Gemma,
    IntegratedPianoT5GemmaConfig,
    IntegratedPianoTransformer,
    _compute_integrated_loss_components,
)
from src.utils.func import filter_valid_args


os.environ["WANDB_PROJECT"] = "pianist-transformer"


def print_model_parameters(model):
    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print(f"Total Parameters:     {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Non-Trainable Parameters: {(total_params - trainable_params):,}")
    print("--------------------------------------------------")
    print(f"Total Parameters (M):     {total_params / 1_000_000:.2f}M")
    print(f"Trainable Parameters (M): {trainable_params / 1_000_000:.2f}M")
    print("--------------------------------------------------")


def load_torch_state_dict(checkpoint_path):
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.is_dir():
        safetensors_path = checkpoint_path / "model.safetensors"
        pytorch_path = checkpoint_path / "pytorch_model.bin"
        if safetensors_path.exists():
            from safetensors.torch import load_file

            return load_file(str(safetensors_path))
        if pytorch_path.exists():
            checkpoint = torch.load(pytorch_path, map_location="cpu")
        else:
            raise FileNotFoundError(f"No model.safetensors or pytorch_model.bin in {checkpoint_path}")
    else:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    return checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint


def score_json_path(refined_dir, score_rel_path):
    score_path = Path(refined_dir) / score_rel_path
    candidates = [
        score_path.with_suffix(".json"),
        score_path.parent / f"{score_path.stem}.node_a.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def make_windows(total_notes, block_notes, overlap_ratio, min_notes):
    total_notes = int(total_notes)
    if total_notes < min_notes:
        return []
    if total_notes <= block_notes:
        return [(0, total_notes)]

    stride = max(1, int(block_notes * (1.0 - overlap_ratio)))
    windows = []
    start = 0
    while start + block_notes <= total_notes:
        windows.append((start, start + block_notes))
        start += stride
    if windows[-1][1] != total_notes and total_notes - start >= min_notes:
        windows.append((total_notes - block_notes, total_notes))

    deduped = []
    seen = set()
    for window in windows:
        if window not in seen:
            deduped.append(window)
            seen.add(window)
    return deduped


def default_input_continuous_dim(task_type, input_feature_mode, score_feature_dim=8, continuous_dim=7):
    if input_feature_mode == "integrated":
        if task_type == "epr":
            return 2 + 3 + score_feature_dim
        if task_type == "csr":
            return 2 + continuous_dim
    return continuous_dim


def infer_input_feature_mode(config):
    mode = config.get("input_feature_mode")
    if mode is not None:
        return str(mode).lower()
    task_type = config.get("task_type", "epr").lower()
    input_dim = config.get("input_continuous_dim")
    if input_dim is not None:
        if task_type == "epr" and int(input_dim) <= 3:
            return "legacy"
        if task_type == "csr" and int(input_dim) <= 7:
            return "legacy"
    return "integrated"


def make_score_note_input(score_continuous, score_feature, has_score_feature, input_feature_mode):
    if input_feature_mode == "legacy":
        return score_continuous
    assert len(score_continuous) == len(score_feature), (
        f"score_continuous/score_feature length mismatch: "
        f"{len(score_continuous)} vs {len(score_feature)}"
    )
    assert len(score_continuous) == len(has_score_feature), (
        f"score_continuous/has_score_feature length mismatch: "
        f"{len(score_continuous)} vs {len(has_score_feature)}"
    )
    rows = []
    for shared, feature, has_feature in zip(score_continuous, score_feature, has_score_feature):
        assert len(shared) >= 3, f"score_continuous row too short: expected >=3, got {len(shared)}"
        assert len(feature) >= 8, f"score_feature row too short: expected >=8, got {len(feature)}"
        has_feature = 1.0 if bool(has_feature) else 0.0
        rows.append([has_feature, 0.0] + list(shared[:3]) + [float(value) * has_feature for value in feature[:8]])
    return rows


def make_performance_note_input(label_continuous, input_feature_mode):
    if input_feature_mode == "legacy":
        return label_continuous
    for row in label_continuous:
        assert len(row) >= 7, f"label_continuous row too short: expected >=7, got {len(row)}"
    return [[0.0, 1.0] + list(row[:7]) for row in label_continuous]


def distributed_info():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(), torch.distributed.get_world_size()
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        return int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    return 0, 1


def build_work_manifest(
    metadata_path,
    refined_dir,
    split,
    block_notes,
    overlap_ratio,
    min_notes,
    max_works=None,
    include_all_performance_dataset=None,
    max_non_asap_performances_per_work=None,
    selection_seed=42,
):
    columns = [
        "tier_a",
        "split",
        "refined_score_midi_path",
        "refined_performance_midi_path",
        "refined_alignment_path",
        "refined_score_note_count",
        "performance_dataset",
    ]
    df = pd.read_csv(metadata_path, usecols=columns)
    df = df[df["tier_a"].fillna(False).astype(bool)]
    df = df[df["split"] == split]
    df = df[df["refined_score_midi_path"].notna()]
    df = df[df["refined_performance_midi_path"].notna()]
    df = df[df["refined_alignment_path"].notna()]
    df = df.sort_values(["refined_score_midi_path", "refined_performance_midi_path"], kind="stable")

    manifest = []
    for score_rel_path, group in df.groupby("refined_score_midi_path", sort=True):
        selected_group = group
        if (
            include_all_performance_dataset is not None
            and max_non_asap_performances_per_work is not None
        ):
            dataset = group["performance_dataset"].fillna("").astype(str)
            always_mask = dataset == str(include_all_performance_dataset)
            always = group[always_mask]
            other = group[~always_mask]
            if len(other) > max_non_asap_performances_per_work:
                rng = random.Random(f"{selection_seed}:{score_rel_path}")
                sampled_indices = rng.sample(list(other.index), max_non_asap_performances_per_work)
                other = other.loc[sampled_indices]
            selected_group = pd.concat([always, other], axis=0).sort_values(
                ["refined_performance_midi_path"],
                kind="stable",
            )

        path = score_json_path(refined_dir, score_rel_path)
        if not path.exists():
            continue
        note_count = int(group["refined_score_note_count"].iloc[0])
        windows = make_windows(note_count, block_notes, overlap_ratio, min_notes)
        if not windows:
            continue
        selected_sources = selected_group["refined_performance_midi_path"].tolist()
        manifest.append(
            {
                "path": str(path),
                "score_source": score_rel_path,
                "note_count": note_count,
                "windows": windows,
                "selected_performance_sources": selected_sources,
                "estimated_performances": int(len(selected_sources)),
                "estimated_examples": int(len(windows) * len(selected_sources)),
            }
        )
    if max_works is not None:
        manifest = manifest[:max_works]
    return manifest


class PianoCoReNodeSFTDataset(Dataset):
    def __init__(
        self,
        manifest,
        split,
        task_type="epr",
        input_feature_mode="integrated",
        shuffle=True,
        seed=42,
        max_performances_per_work=None,
        max_windows_per_work=None,
        cache_size=2,
    ):
        super().__init__()
        self.split = split
        self.task_type = task_type
        self.input_feature_mode = input_feature_mode
        items = list(manifest)
        if shuffle:
            random.Random(seed).shuffle(items)

        self.items = []
        self.cumulative_sizes = []
        total = 0
        for item in items:
            windows = list(item["windows"])
            if max_windows_per_work is not None:
                windows = windows[:max_windows_per_work]
            selected_sources = item.get("selected_performance_sources")
            performance_count = len(selected_sources) if selected_sources is not None else int(item["estimated_performances"])
            if max_performances_per_work is not None:
                performance_count = min(performance_count, max_performances_per_work)
                if selected_sources is not None:
                    selected_sources = selected_sources[:performance_count]
            if not windows or performance_count <= 0:
                continue

            item = dict(item)
            item["windows"] = windows
            if selected_sources is not None:
                item["selected_performance_sources"] = selected_sources
            item["effective_performances"] = performance_count
            item["effective_examples"] = performance_count * len(windows)
            self.items.append(item)
            total += item["effective_examples"]
            self.cumulative_sizes.append(total)

        self.total_examples = total
        self.cache_size = cache_size
        self._cache = OrderedDict()

    def __len__(self):
        return self.total_examples

    def _load_work(self, path):
        if path in self._cache:
            self._cache.move_to_end(path)
            return self._cache[path]

        with open(path, "r", encoding="utf-8") as file:
            work = json.load(file)
        self._cache[path] = work
        self._cache.move_to_end(path)
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return work

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)

        item_idx = bisect.bisect_right(self.cumulative_sizes, index)
        prev_size = 0 if item_idx == 0 else self.cumulative_sizes[item_idx - 1]
        local_index = index - prev_size
        item = self.items[item_idx]

        windows = item["windows"]
        window_count = len(windows)
        perf_slot = local_index // window_count
        window_slot = local_index % window_count
        start, end = windows[window_slot]

        work = self._load_work(item["path"])
        score = work["score"]
        performances = [
            perf for perf in work["performances"]
            if perf.get("split", self.split) == self.split
        ]
        selected_sources = item.get("selected_performance_sources")
        if selected_sources is not None:
            by_source = {perf.get("performance_source"): perf for perf in performances}
            performances = [by_source[source] for source in selected_sources if source in by_source]
        if not performances:
            raise IndexError(f"No performances for split={self.split} in {item['path']}")

        # A tiny number of PianoCoRe-A rows were skipped for pitch mismatch. If
        # metadata counted one of those rows, wrap to a valid performance instead
        # of making DistributedSampler lengths uneven.
        perf = performances[int(perf_slot) % len(performances)]
        labels = perf["label_continuous"]
        interpolated = perf["interpolated"]
        task_type = self.task_type.lower()
        if task_type == "epr":
            score_feature = score.get("score_feature", [[0.0] * 8 for _ in score["pitch"]])
            has_score_feature = score.get("has_score_feature", [0] * len(score["pitch"]))
            continuous = make_score_note_input(
                score["score_continuous"][start:end],
                score_feature[start:end],
                has_score_feature[start:end],
                self.input_feature_mode,
            )
            labels_continuous = labels[start:end]
            label_mask = None
        elif task_type == "csr":
            continuous = make_performance_note_input(labels[start:end], self.input_feature_mode)
            labels_continuous = score["score_feature"][start:end]
            label_mask = score["has_score_feature"][start:end]
        else:
            raise ValueError(f"Unsupported task_type: {self.task_type}")

        sample = {
            "pitch_ids": score["pitch"][start:end],
            "continuous": continuous,
            "labels_continuous": labels_continuous,
            "interpolated": interpolated[start:end],
            "performance_dataset": perf.get("performance_dataset", "unknown"),
            "performance_id": perf.get("performance_id", "unknown"),
        }
        if label_mask is not None:
            sample["label_mask"] = label_mask
        return sample


class NodeSFTTrainer(Trainer):
    def _model_config(self, model):
        return model.module.config if hasattr(model, "module") else model.config

    def _record_loss_components(self, model, outputs, inputs):
        if not hasattr(outputs, "logits"):
            return
        if "labels_continuous" not in inputs or "attention_mask" not in inputs:
            return
        components = _compute_integrated_loss_components(
            self._model_config(model),
            outputs.logits.detach(),
            inputs["labels_continuous"].detach(),
            inputs["attention_mask"].detach(),
        )
        if not getattr(self, "_loss_component_sums", None):
            self._loss_component_sums = {name: 0.0 for name in components}
            self._loss_component_count = 0
        for name, value in components.items():
            self._loss_component_sums[name] += float(value.detach().float().cpu())
        self._loss_component_count += 1

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        loss = outputs.loss
        self._record_loss_components(model, outputs, inputs)
        return (loss, outputs) if return_outputs else loss

    def log(self, logs, *args, **kwargs):
        count = getattr(self, "_loss_component_count", 0)
        if count and "loss" in logs:
            for name, total in self._loss_component_sums.items():
                logs[f"loss_{name}"] = total / count
            self._loss_component_sums = {}
            self._loss_component_count = 0
        if self.is_world_process_zero():
            printable_logs = {"step": self.state.global_step}
            printable_logs.update(logs)
            print(json.dumps(printable_logs, ensure_ascii=False, sort_keys=True), flush=True)
        return super().log(logs, *args, **kwargs)

    def _get_train_sampler(self, train_dataset=None):
        train_dataset = train_dataset if train_dataset is not None else self.train_dataset
        if train_dataset is None or not hasattr(train_dataset, "__len__"):
            return None
        if self.args.world_size <= 1:
            return SequentialSampler(train_dataset)
        return DistributedSampler(
            train_dataset,
            num_replicas=self.args.world_size,
            rank=self.args.process_index,
            shuffle=False,
            drop_last=self.args.dataloader_drop_last,
        )



    def _list_checkpoint_dirs(self):
        out = Path(self.args.output_dir)
        if not out.exists():
            return []
        dirs = [p for p in out.iterdir() if p.is_dir() and p.name.startswith("checkpoint-")]
        # exclude checkpoint-best special folder
        dirs = [d for d in dirs if d.name != "checkpoint-best"]
        def step_of(d):
            m = re.match(r"checkpoint-(\d+)$", d.name)
            return int(m.group(1)) if m else -1
        dirs.sort(key=step_of)
        return dirs

    def _cleanup_checkpoints(self, keep_paths):
        out = Path(self.args.output_dir)
        if not out.exists():
            return
        for p in out.iterdir():
            if not p.is_dir():
                continue
            if p.name == "checkpoint-best":
                # keep if requested
                if str(p) in keep_paths:
                    continue
                # otherwise remove
                shutil.rmtree(p)
                continue
            if p.name.startswith("checkpoint-"):
                if str(p) in keep_paths:
                    continue
                shutil.rmtree(p)

    def evaluate(self, *args, **kwargs):
        # call base evaluate to get metrics
        metrics = super().evaluate(*args, **kwargs)
        if not self.is_world_process_zero():
            return metrics

        # determine metric key
        metric_key = getattr(self.args, "metric_for_best_model", None)
        if metric_key is None:
            metric_key = "eval_loss"

        # possible keys in returned metrics
        candidate_keys = [metric_key, f"eval_{metric_key}", "loss", "eval_loss"]
        metric_value = None
        for k in candidate_keys:
            if k in metrics:
                metric_value = metrics[k]
                break

        # find latest checkpoint (highest step)
        ckpts = self._list_checkpoint_dirs()
        latest_ckpt = str(ckpts[-1]) if ckpts else None

        # init best tracking
        if not hasattr(self, "_best_metric"):
            self._best_metric = None
            self._best_ckpt = None

        # compare metrics
        is_better = False
        if metric_value is not None:
            greater_is_better = getattr(self.args, "greater_is_better", False)
            if self._best_metric is None:
                is_better = True
            else:
                if greater_is_better:
                    is_better = metric_value > self._best_metric
                else:
                    is_better = metric_value < self._best_metric

        # if we have a new best, copy latest checkpoint to checkpoint-best
        out = Path(self.args.output_dir)
        best_dir = out / "checkpoint-best"
        if is_better and latest_ckpt is not None:
            # remove previous best if exists and different
            if self._best_ckpt and best_dir.exists():
                try:
                    shutil.rmtree(best_dir)
                except Exception:
                    pass
            try:
                # copy latest to checkpoint-best
                if best_dir.exists():
                    shutil.rmtree(best_dir)
                shutil.copytree(latest_ckpt, best_dir)
                self._best_metric = metric_value
                self._best_ckpt = str(best_dir)
            except Exception:
                # fallback: just record path
                self._best_metric = metric_value
                self._best_ckpt = latest_ckpt

        # determine keep paths: latest and best (if exist)
        keep = set()
        if latest_ckpt:
            keep.add(str(latest_ckpt))
        if self._best_ckpt:
            keep.add(str(self._best_ckpt))

        # cleanup other checkpoints
        self._cleanup_checkpoints(keep_paths=keep)

        return metrics

class NodeSFTDataCollator:
    def __init__(self, pitch_pad_id=128, task_type="epr"):
        self.pitch_pad_id = pitch_pad_id
        self.task_type = task_type

    def __call__(self, examples):
        pitch_tensors = [torch.tensor(example["pitch_ids"], dtype=torch.long) for example in examples]
        continuous_tensors = [
            torch.tensor(example["continuous"], dtype=torch.float32) for example in examples
        ]
        label_tensors = [
            torch.tensor(example["labels_continuous"], dtype=torch.float32) for example in examples
        ]
        interpolated_tensors = [
            torch.tensor(example["interpolated"], dtype=torch.bool) for example in examples
        ]

        pitch_ids = pad_sequence(pitch_tensors, batch_first=True, padding_value=self.pitch_pad_id)
        continuous = pad_sequence(continuous_tensors, batch_first=True, padding_value=0.0)
        labels_continuous = pad_sequence(label_tensors, batch_first=True, padding_value=0.0)
        interpolated = pad_sequence(interpolated_tensors, batch_first=True, padding_value=False)
        attention_mask = (pitch_ids != self.pitch_pad_id).long()
        if self.task_type == "csr":
            label_mask_tensors = [
                torch.tensor(example["label_mask"], dtype=torch.long) for example in examples
            ]
            loss_mask = pad_sequence(label_mask_tensors, batch_first=True, padding_value=0)
            attention_mask = attention_mask * loss_mask

        return {
            "pitch_ids": pitch_ids,
            "continuous": continuous,
            "labels_continuous": labels_continuous,
            "attention_mask": attention_mask,
            "interpolated": interpolated,
        }


def create_model(train_config):
    dtype = torch.bfloat16 if train_config.get("bf16", False) and torch.cuda.is_available() else torch.float32
    backbone_type = train_config.get("backbone_type", "t5").lower()
    task_type = train_config.get("task_type", "epr").lower()
    input_feature_mode = infer_input_feature_mode(train_config)
    if task_type == "epr":
        missing_beta_keys = [
            key for key in ("epr_distribution", "beta_eps", "beta_kappa_min")
            if key not in train_config
        ]
        if missing_beta_keys:
            warnings.warn(
                "EPR config is missing probabilistic-head keys "
                f"{missing_beta_keys}. Falling back to defaults: "
                f"epr_distribution={train_config.get('epr_distribution', 'point')}, "
                f"beta_eps={train_config.get('beta_eps', 1e-5)}, "
                f"beta_kappa_min={train_config.get('beta_kappa_min', 1e-3)}",
                stacklevel=2,
            )
    score_feature_dim = train_config.get("score_feature_dim", 8)
    input_continuous_dim = train_config.get(
        "input_continuous_dim",
        default_input_continuous_dim(
            task_type,
            input_feature_mode,
            score_feature_dim=score_feature_dim,
            continuous_dim=train_config.get("continuous_dim", 7),
        ),
    )
    model_config = IntegratedPianoT5GemmaConfig(
        backbone_type=backbone_type,
        hidden_size=train_config["hidden_size"],
        intermediate_size=train_config["intermediate_size"],
        num_attention_heads=train_config["num_attention_heads"],
        num_key_value_heads=train_config["num_key_value_heads"],
        head_dim=train_config["head_dim"],
        encoder_layers_num=train_config["encoder_layers_num"],
        decoder_layers_num=train_config["decoder_layers_num"],
        gpt_layers_num=train_config.get("gpt_layers_num"),
        bert_layers_num=train_config.get("bert_layers_num"),
        max_position_embeddings=train_config.get("max_position_embeddings", 4096),
        attention_dropout=train_config.get("attention_dropout", 0.0),
        continuous_dim=train_config["continuous_dim"],
        input_continuous_dim=input_continuous_dim,
        output_continuous_dim=train_config.get("output_continuous_dim", train_config["continuous_dim"]),
        score_feature_dim=score_feature_dim,
        max_time_ms=train_config["max_time_ms"],
        pedal_output_activation=train_config.get("pedal_output_activation", "sigmoid"),
        task_type=task_type,
        time_loss_type=train_config["time_loss_type"],
        value_loss_type=train_config["value_loss_type"],
        csr_grid_loss_type=train_config.get("csr_grid_loss_type", "huber"),
        huber_delta=train_config["huber_delta"],
        loss_weights=train_config["loss_weights"],
        csr_loss_weights=train_config.get("csr_loss_weights"),
        decoder_input_mode=train_config["decoder_input_mode"],
        input_feature_mode=input_feature_mode,
        note_embedding_mode=train_config.get("note_embedding_mode", "fine"),
        special_note_vocab_size=train_config.get("special_note_vocab_size", 5),
        special_note_ids=train_config.get("special_note_ids"),
        pine_partition_dims=train_config.get("pine_partition_dims"),
        use_full_type_embedding=train_config.get("use_full_type_embedding", True),
        use_group_presence_mask=train_config.get("use_group_presence_mask", True),
        head_input_mode=train_config.get("head_input_mode", "full"),
        embedding_depth=train_config.get("embedding_depth", 2),
        head_depth=train_config.get("head_depth", 2),
        head_activation=train_config.get("head_activation", "gelu"),
        epr_distribution=train_config.get("epr_distribution", "point"),
        beta_eps=train_config.get("beta_eps", 1e-5),
        beta_kappa_min=train_config.get("beta_kappa_min", 1e-3),
        prior_token_keep_prob=train_config.get("prior_token_keep_prob", 1.0),
        torch_dtype=dtype,
    )

    resume_path = train_config.get("resume_path")
    if resume_path:
        model = IntegratedPianoT5Gemma(model_config) if backbone_type in {"t5", "t5gemma"} else IntegratedPianoTransformer(model_config)
        state_dict = load_torch_state_dict(resume_path)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"Loaded Integrated {backbone_type} weights from {resume_path}")
        print(f"Missing keys: {len(missing)}")
        print(f"Unexpected keys: {len(unexpected)}")
        return model

    if backbone_type in {"t5", "t5gemma"}:
        model = IntegratedPianoT5Gemma(model_config)
    elif backbone_type in {"bert", "gpt"}:
        model = IntegratedPianoTransformer(model_config)
    else:
        raise ValueError(f"Unsupported backbone_type: {backbone_type}")

    pretrained_model = train_config.get("pretrained_model")
    if pretrained_model and train_config.get("load_pianoformer_backbone", True):
        if backbone_type not in {"t5", "t5gemma"}:
            raise ValueError("load_pianoformer_backbone is only supported for t5 backbones")
        incompatible = model.load_pianoformer_backbone(pretrained_model, torch_dtype=dtype)
        print(f"Loaded PianistTransformer backbone from {pretrained_model}")
        print(f"Missing keys: {len(incompatible.missing_keys)}")
        print(f"Unexpected keys: {len(incompatible.unexpected_keys)}")
    return model


def enable_eval_best_checkpointing(train_config):
    eval_strategy = train_config.get("eval_strategy", train_config.get("evaluation_strategy", "no"))
    save_strategy = train_config.get("save_strategy", "steps")
    if eval_strategy == "no" or save_strategy == "no":
        return

    train_config.setdefault("load_best_model_at_end", True)
    train_config.setdefault("metric_for_best_model", "eval_loss")
    train_config.setdefault("greater_is_better", False)

    if train_config["load_best_model_at_end"] and eval_strategy == "steps" and save_strategy == "steps":
        eval_steps = train_config.get("eval_steps")
        if eval_steps:
            train_config["save_steps"] = eval_steps


def main():
    current_datetime = datetime.datetime.now()
    outname = "inr_" + current_datetime.strftime("%Y-%m-%d-%H-%M-%S")

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/inr_config_pianocore.json")
    parser.add_argument("--deepspeed", type=str, help="Path to DeepSpeed config")
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--limit_works", type=int, default=None)
    parser.add_argument("--limit_performances_per_work", type=int, default=None)
    parser.add_argument("--limit_windows_per_work", type=int, default=None)
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    if torch.cuda.is_available():
        if local_rank >= 0:
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    with open(args.config, "r", encoding="utf-8") as file:
        train_config = json.load(file)
    task_type = train_config.get("task_type", "epr").lower()
    input_feature_mode = infer_input_feature_mode(train_config)
    train_config["input_feature_mode"] = input_feature_mode
    train_config.setdefault(
        "input_continuous_dim",
        default_input_continuous_dim(
            task_type,
            input_feature_mode,
            score_feature_dim=train_config.get("score_feature_dim", 8),
            continuous_dim=train_config.get("continuous_dim", 7),
        ),
    )

    if args.max_steps is not None:
        train_config["max_steps"] = args.max_steps
    if args.limit_works is not None:
        train_config["max_train_works"] = args.limit_works
        train_config["max_eval_works"] = min(args.limit_works, train_config.get("max_eval_works") or args.limit_works)
    if args.limit_performances_per_work is not None:
        train_config["max_performances_per_work"] = args.limit_performances_per_work
    if args.limit_windows_per_work is not None:
        train_config["max_windows_per_work"] = args.limit_windows_per_work

    enable_eval_best_checkpointing(train_config)

    train_config["output_dir"] = os.path.join(train_config["output_dir"], outname)
    train_config["run_name"] = outname
    train_config["logging_dir"] = os.path.join(train_config["logging_dir"], outname)

    output_dir = Path(train_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "train_config.json", "w", encoding="utf-8") as file:
        json.dump(train_config, file, indent=2, ensure_ascii=False)

    train_manifest = build_work_manifest(
        metadata_path=train_config["metadata_path"],
        refined_dir=train_config["refined_dir"],
        split="train",
        block_notes=train_config["block_notes"],
        overlap_ratio=train_config["overlap_ratio"],
        min_notes=train_config["min_notes"],
        max_works=train_config.get("max_train_works"),
    )
    eval_manifest = build_work_manifest(
        metadata_path=train_config["metadata_path"],
        refined_dir=train_config["refined_dir"],
        split="test",
        block_notes=train_config["block_notes"],
        overlap_ratio=train_config["overlap_ratio"],
        min_notes=train_config["min_notes"],
        max_works=train_config.get("max_eval_works"),
        include_all_performance_dataset=train_config.get("eval_include_all_performance_dataset"),
        max_non_asap_performances_per_work=train_config.get("max_eval_non_asap_performances_per_work"),
        selection_seed=train_config.get("seed", 42),
    )
    print(f"Train works: {len(train_manifest)}")
    print(f"Eval works: {len(eval_manifest)}")
    print(f"Estimated train examples: {sum(item['estimated_examples'] for item in train_manifest):,}")
    print(f"Estimated eval examples: {sum(item['estimated_examples'] for item in eval_manifest):,}")

    train_dataset = PianoCoReNodeSFTDataset(
        train_manifest,
        split="train",
        task_type=task_type,
        input_feature_mode=input_feature_mode,
        shuffle=True,
        seed=train_config["seed"],
        max_performances_per_work=train_config.get("max_performances_per_work"),
        max_windows_per_work=train_config.get("max_windows_per_work"),
        cache_size=train_config.get("node_cache_size", 16),
    )
    eval_dataset = PianoCoReNodeSFTDataset(
        eval_manifest,
        split="test",
        task_type=task_type,
        input_feature_mode=input_feature_mode,
        shuffle=False,
        seed=train_config["seed"],
        max_performances_per_work=train_config.get("max_eval_performances_per_work"),
        max_windows_per_work=train_config.get("max_eval_windows_per_work"),
        cache_size=train_config.get("node_cache_size", 16),
    )

    model = create_model(train_config)
    model.to(device)
    print_model_parameters(model)

    training_args_dict = filter_valid_args(train_config, TrainingArguments)
    if args.deepspeed:
        training_args_dict["deepspeed"] = args.deepspeed
    if "accelerator_config" in train_config:
        training_args_dict["accelerator_config"] = train_config["accelerator_config"]
    if torch.cuda.device_count() > 1:
        training_args_dict.setdefault("ddp_find_unused_parameters", True)
        training_args_dict.setdefault("ddp_broadcast_buffers", False)
    training_args = TrainingArguments(**training_args_dict)

    trainer = NodeSFTTrainer(
        model=model,
        args=training_args,
        data_collator=NodeSFTDataCollator(
            pitch_pad_id=train_config["pitch_pad_id"],
            task_type=task_type,
        ),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    resume_path = train_config.get("resume_path")
    trainer.train(resume_from_checkpoint=resume_path if resume_path else None)
    trainer.save_model()


if __name__ == "__main__":
    main()
