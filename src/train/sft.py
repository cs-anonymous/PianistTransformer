import json
import argparse
import datetime
import os
import random
import shutil
from pathlib import Path

from datasets import load_dataset
import pandas as pd
import torch
from transformers import Trainer, TrainingArguments
from torch.nn.utils.rnn import pad_sequence

from src.model.pianoformer import PianoT5GemmaConfig, PianoT5Gemma
from src.utils.func import filter_valid_args

os.environ["WANDB_PROJECT"] = "pianist-transformer"


class BestLastTrainer(Trainer):
    def _cleanup_checkpoints(self, checkpoint_prefix="checkpoint", use_mtime=False):
        if not self.args.should_save:
            return
        output_dir = Path(self.args.output_dir)
        checkpoints = [p for p in output_dir.glob(f"{checkpoint_prefix}-*") if p.name != "checkpoint-best"]
        keep = set()
        if checkpoints:
            key = (lambda p: p.stat().st_mtime) if use_mtime else (lambda p: int(p.name.split("-")[-1]))
            keep.add(str(max(checkpoints, key=key)))
        best_path = getattr(self.state, "best_model_checkpoint", None)
        if best_path:
            keep.add(str(Path(best_path)))
        best_dir = output_dir / "checkpoint-best"
        if best_dir.exists():
            keep.add(str(best_dir))
        for checkpoint in checkpoints:
            if str(checkpoint) not in keep:
                shutil.rmtree(checkpoint, ignore_errors=True)

    def evaluate(self, *args, **kwargs):
        metrics = super().evaluate(*args, **kwargs)
        metric_key = getattr(self.args, "metric_for_best_model", None) or "eval_loss"
        metric_value = None
        for key in (metric_key, f"eval_{metric_key}", "loss", "eval_loss"):
            if key in metrics:
                metric_value = metrics[key]
                break
        output_dir = Path(self.args.output_dir)
        checkpoints = sorted(
            [p for p in output_dir.glob("checkpoint-*") if p.name != "checkpoint-best"],
            key=lambda p: int(p.name.split("-")[-1]),
        )
        latest = checkpoints[-1] if checkpoints else None
        if metric_value is not None and latest is not None:
            if self.state.best_metric is None or metric_value < self.state.best_metric:
                best_dir = output_dir / "checkpoint-best"
                if best_dir.exists():
                    shutil.rmtree(best_dir)
                shutil.copytree(latest, best_dir)
                self.state.best_metric = metric_value
                self.state.best_model_checkpoint = str(best_dir)
        self._cleanup_checkpoints()
        return metrics


def group_ids(examples, block_size, overlap_ratio, include_random_cut=True):
    def slide_window(total_len, window_len):
        window_len = window_len // 8 * 8
        out = []
        start = 0
        while start + window_len <= total_len:
            out.append((start, start + window_len))
            start += int(window_len * (1 - overlap_ratio)) // 8 * 8
        if len(out) == 0 or out[-1][1] != total_len:
            out.append((start, total_len))
        return out
    def random_cut(windows):
        out = []
        for start, end in windows:
            origin_len = end - start
            rand_len = random.randint(8, origin_len) // 8 * 8
            rand_start = random.randint(start, end - rand_len) // 8 * 8
            out.append((rand_start, rand_start + rand_len))
        return out
    xs = []
    labels = []
    for i in range(len(examples["x"])):
        label_ = []
        for j in range(len(examples["label"][i])):
            if j % 8 > 3:
                if examples["label"][i][j] >= 5261 + 64:
                    label_.append(5261 + 127)
                else:
                    label_.append(5261)
            else:
                label_.append(examples["label"][i][j])
        windows = slide_window(len(examples["x"][i]), block_size)
        for start, end in windows:
            x = examples["x"][i][start: end]
            label = label_[start: end]
            xs.append(x)
            labels.append(label)
        if include_random_cut:
            for start, end in random_cut(windows):
                x = examples["x"][i][start: end]
                label = label_[start: end]
                xs.append(x)
                labels.append(label)
    return {"input_ids": xs, "labels": labels}


def build_eval_pair_set(
    metadata_path,
    include_all_performance_dataset="ASAP",
    max_non_asap_performances_per_work=8,
    seed=42,
):
    columns = [
        "tier_a",
        "split",
        "refined_score_midi_path",
        "refined_performance_midi_path",
        "refined_alignment_path",
        "performance_dataset",
    ]
    df = pd.read_csv(metadata_path, usecols=columns)
    df = df[df["tier_a"].fillna(False).astype(bool)]
    df = df[df["split"] == "test"]
    df = df[df["refined_score_midi_path"].notna()]
    df = df[df["refined_performance_midi_path"].notna()]
    df = df[df["refined_alignment_path"].notna()]
    df = df.sort_values(["refined_score_midi_path", "refined_performance_midi_path"], kind="stable")

    pairs = set()
    for score_rel_path, group in df.groupby("refined_score_midi_path", sort=True):
        dataset = group["performance_dataset"].fillna("").astype(str)
        always_mask = dataset == str(include_all_performance_dataset)
        always = group[always_mask]
        other = group[~always_mask]
        if max_non_asap_performances_per_work is not None and len(other) > max_non_asap_performances_per_work:
            rng = random.Random(f"{seed}:{score_rel_path}")
            other = other.loc[rng.sample(list(other.index), max_non_asap_performances_per_work)]
        selected = pd.concat([always, other], axis=0)
        for _, row in selected.iterrows():
            pairs.add((row["refined_score_midi_path"], row["refined_performance_midi_path"]))
    return pairs


class DiffusionSFTDataCollator:
    def __init__(self, config, transposition_range=(-3, 3), prior_token_keep_prob=1.0):
        self.mask_token_id = config.mask_token_id
        self.pad_token_id = config.pad_token_id
        self.bos_token_id = config.bos_token_id
        
        self.pitch_token_start = config.valid_id_range[0][0]
        self.pitch_token_end = config.valid_id_range[0][1]

        self.valid_id_range = config.valid_id_range

        self.transposition_range = transposition_range
        self.prior_token_keep_prob = float(prior_token_keep_prob)

    def _build_decoder_inputs(self, label_tensors):
        original_padded = pad_sequence(label_tensors, batch_first=True, padding_value=self.pad_token_id)
        decoder_input_ids = original_padded.new_full(original_padded.shape, self.pad_token_id)
        decoder_input_ids[:, 0] = self.bos_token_id
        if original_padded.shape[1] > 1:
            decoder_input_ids[:, 1:] = original_padded[:, :-1]

        decoder_attention_mask = (decoder_input_ids != self.pad_token_id)
        if self.prior_token_keep_prob >= 1.0 or decoder_input_ids.shape[1] <= 1:
            return decoder_input_ids, decoder_attention_mask.long()

        positions = torch.arange(decoder_input_ids.shape[1], device=decoder_input_ids.device)
        pitch_positions = positions.gt(0) & (((positions - 1) % 8) == 0)
        keep_mask = torch.rand(
            decoder_input_ids.shape,
            device=decoder_input_ids.device,
        ) < self.prior_token_keep_prob
        keep_mask[:, 0] = True
        keep_mask[:, pitch_positions] = True
        keep_mask |= ~decoder_attention_mask

        dropped_positions = decoder_attention_mask & ~keep_mask
        decoder_input_ids = decoder_input_ids.masked_fill(dropped_positions, self.mask_token_id)
        return decoder_input_ids, decoder_attention_mask.long()


    def __call__(self, examples):
        #len_list = [len(f["input_ids"]) for f in examples]
        #max_length = max(len_list)
        #input_ids = torch.tensor([f["input_ids"] + [self.pad_token_id] * (max_length - len(f["input_ids"])) for f in examples]).long()
        #label_ids = torch.tensor([f["labels"] + [-100] * (max_length - len(f["labels"])) for f in examples]).long()
        #attention_mask = torch.tensor([[1] * len(f["input_ids"]) + [0] * (max_length - len(f["input_ids"])) for f in examples]).long()
        
        input_tensors = [torch.tensor(f["input_ids"]).long() for f in examples]
        label_tensors = [torch.tensor(f["labels"]).long() for f in examples]
        input_ids = pad_sequence(input_tensors, batch_first=True, padding_value=self.pad_token_id)
        label_ids = pad_sequence(label_tensors, batch_first=True, padding_value=-100)
        decoder_input_ids, decoder_attention_mask = self._build_decoder_inputs(label_tensors)

        attention_mask = (input_ids != self.pad_token_id).long()
        
        #seq_len = input_ids.shape[1]
        #positional_mask = (torch.arange(seq_len, device=input_ids.device) % 8 == 0)

        #pitch_mask = positional_mask.unsqueeze(0) & (attention_mask == 1)
        #transpose_values = torch.randint(
        #    self.transposition_range[0],
        #    self.transposition_range[1] + 1,
        #    (input_ids.shape[0], 1),
        #    device=input_ids.device
        #)
        
        #transposition_tensor = torch.zeros_like(input_ids)
        #transposition_tensor[pitch_mask] = transpose_values.expand(-1, seq_len)[pitch_mask]

        #input_ids += transposition_tensor
        #label_ids += transposition_tensor
        #input_ids[pitch_mask] = torch.clamp(input_ids[pitch_mask], self.pitch_token_start, self.pitch_token_end - 1)
        #label_ids[pitch_mask] = torch.clamp(label_ids[pitch_mask], self.pitch_token_start, self.pitch_token_end - 1)

        #label_ids[pitch_mask] = -100
        #batch_size = input_ids.shape[0]
        #t = 1 - torch.rand((batch_size, 1))
        #mask_p = torch.ones_like(input_ids) * t
        #unmask_ind = torch.tensor([[1] * (len_list[i] // 2 + 4) + \
        #                           [1, 0, 0, 0, 0, 0, 0, 0] * ((len_list[i] // 2 - 4) // 8) + \
        #                            [1] * (max_length - len_list[i]) for i in range(batch_size)]).bool()
        #mask_p[unmask_ind] = 0
        #masked_ind = torch.bernoulli(mask_p).bool()
        #label_ids[~masked_ind] = -100
        #input_ids[masked_ind] = self.mask_token_id
        return {
            "input_ids": input_ids,
            "labels": label_ids,
            "attention_mask": attention_mask,
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask,
        }

if __name__ == "__main__":
    current_datetime = datetime.datetime.now()
    outname = "sft_" + current_datetime.strftime("%Y-%m-%d-%H-%M-%S")
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, default="configs/sft_config_pianocore.json")
    parser.add_argument('--deepspeed', type=str, help='Path to DeepSpeed config')
    parser.add_argument('--local_rank', type=int, default=-1, help='local rank passed from deepspeed')

    args = parser.parse_args()

    with open(args.config, "r") as f:
        train_config = json.load(f)
    train_config["output_dir"] = os.path.join(train_config["output_dir"], outname)
    train_config["run_name"] = outname
    train_config["logging_dir"] = os.path.join(train_config["logging_dir"], outname)

    training_args = filter_valid_args(train_config, TrainingArguments)

    # Force DDP for multi-GPU instead of DataParallel.
    if torch.cuda.device_count() > 1:
        training_args["ddp_find_unused_parameters"] = True
        training_args["ddp_broadcast_buffers"] = False

    training_args = TrainingArguments(**training_args)
    print(f"Using device: {training_args.device}")
    
    config = PianoT5GemmaConfig(
        encoder_layers_num=10,
        decoder_layers_num=2,
        torch_dtype=torch.bfloat16   
    )
    
    with training_args.main_process_first(desc="dataset preparation"):
        dataset = load_dataset("json", data_files=train_config["data_paths"])
        dataset = dataset.shuffle(seed=42)

        num_proc = int(train_config.get("dataset_num_proc", min(40, os.cpu_count() or 1)))
        train_dataset = dataset.filter(lambda example: example['split'] == 'train', num_proc=num_proc)
        valid_dataset = dataset.filter(lambda example: example['split'] == 'test', num_proc=num_proc)
        eval_pair_set = None
        if train_config.get("max_eval_non_asap_performances_per_work") is not None:
            eval_pair_set = build_eval_pair_set(
                metadata_path=train_config["metadata_path"],
                include_all_performance_dataset=train_config.get("eval_include_all_performance_dataset", "ASAP"),
                max_non_asap_performances_per_work=train_config.get("max_eval_non_asap_performances_per_work"),
                seed=train_config.get("seed", 42),
            )
            print(f"Selected PT eval performances: {len(eval_pair_set):,}", flush=True)
            valid_dataset = valid_dataset.filter(
                lambda example: (example["score_source"], example["performance_source"]) in eval_pair_set,
                num_proc=num_proc,
            )

        train_dataset = train_dataset.map(
            group_ids,
            fn_kwargs={
                "block_size": train_config["block_size"],
                "overlap_ratio": train_config["overlap_ratio"],
                "include_random_cut": True,
            },
            batched=True,
            num_proc=num_proc,
            remove_columns=['x', 'label', 'score_source', 'performance_source', 'cut', 'split']
        )
        valid_dataset = valid_dataset.map(
            group_ids,
            fn_kwargs={
                "block_size": train_config["block_size"],
                "overlap_ratio": train_config["overlap_ratio"],
                "include_random_cut": False,
            },
            batched=True,
            num_proc=num_proc,
            remove_columns=['x', 'label', 'score_source', 'performance_source', 'cut', 'split']
        )

    data_collator = DiffusionSFTDataCollator(
        config,
        prior_token_keep_prob=train_config.get("prior_token_keep_prob", 1.0),
    )
    if train_config["pretrained_model"] is None:
        model = PianoT5Gemma(config)
        print("Training from scratch (no pretrained model)")
    else:
        model = PianoT5Gemma.from_pretrained(
            train_config["pretrained_model"],
            torch_dtype=torch.bfloat16
        )
        print(f"Loaded pretrained model from {train_config['pretrained_model']}")

    model.to(training_args.device)

    trainer = BestLastTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator, 
        train_dataset=train_dataset["train"],
        eval_dataset=valid_dataset["train"],
    )

    resume_path = train_config.get("resume_path")
    trainer.train(resume_from_checkpoint=resume_path if resume_path else None)
    trainer.save_model()
