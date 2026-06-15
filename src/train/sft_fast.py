import json
import argparse
import datetime
import os
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from src.model.pianoformer import PianoT5GemmaConfig, PianoT5Gemma
from src.utils.func import filter_valid_args

os.environ["WANDB_PROJECT"] = "pianist-transformer"


def group_ids(examples, block_size, overlap_ratio):
    """Split examples into sliding windows."""
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
        random_windows = random_cut(windows)
        for start, end in windows:
            x = examples["x"][i][start: end]
            label = label_[start: end]
            xs.append(x)
            labels.append(label)
        for start, end in random_windows:
            x = examples["x"][i][start: end]
            label = label_[start: end]
            xs.append(x)
            labels.append(label)
    return {"input_ids": xs, "labels": labels}


class FastSFTDataset(Dataset):
    """Fast dataset loader that reads pre-split files."""

    def __init__(self, data_file, block_size, overlap_ratio):
        print(f"Loading from {data_file}...")

        # Read all lines
        with open(data_file, 'r') as f:
            lines = f.readlines()

        print(f"Read {len(lines)} examples, parsing JSON...")

        # Parse JSON
        examples = []
        for line in tqdm(lines, desc="Parsing"):
            examples.append(json.loads(line))

        print(f"Processing windows...")

        # Process into windows
        processed = group_ids(
            {"x": [e['x'] for e in examples],
             "label": [e['label'] for e in examples]},
            block_size, overlap_ratio
        )

        self.input_ids = processed['input_ids']
        self.labels = processed['labels']
        print(f"Created {len(self.input_ids)} windows")

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "labels": self.labels[idx]
        }


class DiffusionSFTDataCollator:
    def __init__(self, config, transposition_range=(-3, 3)):
        self.mask_token_id = config.mask_token_id
        self.pad_token_id = config.pad_token_id
        self.bos_token_id = config.bos_token_id

        self.pitch_token_start = config.valid_id_range[0][0]
        self.pitch_token_end = config.valid_id_range[0][1]

        self.valid_id_range = config.valid_id_range
        self.transposition_range = transposition_range

    def __call__(self, examples):
        input_tensors = [torch.tensor(f["input_ids"]).long() for f in examples]
        label_tensors = [torch.tensor(f["labels"]).long() for f in examples]
        input_ids = pad_sequence(input_tensors, batch_first=True, padding_value=self.pad_token_id)
        label_ids = pad_sequence(label_tensors, batch_first=True, padding_value=-100)

        attention_mask = (input_ids != self.pad_token_id).long()

        return {"input_ids": input_ids, "labels": label_ids, "attention_mask": attention_mask}


if __name__ == "__main__":
    current_datetime = datetime.datetime.now()
    outname = "sft_" + current_datetime.strftime("%Y-%m-%d-%H-%M-%S")

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/sft_pianocore_from_scratch.json")
    parser.add_argument('--deepspeed', type=str, help='Path to DeepSpeed config')
    parser.add_argument('--local_rank', type=int, default=-1, help='local rank passed from deepspeed')
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    with open(args.config, "r") as f:
        train_config = json.load(f)
    train_config["output_dir"] = os.path.join(train_config["output_dir"], outname)
    train_config["run_name"] = outname
    train_config["logging_dir"] = os.path.join(train_config["logging_dir"], outname)

    config = PianoT5GemmaConfig(
        encoder_layers_num=10,
        decoder_layers_num=2,
        torch_dtype=torch.bfloat16
    )

    # Load pre-split data files
    data_dir = Path(train_config["data_paths"][0]).parent / "split"
    train_file = data_dir / "train.jsonl"
    test_file = data_dir / "test.jsonl"

    if not train_file.exists() or not test_file.exists():
        raise FileNotFoundError(
            f"Pre-split files not found. Please run:\n"
            f"  python src/data_process/split_jsonl_by_split.py \\\n"
            f"    --input {train_config['data_paths'][0]} \\\n"
            f"    --output-dir {data_dir}"
        )

    print("Loading train dataset...")
    train_dataset = FastSFTDataset(
        str(train_file),
        block_size=train_config["block_size"],
        overlap_ratio=train_config["overlap_ratio"]
    )

    print("Loading test dataset...")
    valid_dataset = FastSFTDataset(
        str(test_file),
        block_size=train_config["block_size"],
        overlap_ratio=train_config["overlap_ratio"]
    )

    data_collator = DiffusionSFTDataCollator(config)

    if train_config.get("pretrained_model") is None:
        model = PianoT5Gemma(config)
        print("Training from scratch (no pretrained model)")
    else:
        model = PianoT5Gemma.from_pretrained(
            train_config["pretrained_model"],
            torch_dtype=torch.bfloat16
        )
        print(f"Loaded pretrained model from {train_config['pretrained_model']}")

    model.to(device)

    training_args = filter_valid_args(train_config, TrainingArguments)

    if torch.cuda.device_count() > 1:
        training_args["ddp_find_unused_parameters"] = True
        training_args["ddp_broadcast_buffers"] = False

    training_args = TrainingArguments(**training_args)

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
    )

    trainer.train()
    trainer.save_model()
