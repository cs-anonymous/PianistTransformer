"""
Split large JSONL file by 'split' field into separate files.
This avoids loading the entire dataset during training.
"""
import json
import argparse
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial
import os


def process_chunk(chunk, output_dir):
    """Process a chunk of lines and write to appropriate split files."""
    train_file = open(os.path.join(output_dir, "train.jsonl"), "a")
    test_file = open(os.path.join(output_dir, "test.jsonl"), "a")

    train_count = 0
    test_count = 0

    for line in chunk:
        item = json.loads(line)
        split = item.get("split", "train")

        if split == "train":
            train_file.write(line)
            train_count += 1
        elif split == "test":
            test_file.write(line)
            test_count += 1

    train_file.close()
    test_file.close()

    return train_count, test_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Input JSONL file")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--num-workers", type=int, default=16, help="Number of worker processes")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Lines per chunk")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Clear existing files
    train_path = output_dir / "train.jsonl"
    test_path = output_dir / "test.jsonl"
    if train_path.exists():
        train_path.unlink()
    if test_path.exists():
        test_path.unlink()

    print(f"Reading {args.input}...")

    # Read all lines (this is fast, just reading text)
    with open(args.input, 'r') as f:
        lines = f.readlines()

    print(f"Read {len(lines)} lines, splitting into chunks...")

    # Split into chunks
    chunks = [lines[i:i + args.chunk_size] for i in range(0, len(lines), args.chunk_size)]
    print(f"Created {len(chunks)} chunks, processing with {args.num_workers} workers...")

    # Process chunks in parallel
    total_train = 0
    total_test = 0

    with Pool(processes=args.num_workers) as pool:
        results = pool.map(partial(process_chunk, output_dir=args.output_dir), chunks)

        for train_count, test_count in results:
            total_train += train_count
            total_test += test_count

    print(f"\nSplitting complete:")
    print(f"  Train: {total_train} examples -> {train_path}")
    print(f"  Test: {total_test} examples -> {test_path}")


if __name__ == "__main__":
    main()
