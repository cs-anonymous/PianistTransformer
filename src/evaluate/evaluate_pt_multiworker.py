"""
PT Evaluation: Each GPU runs multiple workers concurrently.
Strategy: Split samples into N*3 chunks (N workers per GPU).
"""

import argparse
import json
import os
import sys
from pathlib import Path
import numpy as np
import torch
import torch.multiprocessing as mp
from transformers import LogitsProcessorList
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

DEFAULT_PT_ROOT = Path("/home/kaititech/EPR/third_party/epr/PianistTransformer")
PT_PATH = Path(
    os.environ.get("PT_ROOT", ROOT_DIR if (ROOT_DIR / "models" / "sft").exists() else DEFAULT_PT_ROOT)
)
if str(PT_PATH) not in sys.path:
    sys.path.insert(0, str(PT_PATH))

from src.evaluate.epr_metrics import EPRMetrics, extract_features_from_continuous
from src.evaluate.epr_metrics_extended import ExtendedEPRMetrics
from src.model.generate import BatchSparseForcedTokenProcessor


def load_pt_model_and_config():
    """Load PT model and config."""
    from src.model.pianoformer import PianoT5Gemma, PianoT5GemmaConfig

    model_path = PT_PATH / "models/sft/model.safetensors"
    config_path = PT_PATH / "models/sft/config.json"

    if not model_path.exists() or not config_path.exists():
        raise FileNotFoundError(
            f"PT fine-tuned model not found under {PT_PATH / 'models/sft'}. "
            "Expected both config.json and model.safetensors."
        )

    with open(config_path) as f:
        config_dict = json.load(f)

    config = PianoT5GemmaConfig(
        encoder_layers_num=config_dict['encoder']['num_hidden_layers'],
        decoder_layers_num=config_dict['decoder']['num_hidden_layers'],
        hidden_size=config_dict['hidden_size'],
        intermediate_size=config_dict['encoder']['intermediate_size'],
        num_attention_heads=config_dict['encoder']['num_attention_heads'],
        num_key_value_heads=config_dict['encoder']['num_key_value_heads'],
        head_dim=config_dict['encoder']['head_dim'],
    )

    model = PianoT5Gemma(config)

    from safetensors.torch import load_file
    state_dict = load_file(str(model_path))
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    return model, config


def create_score_input(pitch_ids, score_continuous, config):
    """Recover PT score tokens from JSON note features.

    JSON timing values use log1p(ms) / log1p(max_time_ms). PT timing tokens use
    1 ms units, so we denormalize and round back to integer milliseconds.
    """
    score_tokens = []
    max_time_ms = 10000.0

    for i, pitch in enumerate(pitch_ids):
        # Pitch
        pitch_token = int(config.pitch_start + pitch)

        # Get score continuous values [ioi, duration, velocity, pedal×4]
        cont = score_continuous[i]
        ioi_norm, dur_norm, vel_norm = cont[0], cont[1], cont[2]
        if len(cont) >= 7:
            pedal_norm = cont[3:7]
        else:
            pedal_norm = [0.0, 0.0, 0.0, 0.0]

        # Denormalize JSON features back to PT token values.
        ioi_ms = int(round(np.clip(np.expm1(ioi_norm * np.log1p(max_time_ms)), 0, 4990)))
        dur_ms = int(round(np.clip(np.expm1(dur_norm * np.log1p(max_time_ms)), 0, 4990)))
        vel = int(round(np.clip(vel_norm * 127, 0, 127)))
        pedal_vals = np.clip(np.rint(np.array(pedal_norm) * 127).astype(int), 0, 127)

        # Create PT tokens with range checking
        ioi_token = np.clip(config.timing_start + ioi_ms, config.valid_id_range[1][0], config.valid_id_range[1][1] - 1)
        dur_token = np.clip(config.timing_start + dur_ms, config.valid_id_range[3][0], config.valid_id_range[3][1] - 1)
        vel_token = np.clip(config.velocity_start + vel, config.valid_id_range[2][0], config.valid_id_range[2][1] - 1)
        pedal_tokens = [int(np.clip(config.pedal_start + p, config.valid_id_range[4+j][0], config.valid_id_range[4+j][1] - 1)) for j, p in enumerate(pedal_vals)]

        score_tokens.extend([int(pitch_token), int(ioi_token), int(vel_token), int(dur_token)] + pedal_tokens)

    return score_tokens


def pt_tokens_to_continuous_features(token_ids, config):
    """Convert PT's output tokens to continuous features."""
    num_notes = len(token_ids) // 8
    tokens = np.array(token_ids[:num_notes * 8]).reshape(num_notes, 8)

    ioi_tokens = tokens[:, 1] - config.timing_start
    duration_tokens = tokens[:, 3] - config.timing_start
    velocity_tokens = tokens[:, 2] - config.velocity_start
    pedal_tokens = tokens[:, 4:8] - config.pedal_start

    max_time_ms = 10000.0

    ioi_ms = np.clip(ioi_tokens, 0, max_time_ms)
    ioi_normalized = np.log1p(ioi_ms) / np.log1p(max_time_ms)

    duration_ms = np.clip(duration_tokens, 0, max_time_ms)
    duration_normalized = np.log1p(duration_ms) / np.log1p(max_time_ms)

    velocity_normalized = np.clip(velocity_tokens, 0, 127) / 127.0
    pedal_normalized = np.clip(pedal_tokens, 0, 127) / 127.0

    continuous = np.concatenate([
        ioi_normalized.reshape(-1, 1),
        duration_normalized.reshape(-1, 1),
        velocity_normalized.reshape(-1, 1),
        pedal_normalized
    ], axis=1)

    return continuous


def worker_process(worker_id, gpu_id, sample_indices, dataset, config, result_queue):
    """Worker process for one GPU."""
    device = torch.device(f'cuda:{gpu_id}')

    # Load model
    model, _ = load_pt_model_and_config()
    model = model.to(device)
    model.eval()

    results = []

    for idx in sample_indices:
        try:
            sample = dataset[idx]
            pitch_ids = sample['pitch_ids']
            score_continuous = sample['continuous']  # Score timing/velocity/pedal
            target_continuous = sample['labels_continuous']

            if len(pitch_ids) > 512:
                continue

            score_input = create_score_input(pitch_ids, score_continuous, config)
            input_ids = torch.tensor([score_input], dtype=torch.long).to(device)
            logits_processor = LogitsProcessorList([
                BatchSparseForcedTokenProcessor(
                    input_ids,
                    config,
                    target_len=len(pitch_ids) * 8,
                    origin_len=0,
                    already=0.0,
                    weight=1.0,
                    progress_callback=None,
                )
            ])

            with torch.no_grad():
                generated_ids = model.generate(
                    input_ids=input_ids,
                    max_new_tokens=len(pitch_ids) * 8,
                    do_sample=True,
                    logits_processor=logits_processor,
                    temperature=1.0,
                    top_p=0.95,
                    pad_token_id=config.pad_token_id,
                    eos_token_id=config.eos_token_id,
                )

            pred_tokens = generated_ids[0, 1:len(pitch_ids) * 8 + 1].cpu().numpy().tolist()
            pred_cont = pt_tokens_to_continuous_features(pred_tokens, config)

            target_cont = np.array(target_continuous)

            min_len = min(len(pred_cont), len(target_cont))
            pred_cont = pred_cont[:min_len]
            target_cont = target_cont[:min_len]

            mask = np.ones(min_len, dtype=bool)

            results.append((pred_cont, target_cont, mask))

        except Exception as e:
            continue

    result_queue.put(results)


def evaluate_pt_multiworker(model, config, dataset, pedal_method='binary', max_samples=None, workers_per_gpu=3):
    """
    Multi-worker evaluation: N workers per GPU.
    """
    num_gpus = torch.cuda.device_count()
    total_workers = num_gpus * workers_per_gpu
    print(f"Using {num_gpus} GPUs with {workers_per_gpu} workers per GPU = {total_workers} total workers")

    if max_samples and max_samples < len(dataset):
        dataset = torch.utils.data.Subset(dataset, range(max_samples))

    num_samples = len(dataset)
    samples_per_worker = (num_samples + total_workers - 1) // total_workers

    # Assign samples to workers
    worker_assignments = []
    for worker_id in range(total_workers):
        gpu_id = worker_id % num_gpus
        start_idx = worker_id * samples_per_worker
        end_idx = min(start_idx + samples_per_worker, num_samples)
        if start_idx < num_samples:
            indices = list(range(start_idx, end_idx))
            worker_assignments.append((worker_id, gpu_id, indices))

    print(f"Samples per worker: {[len(w[2]) for w in worker_assignments]}")

    # Create result queue
    mp.set_start_method('spawn', force=True)
    result_queue = mp.Queue()

    # Launch workers
    processes = []
    for worker_id, gpu_id, indices in worker_assignments:
        p = mp.Process(target=worker_process, args=(worker_id, gpu_id, indices, dataset, config, result_queue))
        p.start()
        processes.append(p)

    print(f"Launched {len(processes)} worker processes")

    # Collect results
    all_results = []
    for _ in tqdm(range(len(processes)), desc="Collecting results"):
        worker_results = result_queue.get()
        all_results.extend(worker_results)

    # Wait for all processes
    for p in processes:
        p.join()

    print(f"\nSuccessfully processed {len(all_results)} samples")

    # Extract features
    print("Extracting features...")

    pred_features_list = {'velocity': [], 'duration': [], 'ioi': [], 'pedal': []}
    target_features_list = {'velocity': [], 'duration': [], 'ioi': [], 'pedal': []}

    use_joint_config = (pedal_method == 'binary')

    for pred_cont, target_cont, mask in all_results:
        pred_batch = pred_cont.reshape(1, -1, 7)
        target_batch = target_cont.reshape(1, -1, 7)
        mask_batch = mask.reshape(1, -1)

        pred_feats = extract_features_from_continuous(pred_batch, mask_batch, pedal_as_joint_config=use_joint_config)
        target_feats = extract_features_from_continuous(target_batch, mask_batch, pedal_as_joint_config=use_joint_config)

        for key in pred_features_list.keys():
            pred_features_list[key].append(pred_feats[key])
            target_features_list[key].append(target_feats[key])

    pred_features = {k: np.concatenate(v) for k, v in pred_features_list.items()}
    target_features = {k: np.concatenate(v) for k, v in target_features_list.items()}

    print(f"Collected {len(pred_features['velocity'])} notes")

    return pred_features, target_features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default='results/pt_evaluation_correct')
    parser.add_argument('--pedal-method', type=str, default='binary', choices=['binary', 'continuous'])
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--workers-per-gpu', type=int, default=3)
    args = parser.parse_args()

    print("="*70)
    print("PT CORRECT Evaluation: Multi-Worker per GPU")
    print("="*70)
    print(f"Pedal method: {args.pedal_method}")
    print(f"Workers per GPU: {args.workers_per_gpu}")
    print()

    with open(args.config) as f:
        our_config = json.load(f)

    pt_model, pt_config = load_pt_model_and_config()
    print("✅ PT model loaded successfully")

    from src.train.sft_node import PianoCoReNodeSFTDataset, build_work_manifest

    print("\nBuilding test dataset...")

    test_manifest = build_work_manifest(
        metadata_path=our_config.get('metadata_path', 'data/pianocore/metadata.csv'),
        refined_dir=our_config.get('refined_dir', 'data/pianocore/PianoCoRe/refined'),
        split='test',
        block_notes=our_config.get('block_notes', 512),
        overlap_ratio=our_config.get('overlap_ratio', 0.5),
        min_notes=our_config.get('min_notes', 64),
        max_works=our_config.get('max_eval_works'),
    )

    test_dataset = PianoCoReNodeSFTDataset(
        test_manifest,
        split='test',
        shuffle=False,
        seed=our_config['seed'],
        max_performances_per_work=our_config.get('max_eval_performances_per_work'),
        max_windows_per_work=our_config.get('max_eval_windows_per_work'),
    )

    print(f"Test dataset: {len(test_dataset)} samples")

    try:
        pred_features, target_features = evaluate_pt_multiworker(
            pt_model, pt_config, test_dataset, args.pedal_method, args.max_samples, args.workers_per_gpu
        )

        print("\nComputing metrics...")
        dist_metrics = EPRMetrics(bins=100)
        use_joint_config = (args.pedal_method == 'binary')
        dist_results = dist_metrics.compute_metrics(
            pred_features, target_features,
            pedal_is_joint_config=use_joint_config
        )

        extended_metrics = ExtendedEPRMetrics()
        pedal_feature_name = 'BPedal' if args.pedal_method == 'binary' else 'CPedal'

        extended_results = extended_metrics.compute_metrics_for_features(
            pred_features, target_features,
            pedal_method='binary' if use_joint_config else 'continuous'
        )

        if 'pedal' in extended_results:
            extended_results[pedal_feature_name] = extended_results.pop('pedal')

        overall = {}
        feature_keys = ['velocity', 'duration', 'ioi', pedal_feature_name]
        for metric in ['js', 'ia', 'mae', 'mse', 'rmse', 'pearson']:
            values = [extended_results[f][metric] for f in feature_keys if f in extended_results and metric in extended_results[f]]
            if values:
                overall[metric] = np.mean(values)
        extended_results['overall'] = overall

        print("\n" + "="*70)
        print(f"PT CORRECT EVALUATION RESULTS ({args.pedal_method} pedal)")
        print("="*70)
        print(dist_metrics.format_results(dist_results))
        print()

        print("="*90)
        print(f"Extended EPR Evaluation Results ({pedal_feature_name})")
        print("="*90)
        print()
        header = f"{'Feature':<12} {'JS↓':<10} {'IA↑':<10} {'MAE↓':<10} {'RMSE↓':<10} {'Pearson↑':<10}"
        print(header)
        print("-"*90)
        for feature in ['velocity', 'duration', 'ioi', pedal_feature_name]:
            if feature in extended_results:
                r = extended_results[feature]
                print(f"{feature:<12} {r['js']:<10.4f} {r['ia']:<10.4f} {r['mae']:<10.2f} {r['rmse']:<10.2f} {r['pearson']:<10.4f}")
        print("-"*90)
        if 'overall' in extended_results:
            r = extended_results['overall']
            print(f"{'Overall':<12} {r['js']:<10.4f} {r['ia']:<10.4f} {r['mae']:<10.2f} {r['rmse']:<10.2f} {r['pearson']:<10.4f}")
        print("="*90)

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = {
            'evaluation_type': 'score_to_performance_generation',
            'pedal_method': args.pedal_method,
            'num_samples': args.max_samples if args.max_samples else len(test_dataset),
            'num_notes': len(pred_features['velocity']),
            'distribution_metrics': dist_results,
            'fine_grained_metrics': extended_results,
        }

        output_file = output_dir / f"pt_multiworker_results_{args.pedal_method}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to {output_file}")

    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
