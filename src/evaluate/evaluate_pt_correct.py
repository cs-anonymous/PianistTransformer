"""
CORRECT PT Evaluation: Score -> Performance generation task
Input: Score tokens (pitch only)
Output: Generate performance tokens (pitch + timing + velocity + pedal)
"""

import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

PT_PATH = Path("/home/kaititech/EPR/third_party/epr/PianistTransformer")
sys.path.insert(0, str(PT_PATH))

from src.evaluate.epr_metrics import EPRMetrics, extract_features_from_continuous
from src.evaluate.epr_metrics_extended import ExtendedEPRMetrics


def load_pt_model_and_config():
    """Load PT model and config."""
    from src.model.pianoformer import PianoT5Gemma, PianoT5GemmaConfig

    model_path = PT_PATH / "models/sft/model.safetensors"
    config_path = PT_PATH / "models/sft/config.json"

    print(f"Loading PT config from {config_path}")
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

    print(f"Loading PT model from {model_path}")
    from safetensors.torch import load_file
    state_dict = load_file(str(model_path))
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    return model, config


def create_score_input(pitch_ids, config):
    """
    Create score input: only pitch tokens.
    PT will generate velocity, timing, pedal.
    """
    score_tokens = []
    for pitch in pitch_ids:
        # Score only has pitch
        pitch_token = int(config.pitch_start + pitch)
        score_tokens.append(pitch_token)

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

    # Normalize
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


def evaluate_pt_generation(model, config, dataset, device, pedal_method='binary', max_samples=None, num_workers=40):
    """
    CORRECT evaluation: Score -> Performance generation.
    Uses multi-GPU and multi-threading for faster evaluation.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import torch.multiprocessing as mp

    # Replicate model to all GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Using {num_gpus} GPUs with {num_workers} worker threads")

    all_pred_continuous = []
    all_target_continuous = []
    all_masks = []

    num_samples = len(dataset) if max_samples is None else min(max_samples, len(dataset))
    print(f"Evaluating PT on {num_samples} samples (Score -> Performance generation)...")

    def process_sample(idx):
        """Process one sample on assigned GPU."""
        gpu_id = idx % num_gpus
        local_device = torch.device(f'cuda:{gpu_id}')

        # Clone model to this GPU (done once per thread)
        if not hasattr(process_sample, f'model_{gpu_id}'):
            local_model = model.__class__(config).to(local_device)
            local_model.load_state_dict(model.state_dict())
            local_model.eval()
            setattr(process_sample, f'model_{gpu_id}', local_model)
        else:
            local_model = getattr(process_sample, f'model_{gpu_id}')

        sample = dataset[idx]
        pitch_ids = sample['pitch_ids']
        target_continuous = sample['labels_continuous']

        # Skip if too long
        if len(pitch_ids) > 512:
            return None

        # Create score input (only pitch)
        score_input = create_score_input(pitch_ids, config)

        # Convert to tensor
        input_ids = torch.tensor([score_input], dtype=torch.long).to(local_device)

        # Generate performance
        with torch.no_grad():
            generated_ids = local_model.generate(
                input_ids=input_ids,
                max_new_tokens=len(score_input) * 8,
                do_sample=False,
                pad_token_id=config.pad_token_id,
                eos_token_id=config.eos_token_id,
            )

        # Convert generated tokens to continuous
        try:
            pred_tokens = generated_ids[0].cpu().numpy().tolist()
            pred_cont = pt_tokens_to_continuous_features(pred_tokens, config)

            # Target continuous
            target_cont = np.array(target_continuous)

            # Align lengths (take minimum)
            min_len = min(len(pred_cont), len(target_cont))
            pred_cont = pred_cont[:min_len]
            target_cont = target_cont[:min_len]

            mask = np.ones(min_len, dtype=bool)

            return (pred_cont, target_cont, mask)

        except Exception as e:
            return None

    # Multi-threaded evaluation
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_sample, i): i for i in range(num_samples)}

        for future in tqdm(as_completed(futures), total=num_samples, desc="PT generation"):
            result = future.result()
            if result is not None:
                pred_cont, target_cont, mask = result
                all_pred_continuous.append(pred_cont)
                all_target_continuous.append(target_cont)
                all_masks.append(mask)

    if not all_pred_continuous:
        raise ValueError("No valid predictions collected")

    print(f"\nSuccessfully processed {len(all_pred_continuous)} samples")

    # Extract features
    print("Extracting features...")

    pred_features_list = {'velocity': [], 'duration': [], 'ioi': [], 'pedal': []}
    target_features_list = {'velocity': [], 'duration': [], 'ioi': [], 'pedal': []}

    use_joint_config = (pedal_method == 'binary')

    for pred_cont, target_cont, mask in zip(all_pred_continuous, all_target_continuous, all_masks):
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
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--pedal-method', type=str, default='binary',
                       choices=['binary', 'continuous'])
    parser.add_argument('--max-samples', type=int, default=None)
    args = parser.parse_args()

    print("="*70)
    print("PT CORRECT Evaluation: Score -> Performance Generation")
    print("="*70)
    print(f"Pedal method: {args.pedal_method}")
    print()

    with open(args.config) as f:
        our_config = json.load(f)

    pt_model, pt_config = load_pt_model_and_config()
    print("✅ PT model loaded successfully")

    from src.train.sft_node import PianoCoReNodeSFTDataset, build_work_manifest

    print("\nBuilding test dataset...")

    # Use default paths if not in config
    metadata_path = our_config.get('metadata_path', 'data/pianocore/metadata.csv')
    refined_dir = our_config.get('refined_dir', 'data/pianocore/PianoCoRe/refined')
    block_notes = our_config.get('block_notes', 512)
    overlap_ratio = our_config.get('overlap_ratio', 0.5)
    min_notes = our_config.get('min_notes', 64)

    test_manifest = build_work_manifest(
        metadata_path=metadata_path,
        refined_dir=refined_dir,
        split='test',
        block_notes=block_notes,
        overlap_ratio=overlap_ratio,
        min_notes=min_notes,
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

    if args.max_samples and args.max_samples < len(test_dataset):
        test_dataset = torch.utils.data.Subset(test_dataset, range(args.max_samples))
        print(f"Limited to {args.max_samples} samples")

    try:
        pred_features, target_features = evaluate_pt_generation(
            pt_model, pt_config, test_dataset, args.device, args.pedal_method, args.max_samples, num_workers=40
        )

        print("\nComputing distribution metrics...")
        dist_metrics = EPRMetrics(bins=100)
        use_joint_config = (args.pedal_method == 'binary')
        dist_results = dist_metrics.compute_metrics(
            pred_features, target_features,
            pedal_is_joint_config=use_joint_config
        )

        print("\nComputing fine-grained metrics...")
        extended_metrics = ExtendedEPRMetrics()

        # For reporting, rename pedal based on method
        pedal_feature_name = 'BPedal' if args.pedal_method == 'binary' else 'CPedal'

        pedal_method_arg = 'binary' if use_joint_config else 'continuous'
        extended_results = extended_metrics.compute_metrics_for_features(
            pred_features, target_features,
            pedal_method=pedal_method_arg
        )

        # Rename pedal key for display
        if 'pedal' in extended_results:
            extended_results[pedal_feature_name] = extended_results.pop('pedal')

        # Recompute overall with 5 metrics (velocity, duration, ioi, pedal, and the overall average)
        # Overall = average of velocity, duration, ioi, and pedal (whichever pedal method is used)
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

        # Custom format for extended results with renamed pedal
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

        output_file = output_dir / f"pt_correct_results_{args.pedal_method}.json"
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
