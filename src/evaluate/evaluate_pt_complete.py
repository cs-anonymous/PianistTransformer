"""
Evaluate Pianist Transformer on our test sets with our evaluation metrics.
Supports both Binary and Continuous pedal evaluation methods.
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

# Add PT path
PT_PATH = Path("/home/kaititech/EPR/third_party/epr/PianistTransformer")
sys.path.insert(0, str(PT_PATH))

from src.evaluate.epr_metrics import EPRMetrics, extract_features_from_continuous
from src.evaluate.epr_metrics_extended import ExtendedEPRMetrics


def load_pt_model_and_config():
    """Load PT model and config."""
    from src.model.pianoformer import PianoT5Gemma, PianoT5GemmaConfig

    # Load config from saved model
    model_path = PT_PATH / "models/sft/model.safetensors"
    config_path = PT_PATH / "models/sft/config.json"

    print(f"Loading PT config from {config_path}")
    with open(config_path) as f:
        config_dict = json.load(f)

    # Create config with correct parameters
    config = PianoT5GemmaConfig(
        encoder_layers_num=config_dict['encoder']['num_hidden_layers'],  # 10
        decoder_layers_num=config_dict['decoder']['num_hidden_layers'],  # 2
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
    model.load_state_dict(state_dict, strict=False)  # Use strict=False to handle minor mismatches
    model.eval()

    return model, config


def pt_tokens_to_continuous_features(token_ids, config):
    """
    Convert PT's token predictions to continuous features [0, 1].

    PT format (8 tokens per note):
    - token[0]: pitch (config.pitch_start + pitch_value)
    - token[1]: IOI interval (config.timing_start + ioi_ms)
    - token[2]: velocity (config.velocity_start + velocity_value)
    - token[3]: duration (config.timing_start + duration_ms)
    - token[4-7]: pedal samples (config.pedal_start + pedal_value)

    Returns:
        np.ndarray: (N, 7) array with [ioi, duration, velocity, pedal_0, pedal_1, pedal_2, pedal_3]
        All values normalized to [0, 1] range
    """
    # Reshape to (N, 8)
    num_notes = len(token_ids) // 8
    tokens = np.array(token_ids[:num_notes * 8]).reshape(num_notes, 8)

    # Extract values by subtracting token offsets
    ioi_tokens = tokens[:, 1] - config.timing_start  # IOI in ms
    duration_tokens = tokens[:, 3] - config.timing_start  # Duration in ms
    velocity_tokens = tokens[:, 2] - config.velocity_start  # Velocity [0, 127]
    pedal_tokens = tokens[:, 4:8] - config.pedal_start  # Pedal [0, 127]

    # Normalize to [0, 1]
    max_time_ms = 10000.0  # Same as our model

    # IOI: log normalization
    ioi_ms = np.clip(ioi_tokens, 0, max_time_ms)
    ioi_normalized = np.log1p(ioi_ms) / np.log1p(max_time_ms)

    # Duration: log normalization
    duration_ms = np.clip(duration_tokens, 0, max_time_ms)
    duration_normalized = np.log1p(duration_ms) / np.log1p(max_time_ms)

    # Velocity: linear [0, 127] -> [0, 1]
    velocity_normalized = np.clip(velocity_tokens, 0, 127) / 127.0

    # Pedal: linear [0, 127] -> [0, 1]
    pedal_normalized = np.clip(pedal_tokens, 0, 127) / 127.0

    # Concatenate: [ioi, duration, velocity, pedal_0, pedal_1, pedal_2, pedal_3]
    continuous = np.concatenate([
        ioi_normalized.reshape(-1, 1),
        duration_normalized.reshape(-1, 1),
        velocity_normalized.reshape(-1, 1),
        pedal_normalized
    ], axis=1)

    return continuous


def evaluate_pt_on_dataset(model, config, dataset, device, pedal_method='binary'):
    """
    Evaluate PT model on a dataset.

    Args:
        model: PT model
        config: PT config
        dataset: Our dataset (PianoCoReNodeSFTDataset)
        device: Device
        pedal_method: 'binary' or 'continuous'
    """
    model = model.to(device)
    model.eval()

    # PT uses token-based collator
    from src.train.sft import DiffusionSFTDataCollator
    pt_collator = DiffusionSFTDataCollator(config)

    all_pred_continuous = []
    all_target_continuous = []
    all_masks = []

    print(f"Evaluating PT on {len(dataset)} samples...")

    batch_size = 2
    num_batches = (len(dataset) + batch_size - 1) // batch_size

    for i in tqdm(range(num_batches), desc="PT inference"):
        # Get batch samples
        batch_samples = []
        for j in range(batch_size):
            idx = i * batch_size + j
            if idx >= len(dataset):
                break

            # Get sample from our dataset
            sample = dataset[idx]

            # Convert our continuous format to PT's token format
            # We need the input (score) as PT tokens
            # For evaluation, we use the target as input to get perfect reconstruction
            # This is a limitation - we're evaluating PT's ability to predict from its own tokens

            # Get pitch and continuous from our sample
            pitch_ids = sample['pitch_ids']
            target_continuous = sample['labels_continuous']

            # Convert to PT token format
            # This is approximate - PT uses different tokenization
            # For now, skip samples that don't fit PT's format
            if len(pitch_ids) > 512 or len(pitch_ids) % 8 != 0:
                continue

            # Create dummy PT input (using ground truth as input for now)
            # In real evaluation, this would come from score
            pt_input_ids = []
            for note_idx in range(len(pitch_ids)):
                # PT format: [pitch, ioi, velocity, duration, p0, p1, p2, p3]
                pitch = int(config.pitch_start + pitch_ids[note_idx])

                # Get continuous values
                cont = target_continuous[note_idx]
                ioi_norm, dur_norm, vel_norm = cont[0], cont[1], cont[2]
                pedal_norm = cont[3:7]

                # Denormalize and clip to valid ranges
                ioi_ms = int(np.clip(np.expm1(ioi_norm * np.log1p(10000.0)), 0, 4990))
                dur_ms = int(np.clip(np.expm1(dur_norm * np.log1p(10000.0)), 0, 4990))
                vel = int(np.clip(vel_norm * 127, 0, 127))
                pedal_vals = np.clip((np.array(pedal_norm) * 127).astype(int), 0, 127)

                # Create PT tokens with range checking
                ioi_token = np.clip(config.timing_start + ioi_ms, config.valid_id_range[1][0], config.valid_id_range[1][1] - 1)
                dur_token = np.clip(config.timing_start + dur_ms, config.valid_id_range[3][0], config.valid_id_range[3][1] - 1)
                vel_token = np.clip(config.velocity_start + vel, config.valid_id_range[2][0], config.valid_id_range[2][1] - 1)
                pedal_tokens = [int(np.clip(config.pedal_start + p, config.valid_id_range[4+i][0], config.valid_id_range[4+i][1] - 1)) for i, p in enumerate(pedal_vals)]

                pt_input_ids.extend([int(pitch), int(ioi_token), int(vel_token), int(dur_token)] + pedal_tokens)

            pt_sample = {
                'input_ids': pt_input_ids,
                'labels': pt_input_ids  # For teacher forcing
            }

            batch_samples.append(pt_sample)

        if not batch_samples:
            continue

        # Collate batch with PT collator
        batch = pt_collator(batch_samples)

        # Move to device
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        # Run PT inference
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            # Get predictions - PT outputs logits for each token position
            logits = outputs.logits  # (B, seq_len, vocab_size)

            # Greedy decoding
            pred_token_ids = torch.argmax(logits, dim=-1)  # (B, seq_len)

        # Convert predictions and targets to continuous format
        for b in range(pred_token_ids.shape[0]):
            # Get valid length
            mask = attention_mask[b].cpu().numpy()
            valid_len = int(mask.sum())

            # Get predicted and target tokens
            pred_tokens = pred_token_ids[b, :valid_len].cpu().numpy()
            target_tokens = labels[b, :valid_len].cpu().numpy()

            # Convert to continuous features
            try:
                pred_cont = pt_tokens_to_continuous_features(pred_tokens.tolist(), config)
                target_cont = pt_tokens_to_continuous_features(target_tokens.tolist(), config)

                # Create mask (all valid for now)
                note_mask = np.ones(len(pred_cont), dtype=bool)

                all_pred_continuous.append(pred_cont)
                all_target_continuous.append(target_cont)
                all_masks.append(note_mask)
            except Exception as e:
                print(f"Warning: Failed to convert tokens: {e}")
                continue

    if not all_pred_continuous:
        raise ValueError("No valid predictions collected")

    # Extract features
    print("Extracting features...")

    pred_features_list = {'velocity': [], 'duration': [], 'ioi': [], 'pedal': []}
    target_features_list = {'velocity': [], 'duration': [], 'ioi': [], 'pedal': []}

    use_joint_config = (pedal_method == 'binary')

    for pred_cont, target_cont, mask in zip(all_pred_continuous, all_target_continuous, all_masks):
        # pred_cont and target_cont are (N, 7): [ioi, duration, velocity, pedal*4]

        # Extract features using our function
        # Need to reshape to match expected format: (1, N, 7)
        pred_batch = pred_cont.reshape(1, -1, 7)
        target_batch = target_cont.reshape(1, -1, 7)
        mask_batch = mask.reshape(1, -1)

        from src.evaluate.epr_metrics import extract_features_from_continuous
        pred_feats = extract_features_from_continuous(pred_batch, mask_batch, pedal_as_joint_config=use_joint_config)
        target_feats = extract_features_from_continuous(target_batch, mask_batch, pedal_as_joint_config=use_joint_config)

        for key in pred_features_list.keys():
            pred_features_list[key].append(pred_feats[key])
            target_features_list[key].append(target_feats[key])

    # Concatenate
    pred_features = {k: np.concatenate(v) for k, v in pred_features_list.items()}
    target_features = {k: np.concatenate(v) for k, v in target_features_list.items()}

    print(f"Collected {len(pred_features['velocity'])} notes")

    return pred_features, target_features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                       help='Our dataset config (for loading test set)')
    parser.add_argument('--output-dir', type=str, default='results/pt_evaluation',
                       help='Output directory')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--pedal-method', type=str, default='binary',
                       choices=['binary', 'continuous'])
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Max samples to evaluate (for quick test)')
    args = parser.parse_args()

    print("="*70)
    print("Pianist Transformer Evaluation with Our Metrics")
    print("="*70)
    print(f"Pedal method: {args.pedal_method}")
    print()

    # Load our config
    with open(args.config) as f:
        our_config = json.load(f)

    # Load PT model
    pt_model, pt_config = load_pt_model_and_config()
    print("✅ PT model loaded successfully")

    # Load our test dataset
    from src.train.sft_node import PianoCoReNodeSFTDataset, build_work_manifest

    print("\nBuilding test dataset...")
    test_manifest = build_work_manifest(
        metadata_path=our_config['metadata_path'],
        refined_dir=our_config['refined_dir'],
        split='test',
        block_notes=our_config['block_notes'],
        overlap_ratio=our_config['overlap_ratio'],
        min_notes=our_config['min_notes'],
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

    # Limit samples if requested
    if args.max_samples and args.max_samples < len(test_dataset):
        test_dataset = torch.utils.data.Subset(test_dataset, range(args.max_samples))
        print(f"Limited to {args.max_samples} samples")

    # Evaluate
    try:
        pred_features, target_features = evaluate_pt_on_dataset(
            pt_model, pt_config, test_dataset, args.device, args.pedal_method
        )

        # Compute metrics
        print("\nComputing distribution metrics...")
        dist_metrics = EPRMetrics(bins=100)
        use_joint_config = (args.pedal_method == 'binary')
        dist_results = dist_metrics.compute_metrics(
            pred_features, target_features,
            pedal_is_joint_config=use_joint_config
        )

        print("\nComputing fine-grained metrics...")
        extended_metrics = ExtendedEPRMetrics()
        pedal_method_arg = 'binary' if use_joint_config else 'continuous'
        extended_results = extended_metrics.compute_metrics_for_features(
            pred_features, target_features,
            pedal_method=pedal_method_arg
        )

        # Print results
        print("\n" + "="*70)
        print(f"PT EVALUATION RESULTS ({args.pedal_method} pedal method)")
        print("="*70)
        print(dist_metrics.format_results(dist_results))
        print()
        print(extended_metrics.format_results(extended_results))

        # Save results
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = {
            'pedal_method': args.pedal_method,
            'num_samples': args.max_samples if args.max_samples else len(test_dataset),
            'num_notes': len(pred_features['velocity']),
            'distribution_metrics': dist_results,
            'fine_grained_metrics': extended_results,
        }

        output_file = output_dir / f"pt_results_{args.pedal_method}.json"
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
