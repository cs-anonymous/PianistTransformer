"""
Evaluate PT on its own ASAP test set using the processed data.
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


def load_asap_test_data(data_path, max_samples=None):
    """Load ASAP test set from PT's processed data."""
    test_samples = []

    with open(data_path) as f:
        for line in f:
            sample = json.loads(line)
            if sample['split'] == 'test':
                test_samples.append(sample)
                if max_samples and len(test_samples) >= max_samples:
                    break

    return test_samples


def evaluate_pt_on_asap(model, config, test_samples, device, pedal_method='binary'):
    """Evaluate PT on ASAP test set."""
    model = model.to(device)
    model.eval()

    all_pred_continuous = []
    all_target_continuous = []
    all_masks = []

    print(f"Evaluating PT on {len(test_samples)} ASAP test samples...")

    for sample in tqdm(test_samples, desc="PT ASAP evaluation"):
        try:
            score_tokens = sample['x']
            label_tokens = sample['label']

            # Convert to tensor
            input_ids = torch.tensor([score_tokens], dtype=torch.long).to(device)

            # Generate
            with torch.no_grad():
                generated_ids = model.generate(
                    input_ids=input_ids,
                    max_new_tokens=len(score_tokens),
                    do_sample=False,
                    pad_token_id=config.pad_token_id,
                    eos_token_id=config.eos_token_id,
                )

            # Convert to continuous
            pred_tokens = generated_ids[0].cpu().numpy().tolist()
            pred_cont = pt_tokens_to_continuous_features(pred_tokens, config)
            target_cont = pt_tokens_to_continuous_features(label_tokens, config)

            # Align lengths
            min_len = min(len(pred_cont), len(target_cont))
            pred_cont = pred_cont[:min_len]
            target_cont = target_cont[:min_len]

            mask = np.ones(min_len, dtype=bool)

            all_pred_continuous.append(pred_cont)
            all_target_continuous.append(target_cont)
            all_masks.append(mask)

        except Exception as e:
            print(f"Warning: Sample failed: {e}")
            continue

    if not all_pred_continuous:
        raise ValueError("No valid predictions")

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
    parser.add_argument('--data-path', type=str,
                       default='/home/kaititech/EPR/third_party/epr/PianistTransformer/data/processed/sft/sft.jsonl')
    parser.add_argument('--output-dir', type=str, default='results/pt_asap_evaluation')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--pedal-method', type=str, default='binary', choices=['binary', 'continuous'])
    parser.add_argument('--max-samples', type=int, default=None)
    args = parser.parse_args()

    print("="*70)
    print("PT Evaluation on ASAP Test Set")
    print("="*70)
    print(f"Pedal method: {args.pedal_method}")
    print()

    pt_model, pt_config = load_pt_model_and_config()
    print("✅ PT model loaded")

    print("\nLoading ASAP test data...")
    test_samples = load_asap_test_data(args.data_path, args.max_samples)
    print(f"Loaded {len(test_samples)} test samples")

    try:
        pred_features, target_features = evaluate_pt_on_asap(
            pt_model, pt_config, test_samples, args.device, args.pedal_method
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
        print(f"PT ASAP TEST RESULTS ({args.pedal_method} pedal)")
        print("="*70)
        print(dist_metrics.format_results(dist_results))
        print()

        print("="*90)
        print(f"Extended Results ({pedal_feature_name})")
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
            'evaluation_type': 'asap_test_set',
            'pedal_method': args.pedal_method,
            'num_samples': len(test_samples),
            'num_notes': len(pred_features['velocity']),
            'distribution_metrics': dist_results,
            'fine_grained_metrics': extended_results,
        }

        output_file = output_dir / f"pt_asap_results_{args.pedal_method}.json"
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
