"""
Evaluate PT on ASAP test set by processing metadata directly.
Uses the same test split logic as PT (10% of unique scores, seed=42).
"""

import argparse
import json
import sys
import os
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from miditoolkit import MidiFile

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

PT_PATH = Path("/home/kaititech/EPR/third_party/epr/PianistTransformer")
sys.path.insert(0, str(PT_PATH))

from src.evaluate.epr_metrics import EPRMetrics, extract_features_from_continuous
from src.evaluate.epr_metrics_extended import ExtendedEPRMetrics
from src.utils.midi import align_score_and_performance


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


def get_test_scores(metadata_path):
    """Get test score set using PT's logic: 10% of unique scores, seed=42."""
    metadata = pd.read_csv(metadata_path)

    scores_set = set()
    for i in range(len(metadata)):
        scores_set.add(metadata["midi_score"][i])

    random.seed(42)
    scores_set = sorted(list(scores_set))
    random.shuffle(scores_set)

    test_set = scores_set[:int(0.1 * len(scores_set))]

    print(f"Total unique scores: {len(scores_set)}")
    print(f"Test scores: {len(test_set)}")

    return set(test_set)


def evaluate_pt_on_asap(model, config, metadata_path, asap_dir, device, pedal_method='binary', max_samples=None):
    """Evaluate PT on ASAP test set."""
    model = model.to(device)
    model.eval()

    metadata = pd.read_csv(metadata_path)
    test_scores = get_test_scores(metadata_path)

    # Filter test samples
    test_indices = [i for i in range(len(metadata)) if metadata["midi_score"][i] in test_scores]

    if max_samples:
        test_indices = test_indices[:max_samples]

    print(f"Total test samples: {len(test_indices)}")

    all_pred_continuous = []
    all_target_continuous = []
    all_masks = []

    processed = 0
    failed = 0

    for idx in tqdm(test_indices, desc="PT ASAP evaluation"):
        try:
            score_path = os.path.join(asap_dir, metadata["midi_score"][idx])
            perf_path = os.path.join(asap_dir, metadata["midi_performance"][idx])

            score_midi = MidiFile(score_path)
            perf_midi = MidiFile(perf_path)

            # Align and convert to PT tokens
            xs, labels = align_score_and_performance(config, score_midi, perf_midi)

            # Evaluate each segment
            for seg_idx in range(len(xs)):
                score_tokens = xs[seg_idx]
                label_tokens = labels[seg_idx]

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

                processed += 1

        except Exception as e:
            failed += 1
            continue

    if not all_pred_continuous:
        raise ValueError("No valid predictions")

    print(f"\nSuccessfully processed {processed} segments from {len(test_indices)} samples ({failed} failed)")

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
    parser.add_argument('--metadata-path', type=str, default='data/midis/asap-dataset-master/metadata.csv')
    parser.add_argument('--asap-dir', type=str, default='data/midis/asap-dataset-master/')
    parser.add_argument('--output-dir', type=str, default='results/pt_asap_evaluation')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--pedal-method', type=str, default='binary', choices=['binary', 'continuous'])
    parser.add_argument('--max-samples', type=int, default=None)
    args = parser.parse_args()

    print("="*70)
    print("PT Evaluation on ASAP Test Set (from metadata)")
    print("="*70)
    print(f"Pedal method: {args.pedal_method}")
    print()

    pt_model, pt_config = load_pt_model_and_config()
    print("✅ PT model loaded")

    try:
        pred_features, target_features = evaluate_pt_on_asap(
            pt_model, pt_config, args.metadata_path, args.asap_dir, args.device, args.pedal_method, args.max_samples
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
            'evaluation_type': 'asap_test_set_from_metadata',
            'pedal_method': args.pedal_method,
            'num_notes': len(pred_features['velocity']),
            'distribution_metrics': dist_results,
            'fine_grained_metrics': extended_results,
        }

        output_file = output_dir / f"pt_asap_direct_results_{args.pedal_method}.json"
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
