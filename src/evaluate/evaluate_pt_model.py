"""
Evaluate Pianist Transformer model using our evaluation framework.
This allows us to evaluate PT with BOTH binary and continuous pedal methods.
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


def load_pt_model(model_path):
    """Load PT model."""
    # Import PT's model classes
    from src.model.pianoformer import PianoT5Gemma, PianoT5GemmaConfig

    print(f"Loading PT model from {model_path}")

    # Load config (use default PT config)
    config = PianoT5GemmaConfig()
    model = PianoT5Gemma(config)

    # Load weights
    if str(model_path).endswith('.safetensors'):
        from safetensors.torch import load_file
        state_dict = load_file(str(model_path))
        model.load_state_dict(state_dict)
    else:
        checkpoint = torch.load(model_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

    model.eval()
    return model


def load_pt_dataset(split='test'):
    """Load PT's ASAP dataset."""
    from src.train.sft import ASAPSFTDataset, build_work_manifest

    # Use PT's default config paths
    metadata_path = PT_PATH / "data/asap-dataset-master/metadata.csv"
    refined_dir = PT_PATH / "data/asap-dataset-master"

    manifest = build_work_manifest(
        metadata_path=str(metadata_path),
        refined_dir=str(refined_dir),
        split=split,
        block_notes=512,
        overlap_ratio=0.5,
        min_notes=64,
    )

    dataset = ASAPSFTDataset(
        manifest,
        split=split,
        shuffle=False,
        seed=42,
    )

    return dataset


def pt_output_to_continuous(pt_output):
    """
    Convert PT's token-based output to continuous features.

    PT outputs 8 tokens per note:
    - velocity_class (127 classes)
    - duration_class (256 classes)
    - ioi_class (256 classes)
    - pedal_0, pedal_1, pedal_2, pedal_3 (binary)

    We need to convert to continuous [0, 1] range.
    """
    # Extract predictions from PT output
    # This requires understanding PT's output format
    # For now, return placeholder
    raise NotImplementedError("Need to implement PT output conversion")


def evaluate_pt_on_asap(model, dataset, device, pedal_method='binary'):
    """
    Evaluate PT model on ASAP test set.

    Note: This is complex because PT uses token-based representation.
    We need to:
    1. Run PT inference to get token predictions
    2. Convert tokens back to continuous values
    3. Extract features for evaluation
    """
    print(f"Evaluating PT model on ASAP test set...")
    print(f"Pedal method: {pedal_method}")
    print(f"Dataset size: {len(dataset)} samples")

    # TODO: Implement PT inference and conversion
    # This requires understanding PT's:
    # - Input format
    # - Output format
    # - Token-to-continuous conversion

    raise NotImplementedError(
        "PT evaluation requires implementing:\n"
        "1. PT inference loop\n"
        "2. Token-to-continuous conversion\n"
        "3. Feature extraction compatible with our metrics"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str,
                       default='/home/kaititech/EPR/third_party/epr/PianistTransformer/models/sft/model.safetensors',
                       help='PT model path')
    parser.add_argument('--output-dir', type=str, default='results/pt_evaluation',
                       help='Output directory')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--pedal-method', type=str, default='binary',
                       choices=['binary', 'continuous'])
    args = parser.parse_args()

    print("="*70)
    print("Pianist Transformer Evaluation with Our Metrics")
    print("="*70)
    print(f"Model: {args.model}")
    print(f"Pedal method: {args.pedal_method}")
    print()

    # Load model
    try:
        model = load_pt_model(args.model)
        print("✅ PT model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load PT model: {e}")
        print("\nThis script requires:")
        print("1. PT's model code to be importable")
        print("2. Understanding PT's model architecture")
        print("3. Compatible with PT's checkpoint format")
        return

    # Load dataset
    try:
        dataset = load_pt_dataset('test')
        print(f"✅ PT dataset loaded: {len(dataset)} samples")
    except Exception as e:
        print(f"❌ Failed to load PT dataset: {e}")
        return

    # Evaluate
    try:
        results = evaluate_pt_on_asap(
            model, dataset, args.device, args.pedal_method
        )

        # Save results
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / f"pt_results_{args.pedal_method}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to {output_file}")

    except NotImplementedError as e:
        print(f"\n❌ {e}")
        print("\n" + "="*70)
        print("NEXT STEPS")
        print("="*70)
        print("\nTo evaluate PT with continuous pedal method, we need to:")
        print("\n1. Study PT's output format:")
        print("   - Read PT's inference code")
        print("   - Understand token prediction format")
        print("   - 8 tokens per note: [vel_class, dur_class, ioi_class, p0, p1, p2, p3]")
        print("\n2. Implement token-to-continuous conversion:")
        print("   - velocity_class (0-126) → velocity [0, 1]")
        print("   - duration_class (0-255) → duration [0, 1]")
        print("   - ioi_class (0-255) → ioi [0, 1]")
        print("   - pedal binary → pedal [0, 1] for continuous evaluation")
        print("\n3. Run inference and extract features")
        print("\nAlternatively:")
        print("- Check if PT provides evaluation scripts")
        print("- Check if PT outputs continuous predictions anywhere")
        print("- Contact PT authors for evaluation protocol")


if __name__ == '__main__':
    main()
