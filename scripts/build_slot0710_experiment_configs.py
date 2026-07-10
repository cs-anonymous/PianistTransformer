#!/usr/bin/env python3
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"
METADATA = ROOT.parent / "PianoCoRe" / "metadata.csv"
REFINED = ROOT.parent / "PianoCoRe" / "processed"

BASE_NOMUS = CONFIG_DIR / "inr0624_note_rawlog_slot8_nomus_clean_20260709.json"
BASE_MUSICAL = CONFIG_DIR / "inr0624_note_rawlog_slot12_m51_clean_20260709.json"

EXPERIMENTS = {
    "slot0710_pt5_abs.json": {
        "base": BASE_NOMUS,
        "slot_version": "slot5",
        "target": "absolute_log",
        "musical_feature_mode": "none",
        "disable_musical_features": True,
        "run_name": "PT5-Abs",
    },
    "slot0710_pt5_dev.json": {
        "base": BASE_NOMUS,
        "slot_version": "slot5",
        "target": "raw_log_deviation",
        "musical_feature_mode": "none",
        "disable_musical_features": True,
        "run_name": "PT5-Dev",
    },
    "slot0710_inr8_abs.json": {
        "base": BASE_NOMUS,
        "slot_version": "slot8",
        "target": "absolute_log",
        "musical_feature_mode": "none",
        "disable_musical_features": True,
        "run_name": "INR8-Abs",
    },
    "slot0710_inr8_dev.json": {
        "base": BASE_NOMUS,
        "slot_version": "slot8",
        "target": "raw_log_deviation",
        "target_dim": 9,
        "input_dim": 16,
        "raw_timing_head_type": "regression",
        "musical_feature_mode": "none",
        "disable_musical_features": True,
        "run_name": "INR8-Dev",
    },
    "slot0710_pt9_abs.json": {
        "base": BASE_MUSICAL,
        "slot_version": "slot9",
        "target": "absolute_log",
        "musical_feature_mode": "musical51",
        "disable_musical_features": False,
        "run_name": "PT9-Abs",
    },
    "slot0710_inr12_dev.json": {
        "base": BASE_MUSICAL,
        "slot_version": "slot12",
        "target": "raw_log_deviation",
        "musical_feature_mode": "musical51",
        "disable_musical_features": False,
        "run_name": "INR12-Dev",
    },
}


def build_config(spec):
    config = json.loads(spec["base"].read_text(encoding="utf-8"))
    for key in (
        "eval_every_steps",
        "eval_every_epochs",
        "save_every_steps",
        "eval_steps",
        "save_steps",
        "evaluation_strategy",
        "eval_strategy",
        "save_strategy",
    ):
        config.pop(key, None)
    target_dim = int(spec.get("target_dim", 7))
    config.update(
        {
            "metadata_path": str(METADATA),
            "refined_dir": str(REFINED),
            "note_embedding_mode": "slot_attribute",
            "slot_version": spec["slot_version"],
            "slot_dim": 128,
            "epr_timing_target": spec["target"],
            "continuous_dim": target_dim,
            "output_continuous_dim": target_dim,
            "musical_feature_mode": spec["musical_feature_mode"],
            "disable_musical_features": spec["disable_musical_features"],
            "prior_property_dropout_prob": 0.5,
            "prior_token_keep_prob": 1.0,
            "loss_weights": {
                "ioi": 1.0,
                "duration": 1.0,
                "velocity": 1.0,
                "pedal": 1.0,
            },
            "per_device_train_batch_size": 32,
            "per_device_eval_batch_size": 32,
            "gradient_accumulation_steps": 2,
            "global_batch_size": 64,
            "run_name": spec["run_name"],
            "resume_path": None,
            "resume_trainer_state": False,
            "overwrite_output_dir": True,
        }
    )
    if spec.get("input_dim") is not None:
        config["input_continuous_dim"] = int(spec["input_dim"])
    config.pop("raw_timing_loss_lambda", None)
    config.pop("raw_timing_head_type", None)
    if spec["target"] == "raw_log_deviation":
        config["raw_timing_loss_lambda"] = 0.25
    if spec.get("raw_timing_head_type"):
        config["raw_timing_head_type"] = spec["raw_timing_head_type"]
    return config


def main():
    for filename, spec in EXPERIMENTS.items():
        path = CONFIG_DIR / filename
        path.write_text(
            json.dumps(build_config(spec), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
