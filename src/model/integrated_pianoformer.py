import math
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.cache_utils import EncoderDecoderCache
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput, Seq2SeqModelOutput
from transformers.models.t5gemma.modeling_t5gemma import (
    GenerationMixin,
    T5GemmaDecoder,
    T5GemmaEncoderLayer,
    T5GemmaPreTrainedModel,
    T5GemmaRMSNorm,
    T5GemmaRotaryEmbedding,
    T5GemmaSelfAttention,
    bidirectional_mask_function,
    create_causal_mask,
    create_sliding_window_causal_mask,
    make_default_2d_attention_mask,
    sliding_window_bidirectional_mask_function,
)
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, is_torchdynamo_compiling
from transformers.utils import logging

from src.model.pianoformer import PianoT5Gemma, PianoT5GemmaConfig


logger = logging.get_logger(__name__)


ALN_DISTRIBUTIONS = {"lan"}
ACN_DISTRIBUTIONS = {"can"}
IACN_DISTRIBUTIONS = {"ican"}
ILN_DISTRIBUTIONS = {"iln"}
SN_DISTRIBUTIONS = {"sn", "skew_normal"}
DLM_DISTRIBUTIONS = {"dlm", "discretized_logistic_mixture"}
DINR_DISTRIBUTIONS = {"dinr", "dinr_categorical", "metric_categorical"}
TANH_T_DISTRIBUTIONS = {"tanh_student_t", "bounded_tanh_student_t"}
BOUNDED_SN_DISTRIBUTIONS = {"bounded_skew_normal", "bounded_sn"}
DISCRETE_LN_DISTRIBUTIONS = {"discrete_logistic_normal", "discretized_logistic_normal"}
DISCRETE_BETA_DISTRIBUTIONS = {"discrete_beta", "discretized_beta"}
TRUNCATED_LOGISTIC_DISTRIBUTIONS = {"truncated_logistic", "discrete_truncated_logistic"}
DISCRETE_BOUNDED_DISTRIBUTIONS = {
    *DISCRETE_LN_DISTRIBUTIONS,
    *DISCRETE_BETA_DISTRIBUTIONS,
    *TRUNCATED_LOGISTIC_DISTRIBUTIONS,
}


def _is_scalar_distribution(distribution):
    return distribution in {
        "logistic_normal",
        "mixture_logistic_normal",
        "mixture_beta",
        *DISCRETE_BOUNDED_DISTRIBUTIONS,
        *TANH_T_DISTRIBUTIONS,
        *BOUNDED_SN_DISTRIBUTIONS,
        *SN_DISTRIBUTIONS,
        *ALN_DISTRIBUTIONS,
        *ACN_DISTRIBUTIONS,
        *IACN_DISTRIBUTIONS,
        *ILN_DISTRIBUTIONS,
    }


def _scalar_distribution_dim(distribution):
    if distribution in {*ACN_DISTRIBUTIONS, "logistic_normal", "mixture_logistic_normal", "mixture_beta", *DISCRETE_BOUNDED_DISTRIBUTIONS, *SN_DISTRIBUTIONS, *TANH_T_DISTRIBUTIONS, *BOUNDED_SN_DISTRIBUTIONS}:
        return 3
    if distribution in IACN_DISTRIBUTIONS:
        return 5
    if distribution in ILN_DISTRIBUTIONS:
        return 4
    if distribution in ALN_DISTRIBUTIONS:
        return 4
    return 1


def _scalar_distribution_components(config, distribution):
    if distribution in {*SN_DISTRIBUTIONS, *TANH_T_DISTRIBUTIONS, *BOUNDED_SN_DISTRIBUTIONS, *ALN_DISTRIBUTIONS, *ACN_DISTRIBUTIONS, *IACN_DISTRIBUTIONS, *ILN_DISTRIBUTIONS, "logistic_normal"}:
        return 1
    return int(getattr(config, "epr_mixture_components", 1))


def resolve_timing_control_mode(timing_control_mode="dinr_floor_log", use_timing_scale_bit=False):
    mode = "dinr_floor_log" if timing_control_mode is None else str(timing_control_mode).lower()
    if mode != "dinr_floor_log":
        raise ValueError("Only timing_control_mode=dinr_floor_log is supported")
    return "dinr_floor_log"


def timing_control_feature_dim(timing_control_mode="dinr_floor_log", use_timing_scale_bit=False):
    resolve_timing_control_mode(
        timing_control_mode=timing_control_mode,
        use_timing_scale_bit=use_timing_scale_bit,
    )
    return 3


def musical_feature_dim(musical_feature_mode="musical4slot"):
    mode = str(musical_feature_mode).lower()
    if mode in {"none", "nomus", "no_musical", "disabled"}:
        return 0
    if mode in {
        "musical4slot",
        "musical4slot_full",
        "musical4slot_idx145",
        "musical4slot_no_onset",
        "musical4slot_no_duration",
        "musical4slot_no_annotation",
        "musical4slot_no_length",
        "musical9",
        "asap4slot",
        "compact4slot",
    }:
        return 9
    raise ValueError(
        f"Unsupported musical_feature_mode={musical_feature_mode}; use musical4slot for ASAP_processed 9D features"
    )


def normalize_slot_version(slot_version=None):
    if slot_version is None:
        return "slot8"
    value = str(slot_version).lower()
    if value in {"slot5", "5slot", "pt5", "a5", "a5_pt_absolute_nomus"}:
        return "slot5"
    if value in {"slot6", "6slot", "pt6", "a6", "a6_pt_musical"}:
        return "slot6"
    if value in {"slot7", "7slot", "pt7", "a7", "a7_pt_onset_annotation"}:
        return "slot7"
    if value in {"slot8", "8slot", "8slot_rawlog_nomus_0709", "inr8", "b8", "b8_inr_logdev_nomus"}:
        return "slot8"
    if value in {"slot9", "9slot", "pt9", "a9", "a9_pt_absolute_musical"}:
        return "slot9"
    if value in {"slot12", "12slot", "12slot_rawlog_musical_0709", "inr12", "b12", "b12_inr_logdev_musical"}:
        return "slot12"
    raise ValueError(f"Unsupported slot_version={slot_version}")


def slot_version_num_slots(slot_version):
    value = normalize_slot_version(slot_version)
    return {"slot5": 5, "slot6": 6, "slot7": 7, "slot8": 8, "slot9": 9, "slot12": 12}[value]


def slot_version_is_pt(slot_version):
    return normalize_slot_version(slot_version) in {"slot5", "slot6", "slot7", "slot9"}


def normalize_pedal_representation(pedal_representation="binary_4"):
    value = str(pedal_representation or "binary_4").lower()
    aliases = {"binary4": "binary_4", "pedal4_binary": "binary_4"}
    value = aliases.get(value, value)
    if value != "binary_4":
        raise ValueError("Only pedal_representation=binary_4 is supported")
    return value


def pedal_representation_dim(pedal_representation="binary_4"):
    normalize_pedal_representation(pedal_representation)
    return 4


def score_note_input_schema(config_or_value=None):
    if isinstance(config_or_value, dict):
        mode = str(config_or_value.get("score_note_input_schema", "integrated")).lower()
    elif hasattr(config_or_value, "score_note_input_schema"):
        mode = str(getattr(config_or_value, "score_note_input_schema", "integrated")).lower()
    elif config_or_value is None:
        mode = "integrated"
    else:
        mode = str(config_or_value).lower()
    if mode != "integrated":
        raise ValueError(f"Only integrated INR score-note schema is supported, got {mode}")
    return mode


def decoder_note_input_schema(config_or_value=None):
    if isinstance(config_or_value, dict):
        mode = str(config_or_value.get("decoder_note_input_schema", "integrated")).lower()
    elif hasattr(config_or_value, "decoder_note_input_schema"):
        mode = str(getattr(config_or_value, "decoder_note_input_schema", "integrated")).lower()
    elif config_or_value is None:
        mode = "integrated"
    else:
        mode = str(config_or_value).lower()
    if mode not in {"integrated", "perf_target"}:
        raise ValueError(f"Only integrated/perf_target INR decoder-note schemas are supported, got {mode}")
    return mode


class IntegratedPianoT5GemmaConfig(PianoT5GemmaConfig):
    def __init__(
        self,
        backbone_type="t5",
        continuous_dim=5,
        input_continuous_dim=None,
        score_input_continuous_dim=None,
        decoder_input_continuous_dim=None,
        output_continuous_dim=None,
        pitch_vocab_size=128,
        pitch_pad_id=128,
        max_time_ms=10000.0,
        pedal_output_activation="sigmoid",
        task_type="epr",
        input_feature_mode="integrated",
        score_feature_dim=8,
        time_loss_type="huber",
        value_loss_type="mse",
        removed_task_grid_loss_type="huber",
        removed_task_grid_step=1.0 / 24.0,
        removed_task_grid_soft_ce_tau=1.5,
        removed_task_mo_max=6.0,
        removed_task_md_max=6.0,
        removed_task_ml_max=6.0,
        huber_delta=0.05,
        loss_normalization=False,
        loss_weights=None,
        removed_task_loss_weights=None,
        decoder_input_mode="score",
        note_embedding_mode="sine",
        score_note_input_schema="integrated",
        decoder_note_input_schema="integrated",
        special_note_vocab_size=5,
        special_note_ids=None,
        use_full_type_embedding=True,
        use_group_presence_mask=True,
        head_input_mode="full",
        embedding_depth=2,
        head_depth=2,
        head_width_multiplier=1.0,
        head_activation="gelu",
        decoder_head_layout="pyramid4",
        decoder_head_expand_ratio=2.0,
        decoder_head_shrink_ratio=0.5,
        gpt_layers_num=None,
        bert_layers_num=None,
        max_position_embeddings=4096,
        attention_dropout=0.0,
        epr_distribution="point",
        pedal_distribution=None,
        epr_mixture_components=1,
        epr_distribution_eps=None,
        logistic_normal_sigma_min=1e-3,
        logistic_normal_sigma_max=10.0,
        beta_eps=1e-5,
        beta_kappa_min=1e-3,
        beta_alpha_min=1e-4,
        mixture_beta_parameterization="alpha_beta",
        mixture_beta_kappa_min=1e-3,
        predictive_variance_lambda=0.0,
        predictive_timing_radius=0.05,
        predictive_velocity_radius=0.05,
        skew_normal_sigma_min=1e-4,
        skew_normal_sigma_max=1e4,
        bounded_floorlog_support=False,
        velocity_distribution=None,
        dlm_components=None,
        dlm_timing_bins=256,
        dlm_velocity_bins=128,
        dlm_ioi_zero_min=0.0,
        dlm_ioi_zero_max=5.0,
        dlm_ioi_nonzero_min=-2.0,
        dlm_ioi_nonzero_max=1.0,
        dlm_duration_min=-2.0,
        dlm_duration_max=1.0,
        dlm_velocity_min=-0.5,
        dlm_velocity_max=127.5,
        dlm_scale_min=1e-3,
        dlm_scale_max=10.0,
        dlm_timing_scale_min=None,
        dlm_timing_scale_max=None,
        dlm_timing_scale_parameterization="legacy_clamp",
        dlm_ioi_nonzero_scale_max=None,
        dlm_ioi_zero_scale_max=None,
        dlm_duration_scale_max=None,
        dlm_velocity_scale_min=None,
        dlm_velocity_scale_max=None,
        dlm_velocity_scale_parameterization="legacy_clamp",
        dlm_tail_loss_lambda=0.0,
        dlm_tail_radius=0.05,
        dlm_target_tail_loss_lambda=0.0,
        dlm_target_tail_radius_frac=0.0,
        dlm_target_tail_ioi_radius=None,
        dlm_target_tail_duration_radius=None,
        dlm_timing_weighted_nll_alpha=0.0,
        dlm_timing_weight_min=0.5,
        dlm_timing_weight_max=4.0,
        dlm_raw_ms_crps_lambda=0.0,
        dlm_raw_ms_crps_scale_ms=1000.0,
        dlm_sampling_temperature=1.0,
        dlm_sampling_top_p=1.0,
        dlm_sampling_top_k=0,
        dlm_ioi_zero_inflated=False,
        dlm_pedal_zero_one_inflated=False,
        dlm_pedal_inflated_eps=0.5,
        dlm_timing_sample_truncate_radius=0.0,
        dlm_timing_sample_truncate_center="mean",
        timing_sample_truncate_radius=None,
        timing_sample_truncate_center=None,
        timing_sample_shrink_mode="none",
        timing_sample_shrink_factor=1.0,
        timing_sample_shrink_radius=0.0,
        dlm_ioi_zero_sample_shrink_factor=None,
        dlm_ioi_nonzero_sample_shrink_factor=None,
        dlm_duration_sample_shrink_factor=None,
        dlm_velocity_sample_shrink_factor=None,
        pn_mean_loss_lambda=0.0,
        pn_var_ioi_zero_lambda=0.0,
        pn_var_ioi_nonzero_lambda=0.0,
        pn_var_duration_lambda=0.0,
        pn_var_velocity_lambda=0.0,
        pn_variance_shrinkage_tau=4.0,
        pn_variance_epsilon=1e-4,
        raw_timing_loss_lambda=0.5,
        legacy_dual_timing_head=False,
        raw_timing_head_type=None,
        epr_inflated_features=None,
        epr_timing_bins=5000,
        epr_value_bins=128,
        dinr_timing_bins=512,
        dinr_zero_bin=93,
        dinr_timing_step=2.0 / 93.0,
        dinr_output_timing_bins=None,
        dinr_output_zero_bin=None,
        dinr_output_timing_step=None,
        dinr_vocabulary_mode="unified",
        dinr_absolute_max_ms=8000.0,
        dinr_deviation_min=-2.0,
        dinr_deviation_max=2.0,
        dinr_zero_ioi_min=0.0,
        dinr_zero_ioi_max=5.0,
        dinr_sampling_temperature=1.0,
        dinr_sampling_top_p=1.0,
        dinr_sampling_top_k=0,
        dinr_numerical_frequencies=16,
        dinr_input_numerical_coordinates=True,
        dinr_input_velocity_numerical_coordinates=True,
        dinr_output_deviation_numerical_coordinates=True,
        dinr_velocity_numerical_coordinates=True,
        epr_timing_target="floor_log_deviation",
        timing_control_mode="dinr_floor_log",
        timing_log_scale=50.0,
        use_timing_scale_bit=False,
        soft_ce_tau=None,
        timing_input_normalization="linear_5000",
        musical_feature_mode="musical4slot",
        slot_version=None,
        slot_dim=None,
        slot_fusion="mlp",
        musical_slot_fusion="sum",
        slot_gates=False,
        slot_gate_scope="all",
        slot_gate_init=1.0,
        slot_share_role_encoders=True,
        musical_gate_init=1.0,
        musical_component_gates=False,
        musical_component_gate_init=1.0,
        additive_embedding_gates=False,
        additive_gate_init=1.0,
        additive_musical_gate_init=1.0,
        prior_token_keep_prob=1.0,
        prior_token_dropout_mode="mask",
        prior_attribute_keep_probs=None,
        prior_attribute_noise_std=0.05,
        prior_property_dropout_prob=None,
        prior_property_dropout_pattern="independent",
        prior_property_dropout_replacement="pad",
        prior_property_visible_prob=0.50,
        prior_property_all_dropout_prob=0.25,
        stable_force_all_properties_visible=False,
        tf_embedding_mask_keep_prob=1.0,
        tf_embedding_mask_score=False,
        tf_embedding_mask_decoder=False,
        slot_decoder_mask_mode="property",
        stable_contract_loss=False,
        stable_contract_alpha=1.0,
        stable_contract_lambda=0.0,
        stable_contract_ioi_alpha=None,
        stable_contract_duration_alpha=None,
        stable_contract_ioi_lambda=None,
        stable_contract_duration_lambda=None,
        stable_contract_eps=1e-6,
        zero_ioi_transform=None,
        zero_ioi_positive_support=False,
        zero_ioi_support_eps=1e-6,
        zero_ioi_residual=False,
        zero_ioi_residual_targets=None,
        zero_score_ioi_embedding=False,
        zero_timing_head_condition=False,
        zero_ioi_dual_distribution_mode="none",
        zero_ioi_dual_duration=True,
        piano_pitch_min=21,
        pedal_representation="binary_4",
        use_style_tokens=False,
        style_creator_vocab_size=1,
        style_source_vocab_size=1,
        style_score_stat_dim=18,
        style_perf_stat_dim=18,
        style_integration_mode="prepend",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.backbone_type = backbone_type
        self.continuous_dim = continuous_dim
        self.input_continuous_dim = input_continuous_dim or continuous_dim
        self.score_input_continuous_dim = score_input_continuous_dim or self.input_continuous_dim
        self.decoder_input_continuous_dim = decoder_input_continuous_dim or self.input_continuous_dim
        self.output_continuous_dim = output_continuous_dim or continuous_dim
        self.pitch_vocab_size = pitch_vocab_size
        self.pitch_pad_id = pitch_pad_id
        self.intermediate_size = self.encoder.intermediate_size
        self.num_attention_heads = self.encoder.num_attention_heads
        self.num_key_value_heads = self.encoder.num_key_value_heads
        self.head_dim = self.encoder.head_dim
        self.max_time_ms = max_time_ms
        self.pedal_output_activation = pedal_output_activation
        self.task_type = task_type
        self.input_feature_mode = input_feature_mode
        self.score_feature_dim = score_feature_dim
        self.time_loss_type = time_loss_type
        self.value_loss_type = value_loss_type
        self.removed_task_grid_loss_type = removed_task_grid_loss_type
        self.removed_task_grid_step = float(removed_task_grid_step)
        self.removed_task_grid_soft_ce_tau = float(removed_task_grid_soft_ce_tau)
        self.removed_task_mo_max = float(removed_task_mo_max)
        self.removed_task_md_max = float(removed_task_md_max)
        self.removed_task_ml_max = float(removed_task_ml_max)
        self.huber_delta = huber_delta
        self.loss_normalization = bool(loss_normalization)
        self.loss_weights = loss_weights or {
            "ioi": 1.0,
            "duration": 1.0,
            "velocity": 1.0,
            "pedal": 1.0,
        }
        self.removed_task_loss_weights = removed_task_loss_weights or {
            "mo": 1.0,
            "ioi_zero": 1.0,
            "md": 1.0,
            "ml": 1.0,
            "tempo": 1.0,
            "first": 1.0,
            "grace": 0.4,
            "hand": 0.5,
            "trill": 0.4,
            "stacc": 0.3,
            "stem": 0.2,
        }
        self.decoder_input_mode = decoder_input_mode
        self.note_embedding_mode = note_embedding_mode
        self.score_note_input_schema = score_note_input_schema
        self.decoder_note_input_schema = decoder_note_input_schema
        self.special_note_vocab_size = special_note_vocab_size
        self.special_note_ids = special_note_ids or {
            "pad": 0,
            "mask": 1,
            "bos": 2,
            "eos": 3,
            "play": 4,
        }
        self.use_full_type_embedding = use_full_type_embedding
        self.use_group_presence_mask = use_group_presence_mask
        self.head_input_mode = head_input_mode
        self.embedding_depth = embedding_depth
        self.head_depth = head_depth
        self.head_width_multiplier = float(head_width_multiplier)
        self.head_activation = head_activation
        self.decoder_head_layout = str(decoder_head_layout).lower()
        self.decoder_head_expand_ratio = float(decoder_head_expand_ratio)
        self.decoder_head_shrink_ratio = float(decoder_head_shrink_ratio)
        self.gpt_layers_num = gpt_layers_num
        self.bert_layers_num = bert_layers_num
        self.max_position_embeddings = max_position_embeddings
        self.attention_dropout = attention_dropout
        self.epr_distribution = epr_distribution
        self.pedal_distribution = pedal_distribution or epr_distribution
        self.epr_mixture_components = int(epr_mixture_components)
        self.epr_distribution_eps = beta_eps if epr_distribution_eps is None else epr_distribution_eps
        self.logistic_normal_sigma_min = logistic_normal_sigma_min
        self.logistic_normal_sigma_max = logistic_normal_sigma_max
        self.beta_eps = beta_eps
        self.beta_kappa_min = beta_kappa_min
        self.beta_alpha_min = beta_alpha_min
        self.mixture_beta_parameterization = str(mixture_beta_parameterization).lower()
        self.mixture_beta_kappa_min = float(mixture_beta_kappa_min)
        self.predictive_variance_lambda = float(predictive_variance_lambda)
        self.predictive_timing_radius = float(predictive_timing_radius)
        self.predictive_velocity_radius = float(predictive_velocity_radius)
        self.skew_normal_sigma_min = skew_normal_sigma_min
        self.skew_normal_sigma_max = skew_normal_sigma_max
        self.bounded_floorlog_support = bool(bounded_floorlog_support)
        default_velocity_distribution = (
            "skew_normal"
            if str(epr_distribution).lower() in DLM_DISTRIBUTIONS
            else epr_distribution
        )
        self.velocity_distribution = str(velocity_distribution or default_velocity_distribution).lower()
        self.dlm_components = int(dlm_components if dlm_components is not None else epr_mixture_components)
        self.dlm_timing_bins = int(dlm_timing_bins)
        self.dlm_velocity_bins = int(dlm_velocity_bins)
        self.dlm_ioi_zero_min = float(dlm_ioi_zero_min)
        self.dlm_ioi_zero_max = float(dlm_ioi_zero_max)
        self.dlm_ioi_nonzero_min = float(dlm_ioi_nonzero_min)
        self.dlm_ioi_nonzero_max = float(dlm_ioi_nonzero_max)
        self.dlm_duration_min = float(dlm_duration_min)
        self.dlm_duration_max = float(dlm_duration_max)
        self.dlm_velocity_min = float(dlm_velocity_min)
        self.dlm_velocity_max = float(dlm_velocity_max)
        self.dlm_scale_min = float(dlm_scale_min)
        self.dlm_scale_max = float(dlm_scale_max)
        self.dlm_timing_scale_min = float(
            dlm_scale_min if dlm_timing_scale_min is None else dlm_timing_scale_min
        )
        self.dlm_timing_scale_max = float(
            dlm_scale_max if dlm_timing_scale_max is None else dlm_timing_scale_max
        )
        self.dlm_timing_scale_parameterization = str(dlm_timing_scale_parameterization).lower()
        self.dlm_ioi_nonzero_scale_max = float(
            self.dlm_timing_scale_max if dlm_ioi_nonzero_scale_max is None else dlm_ioi_nonzero_scale_max
        )
        self.dlm_ioi_zero_scale_max = float(
            self.dlm_timing_scale_max if dlm_ioi_zero_scale_max is None else dlm_ioi_zero_scale_max
        )
        self.dlm_duration_scale_max = float(
            self.dlm_timing_scale_max if dlm_duration_scale_max is None else dlm_duration_scale_max
        )
        self.dlm_velocity_scale_min = float(
            self.dlm_scale_min if dlm_velocity_scale_min is None else dlm_velocity_scale_min
        )
        self.dlm_velocity_scale_max = float(
            self.dlm_scale_max if dlm_velocity_scale_max is None else dlm_velocity_scale_max
        )
        self.dlm_velocity_scale_parameterization = str(dlm_velocity_scale_parameterization).lower()
        self.dlm_tail_loss_lambda = float(dlm_tail_loss_lambda)
        self.dlm_tail_radius = float(dlm_tail_radius)
        self.dlm_target_tail_loss_lambda = float(dlm_target_tail_loss_lambda)
        self.dlm_target_tail_radius_frac = float(dlm_target_tail_radius_frac)
        self.dlm_target_tail_ioi_radius = (
            None if dlm_target_tail_ioi_radius is None else float(dlm_target_tail_ioi_radius)
        )
        self.dlm_target_tail_duration_radius = (
            None if dlm_target_tail_duration_radius is None else float(dlm_target_tail_duration_radius)
        )
        self.dlm_timing_weighted_nll_alpha = float(dlm_timing_weighted_nll_alpha)
        self.dlm_timing_weight_min = float(dlm_timing_weight_min)
        self.dlm_timing_weight_max = float(dlm_timing_weight_max)
        self.dlm_raw_ms_crps_lambda = float(dlm_raw_ms_crps_lambda)
        self.dlm_raw_ms_crps_scale_ms = float(dlm_raw_ms_crps_scale_ms)
        self.dlm_ioi_zero_inflated = bool(dlm_ioi_zero_inflated)
        self.dlm_pedal_zero_one_inflated = bool(dlm_pedal_zero_one_inflated)
        self.dlm_pedal_inflated_eps = float(dlm_pedal_inflated_eps)
        if self.dlm_timing_weighted_nll_alpha < 0.0:
            raise ValueError("dlm_timing_weighted_nll_alpha must be >= 0")
        if self.dlm_timing_weight_min <= 0.0 or self.dlm_timing_weight_max < self.dlm_timing_weight_min:
            raise ValueError("DLM timing weight bounds require 0 < min <= max")
        if self.dlm_raw_ms_crps_lambda < 0.0 or self.dlm_raw_ms_crps_scale_ms <= 0.0:
            raise ValueError("DLM raw-ms CRPS requires lambda >= 0 and scale_ms > 0")
        if self.dlm_target_tail_loss_lambda < 0.0 or self.dlm_target_tail_radius_frac < 0.0:
            raise ValueError("DLM target-tail loss requires lambda >= 0 and radius_frac >= 0")
        self.dlm_sampling_temperature = float(dlm_sampling_temperature)
        self.dlm_sampling_top_p = float(dlm_sampling_top_p)
        self.dlm_sampling_top_k = int(dlm_sampling_top_k)
        if not math.isfinite(self.dlm_sampling_temperature) or self.dlm_sampling_temperature < 0.0:
            raise ValueError(
                "dlm_sampling_temperature must be finite and >= 0, got "
                f"{self.dlm_sampling_temperature}"
            )
        if timing_sample_truncate_radius is None:
            timing_sample_truncate_radius = dlm_timing_sample_truncate_radius
        if timing_sample_truncate_center is None:
            timing_sample_truncate_center = dlm_timing_sample_truncate_center
        self.timing_sample_truncate_radius = float(timing_sample_truncate_radius or 0.0)
        self.timing_sample_truncate_center = str(timing_sample_truncate_center or "mean").lower()
        self.dlm_timing_sample_truncate_radius = self.timing_sample_truncate_radius
        self.dlm_timing_sample_truncate_center = self.timing_sample_truncate_center
        self.timing_sample_shrink_mode = str(timing_sample_shrink_mode or "none").lower()
        self.timing_sample_shrink_factor = float(timing_sample_shrink_factor)
        self.timing_sample_shrink_radius = float(timing_sample_shrink_radius or 0.0)
        self.dlm_ioi_zero_sample_shrink_factor = dlm_ioi_zero_sample_shrink_factor
        self.dlm_ioi_nonzero_sample_shrink_factor = dlm_ioi_nonzero_sample_shrink_factor
        self.dlm_duration_sample_shrink_factor = dlm_duration_sample_shrink_factor
        self.dlm_velocity_sample_shrink_factor = dlm_velocity_sample_shrink_factor
        self.pn_mean_loss_lambda = float(pn_mean_loss_lambda)
        self.pn_var_ioi_zero_lambda = float(pn_var_ioi_zero_lambda)
        self.pn_var_ioi_nonzero_lambda = float(pn_var_ioi_nonzero_lambda)
        self.pn_var_duration_lambda = float(pn_var_duration_lambda)
        self.pn_var_velocity_lambda = float(pn_var_velocity_lambda)
        self.pn_variance_shrinkage_tau = float(pn_variance_shrinkage_tau)
        self.pn_variance_epsilon = float(pn_variance_epsilon)
        self.raw_timing_loss_lambda = raw_timing_loss_lambda
        self.legacy_dual_timing_head = bool(legacy_dual_timing_head)
        self.raw_timing_head_type = str(
            raw_timing_head_type
            or ("distribution" if self.legacy_dual_timing_head else "none")
        ).lower()
        if self.raw_timing_head_type not in {"none", "distribution", "regression"}:
            raise ValueError(
                "raw_timing_head_type must be none, distribution, or regression; "
                f"got {self.raw_timing_head_type}"
            )
        self.epr_inflated_features = epr_inflated_features or {
            "ioi": "zero",
            "pedal": "zero_one",
        }
        self.epr_timing_bins = int(epr_timing_bins)
        self.epr_value_bins = int(epr_value_bins)
        self.dinr_timing_bins = int(dinr_timing_bins)
        self.dinr_zero_bin = int(dinr_zero_bin)
        self.dinr_timing_step = float(dinr_timing_step)
        self.dinr_output_timing_bins = int(dinr_output_timing_bins or self.dinr_timing_bins)
        self.dinr_output_zero_bin = int(self.dinr_zero_bin if dinr_output_zero_bin is None else dinr_output_zero_bin)
        self.dinr_output_timing_step = float(self.dinr_timing_step if dinr_output_timing_step is None else dinr_output_timing_step)
        self.dinr_vocabulary_mode = str(dinr_vocabulary_mode or "unified").lower()
        self.dinr_absolute_max_ms = float(dinr_absolute_max_ms)
        self.dinr_deviation_min = float(dinr_deviation_min)
        self.dinr_deviation_max = float(dinr_deviation_max)
        self.dinr_zero_ioi_min = float(dinr_zero_ioi_min)
        self.dinr_zero_ioi_max = float(dinr_zero_ioi_max)
        self.dinr_sampling_temperature = float(dinr_sampling_temperature)
        self.dinr_sampling_top_p = float(dinr_sampling_top_p)
        self.dinr_sampling_top_k = int(dinr_sampling_top_k)
        self.dinr_numerical_frequencies = int(dinr_numerical_frequencies)
        self.dinr_input_numerical_coordinates = bool(dinr_input_numerical_coordinates)
        self.dinr_input_velocity_numerical_coordinates = bool(dinr_input_velocity_numerical_coordinates)
        self.dinr_output_deviation_numerical_coordinates = bool(
            dinr_output_deviation_numerical_coordinates
        )
        self.dinr_velocity_numerical_coordinates = bool(dinr_velocity_numerical_coordinates)
        if self.dinr_timing_bins < 2 or self.dinr_timing_step <= 0.0:
            raise ValueError("DINR requires at least two timing bins and a positive timing step")
        if not (0 <= self.dinr_zero_bin < self.dinr_timing_bins):
            raise ValueError("dinr_zero_bin must index the DINR timing vocabulary")
        if self.dinr_output_timing_bins < 2 or self.dinr_output_timing_step <= 0.0:
            raise ValueError("DINR requires at least two output timing bins and a positive output timing step")
        if not (0 <= self.dinr_output_zero_bin < self.dinr_output_timing_bins):
            raise ValueError("dinr_output_zero_bin must index the DINR output timing vocabulary")
        self.epr_timing_target = str(epr_timing_target).lower()
        self.timing_log_scale = float(timing_log_scale)
        self.timing_control_mode = resolve_timing_control_mode(
            timing_control_mode=timing_control_mode,
            use_timing_scale_bit=use_timing_scale_bit,
        )
        self.use_timing_scale_bit = self.timing_control_mode == "piecewise_scale_bit"
        self.control_feature_dim = timing_control_feature_dim(
            timing_control_mode=self.timing_control_mode,
            use_timing_scale_bit=self.use_timing_scale_bit,
        )
        self.score_control_feature_dim = self.control_feature_dim
        self.pedal_representation = normalize_pedal_representation(pedal_representation)
        self.performance_control_feature_dim = self.control_feature_dim + pedal_representation_dim(self.pedal_representation)
        self.musical_feature_mode = str(musical_feature_mode).lower()
        self.musical_feature_dim = musical_feature_dim(self.musical_feature_mode)
        self.mask_feature_dim = 2 if self.musical_feature_dim == 0 else 3
        self.slot_version = normalize_slot_version(slot_version) if str(note_embedding_mode).lower() == "slot_attribute" else None
        self.slot_dim = None if slot_dim is None else int(slot_dim)
        self.slot_fusion = str(slot_fusion).lower()
        self.musical_slot_fusion = str(musical_slot_fusion or "sum").lower()
        self.slot_gates = bool(slot_gates)
        self.slot_gate_scope = str(slot_gate_scope or "all").lower()
        self.slot_gate_init = float(slot_gate_init)
        self.slot_share_role_encoders = bool(slot_share_role_encoders)
        self.musical_gate_init = float(musical_gate_init)
        self.musical_component_gates = bool(musical_component_gates)
        self.musical_component_gate_init = float(musical_component_gate_init)
        self.additive_embedding_gates = bool(additive_embedding_gates)
        self.additive_gate_init = float(additive_gate_init)
        self.additive_musical_gate_init = float(additive_musical_gate_init)
        self.soft_ce_tau = soft_ce_tau or {
            "ioi": 10.0,
            "duration": 30.0,
            "velocity": 6.0,
            "pedal": 2.0,
        }
        self.timing_input_normalization = timing_input_normalization
        self.prior_token_keep_prob = prior_token_keep_prob
        self.prior_token_dropout_mode = prior_token_dropout_mode
        self.prior_attribute_keep_probs = prior_attribute_keep_probs
        self.prior_attribute_noise_std = prior_attribute_noise_std
        self.prior_property_dropout_prob = prior_property_dropout_prob
        self.prior_property_dropout_pattern = str(prior_property_dropout_pattern or "independent").lower()
        if self.prior_property_dropout_pattern not in {"independent", "correlated", "mixed"}:
            raise ValueError(
                "prior_property_dropout_pattern must be independent, correlated, or mixed; "
                f"got {self.prior_property_dropout_pattern}"
            )
        self.prior_property_dropout_replacement = str(
            prior_property_dropout_replacement or "pad"
        ).lower()
        if self.prior_property_dropout_replacement not in {"pad", "mask"}:
            raise ValueError(
                "prior_property_dropout_replacement must be pad or mask; "
                f"got {self.prior_property_dropout_replacement}"
            )
        self.prior_property_visible_prob = float(prior_property_visible_prob)
        self.prior_property_all_dropout_prob = float(prior_property_all_dropout_prob)
        if (
            self.prior_property_visible_prob < 0.0
            or self.prior_property_all_dropout_prob < 0.0
            or self.prior_property_visible_prob + self.prior_property_all_dropout_prob > 1.0
        ):
            raise ValueError(
                "prior_property_visible_prob and prior_property_all_dropout_prob must be non-negative "
                "and sum to at most 1"
            )
        self.stable_force_all_properties_visible = bool(stable_force_all_properties_visible)
        self.tf_embedding_mask_keep_prob = float(tf_embedding_mask_keep_prob)
        self.tf_embedding_mask_score = bool(tf_embedding_mask_score)
        self.tf_embedding_mask_decoder = bool(tf_embedding_mask_decoder)
        self.slot_decoder_mask_mode = str(slot_decoder_mask_mode or "property").lower()
        if self.slot_decoder_mask_mode not in {"property", "whole_token", "none"}:
            raise ValueError(
                "slot_decoder_mask_mode must be property, whole_token, or none; "
                f"got {self.slot_decoder_mask_mode}"
            )
        self.stable_contract_loss = bool(stable_contract_loss)
        self.stable_contract_alpha = float(stable_contract_alpha)
        self.stable_contract_lambda = float(stable_contract_lambda)
        self.stable_contract_ioi_alpha = (
            None if stable_contract_ioi_alpha is None else float(stable_contract_ioi_alpha)
        )
        self.stable_contract_duration_alpha = (
            None if stable_contract_duration_alpha is None else float(stable_contract_duration_alpha)
        )
        self.stable_contract_ioi_lambda = (
            None if stable_contract_ioi_lambda is None else float(stable_contract_ioi_lambda)
        )
        self.stable_contract_duration_lambda = (
            None if stable_contract_duration_lambda is None else float(stable_contract_duration_lambda)
        )
        self.stable_contract_eps = float(stable_contract_eps)
        self.zero_ioi_transform = str(
            zero_ioi_transform
            or ("softplus" if bool(zero_ioi_positive_support) else "none")
        ).lower()
        if self.zero_ioi_transform not in {"none", "softplus", "folded_abs", "squared"}:
            raise ValueError(
                "zero_ioi_transform must be none, softplus, folded_abs, or squared; "
                f"got {self.zero_ioi_transform}"
            )
        self.zero_ioi_positive_support = self.zero_ioi_transform != "none"
        self.zero_ioi_support_eps = float(zero_ioi_support_eps)
        self.zero_ioi_residual = bool(zero_ioi_residual)
        targets = zero_ioi_residual_targets if zero_ioi_residual_targets is not None else ["ioi"]
        if isinstance(targets, str):
            targets = [part.strip() for part in targets.split(",") if part.strip()]
        self.zero_ioi_residual_targets = tuple(str(target).lower() for target in targets)
        invalid_targets = set(self.zero_ioi_residual_targets) - {"ioi", "duration"}
        if invalid_targets:
            raise ValueError(f"zero_ioi_residual_targets supports ioi/duration only, got {sorted(invalid_targets)}")
        self.zero_score_ioi_embedding = bool(zero_score_ioi_embedding)
        self.zero_timing_head_condition = bool(zero_timing_head_condition)
        self.zero_ioi_dual_distribution_mode = str(zero_ioi_dual_distribution_mode or "none").lower()
        if self.zero_ioi_dual_distribution_mode == "folded_abs":
            self.zero_ioi_dual_distribution_mode = "zero_folded"
        if self.zero_ioi_dual_distribution_mode not in {"none", "skew_normal", "zero_folded"}:
            raise ValueError(
                "zero_ioi_dual_distribution_mode must be none, skew_normal, or zero_folded; "
                f"got {self.zero_ioi_dual_distribution_mode}"
            )
        self.zero_ioi_dual_duration = bool(zero_ioi_dual_duration)
        self.piano_pitch_min = int(piano_pitch_min)
        if bool(use_style_tokens):
            raise ValueError("use_style_tokens is disabled for the simplified EPR/removed_task pipelines")
        self.use_style_tokens = bool(use_style_tokens)
        self.style_creator_vocab_size = int(style_creator_vocab_size)
        self.style_source_vocab_size = int(style_source_vocab_size)
        self.style_score_stat_dim = int(style_score_stat_dim)
        self.style_perf_stat_dim = int(style_perf_stat_dim)
        self.style_integration_mode = str(style_integration_mode or "prepend").lower()
        valid_style_modes = {
            "prepend",
            "add",
            "film",
            "add_film",
            "prepend_add",
            "prepend_film",
            "dec_add",
            "dec_film",
            "dec_add_film",
            "prepend_dec_add",
            "prepend_dec_film",
            "prepend_dec_add_film",
        }
        if self.style_integration_mode not in valid_style_modes:
            raise ValueError(f"Unsupported style_integration_mode={style_integration_mode}")
        self.style_token_count = (
            4 if self.use_style_tokens and self.style_integration_mode.startswith("prepend") else 0
        )


def _activation(name):
    name = str(name).lower()
    if name == "gelu":
        return nn.GELU()
    if name == "relu":
        return nn.ReLU()
    if name == "silu":
        return nn.SiLU()
    raise ValueError(f"Unsupported activation: {name}")


def _make_mlp(input_dim, output_dim, hidden_dim, depth=2, activation="gelu"):
    if int(output_dim) == 0:
        return None
    depth = int(depth)
    if depth <= 1:
        return nn.Linear(input_dim, output_dim)
    if depth == 3:
        mid_dim = max(1, int(round(float(hidden_dim) * 0.5)))
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            _activation(activation),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, mid_dim),
            _activation(activation),
            nn.Linear(mid_dim, output_dim),
        )
    layers = [nn.Linear(input_dim, hidden_dim), _activation(activation)]
    for _ in range(depth - 2):
        layers.extend([nn.Linear(hidden_dim, hidden_dim), _activation(activation)])
    layers.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*layers)


def _gate_logit_from_value(value, eps=1e-6):
    value = min(max(float(value), eps), 1.0 - eps)
    return math.log(value / (1.0 - value))


def _make_decoder_head(
    input_dim,
    output_dim,
    hidden_dim,
    depth=2,
    activation="gelu",
    layout="pyramid4",
    expand_ratio=2.0,
    shrink_ratio=0.5,
):
    if int(output_dim) == 0:
        return None

    layout = str(layout).lower()
    if layout in {"mlp", "default"}:
        return _make_mlp(input_dim, output_dim, hidden_dim, depth=depth, activation=activation)
    if layout != "pyramid4":
        raise ValueError(f"Unsupported decoder_head_layout: {layout}")

    expand_dim = max(1, int(round(float(input_dim) * float(expand_ratio))))
    shrink_dim = max(1, int(round(float(input_dim) * float(shrink_ratio))))
    return nn.Sequential(
        nn.Linear(input_dim, expand_dim),
        _activation(activation),
        nn.Linear(expand_dim, input_dim),
        _activation(activation),
        nn.Linear(input_dim, shrink_dim),
        _activation(activation),
        nn.Linear(shrink_dim, output_dim),
    )


class IntegratedNoteEncoder(nn.Module):
    def __init__(self, config, continuous_dim=None, role="score"):
        super().__init__()
        self.config = config
        role = str(role).lower()
        if continuous_dim is None:
            if role == "decoder":
                continuous_dim = getattr(config, "decoder_input_continuous_dim", config.input_continuous_dim)
            else:
                continuous_dim = getattr(config, "score_input_continuous_dim", config.input_continuous_dim)
        self.continuous_dim = continuous_dim
        self.role = role
        self.mode = getattr(config, "note_embedding_mode", "sine").lower()
        self.schema = (
            decoder_note_input_schema(config)
            if self.role == "decoder"
            else score_note_input_schema(config)
        )
        self.special_note_embeddings = nn.Embedding(
            config.special_note_vocab_size,
            config.hidden_size,
        )
        self.embedding_depth = getattr(config, "embedding_depth", 2)
        self.activation = getattr(config, "head_activation", "gelu")
        self.pitch_factor_dim = 20
        self.slot_pitch_embedding = None
        self.slot_version = normalize_slot_version(getattr(config, "slot_version", None)) if self.mode == "slot_attribute" else None
        self.slot_num = slot_version_num_slots(self.slot_version) if self.slot_version is not None else 0
        self.slot_is_pt = slot_version_is_pt(self.slot_version) if self.slot_version is not None else False
        self.slot_fusion_mode = str(getattr(config, "slot_fusion", "mlp") or "mlp").lower()
        self.musical_slot_fusion_mode = str(getattr(config, "musical_slot_fusion", "sum") or "sum").lower()
        if self.musical_slot_fusion_mode not in {"sum", "mlp"}:
            raise ValueError(
                f"Unsupported musical_slot_fusion={self.musical_slot_fusion_mode}; expected sum or mlp"
            )
        self.slot_dim = (
            int(getattr(config, "slot_dim", 0))
            if getattr(config, "slot_dim", None) is not None
            else (128 if self.slot_num > 0 else 0)
        )
        self.score_control_dim = int(
            getattr(config, "score_control_feature_dim", getattr(config, "control_feature_dim", 5))
        )
        self.pedal_dim = pedal_representation_dim(getattr(config, "pedal_representation", "binary_4"))
        self.performance_control_dim = int(
            getattr(
                config,
                "performance_control_feature_dim",
                getattr(config, "control_feature_dim", 5) + self.pedal_dim,
            )
        )
        self.musical_dim = int(getattr(config, "musical_feature_dim", 12))
        self.musical_feature_mode = str(getattr(config, "musical_feature_mode", "musical4slot")).lower()
        self.mask_dim = int(getattr(config, "mask_feature_dim", 3))
        self.decoder_target_dim = (
            max(0, int(self.continuous_dim) - self.mask_dim)
            if self.role == "decoder" and self.schema == "perf_target"
            else 7
        )
        self.decoder_target_mask_dim = 3
        shared_dinr_encoder = str(getattr(config, "epr_distribution", "point")).lower() in DINR_DISTRIBUTIONS
        if (self.role == "decoder" or shared_dinr_encoder) and self.schema == "integrated":
            self.performance_missing_embeddings = nn.Parameter(
                torch.zeros(self.performance_control_dim, config.hidden_size)
            )
            nn.init.normal_(self.performance_missing_embeddings, mean=0.0, std=0.02)
        else:
            self.register_parameter("performance_missing_embeddings", None)

        self.pitch_input_dim = self.pitch_factor_dim
        if self.mode not in {"sine", "cine", "slot_attribute"}:
            raise ValueError(
                f"Unsupported note_embedding_mode: {self.mode}. Expected one of: sine, cine, slot_attribute"
            )
        if self.mode == "slot_attribute":
            if self.schema == "perf_target":
                raise ValueError("slot_attribute only supports decoder_note_input_schema='integrated'")
            if self.slot_num <= 0 or self.slot_dim <= 0:
                raise ValueError(
                    f"slot_attribute requires a valid slot_version/slot_dim, got slot_version={self.slot_version}, "
                    f"slot_dim={self.slot_dim}"
                )
            if self.slot_fusion_mode not in {"mlp", "direct_concat", "sum"}:
                raise ValueError(
                    f"slot_attribute slot_fusion must be mlp or direct_concat, got {self.slot_fusion_mode}"
                )
            if self.slot_fusion_mode == "direct_concat" and self.slot_num * self.slot_dim != int(config.hidden_size):
                raise ValueError(
                    "slot_attribute direct_concat requires slot_num * slot_dim == hidden_size, got "
                    f"{self.slot_num} * {self.slot_dim} != {config.hidden_size}"
                )
        pitch_output_dim = (
            int(config.hidden_size)
            if self.mode == "slot_attribute" and self.slot_fusion_mode == "sum"
            else self.slot_dim
            if self.mode == "slot_attribute"
            else config.hidden_size
        )
        if self.mode == "slot_attribute":
            pitch_table_size = max(int(config.pitch_vocab_size), int(config.pitch_pad_id) + 1) + 1
            self.slot_pitch_embedding = nn.Embedding(pitch_table_size, self.slot_dim)
            self.pitch_projection = None
        else:
            self.pitch_projection = _make_mlp(
                self.pitch_input_dim,
                pitch_output_dim,
                pitch_output_dim,
                self.embedding_depth,
                self.activation,
            )

        self.score_control_projection = None
        self.performance_control_projection = None
        self.musical_projection = None
        self.mask_projection = None
        self.continuous_mlp = None
        self.decoder_target_projection = None
        self.slot_score_ioi_projection = None
        self.slot_score_duration_projection = None
        self.slot_score_velocity_projection = None
        self.slot_perf_ioi_projection = None
        self.slot_perf_duration_projection = None
        self.slot_perf_velocity_projection = None
        self.slot_perf_pedal_projection = None
        self.slot_musical_onset_projection = None
        self.slot_musical_duration_projection = None
        self.slot_musical_length_projection = None
        self.slot_musical_binary_projection = None
        self.slot_null_embeddings = None
        self.slot_mask_embeddings = None
        self.slot_pad_embeddings = None
        self.slot_zero_score_ioi_embedding = None
        self.slot_musical_onset_embedding = None
        self.slot_musical_duration_embedding = None
        self.slot_musical_length_embedding = None
        self.slot_musical_onset_scalar_projection = None
        self.slot_musical_duration_scalar_projection = None
        self.slot_musical_length_scalar_projection = None
        self.slot_musical_compact_norm = None
        self.slot_musical_fusion = None
        self.slot_fusion = None
        self.slot_gate_logits = None
        self.slot_gate_mask = None
        self.slot_musical_component_gate_logits = None
        self.additive_embedding_gate_logits = None
        self.slot_timing_dim = max(1, (self.score_control_dim - 1) // 2) if self.mode == "slot_attribute" else 0
        self.dinr_enabled = (
            self.mode == "slot_attribute"
            and str(getattr(config, "epr_distribution", "point")).lower() in DINR_DISTRIBUTIONS
        )
        self.dinr_timing_table = None
        self.dinr_duration_table = None
        self.dinr_velocity_table = None
        self.dinr_field_embedding = None
        self.dinr_role_embedding = None
        if self.dinr_enabled:
            timing_coordinates = (
                torch.arange(int(config.dinr_timing_bins), dtype=torch.float32)
                - int(config.dinr_zero_bin)
            ) * float(config.dinr_timing_step)
            self.dinr_timing_table = DINRValueTable(
                timing_coordinates,
                self.slot_dim,
                frequencies=int(getattr(config, "dinr_numerical_frequencies", 16)),
            )
            self.dinr_velocity_table = DINRValueTable(
                torch.arange(128, dtype=torch.float32) / 127.0,
                self.slot_dim,
                frequencies=int(getattr(config, "dinr_numerical_frequencies", 16)),
            )
            self.dinr_field_embedding = nn.Embedding(2, self.slot_dim)
            self.dinr_role_embedding = nn.Embedding(2, self.slot_dim)

        if self.role == "decoder" and self.schema == "perf_target":
            self.decoder_target_projection = _make_mlp(
                self.decoder_target_dim,
                config.hidden_size,
                config.hidden_size,
                self.embedding_depth,
                self.activation,
            )
            self.mask_projection = _make_mlp(
                self.mask_dim,
                config.hidden_size,
                config.hidden_size,
                self.embedding_depth,
                self.activation,
                )
        elif self.mode == "sine":
            self.score_control_projection = _make_mlp(
                self.score_control_dim,
                config.hidden_size,
                config.hidden_size,
                self.embedding_depth,
                self.activation,
            )
            self.performance_control_projection = _make_mlp(
                self.performance_control_dim,
                config.hidden_size,
                config.hidden_size,
                self.embedding_depth,
                self.activation,
            )
            self.musical_projection = (
                _make_mlp(
                    self.musical_dim,
                    config.hidden_size,
                    config.hidden_size,
                    self.embedding_depth,
                    self.activation,
                )
                if self.musical_dim > 0
                else None
            )
            self.mask_projection = _make_mlp(
                self.mask_dim,
                config.hidden_size,
                config.hidden_size,
                self.embedding_depth,
                self.activation,
            )
            if bool(getattr(config, "additive_embedding_gates", False)):
                gate_logit = _gate_logit_from_value(getattr(config, "additive_gate_init", 1.0))
                musical_logit = _gate_logit_from_value(getattr(config, "additive_musical_gate_init", 1.0))
                self.additive_embedding_gate_logits = nn.Parameter(torch.full((5,), gate_logit))
                with torch.no_grad():
                    self.additive_embedding_gate_logits[3] = musical_logit
        elif self.mode == "cine":
            flat_dim = (
                self.pitch_factor_dim
                + self.score_control_dim
                + self.performance_control_dim
                + self.musical_dim
                + self.mask_dim
            )
            self.continuous_mlp = _make_mlp(
                flat_dim,
                config.hidden_size,
                config.hidden_size,
                self.embedding_depth,
                self.activation,
            )
        else:
            expected_perf_dim = self.slot_timing_dim * 2 + 1 + pedal_representation_dim(
                getattr(config, "pedal_representation", "binary_4")
            )
            if self.score_control_dim != self.slot_timing_dim * 2 + 1:
                raise ValueError(
                    f"slot_attribute expects score_control_dim = 2 * timing_dim + 1, got {self.score_control_dim}"
                )
            if self.performance_control_dim != expected_perf_dim:
                raise ValueError(
                    f"slot_attribute expects performance_control_dim={expected_perf_dim}, got {self.performance_control_dim}"
                )
            if self.slot_version in {"slot6", "slot9", "slot12"} and self.musical_dim not in {0, 9}:
                raise ValueError(
                    f"{self.slot_version} expects musical_feature_dim in {{0, 9}}, got {self.musical_dim}. "
                    "Use musical_feature_mode='musical4slot' or disable musical features."
                )
            if self.slot_version == "slot7" and self.musical_dim != 0:
                raise ValueError(
                    "slot7 is legacy musical layout and no longer accepts musical features. Use slot6 with musical4slot."
                )
            self.slot_score_ioi_projection = _make_mlp(
                self.slot_timing_dim,
                self.slot_dim,
                self.slot_dim,
                self.embedding_depth,
                self.activation,
            )
            self.slot_score_duration_projection = _make_mlp(
                self.slot_timing_dim,
                self.slot_dim,
                self.slot_dim,
                self.embedding_depth,
                self.activation,
            )
            self.slot_score_velocity_projection = _make_mlp(
                1,
                self.slot_dim,
                self.slot_dim,
                self.embedding_depth,
                self.activation,
            )
            self.slot_perf_ioi_projection = _make_mlp(
                self.slot_timing_dim,
                self.slot_dim,
                self.slot_dim,
                self.embedding_depth,
                self.activation,
            )
            self.slot_perf_duration_projection = _make_mlp(
                self.slot_timing_dim,
                self.slot_dim,
                self.slot_dim,
                self.embedding_depth,
                self.activation,
            )
            self.slot_perf_velocity_projection = _make_mlp(
                1,
                self.slot_dim,
                self.slot_dim,
                self.embedding_depth,
                self.activation,
            )
            self.slot_perf_pedal_projection = _make_mlp(
                self.pedal_dim,
                self.slot_dim,
                self.slot_dim,
                self.embedding_depth,
                self.activation,
            )
            if self.slot_version in {"slot6", "slot7", "slot9", "slot12"}:
                onset_embeddings = 147
                self.slot_musical_onset_embedding = nn.Embedding(onset_embeddings, self.slot_dim)
                self.slot_musical_binary_projection = _make_mlp(
                    6,
                    self.slot_dim,
                    self.slot_dim,
                    self.embedding_depth,
                    self.activation,
                )
                if self.slot_version != "slot7":
                    self.slot_musical_duration_embedding = nn.Embedding(147, self.slot_dim)
                    self.slot_musical_length_embedding = nn.Embedding(147, self.slot_dim)
                if self.musical_dim == 9:
                    self.slot_musical_compact_norm = nn.LayerNorm(self.slot_dim)
                    if self.musical_slot_fusion_mode == "mlp":
                        self.slot_musical_fusion = _make_mlp(
                            self.slot_dim * 4,
                            self.slot_dim,
                            self.slot_dim * 2,
                            self.embedding_depth,
                            self.activation,
                        )
            self.slot_null_embeddings = nn.Parameter(torch.zeros(self.slot_num, self.slot_dim))
            self.slot_mask_embeddings = nn.Parameter(torch.zeros(self.slot_num, self.slot_dim))
            self.slot_pad_embeddings = nn.Parameter(torch.zeros(self.slot_num, self.slot_dim))
            if bool(getattr(config, "zero_score_ioi_embedding", False)):
                self.slot_zero_score_ioi_embedding = nn.Embedding(2, self.slot_dim)
                nn.init.normal_(self.slot_zero_score_ioi_embedding.weight, mean=0.0, std=0.02)
            nn.init.normal_(self.slot_null_embeddings, mean=0.0, std=0.02)
            nn.init.normal_(self.slot_mask_embeddings, mean=0.0, std=0.02)
            nn.init.normal_(self.slot_pad_embeddings, mean=0.0, std=0.02)
            if self.slot_fusion_mode == "mlp":
                self.slot_fusion = _make_mlp(
                    self.slot_num * self.slot_dim,
                    config.hidden_size,
                    max(config.hidden_size * 2, self.slot_dim * 12),
                    max(2, self.embedding_depth),
                    self.activation,
                )
            elif self.slot_fusion_mode == "sum" and self.slot_dim != int(config.hidden_size):
                raise ValueError(
                    "slot_attribute sum fusion requires slot_dim == hidden_size, got "
                    f"{self.slot_dim} != {config.hidden_size}"
                )
            if bool(getattr(config, "musical_component_gates", False)) and self.musical_dim == 9:
                component_gate_init = _gate_logit_from_value(
                    getattr(config, "musical_component_gate_init", 1.0)
                )
                self.slot_musical_component_gate_logits = nn.Parameter(
                    torch.full((4,), component_gate_init)
                )
            if bool(getattr(config, "slot_gates", False)):
                gate_init = _gate_logit_from_value(getattr(config, "slot_gate_init", 1.0))
                self.slot_gate_logits = nn.Parameter(torch.full((self.slot_num,), gate_init))
                scope = str(getattr(config, "slot_gate_scope", "all") or "all").lower()
                if scope not in {"all", "musical_only"}:
                    raise ValueError(f"Unsupported slot_gate_scope={scope}; expected all or musical_only")
                if self.slot_version in {"slot6", "slot7", "slot9", "slot12"} and self.musical_dim > 0:
                    musical_base = 5 if self.slot_is_pt else 8
                    musical_logit = _gate_logit_from_value(getattr(config, "musical_gate_init", 1.0))
                    with torch.no_grad():
                        self.slot_gate_logits[musical_base:] = musical_logit
                    if scope == "musical_only":
                        self.slot_gate_mask = torch.zeros(self.slot_num, dtype=torch.bool)
                        self.slot_gate_mask[musical_base:] = True
                elif scope == "musical_only":
                    self.slot_gate_mask = torch.zeros(self.slot_num, dtype=torch.bool)
        self.norm = nn.LayerNorm(config.hidden_size)

    def _pitch_factors(self, pitch_ids):
        pitch_value = pitch_ids.long()
        pitch_min = int(getattr(self.config, "piano_pitch_min", 21))
        register_index = (pitch_value - pitch_min).clamp(min=0) // 12
        valid_register = (register_index >= 0) & (register_index < 8)
        register_index = register_index.clamp(0, 7)
        pitch_class = torch.remainder(pitch_value, 12).clamp(0, 11)
        valid_pitch = pitch_value != int(self.config.pitch_pad_id)
        pitch_class_one_hot = F.one_hot(pitch_class, num_classes=12).to(
            dtype=self.special_note_embeddings.weight.dtype,
            device=pitch_ids.device,
        )
        register_one_hot = F.one_hot(register_index, num_classes=8).to(
            dtype=pitch_class_one_hot.dtype,
            device=pitch_ids.device,
        )
        return torch.cat(
            [
                pitch_class_one_hot * valid_pitch.unsqueeze(-1).to(dtype=pitch_class_one_hot.dtype),
                register_one_hot * valid_register.unsqueeze(-1).to(dtype=register_one_hot.dtype),
            ],
            dim=-1,
        )

    def _split_inr0624(self, continuous):
        start = 0
        score_control = continuous[..., start : start + self.score_control_dim]
        start += self.score_control_dim
        performance_control = continuous[..., start : start + self.performance_control_dim]
        start += self.performance_control_dim
        musical = continuous[..., start : start + self.musical_dim]
        start += self.musical_dim
        masks = continuous[..., start : start + self.mask_dim]
        if continuous.shape[-1] != start + self.mask_dim:
            raise ValueError(
                f"Unexpected INR0624 continuous dim {continuous.shape[-1]}, expected {start + self.mask_dim}"
            )
        return score_control, performance_control, musical, masks

    def _split_perf_target(self, continuous):
        target_features = continuous[..., : self.decoder_target_dim]
        masks = continuous[..., self.decoder_target_dim : self.decoder_target_dim + self.mask_dim]
        if continuous.shape[-1] != self.decoder_target_dim + self.mask_dim:
            raise ValueError(
                f"Unexpected perf-target decoder dim {continuous.shape[-1]}, "
                f"expected {self.decoder_target_dim + self.mask_dim}"
            )
        return target_features, masks

    def _slot_mask_embedding(self, slot_index, reference):
        value = self.slot_mask_embeddings[slot_index].to(dtype=reference.dtype, device=reference.device)
        return value.view(*([1] * (reference.ndim - 1)), self.slot_dim)

    def _slot_null_embedding(self, slot_index, reference):
        value = self.slot_null_embeddings[slot_index].to(dtype=reference.dtype, device=reference.device)
        return value.view(*([1] * (reference.ndim - 1)), self.slot_dim)

    def _slot_pad_embedding(self, slot_index, reference):
        value = self.slot_pad_embeddings[slot_index].to(dtype=reference.dtype, device=reference.device)
        return value.view(*([1] * (reference.ndim - 1)), self.slot_dim)

    def _slot_encode(self, projection, features, mask, slot_index):
        encoded = projection(features)
        if mask is None:
            return encoded
        mask = mask.to(dtype=encoded.dtype, device=encoded.device)
        return encoded * mask + self._slot_mask_embedding(slot_index, encoded) * (1.0 - mask)

    def _dinr_slot_encode(self, features, mask, slot_index, field, role, velocity=False):
        if velocity:
            ids = torch.round(features[..., 0].float().clamp(0.0, 1.0) * 127.0).long()
            encoded = self.dinr_velocity_table.encode_ids(
                ids,
                include_numerical_coordinates=bool(
                    getattr(self.config, "dinr_input_velocity_numerical_coordinates", True)
                ),
            )
        else:
            coordinate = features[..., 0].float()
            ids = torch.round(
                coordinate / float(self.config.dinr_timing_step)
                + int(self.config.dinr_zero_bin)
            ).long().clamp(0, int(self.config.dinr_timing_bins) - 1)
            encoded = self.dinr_timing_table.encode_ids(
                ids,
                include_numerical_coordinates=bool(
                    getattr(self.config, "dinr_input_numerical_coordinates", True)
                ),
            )
        field_ids = torch.full_like(ids, int(field))
        role_ids = torch.full_like(ids, int(role))
        encoded = (
            encoded
            + self.dinr_field_embedding(field_ids)
            + self.dinr_role_embedding(role_ids)
        )
        if mask is None:
            return encoded
        mask = mask.to(dtype=encoded.dtype, device=encoded.device)
        return encoded * mask + self._slot_mask_embedding(slot_index, encoded) * (1.0 - mask)

    def _slot_value_encode(self, projection, features, mask, slot_index, field, role, velocity=False):
        if self.dinr_enabled:
            return self._dinr_slot_encode(
                features, mask, slot_index, field=field, role=role, velocity=velocity
            )
        return self._slot_encode(projection, features, mask, slot_index)

    def _slot_encode_with_missing(
        self,
        projection,
        features,
        mask,
        slot_index,
        missing_mask=None,
        missing_replacement="pad",
    ):
        encoded = self._slot_encode(projection, features, mask, slot_index)
        if missing_mask is None:
            return encoded
        missing = missing_mask.to(dtype=encoded.dtype, device=encoded.device).clamp(0.0, 1.0)
        if missing.shape[-1] != 1:
            missing = missing.amax(dim=-1, keepdim=True)
        replacement = str(missing_replacement or "pad").lower()
        if replacement == "mask":
            missing_embedding = self._slot_mask_embedding(slot_index, encoded)
        elif replacement == "pad":
            missing_embedding = self._slot_pad_embedding(slot_index, encoded)
        else:
            raise ValueError(f"Unsupported slot missing replacement: {missing_replacement}")
        return encoded * (1.0 - missing) + missing_embedding * missing

    def _slot_value_encode_with_missing(
        self,
        projection,
        features,
        mask,
        slot_index,
        field,
        role,
        velocity=False,
        missing_mask=None,
        missing_replacement="pad",
    ):
        encoded = self._slot_value_encode(
            projection, features, mask, slot_index, field=field, role=role, velocity=velocity
        )
        if missing_mask is None:
            return encoded
        missing = missing_mask.to(dtype=encoded.dtype, device=encoded.device).clamp(0.0, 1.0)
        if missing.shape[-1] != 1:
            missing = missing.amax(dim=-1, keepdim=True)
        replacement = str(missing_replacement or "pad").lower()
        missing_embedding = (
            self._slot_mask_embedding(slot_index, encoded)
            if replacement == "mask"
            else self._slot_pad_embedding(slot_index, encoded)
        )
        return encoded * (1.0 - missing) + missing_embedding * missing

    def _zero_score_ioi_condition(self, score_ioi, reference):
        if self.dinr_enabled or self.slot_zero_score_ioi_embedding is None:
            return reference.new_zeros(reference.shape)
        zero_bit = (
            score_ioi[..., 0].float().abs()
            <= float(getattr(self.config, "zero_ioi_support_eps", 1e-6))
        ).long()
        return self.slot_zero_score_ioi_embedding(zero_bit).to(
            dtype=reference.dtype,
            device=reference.device,
        )

    def _slot_apply_gate(self, slot_index, embedding):
        if self.slot_gate_logits is None:
            return embedding
        if self.slot_gate_mask is not None and not bool(self.slot_gate_mask[slot_index].item()):
            return embedding
        gate = torch.sigmoid(self.slot_gate_logits[slot_index]).to(dtype=embedding.dtype, device=embedding.device)
        return embedding * gate

    def _split_slot_performance(self, performance_control):
        timing = self.slot_timing_dim
        perf_ioi = performance_control[..., 0:timing]
        perf_duration = performance_control[..., timing : timing * 2]
        perf_velocity = performance_control[..., timing * 2 : timing * 2 + 1]
        pedal_start = timing * 2 + 1
        perf_pedal = performance_control[..., pedal_start : pedal_start + self.pedal_dim]
        return perf_ioi, perf_duration, perf_velocity, perf_pedal

    def _split_slot_score(self, score_control):
        timing = self.slot_timing_dim
        score_ioi = score_control[..., 0:timing]
        score_duration = score_control[..., timing : timing * 2]
        score_velocity = score_control[..., timing * 2 : timing * 2 + 1]
        return score_ioi, score_duration, score_velocity

    def _split_slot_musical145_onset_annotation(self, musical):
        if musical.shape[-1] < 151:
            raise ValueError(f"slot7 expects musical145_onset_annotation rows, got dim={musical.shape[-1]}")
        musical_onset = musical[..., 0:145]
        musical_binary = musical[..., 145:151]
        return musical_onset, musical_binary

    def _split_slot_musical4slot(self, musical):
        if musical.shape[-1] < 9:
            raise ValueError(
                "musical4slot expects rows "
                "[mo_idx, md_idx, ml_idx, staff, trill, grace, staccato, stem_up, stem_down], "
                f"got dim={musical.shape[-1]}"
            )
        mo_idx = musical[..., 0].float().round().long().clamp(0, 144)
        md_idx = musical[..., 1].float().round().long().clamp(0, 144)
        ml_idx = musical[..., 2].float().round().long().clamp(0, 144)
        annotation = musical[..., 3:9].float().clamp(0.0, 1.0)
        return mo_idx, md_idx, ml_idx, annotation

    def _musical4slot_embedding(self, musical, mask, slot_index):
        mo_idx, md_idx, ml_idx, annotation = self._split_slot_musical4slot(musical)
        active = (
            mask.squeeze(-1).to(device=mo_idx.device) > 0.5
            if mask is not None
            else torch.ones_like(mo_idx, dtype=torch.bool)
        )
        pad_id = 146
        no_value_id = 145
        mo_ids = torch.where(active, mo_idx, mo_idx.new_full(mo_idx.shape, pad_id))
        md_ids = torch.where(active, md_idx, md_idx.new_full(md_idx.shape, pad_id))
        ml_has_value = ml_idx > 0
        ml_ids = torch.where(ml_has_value, ml_idx, ml_idx.new_full(ml_idx.shape, no_value_id))
        ml_ids = torch.where(active, ml_ids, ml_idx.new_full(ml_idx.shape, pad_id))
        onset_embedding = self.slot_musical_onset_embedding(mo_ids).to(dtype=annotation.dtype)
        duration_embedding = self.slot_musical_duration_embedding(md_ids).to(dtype=annotation.dtype)
        length_embedding = self.slot_musical_length_embedding(ml_ids).to(dtype=annotation.dtype)
        annotation_embedding = self._slot_encode(
            self.slot_musical_binary_projection,
            annotation,
            mask,
            slot_index,
        )
        if self.slot_musical_component_gate_logits is not None:
            component_gates = torch.sigmoid(
                self.slot_musical_component_gate_logits.to(
                    dtype=annotation_embedding.dtype,
                    device=annotation_embedding.device,
                )
            )
            onset_embedding = onset_embedding * component_gates[0]
            duration_embedding = duration_embedding * component_gates[1]
            length_embedding = length_embedding * component_gates[2]
            annotation_embedding = annotation_embedding * component_gates[3]
        if self.slot_musical_fusion is not None and self.musical_slot_fusion_mode == "mlp":
            musical_slot = self.slot_musical_fusion(
                torch.cat(
                    [
                        onset_embedding,
                        duration_embedding,
                        length_embedding,
                        annotation_embedding,
                    ],
                    dim=-1,
                )
            )
        else:
            musical_slot = onset_embedding + duration_embedding + length_embedding + annotation_embedding
        if self.slot_musical_compact_norm is not None:
            musical_slot = self.slot_musical_compact_norm(musical_slot)
        return musical_slot

    def _categorical_slot_embedding(self, table, one_hot, scalar_features, scalar_projection, slot_index, mask):
        category = one_hot.float().argmax(dim=-1).clamp(0, table.num_embeddings - 3)
        no_value_id = table.num_embeddings - 2
        pad_id = table.num_embeddings - 1
        if mask is None:
            ids = category
            active = torch.ones_like(category, dtype=torch.bool)
        else:
            active = mask.squeeze(-1).to(device=category.device) > 0.5
            ids = torch.where(active, category, category.new_full(category.shape, pad_id))
        emb = table(ids).to(dtype=scalar_features.dtype)
        if scalar_projection is not None and scalar_features.shape[-1] > 0:
            emb = emb + scalar_projection(scalar_features) * active.unsqueeze(-1).to(dtype=emb.dtype)
        return emb

    def _category_only_slot_embedding(self, table, one_hot, slot_index, mask):
        category_count = table.num_embeddings - 2
        category_scores = one_hot[..., :category_count].float()
        category = category_scores.argmax(dim=-1).clamp(0, category_count - 1)
        has_value = category_scores.sum(dim=-1).to(device=category.device) > 0.5
        no_value_id = table.num_embeddings - 2
        pad_id = table.num_embeddings - 1
        ids = torch.where(has_value, category, category.new_full(category.shape, no_value_id))
        if mask is not None:
            active = mask.squeeze(-1).to(device=category.device) > 0.5
            ids = torch.where(active, ids, category.new_full(category.shape, pad_id))
        return table(ids).to(dtype=one_hot.dtype)

    def _musical_length_slot_embedding(self, one_hot, scalar, present, slot_index, mask):
        category = one_hot.float().argmax(dim=-1).clamp(0, self.slot_musical_length_embedding.num_embeddings - 3)
        no_value_id = self.slot_musical_length_embedding.num_embeddings - 2
        pad_id = self.slot_musical_length_embedding.num_embeddings - 1
        active = mask.squeeze(-1).to(device=category.device) > 0.5 if mask is not None else torch.ones_like(category, dtype=torch.bool)
        has_length = present.squeeze(-1).to(device=category.device) > 0.5
        ids = torch.where(has_length, category, category.new_full(category.shape, no_value_id))
        ids = torch.where(active, ids, category.new_full(category.shape, pad_id))
        residual_mask = (active & has_length).unsqueeze(-1).to(dtype=scalar.dtype)
        return self.slot_musical_length_embedding(ids).to(dtype=scalar.dtype) + self.slot_musical_length_scalar_projection(
            torch.cat([scalar, present], dim=-1)
        ) * residual_mask

    def _forward_slot_attribute(
        self,
        pitch_embeds,
        score_control,
        performance_control,
        musical,
        masks,
        performance_missing_mask=None,
        role=None,
    ):
        effective_role = str(role or self.role).lower()
        is_decoder = effective_role == "decoder"
        m_score = masks[..., 0:1] if self.mask_dim >= 1 else None
        m_perf = masks[..., 1:2] if self.mask_dim >= 2 else None
        m_musical = masks[..., 2:3] if self.mask_dim >= 3 else None

        score_ioi, score_duration, score_velocity = self._split_slot_score(score_control)
        perf_ioi, perf_duration, perf_velocity, perf_pedal = self._split_slot_performance(performance_control)
        missing_score = None
        missing_perf = None
        missing_replacement = str(
            getattr(self.config, "prior_property_dropout_replacement", "pad") or "pad"
        ).lower()
        if performance_missing_mask is not None:
            missing_perf = self._split_slot_performance(performance_missing_mask)

        if self.slot_is_pt:
            value_mask = m_perf if is_decoder else m_score
            ioi_projection = self.slot_perf_ioi_projection if is_decoder else self.slot_score_ioi_projection
            duration_projection = (
                self.slot_perf_duration_projection if is_decoder else self.slot_score_duration_projection
            )
            velocity_projection = (
                self.slot_perf_velocity_projection if is_decoder else self.slot_score_velocity_projection
            )
            ioi_features = perf_ioi if is_decoder else score_ioi
            duration_features = perf_duration if is_decoder else score_duration
            velocity_features = perf_velocity if is_decoder else score_velocity
            pedal_mask = m_perf if is_decoder else None
            value_role = 1 if is_decoder else 0
            pedal_embedding = (
                self._slot_encode_with_missing(
                    self.slot_perf_pedal_projection,
                    perf_pedal,
                    pedal_mask,
                    4,
                    missing_perf[3] if is_decoder and missing_perf is not None else None,
                    missing_replacement,
                )
                if is_decoder
                else self._slot_mask_embedding(4, pitch_embeds).expand_as(pitch_embeds)
            )
            ioi_embedding = self._slot_value_encode_with_missing(
                ioi_projection,
                ioi_features,
                value_mask,
                1,
                field=0,
                role=value_role,
                missing_mask=missing_perf[0] if is_decoder and missing_perf is not None else None,
                missing_replacement=missing_replacement,
            )
            ioi_embedding = ioi_embedding + self._zero_score_ioi_condition(score_ioi, ioi_embedding)
            slot_embeddings = [
                self._slot_apply_gate(0, pitch_embeds),
                self._slot_apply_gate(1, ioi_embedding),
                self._slot_apply_gate(2, self._slot_value_encode_with_missing(duration_projection, duration_features, value_mask, 2, field=1, role=value_role, missing_mask=missing_perf[1] if is_decoder and missing_perf is not None else None, missing_replacement=missing_replacement)),
                self._slot_apply_gate(3, self._slot_value_encode_with_missing(velocity_projection, velocity_features, value_mask, 3, field=0, role=value_role, velocity=True, missing_mask=missing_perf[2] if is_decoder and missing_perf is not None else None, missing_replacement=missing_replacement)),
                self._slot_apply_gate(4, pedal_embedding),
            ]
        else:
            if not is_decoder:
                perf_slots = [
                    self._slot_null_embedding(slot_index, pitch_embeds).expand_as(pitch_embeds)
                    for slot_index in range(4, 8)
                ]
            else:
                perf_slots = [
                    self._slot_value_encode_with_missing(self.slot_perf_ioi_projection, perf_ioi, m_perf, 4, field=0, role=1, missing_mask=missing_perf[0] if missing_perf is not None else None, missing_replacement=missing_replacement),
                    self._slot_value_encode_with_missing(self.slot_perf_duration_projection, perf_duration, m_perf, 5, field=1, role=1, missing_mask=missing_perf[1] if missing_perf is not None else None, missing_replacement=missing_replacement),
                    self._slot_value_encode_with_missing(self.slot_perf_velocity_projection, perf_velocity, m_perf, 6, field=0, role=1, velocity=True, missing_mask=missing_perf[2] if missing_perf is not None else None, missing_replacement=missing_replacement),
                    self._slot_encode_with_missing(self.slot_perf_pedal_projection, perf_pedal, m_perf, 7, missing_perf[3] if missing_perf is not None else None, missing_replacement),
                ]
            score_ioi_embedding = self._slot_value_encode(self.slot_score_ioi_projection, score_ioi, m_score, 1, field=0, role=0)
            score_ioi_embedding = score_ioi_embedding + self._zero_score_ioi_condition(score_ioi, score_ioi_embedding)
            slot_embeddings = [
                self._slot_apply_gate(0, pitch_embeds),
                self._slot_apply_gate(1, score_ioi_embedding),
                self._slot_apply_gate(2, self._slot_value_encode(self.slot_score_duration_projection, score_duration, m_score, 2, field=1, role=0)),
                self._slot_apply_gate(3, self._slot_value_encode(self.slot_score_velocity_projection, score_velocity, m_score, 3, field=0, role=0, velocity=True)),
                self._slot_apply_gate(4, perf_slots[0]),
                self._slot_apply_gate(5, perf_slots[1]),
                self._slot_apply_gate(6, perf_slots[2]),
                self._slot_apply_gate(7, perf_slots[3]),
            ]
        if self.slot_version in {"slot6", "slot7", "slot9", "slot12"}:
            musical_base = len(slot_embeddings)
            if self.musical_dim == 9:
                if is_decoder:
                    musical_slot = self._slot_mask_embedding(musical_base, pitch_embeds).expand_as(pitch_embeds)
                else:
                    musical_slot = self._musical4slot_embedding(musical, m_musical, musical_base)
                if self.slot_version == "slot6":
                    slot_embeddings.append(self._slot_apply_gate(musical_base, musical_slot))
                else:
                    slot_embeddings.extend(
                        [
                            self._slot_apply_gate(musical_base, musical_slot),
                            self._slot_apply_gate(
                                musical_base + 1,
                                self._slot_mask_embedding(musical_base + 1, pitch_embeds).expand_as(pitch_embeds),
                            ),
                            self._slot_apply_gate(
                                musical_base + 2,
                                self._slot_mask_embedding(musical_base + 2, pitch_embeds).expand_as(pitch_embeds),
                            ),
                            self._slot_apply_gate(
                                musical_base + 3,
                                self._slot_mask_embedding(musical_base + 3, pitch_embeds).expand_as(pitch_embeds),
                            ),
                        ]
                    )
                fused = torch.cat(slot_embeddings, dim=-1)
                if self.slot_fusion_mode == "sum":
                    return torch.stack(slot_embeddings, dim=0).sum(dim=0)
                if self.slot_fusion is None:
                    return fused
                return self.slot_fusion(fused)
            if self.musical_dim > 0:
                raise ValueError(
                    f"{self.slot_version} only supports musical_feature_dim=9 for ASAP_processed musical slots"
                )
            musical_slot = self._slot_mask_embedding(musical_base, pitch_embeds).expand_as(pitch_embeds)
            if self.slot_version == "slot6":
                slot_embeddings.append(self._slot_apply_gate(musical_base, musical_slot))
            else:
                slot_embeddings.extend(
                    [
                        self._slot_apply_gate(musical_base, musical_slot),
                        self._slot_apply_gate(
                            musical_base + 1,
                            self._slot_mask_embedding(musical_base + 1, pitch_embeds).expand_as(pitch_embeds),
                        ),
                        self._slot_apply_gate(
                            musical_base + 2,
                            self._slot_mask_embedding(musical_base + 2, pitch_embeds).expand_as(pitch_embeds),
                        ),
                        self._slot_apply_gate(
                            musical_base + 3,
                            self._slot_mask_embedding(musical_base + 3, pitch_embeds).expand_as(pitch_embeds),
                        ),
                    ]
                )
        fused = torch.cat(slot_embeddings, dim=-1)
        if self.slot_fusion_mode == "sum":
            return torch.stack(slot_embeddings, dim=0).sum(dim=0)
        if self.slot_fusion is None:
            return fused
        return self.slot_fusion(fused)

    def forward(self, pitch_ids, continuous, special_note_ids=None, performance_missing_mask=None, role=None):
        effective_role = str(role or self.role).lower()
        try:
            projection_dtype = next(self.parameters()).dtype
        except StopIteration:
            projection_dtype = continuous.dtype
        continuous = continuous.to(dtype=projection_dtype)
        pitch_factors = self._pitch_factors(pitch_ids).to(dtype=projection_dtype)
        if self.slot_pitch_embedding is not None:
            pitch_embeds = self.slot_pitch_embedding(
                pitch_ids.long().clamp(0, self.slot_pitch_embedding.num_embeddings - 1)
            ).to(dtype=projection_dtype)
        else:
            pitch_embeds = self.pitch_projection(pitch_factors)

        if effective_role == "decoder" and self.schema == "perf_target":
            target_features, masks = self._split_perf_target(continuous)
            target_embeds = self.decoder_target_projection(target_features)
            mask_embeds = self.mask_projection(masks)
            embeddings = pitch_embeds + target_embeds + mask_embeds
            embeddings = self.norm(embeddings)
            return self._apply_special_embeddings(embeddings, special_note_ids)

        score_control, performance_control, musical, masks = self._split_inr0624(continuous)
        if performance_missing_mask is not None:
            if self.performance_missing_embeddings is None:
                raise ValueError("performance_missing_mask is only supported for decoder note inputs")
            performance_missing_mask = performance_missing_mask.to(
                dtype=projection_dtype,
                device=performance_control.device,
            )
            if performance_missing_mask.shape != performance_control.shape:
                raise ValueError(
                    "performance_missing_mask shape mismatch: "
                    f"got {tuple(performance_missing_mask.shape)}, expected {tuple(performance_control.shape)}"
                )
            performance_missing_mask = performance_missing_mask.clamp(0.0, 1.0)
            performance_control = performance_control * (1.0 - performance_missing_mask)
        if self.mode == "cine":
            embeddings = self.continuous_mlp(
                torch.cat([pitch_factors, score_control, performance_control, musical, masks], dim=-1)
            )
        elif self.mode == "slot_attribute":
            embeddings = self._forward_slot_attribute(
                pitch_embeds,
                score_control,
                performance_control,
                musical,
                masks,
                performance_missing_mask=performance_missing_mask,
                role=effective_role,
            )
        else:
            m_score_control = masks[..., 0:1]
            m_performance_control = masks[..., 1:2]
            score_control_embeds = self.score_control_projection(score_control) * m_score_control
            performance_control_embeds = self.performance_control_projection(performance_control) * m_performance_control
            musical_embeds = score_control_embeds.new_zeros(score_control_embeds.shape)
            if self.musical_projection is not None:
                musical_embeds = self.musical_projection(musical) * masks[..., 2:3]
            mask_embeds = self.mask_projection(masks)
            if self.additive_embedding_gate_logits is not None:
                additive_gates = torch.sigmoid(
                    self.additive_embedding_gate_logits.to(
                        dtype=pitch_embeds.dtype,
                        device=pitch_embeds.device,
                    )
                )
                pitch_embeds = pitch_embeds * additive_gates[0]
                score_control_embeds = score_control_embeds * additive_gates[1]
                performance_control_embeds = performance_control_embeds * additive_gates[2]
                musical_embeds = musical_embeds * additive_gates[3]
                mask_embeds = mask_embeds * additive_gates[4]
            embeddings = (
                pitch_embeds
                + score_control_embeds
                + performance_control_embeds
                + musical_embeds
                + mask_embeds
            )
        if performance_missing_mask is not None and self.mode != "slot_attribute":
            missing_embeds = torch.matmul(
                performance_missing_mask,
                self.performance_missing_embeddings.to(dtype=projection_dtype),
            )
            embeddings = embeddings + missing_embeds
        embeddings = self.norm(embeddings)
        return self._apply_special_embeddings(embeddings, special_note_ids)

    def pad_embedding(self):
        pad_id = int(self.config.special_note_ids.get("pad", 0))
        return self.special_note_embeddings.weight[pad_id]

    def mask_embedding(self):
        mask_id = int(self.config.special_note_ids.get("mask", 1))
        return self.special_note_embeddings.weight[mask_id]

    def _apply_special_embeddings(self, embeddings, special_note_ids):
        if special_note_ids is None:
            return embeddings
        special_mask = special_note_ids >= 0
        if not special_mask.any():
            return embeddings
        safe_ids = special_note_ids.clamp_min(0)
        special_embeds = self.special_note_embeddings(safe_ids).to(dtype=embeddings.dtype)
        return torch.where(special_mask.unsqueeze(-1), special_embeds, embeddings)


class IntegratedStyleTokenEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        hidden_size = int(config.hidden_size)
        activation = getattr(config, "head_activation", "gelu")
        depth = getattr(config, "embedding_depth", 2)
        self.creator_embedding = nn.Embedding(
            max(1, int(getattr(config, "style_creator_vocab_size", 1))),
            hidden_size,
        )
        self.source_embedding = nn.Embedding(
            max(1, int(getattr(config, "style_source_vocab_size", 1))),
            hidden_size,
        )
        self.score_projection = _make_mlp(
            max(1, int(getattr(config, "style_score_stat_dim", 18))),
            hidden_size,
            hidden_size,
            depth=depth,
            activation=activation,
        )
        self.perf_projection = _make_mlp(
            max(1, int(getattr(config, "style_perf_stat_dim", 18))),
            hidden_size,
            hidden_size,
            depth=depth,
            activation=activation,
        )
        self.perf_pad_embedding = nn.Embedding(1, hidden_size)
        self.token_type_embedding = nn.Embedding(4, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size * 4, hidden_size),
            _activation(activation),
            nn.LayerNorm(hidden_size),
        )
        self.film = nn.Linear(hidden_size, hidden_size * 2)

    def forward(
        self,
        creator_ids,
        source_ids,
        score_stats,
        perf_stats,
        perf_is_pad=None,
    ):
        projection_dtype = self.creator_embedding.weight.dtype
        creator_ids = creator_ids.long().clamp(0, self.creator_embedding.num_embeddings - 1)
        source_ids = source_ids.long().clamp(0, self.source_embedding.num_embeddings - 1)
        score_stats = score_stats.to(dtype=projection_dtype)
        perf_stats = perf_stats.to(dtype=projection_dtype)
        creator = self.creator_embedding(creator_ids)
        source = self.source_embedding(source_ids)
        score = self.score_projection(score_stats)
        perf = self.perf_projection(perf_stats)
        if perf_is_pad is not None:
            pad = self.perf_pad_embedding(
                torch.zeros_like(creator_ids, dtype=torch.long, device=creator_ids.device)
            ).to(dtype=projection_dtype)
            perf = torch.where(perf_is_pad.bool().unsqueeze(-1), pad, perf)
        style_tokens = torch.stack([creator, source, score, perf], dim=1)
        token_types = torch.arange(4, device=style_tokens.device).unsqueeze(0)
        style_tokens = style_tokens + self.token_type_embedding(token_types).to(dtype=style_tokens.dtype)
        return self.norm(style_tokens)

    def summary(
        self,
        creator_ids,
        source_ids,
        score_stats,
        perf_stats,
        perf_is_pad=None,
    ):
        style_tokens = self.forward(
            creator_ids,
            source_ids,
            score_stats,
            perf_stats,
            perf_is_pad=perf_is_pad,
        )
        return self.fusion(style_tokens.reshape(style_tokens.shape[0], -1))

    def apply_to_notes(self, note_embeds, style_vec, mode):
        mode = str(mode).lower()
        styled = note_embeds
        if "add" in mode:
            styled = styled + style_vec.unsqueeze(1).to(dtype=styled.dtype)
        if "film" in mode:
            gamma, beta = self.film(style_vec).chunk(2, dim=-1)
            gamma = 0.1 * torch.tanh(gamma).unsqueeze(1).to(dtype=styled.dtype)
            beta = beta.unsqueeze(1).to(dtype=styled.dtype)
            styled = styled * (1.0 + gamma) + beta
        return styled


class IntegratedStyleTokenMixin:
    def _style_tokens_enabled(self):
        return bool(getattr(self.config, "use_style_tokens", False))

    def _style_integration_mode(self):
        return str(getattr(self.config, "style_integration_mode", "prepend") or "prepend").lower()

    def _style_should_prepend(self):
        return self._style_tokens_enabled() and self._style_integration_mode().startswith("prepend")

    def _style_should_apply_to_notes(self):
        mode = self._style_integration_mode()
        note_modes = {"add", "film", "add_film", "prepend_add", "prepend_film"}
        return self._style_tokens_enabled() and mode in note_modes

    def _style_should_apply_to_decoder_inputs(self):
        mode = self._style_integration_mode()
        decoder_input_modes = {"dec_add", "dec_add_film", "prepend_dec_add", "prepend_dec_add_film"}
        return self._style_tokens_enabled() and mode in decoder_input_modes

    def _style_should_apply_to_decoder_hidden(self):
        mode = self._style_integration_mode()
        decoder_hidden_modes = {"dec_film", "dec_add_film", "prepend_dec_film", "prepend_dec_add_film"}
        return self._style_tokens_enabled() and mode in decoder_hidden_modes

    def _build_style_tokens(
        self,
        style_creator_ids=None,
        style_source_ids=None,
        style_score_stats=None,
        style_perf_stats=None,
        style_perf_is_pad=None,
    ):
        if not self._style_tokens_enabled():
            return None
        missing = [
            name
            for name, value in (
                ("style_creator_ids", style_creator_ids),
                ("style_source_ids", style_source_ids),
                ("style_score_stats", style_score_stats),
                ("style_perf_stats", style_perf_stats),
            )
            if value is None
        ]
        if missing:
            raise ValueError(f"Style tokens are enabled but missing inputs: {missing}")
        return self.style_token_encoder(
            style_creator_ids,
            style_source_ids,
            style_score_stats,
            style_perf_stats,
            perf_is_pad=style_perf_is_pad,
        )

    def _prepend_style_tokens(self, note_embeds, attention_mask, **style_kwargs):
        if not self._style_should_prepend():
            return note_embeds, attention_mask, 0
        style_tokens = self._build_style_tokens(**style_kwargs)
        if style_tokens is None:
            return note_embeds, attention_mask, 0
        style_mask = attention_mask.new_ones((attention_mask.shape[0], style_tokens.shape[1]))
        return (
            torch.cat([style_tokens.to(dtype=note_embeds.dtype), note_embeds], dim=1),
            torch.cat([style_mask, attention_mask], dim=1),
            style_tokens.shape[1],
        )

    def _apply_style_to_note_embeds(self, note_embeds, **style_kwargs):
        if not self._style_should_apply_to_notes():
            return note_embeds
        missing = [
            name
            for name, value in style_kwargs.items()
            if name != "style_perf_is_pad" and value is None
        ]
        if missing:
            raise ValueError(f"Style note conditioning is enabled but missing inputs: {missing}")
        style_vec = self.style_token_encoder.summary(
            style_kwargs["style_creator_ids"],
            style_kwargs["style_source_ids"],
            style_kwargs["style_score_stats"],
            style_kwargs["style_perf_stats"],
            perf_is_pad=style_kwargs.get("style_perf_is_pad"),
        )
        return self.style_token_encoder.apply_to_notes(
            note_embeds,
            style_vec,
            self._style_integration_mode(),
        )

    def _style_summary_vector(self, **style_kwargs):
        missing = [
            name
            for name, value in style_kwargs.items()
            if name != "style_perf_is_pad" and value is None
        ]
        if missing:
            raise ValueError(f"Style conditioning is enabled but missing inputs: {missing}")
        return self.style_token_encoder.summary(
            style_kwargs["style_creator_ids"],
            style_kwargs["style_source_ids"],
            style_kwargs["style_score_stats"],
            style_kwargs["style_perf_stats"],
            perf_is_pad=style_kwargs.get("style_perf_is_pad"),
        )

    def _apply_style_to_decoder_inputs(self, decoder_inputs_embeds, **style_kwargs):
        if not self._style_should_apply_to_decoder_inputs():
            return decoder_inputs_embeds
        style_vec = self._style_summary_vector(**style_kwargs)
        return self.style_token_encoder.apply_to_notes(decoder_inputs_embeds, style_vec, "add")

    def _apply_style_to_decoder_hidden(self, decoder_hidden, **style_kwargs):
        if not self._style_should_apply_to_decoder_hidden():
            return decoder_hidden
        style_vec = self._style_summary_vector(**style_kwargs)
        return self.style_token_encoder.apply_to_notes(decoder_hidden, style_vec, "film")

    def _apply_zero_ioi_residual(self, raw_outputs, score_shared_raw):
        residual = getattr(self.continuous_decoder, "zero_ioi_residual", None)
        if residual is None or score_shared_raw is None:
            return raw_outputs
        if not (
            _uses_epr_targets(self.config)
            and str(getattr(self.config, "epr_distribution", "point")).lower() in SN_DISTRIBUTIONS
        ):
            return raw_outputs
        zero_mask = _zero_score_ioi_mask(self.config, score_shared_raw)
        if zero_mask is None:
            return raw_outputs
        mask = zero_mask.to(dtype=raw_outputs.dtype, device=raw_outputs.device).unsqueeze(-1)
        residual = residual.to(dtype=raw_outputs.dtype, device=raw_outputs.device)
        adjusted = raw_outputs.clone()
        timing_feature_dim = (
            3 * _timing_distribution_count(self.config)
            + (1 if _uses_raw_timing_regression_head(self.config) else 0)
        )
        targets = tuple(getattr(self.config, "zero_ioi_residual_targets", ("ioi",)))
        cursor = 0
        for target in targets:
            update = residual[cursor : cursor + 3].view(*([1] * (raw_outputs.ndim - 1)), 3)
            cursor += 3
            if target == "ioi":
                start = 0
            elif target == "duration":
                start = timing_feature_dim
            else:
                continue
            end = start + 3
            adjusted[..., start:end] = adjusted[..., start:end] + mask * update
        return adjusted


class DINRValueTable(nn.Module):
    """Categorical identities augmented with an explicit numerical coordinate."""

    def __init__(self, coordinates, dim, frequencies=16):
        super().__init__()
        coordinates = torch.as_tensor(coordinates, dtype=torch.float32)
        self.register_buffer("coordinates", coordinates, persistent=True)
        self.lookup = nn.Embedding(int(coordinates.numel()), int(dim))
        self.frequencies = max(1, int(frequencies))
        self.numeric_projection = nn.Linear(1 + 2 * self.frequencies, int(dim), bias=False)

    def numerical_features(self, values):
        values = values.float().unsqueeze(-1)
        frequencies = torch.arange(
            1,
            self.frequencies + 1,
            device=values.device,
            dtype=values.dtype,
        )
        angles = values * frequencies * math.pi
        return torch.cat([values, torch.sin(angles), torch.cos(angles)], dim=-1)

    def encode_ids(self, ids, include_numerical_coordinates=True):
        ids = ids.long().clamp(0, self.lookup.num_embeddings - 1)
        if not include_numerical_coordinates:
            return self.lookup(ids)
        coordinates = self.coordinates.to(device=ids.device)[ids]
        return self.lookup(ids) + self.numeric_projection(self.numerical_features(coordinates))

    def prototypes(self, include_numerical_coordinates=True):
        coordinates = self.coordinates.to(device=self.lookup.weight.device)
        if not include_numerical_coordinates:
            return self.lookup.weight
        return self.lookup.weight + self.numeric_projection(self.numerical_features(coordinates))


class DINRQueryHead(nn.Module):
    def __init__(
        self,
        input_dim,
        prototype_dim,
        output_bins,
        hidden_dim,
        depth,
        activation,
        value_table,
        use_numerical_coordinates=True,
    ):
        super().__init__()
        self.query = _make_decoder_head(
            input_dim,
            prototype_dim,
            hidden_dim,
            depth=depth,
            activation=activation,
            layout="mlp",
            expand_ratio=1.0,
            shrink_ratio=1.0,
        )
        self.bias = nn.Parameter(torch.zeros(int(output_bins)))
        self.value_table = value_table
        self.use_numerical_coordinates = bool(use_numerical_coordinates)
        self.alternate_value_table = None
        self.alternate_bias = None
        self.alternate_use_numerical_coordinates = True

    def forward(self, hidden_states, use_alternate=None):
        query = self.query(hidden_states)
        prototypes = self.value_table.prototypes(
            include_numerical_coordinates=self.use_numerical_coordinates
        ).to(dtype=query.dtype, device=query.device)
        logits = torch.matmul(query, prototypes.transpose(0, 1)) + self.bias.to(dtype=query.dtype)
        if use_alternate is None or self.alternate_value_table is None:
            return logits
        alternate_prototypes = self.alternate_value_table.prototypes(
            include_numerical_coordinates=self.alternate_use_numerical_coordinates
        ).to(dtype=query.dtype, device=query.device)
        alternate_logits = torch.matmul(query, alternate_prototypes.transpose(0, 1))
        alternate_logits = alternate_logits + self.alternate_bias.to(dtype=query.dtype)
        return torch.where(use_alternate.unsqueeze(-1), alternate_logits, logits)


class IntegratedContinuousDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.output_dim = config.output_continuous_dim
        self.epr_distribution = getattr(config, "epr_distribution", "point").lower()
        self.pedal_distribution = getattr(config, "pedal_distribution", self.epr_distribution).lower()
        self.pedal_representation = normalize_pedal_representation(getattr(config, "pedal_representation", "binary_4"))
        self.pedal_dim = pedal_representation_dim(self.pedal_representation)
        head_depth = getattr(config, "head_depth", 2)
        head_width_multiplier = float(getattr(config, "head_width_multiplier", 1.0))
        activation = getattr(config, "head_activation", "gelu")
        decoder_head_layout = getattr(config, "decoder_head_layout", "pyramid4")
        decoder_head_expand_ratio = float(getattr(config, "decoder_head_expand_ratio", 2.0))
        decoder_head_shrink_ratio = float(getattr(config, "decoder_head_shrink_ratio", 0.5))
        full_dim = config.hidden_size
        head_hidden_dim = max(1, int(round(full_dim * head_width_multiplier)))
        self.shared_slice = self.score_slice = self.perf_slice = slice(None)
        shared_dim = score_dim = perf_dim = full_dim
        shared_pack_mode = "concat"
        if (
            getattr(config, "task_type", "epr") == "epr"
            and self.epr_distribution in DINR_DISTRIBUTIONS
        ):
            ioi_output_dim = int(config.dinr_output_timing_bins)
            duration_output_dim = int(config.dinr_output_timing_bins)
            velocity_output_dim = 128
            pedal_output_dim = self.pedal_dim
        elif (
            getattr(config, "task_type", "epr") == "epr"
            and self.epr_distribution in {"categorical", "hard_categorical", "soft_categorical"}
        ):
            ioi_output_dim = int(config.epr_timing_bins)
            duration_output_dim = int(config.epr_timing_bins)
            velocity_output_dim = int(config.epr_value_bins)
            pedal_output_dim = int(config.epr_value_bins) * 4
            if self.pedal_representation == "binary_4":
                pedal_output_dim = self.pedal_dim
        elif getattr(config, "task_type", "epr") == "epr" and self.epr_distribution == "beta_mu_kappa":
            ioi_output_dim = duration_output_dim = velocity_output_dim = 2
            shared_pack_mode = "beta_mu_kappa"
            pedal_output_dim = self.pedal_dim
        elif (
            getattr(config, "task_type", "epr") == "epr"
            and _is_scalar_distribution(self.epr_distribution)
        ):
            components = _scalar_distribution_components(config, self.epr_distribution)
            if components < 1:
                raise ValueError(f"epr_mixture_components must be >= 1, got {components}")
            per_component_dim = _scalar_distribution_dim(self.epr_distribution)
            per_feature_dim = components * per_component_dim
            if self.epr_distribution in SN_DISTRIBUTIONS:
                dual_timing = str(
                    getattr(config, "zero_ioi_dual_distribution_mode", "none") or "none"
                ).lower() not in {"none", "off", "false", "0"}
                base_timing_count = _timing_distribution_count(config)
                ioi_timing_count = base_timing_count + (1 if dual_timing else 0)
                duration_timing_count = base_timing_count + (
                    1 if dual_timing and bool(getattr(config, "zero_ioi_dual_duration", True)) else 0
                )
                ioi_output_dim = (
                    per_feature_dim * ioi_timing_count
                    + (1 if _uses_raw_timing_regression_head(config) else 0)
                )
                duration_output_dim = (
                    per_feature_dim * duration_timing_count
                    + (1 if _uses_raw_timing_regression_head(config) else 0)
                )
                velocity_output_dim = per_feature_dim
            else:
                ioi_output_dim = duration_output_dim = velocity_output_dim = per_feature_dim
            if self.epr_distribution in ALN_DISTRIBUTIONS:
                # ALN is timing-only split-normal with one component.
                velocity_output_dim = components * 3
            pedal_components = _scalar_distribution_components(config, self.pedal_distribution)
            pedal_output_dim = pedal_components * _scalar_distribution_dim(self.pedal_distribution) * 4
            if self.pedal_representation == "binary_4":
                pedal_output_dim = self.pedal_dim
        elif (
            getattr(config, "task_type", "epr") == "epr"
            and self.epr_distribution in DLM_DISTRIBUTIONS
        ):
            components = int(getattr(config, "dlm_components", getattr(config, "epr_mixture_components", 8)))
            if components < 1:
                raise ValueError(f"dlm_components must be >= 1, got {components}")
            per_feature_dim = components * 3
            ioi_output_dim = per_feature_dim + (1 if bool(getattr(config, "dlm_ioi_zero_inflated", False)) else 0)
            duration_output_dim = per_feature_dim
            velocity_distribution = str(getattr(config, "velocity_distribution", "skew_normal")).lower()
            velocity_output_dim = per_feature_dim if velocity_distribution in DLM_DISTRIBUTIONS else 3
            pedal_output_dim = self.pedal_dim
        else:
            ioi_output_dim = duration_output_dim = velocity_output_dim = 1
            pedal_output_dim = self.pedal_dim

        generic_output_dim = self.output_dim
        if getattr(config, "task_type", "epr") == "removed_task" and _removed_task_uses_grid_head(config):
            generic_output_dim = _removed_task_grid_raw_output_dim(config)

        self.split_shared_heads = getattr(config, "task_type", "epr") == "epr"
        self.shared_pack_mode = shared_pack_mode
        self.zero_timing_head_condition = None
        if (
            getattr(config, "task_type", "epr") == "epr"
            and bool(getattr(config, "zero_timing_head_condition", False))
        ):
            self.zero_timing_head_condition = nn.Embedding(2, shared_dim)
            nn.init.normal_(self.zero_timing_head_condition.weight, mean=0.0, std=0.02)
        self.zero_ioi_residual = None
        if (
            getattr(config, "task_type", "epr") == "epr"
            and self.epr_distribution in SN_DISTRIBUTIONS
            and bool(getattr(config, "zero_ioi_residual", False))
        ):
            residual_targets = tuple(getattr(config, "zero_ioi_residual_targets", ("ioi",)))
            self.zero_ioi_residual = nn.Parameter(torch.zeros(3 * len(residual_targets)))
        self.dinr_timing_table = None
        self.dinr_duration_table = None
        self.dinr_absolute_timing_table = None
        self.dinr_velocity_table = None
        if self.split_shared_heads and self.epr_distribution in DINR_DISTRIBUTIONS:
            timing_coordinates = (
                torch.arange(int(config.dinr_output_timing_bins), dtype=torch.float32)
                - int(config.dinr_output_zero_bin)
            ) * float(config.dinr_output_timing_step)
            velocity_coordinates = torch.arange(128, dtype=torch.float32) / 127.0
            prototype_dim = int(getattr(config, "slot_dim", None) or 128)
            frequencies = int(getattr(config, "dinr_numerical_frequencies", 16))
            self.dinr_timing_table = DINRValueTable(
                timing_coordinates, prototype_dim, frequencies=frequencies
            )
            if bool(getattr(config, "dinr_separate_timing_tables", False)):
                self.dinr_duration_table = DINRValueTable(
                    timing_coordinates, prototype_dim, frequencies=frequencies
                )
            separated_vocabulary = str(getattr(config, "dinr_vocabulary_mode", "unified")).lower() == "separated"
            if separated_vocabulary:
                absolute_coordinates = (
                    torch.arange(int(config.dinr_timing_bins), dtype=torch.float32)
                    - int(config.dinr_zero_bin)
                ) * float(config.dinr_timing_step)
                if int(config.dinr_timing_bins) != int(config.dinr_output_timing_bins):
                    raise ValueError("Separated DINR currently requires equal absolute/deviation vocabulary sizes")
                self.dinr_absolute_timing_table = DINRValueTable(
                    absolute_coordinates, prototype_dim, frequencies=frequencies
                )
            self.dinr_velocity_table = DINRValueTable(
                velocity_coordinates, prototype_dim, frequencies=frequencies
            )
            head_args = dict(
                input_dim=shared_dim,
                prototype_dim=prototype_dim,
                hidden_dim=head_hidden_dim,
                depth=head_depth,
                activation=activation,
            )
            self.ioi_head = DINRQueryHead(
                output_bins=int(config.dinr_output_timing_bins),
                value_table=self.dinr_timing_table,
                use_numerical_coordinates=bool(
                    getattr(config, "dinr_output_deviation_numerical_coordinates", True)
                ),
                **head_args,
            )
            if separated_vocabulary:
                self.ioi_head.alternate_value_table = self.dinr_absolute_timing_table
                self.ioi_head.alternate_bias = nn.Parameter(torch.zeros(int(config.dinr_timing_bins)))
            self.duration_head = DINRQueryHead(
                output_bins=int(config.dinr_output_timing_bins),
                value_table=self.dinr_duration_table or self.dinr_timing_table,
                use_numerical_coordinates=bool(
                    getattr(config, "dinr_output_deviation_numerical_coordinates", True)
                ),
                **head_args,
            )
            self.velocity_head = DINRQueryHead(
                output_bins=128,
                value_table=self.dinr_velocity_table,
                use_numerical_coordinates=bool(
                    getattr(config, "dinr_velocity_numerical_coordinates", True)
                ),
                **head_args,
            )
            self.shared_extra_head = None
        elif self.split_shared_heads:
            self.ioi_head = _make_decoder_head(
                shared_dim,
                ioi_output_dim,
                head_hidden_dim,
                depth=head_depth,
                activation=activation,
                layout=decoder_head_layout,
                expand_ratio=decoder_head_expand_ratio,
                shrink_ratio=decoder_head_shrink_ratio,
            )
            self.duration_head = _make_decoder_head(
                shared_dim,
                duration_output_dim,
                head_hidden_dim,
                depth=head_depth,
                activation=activation,
                layout=decoder_head_layout,
                expand_ratio=decoder_head_expand_ratio,
                shrink_ratio=decoder_head_shrink_ratio,
            )
            self.velocity_head = _make_decoder_head(
                shared_dim,
                velocity_output_dim,
                head_hidden_dim,
                depth=head_depth,
                activation=activation,
                layout=decoder_head_layout,
                expand_ratio=decoder_head_expand_ratio,
                shrink_ratio=decoder_head_shrink_ratio,
            )
            self.shared_extra_head = (
                None
            )
        else:
            shared_output_dim = ioi_output_dim + duration_output_dim + velocity_output_dim
            self.shared_head = _make_decoder_head(
                shared_dim,
                shared_output_dim,
                head_hidden_dim,
                depth=head_depth,
                activation=activation,
                layout=decoder_head_layout,
                expand_ratio=decoder_head_expand_ratio,
                shrink_ratio=decoder_head_shrink_ratio,
            )
        self.pedal_head = _make_decoder_head(
            perf_dim,
            pedal_output_dim,
            head_hidden_dim,
            depth=head_depth,
            activation=activation,
            layout=decoder_head_layout,
            expand_ratio=decoder_head_expand_ratio,
            shrink_ratio=decoder_head_shrink_ratio,
        )
        self.generic_head = _make_decoder_head(
            score_dim,
            generic_output_dim,
            head_hidden_dim,
            depth=head_depth,
            activation=activation,
            layout=decoder_head_layout,
            expand_ratio=decoder_head_expand_ratio,
            shrink_ratio=decoder_head_shrink_ratio,
        )

    def _timing_conditioned_hidden(self, shared_hidden, score_shared_raw=None):
        if self.zero_timing_head_condition is None or score_shared_raw is None:
            return shared_hidden
        zero_bit = (
            score_shared_raw[..., 0].float().abs()
            <= float(getattr(self.config, "zero_ioi_support_eps", 1e-6))
        ).long()
        condition = self.zero_timing_head_condition(zero_bit).to(
            dtype=shared_hidden.dtype,
            device=shared_hidden.device,
        )
        return shared_hidden + condition

    def _shared_outputs(self, hidden_states, score_shared_raw=None):
        shared_hidden = hidden_states[..., self.shared_slice]
        if not self.split_shared_heads:
            return self.shared_head(shared_hidden)

        timing_hidden = self._timing_conditioned_hidden(shared_hidden, score_shared_raw)
        if self.epr_distribution in DINR_DISTRIBUTIONS and self.ioi_head.alternate_value_table is not None:
            if score_shared_raw is None:
                raise ValueError("Separated DINR IOI routing requires score_shared_raw")
            zero_score_ioi = score_shared_raw[..., 0].abs() <= float(
                getattr(self.config, "zero_ioi_support_eps", 1e-6)
            )
            ioi = self.ioi_head(timing_hidden, use_alternate=zero_score_ioi)
        else:
            ioi = self.ioi_head(timing_hidden)
        duration = (
            self.duration_head(timing_hidden)
            if self.duration_head is not None
            else shared_hidden.new_empty(*shared_hidden.shape[:-1], 0)
        )
        velocity = self.velocity_head(shared_hidden)
        if self.shared_pack_mode == "beta_mu_kappa":
            return torch.cat(
                [
                    ioi[..., 0:1],
                    duration[..., 0:1],
                    velocity[..., 0:1],
                    ioi[..., 1:2],
                    duration[..., 1:2],
                    velocity[..., 1:2],
                ],
                dim=-1,
            )

        parts = [ioi, duration, velocity]
        if self.shared_extra_head is not None:
            parts.append(self.shared_extra_head(shared_hidden))
        return torch.cat(parts, dim=-1)

    def forward(self, hidden_states, score_shared_raw=None):
        if _uses_epr_targets(self.config):
            shared = self._shared_outputs(hidden_states, score_shared_raw=score_shared_raw)
            pedal = self.pedal_head(hidden_states[..., self.perf_slice])
            if self.epr_distribution in {
                "beta_mu_kappa",
                "lan",
                "can",
                "ican",
                "iln",
                "sn",
                "skew_normal",
                "logistic_normal",
                "mixture_logistic_normal",
                "mixture_beta",
                *DISCRETE_BOUNDED_DISTRIBUTIONS,
                *DLM_DISTRIBUTIONS,
                *DINR_DISTRIBUTIONS,
            }:
                return torch.cat([shared, pedal], dim=-1)
            shared = torch.sigmoid(shared)
            if self.pedal_representation == "binary_4":
                return torch.cat([shared, pedal], dim=-1)
            if self.config.pedal_output_activation == "sigmoid":
                pedal = torch.sigmoid(pedal)
            elif self.config.pedal_output_activation != "linear":
                raise ValueError(f"Unsupported pedal_output_activation: {self.config.pedal_output_activation}")
            return torch.cat([shared, pedal], dim=-1)

        if self.output_dim != 7:
            return self.generic_head(hidden_states[..., self.score_slice])

        shared = self._shared_outputs(hidden_states, score_shared_raw=score_shared_raw)
        pedal = self.pedal_head(hidden_states[..., self.perf_slice])
        if self.epr_distribution in {
            "beta_mu_kappa",
            "categorical",
            "hard_categorical",
            "soft_categorical",
            "lan",
            "can",
            "ican",
            "iln",
            "sn",
            "skew_normal",
            "logistic_normal",
            "mixture_logistic_normal",
            "mixture_beta",
            *DISCRETE_BOUNDED_DISTRIBUTIONS,
            *DLM_DISTRIBUTIONS,
            *DINR_DISTRIBUTIONS,
        }:
            return torch.cat([shared, pedal], dim=-1)

        shared = torch.sigmoid(shared)
        if self.config.pedal_output_activation == "sigmoid":
            pedal = torch.sigmoid(pedal)
        elif self.config.pedal_output_activation != "linear":
            raise ValueError(f"Unsupported pedal_output_activation: {self.config.pedal_output_activation}")
        return torch.cat([shared, pedal], dim=-1)


def _split_epr_distribution_params(raw_outputs):
    return {
        "shared_mu": raw_outputs[..., 0:3],
        "shared_kappa": raw_outputs[..., 3:6],
        "pedal_mu": raw_outputs[..., 6:10],
        "pedal_kappa": raw_outputs[..., 10:14],
    }


def _split_epr_categorical_logits(config, raw_outputs):
    timing_bins = int(config.epr_timing_bins)
    value_bins = int(config.epr_value_bins)
    ioi_end = timing_bins
    duration_end = ioi_end + timing_bins
    velocity_end = duration_end + value_bins
    pedal_start = velocity_end
    pedal_end = pedal_start + 4 * value_bins
    return {
        "ioi": raw_outputs[..., :ioi_end],
        "duration": raw_outputs[..., ioi_end:duration_end],
        "velocity": raw_outputs[..., duration_end:velocity_end],
        "pedal": raw_outputs[..., pedal_start:pedal_end].reshape(*raw_outputs.shape[:-1], 4, value_bins),
    }


def _split_dinr_logits(config, raw_outputs):
    timing_bins = int(config.dinr_output_timing_bins)
    ioi_end = timing_bins
    duration_end = ioi_end + timing_bins
    velocity_end = duration_end + 128
    pedal_dim = pedal_representation_dim(getattr(config, "pedal_representation", "binary_4"))
    pedal_end = velocity_end + pedal_dim
    if raw_outputs.shape[-1] != pedal_end:
        raise ValueError(
            f"Unexpected DINR output width {raw_outputs.shape[-1]}, expected {pedal_end}"
        )
    pedal = raw_outputs[..., velocity_end:pedal_end]
    return {
        "ioi": raw_outputs[..., :ioi_end],
        "duration": raw_outputs[..., ioi_end:duration_end],
        "velocity": raw_outputs[..., duration_end:velocity_end],
        "pedal": pedal,
    }


def _dinr_coordinates(config, reference, vocabulary="deviation"):
    if vocabulary == "absolute":
        bins = int(config.dinr_timing_bins)
        zero_bin = int(config.dinr_zero_bin)
        step = float(config.dinr_timing_step)
    else:
        bins = int(config.dinr_output_timing_bins)
        zero_bin = int(config.dinr_output_zero_bin)
        step = float(config.dinr_output_timing_step)
    return (
        torch.arange(bins, device=reference.device, dtype=torch.float32) - zero_bin
    ) * step


def _dinr_support_mask(config, logits, feature, score_shared_raw=None):
    coordinates = _dinr_coordinates(config, logits)
    if feature == "duration":
        valid = (coordinates >= float(config.dinr_deviation_min)) & (
            coordinates <= float(config.dinr_deviation_max)
        )
        return logits.masked_fill(~valid.view(*([1] * (logits.ndim - 1)), -1), float("-inf"))
    if feature != "ioi":
        return logits
    if score_shared_raw is None:
        raise ValueError("DINR IOI support masking requires score_shared_raw")
    nonzero_valid = (coordinates >= float(config.dinr_deviation_min)) & (
        coordinates <= float(config.dinr_deviation_max)
    )
    zero_coordinates = (
        _dinr_coordinates(config, logits, vocabulary="absolute")
        if str(getattr(config, "dinr_vocabulary_mode", "unified")).lower() == "separated"
        else coordinates
    )
    zero_valid = (zero_coordinates >= float(config.dinr_zero_ioi_min)) & (
        zero_coordinates <= float(config.dinr_zero_ioi_max)
    )
    zero_score = score_shared_raw[..., 0].abs() <= float(getattr(config, "zero_ioi_support_eps", 1e-6))
    valid = torch.where(
        zero_score.unsqueeze(-1),
        zero_valid.view(*([1] * (logits.ndim - 1)), -1),
        nonzero_valid.view(*([1] * (logits.ndim - 1)), -1),
    )
    return logits.masked_fill(~valid, float("-inf"))


def _beta_params(raw_mu, raw_kappa, eps=1e-5, kappa_min=1e-3):
    mu = raw_mu.sigmoid()
    kappa = F.softplus(raw_kappa) + kappa_min
    alpha = mu * kappa + eps
    beta = (1.0 - mu) * kappa + eps
    return mu, kappa, alpha, beta


def _epr_mixture_components(config):
    components = int(getattr(config, "epr_mixture_components", 1))
    if components < 1:
        raise ValueError(f"epr_mixture_components must be >= 1, got {components}")
    return components


def _uses_binary4_pedal(config):
    return normalize_pedal_representation(getattr(config, "pedal_representation", "binary_4")) == "binary_4"


def _attach_simple_pedal_params(config, params, raw_outputs, cursor):
    representation = normalize_pedal_representation(getattr(config, "pedal_representation", "binary_4"))
    if representation == "binary_4":
        params["pedal_binary_logits"] = raw_outputs[..., cursor : cursor + 4]
        return params
    raise ValueError("Only pedal_representation=binary_4 is supported")


def _mask_count(mask):
    return mask.to(dtype=torch.float32).sum().clamp_min(1.0)


def _split_epr_mixture_params(config, raw_outputs):
    components = _epr_mixture_components(config)
    distribution = getattr(config, "epr_distribution", "point").lower()
    if distribution in DLM_DISTRIBUTIONS:
        components = int(getattr(config, "dlm_components", components))
        per_feature_dim = components * 3
        cursor = 0
        ioi_base = raw_outputs[..., cursor : cursor + per_feature_dim].reshape(
            *raw_outputs.shape[:-1],
            3,
            components,
        )
        cursor += per_feature_dim
        ioi_zero_logit = None
        if bool(getattr(config, "dlm_ioi_zero_inflated", False)):
            ioi_zero_logit = raw_outputs[..., cursor]
            cursor += 1
        duration_base = raw_outputs[..., cursor : cursor + per_feature_dim].reshape(
            *raw_outputs.shape[:-1],
            3,
            components,
        )
        cursor += per_feature_dim
        velocity_distribution = str(getattr(config, "velocity_distribution", "skew_normal")).lower()
        if velocity_distribution in DLM_DISTRIBUTIONS:
            velocity_base = raw_outputs[..., cursor : cursor + per_feature_dim].reshape(
                *raw_outputs.shape[:-1],
                3,
                components,
            )
            cursor += per_feature_dim
            params = {
                "ioi_logits": ioi_base[..., 0, :],
                "ioi_loc": ioi_base[..., 1, :],
                "ioi_log_scale": ioi_base[..., 2, :],
                "duration_logits": duration_base[..., 0, :],
                "duration_loc": duration_base[..., 1, :],
                "duration_log_scale": duration_base[..., 2, :],
                "velocity_logits": velocity_base[..., 0, :],
                "velocity_loc": velocity_base[..., 1, :],
                "velocity_log_scale": velocity_base[..., 2, :],
            }
        else:
            velocity_base = raw_outputs[..., cursor : cursor + 3]
            cursor += 3
            params = {
                "ioi_logits": ioi_base[..., 0, :],
                "ioi_loc": ioi_base[..., 1, :],
                "ioi_log_scale": ioi_base[..., 2, :],
                "duration_logits": duration_base[..., 0, :],
                "duration_loc": duration_base[..., 1, :],
                "duration_log_scale": duration_base[..., 2, :],
                "velocity_loc": velocity_base[..., 0],
                "velocity_log_scale": velocity_base[..., 1],
                "velocity_alpha": velocity_base[..., 2],
            }
        if ioi_zero_logit is not None:
            params["ioi_zero_logit"] = ioi_zero_logit
        return _attach_simple_pedal_params(config, params, raw_outputs, cursor)
    if distribution in SN_DISTRIBUTIONS:
        feature_dim = 3
        base_timing_count = _timing_distribution_count(config)
        dual_timing = str(
            _config_value(config, "zero_ioi_dual_distribution_mode", "none") or "none"
        ).lower() not in {"none", "off", "false", "0"}
        dual_duration = dual_timing and bool(_config_value(config, "zero_ioi_dual_duration", True))
        ioi_timing_count = base_timing_count + (1 if dual_timing else 0)
        duration_timing_count = base_timing_count + (1 if dual_duration else 0)
        ioi_feature_dim = feature_dim * ioi_timing_count + (
            1 if _uses_raw_timing_regression_head(config) else 0
        )
        duration_feature_dim = feature_dim * duration_timing_count + (
            1 if _uses_raw_timing_regression_head(config) else 0
        )
        ioi_raw = raw_outputs[..., :ioi_feature_dim]
        ioi_base = ioi_raw[..., : feature_dim * ioi_timing_count].reshape(
            *raw_outputs.shape[:-1], ioi_timing_count, feature_dim
        )
        cursor = ioi_feature_dim
        duration_raw = raw_outputs[..., cursor : cursor + duration_feature_dim]
        duration_base = duration_raw[..., : feature_dim * duration_timing_count].reshape(
            *raw_outputs.shape[:-1],
            duration_timing_count,
            feature_dim,
        )
        cursor += duration_feature_dim
        velocity_base = raw_outputs[..., cursor : cursor + feature_dim]
        cursor += feature_dim
        params = {
            "timing_log_loc": torch.stack([ioi_base[..., 0, 0], duration_base[..., 0, 0]], dim=-1),
            "timing_log_log_scale": torch.stack([ioi_base[..., 0, 1], duration_base[..., 0, 1]], dim=-1),
            "timing_log_alpha": torch.stack([ioi_base[..., 0, 2], duration_base[..., 0, 2]], dim=-1),
            "velocity_loc": velocity_base[..., 0],
            "velocity_log_scale": velocity_base[..., 1],
            "velocity_alpha": velocity_base[..., 2],
        }
        params = _attach_simple_pedal_params(config, params, raw_outputs, cursor)
        if dual_timing:
            zero_duration_base = (
                duration_base[..., base_timing_count, :]
                if dual_duration
                else duration_base[..., 0, :]
            )
            params.update(
                {
                    "timing_zero_log_loc": torch.stack(
                        [ioi_base[..., base_timing_count, 0], zero_duration_base[..., 0]],
                        dim=-1,
                    ),
                    "timing_zero_log_log_scale": torch.stack(
                        [ioi_base[..., base_timing_count, 1], zero_duration_base[..., 1]],
                        dim=-1,
                    ),
                    "timing_zero_log_alpha": torch.stack(
                        [ioi_base[..., base_timing_count, 2], zero_duration_base[..., 2]],
                        dim=-1,
                    ),
                }
            )
        if base_timing_count > 1:
            params.update(
                {
                    "timing_raw_loc": torch.stack(
                        [ioi_base[..., 1, 0], duration_base[..., 1, 0]], dim=-1
                    ),
                    "timing_raw_log_scale": torch.stack(
                        [ioi_base[..., 1, 1], duration_base[..., 1, 1]], dim=-1
                    ),
                    "timing_raw_alpha": torch.stack(
                        [ioi_base[..., 1, 2], duration_base[..., 1, 2]], dim=-1
                    ),
                }
            )
        elif _uses_raw_timing_regression_head(config):
            params["timing_raw_regression"] = torch.stack(
                [ioi_raw[..., -1], duration_raw[..., -1]],
                dim=-1,
            )
        return params
    if distribution in {*ACN_DISTRIBUTIONS, *IACN_DISTRIBUTIONS}:
        per_feature_dim = _scalar_distribution_dim(distribution)
        shared_feature_count = 3
        shared_base_dim = per_feature_dim * shared_feature_count
        shared_base = raw_outputs[..., :shared_base_dim].reshape(
            *raw_outputs.shape[:-1],
            shared_feature_count,
            per_feature_dim,
        )
        duration_idx = 1
        velocity_idx = 2
        params = {
            "shared_a": torch.stack(
                [
                    shared_base[..., 0, -3],
                    shared_base[..., duration_idx, -3],
                    shared_base[..., velocity_idx, -3],
                ],
                dim=-1,
            ),
            "shared_b": torch.stack(
                [
                    shared_base[..., 0, -2],
                    shared_base[..., duration_idx, -2],
                    shared_base[..., velocity_idx, -2],
                ],
                dim=-1,
            ),
            "shared_c": torch.stack(
                [
                    shared_base[..., 0, -1],
                    shared_base[..., duration_idx, -1],
                    shared_base[..., velocity_idx, -1],
                ],
                dim=-1,
            ),
        }
        if distribution in IACN_DISTRIBUTIONS:
            params["shared_mode_logits"] = torch.stack(
                [
                    shared_base[..., 0, 0:2],
                    shared_base[..., duration_idx, 0:2],
                    shared_base[..., velocity_idx, 0:2],
                ],
                dim=-2,
            )
        pedal_representation = str(getattr(config, "pedal_representation", "binary_4")).lower()
        if pedal_representation == "binary_4":
            pedal_start = shared_base_dim
            params["pedal_binary_logits"] = raw_outputs[..., pedal_start : pedal_start + 4]
            return params
        if normalize_pedal_representation(pedal_representation) == "binary_4":
            return _attach_simple_pedal_params(config, params, raw_outputs, shared_base_dim)
        pedal_distribution = str(getattr(config, "pedal_distribution", distribution)).lower()
        pedal_feature_dim = _scalar_distribution_dim(pedal_distribution)
        pedal_base_dim = pedal_feature_dim * 4
        pedal_start = shared_base_dim
        pedal_base = raw_outputs[..., pedal_start : pedal_start + pedal_base_dim].reshape(
            *raw_outputs.shape[:-1],
            4,
            pedal_feature_dim,
        )
        params["pedal_a"] = pedal_base[..., -3]
        params["pedal_b"] = pedal_base[..., -2]
        params["pedal_c"] = pedal_base[..., -1]
        if pedal_distribution in IACN_DISTRIBUTIONS:
            params["pedal_mode_logits"] = pedal_base[..., 0:2]
        return params

    if distribution in ALN_DISTRIBUTIONS:
        components = 1
        timing_dim = components * 4
        scalar_dim = components * 3
        ioi_base = raw_outputs[..., :timing_dim].reshape(*raw_outputs.shape[:-1], 4, components)
        cursor = timing_dim
        duration_base = raw_outputs[..., cursor : cursor + timing_dim].reshape(
            *raw_outputs.shape[:-1],
            4,
            components,
        )
        cursor += timing_dim
        velocity_base = raw_outputs[..., cursor : cursor + scalar_dim].reshape(
            *raw_outputs.shape[:-1],
            3,
            components,
        )
        cursor += scalar_dim
        params = {
            "shared_logits": torch.stack([ioi_base[..., 0, :], duration_base[..., 0, :], velocity_base[..., 0, :]], dim=-2),
            "shared_a": torch.stack([ioi_base[..., 1, :], duration_base[..., 1, :], velocity_base[..., 1, :]], dim=-2),
            "shared_b": torch.stack([ioi_base[..., 2, :], duration_base[..., 2, :], velocity_base[..., 2, :]], dim=-2),
            "shared_c": torch.stack(
                [
                    ioi_base[..., 3, :],
                    duration_base[..., 3, :],
                    torch.zeros_like(velocity_base[..., 2, :]),
                ],
                dim=-2,
            ),
        }
        pedal_representation = str(getattr(config, "pedal_representation", "binary_4")).lower()
        if pedal_representation == "binary_4":
            params["pedal_binary_logits"] = raw_outputs[..., cursor : cursor + 4]
            return params
        if normalize_pedal_representation(pedal_representation) == "binary_4":
            return _attach_simple_pedal_params(config, params, raw_outputs, cursor)
        pedal_base = raw_outputs[..., cursor : cursor + scalar_dim * 4].reshape(
            *raw_outputs.shape[:-1],
            4,
            3,
            components,
        )
        params["pedal_logits"] = pedal_base[..., 0, :]
        params["pedal_a"] = pedal_base[..., 1, :]
        params["pedal_b"] = pedal_base[..., 2, :]
        return params

    per_component_dim = 3
    per_feature_dim = components * per_component_dim
    shared_feature_count = 3
    shared_base_dim = per_feature_dim * shared_feature_count
    pedal_base_dim = per_feature_dim * 4
    shared_base = raw_outputs[..., :shared_base_dim].reshape(
        *raw_outputs.shape[:-1],
        shared_feature_count,
        per_component_dim,
        components,
    )
    pedal_representation = str(getattr(config, "pedal_representation", "binary_4")).lower()
    duration_idx = 1
    velocity_idx = 2
    params = {
        "shared_logits": torch.stack(
            [
                shared_base[..., 0, 0, :],
                shared_base[..., duration_idx, 0, :],
                shared_base[..., velocity_idx, 0, :],
            ],
            dim=-2,
        ),
        "shared_a": torch.stack(
            [
                shared_base[..., 0, 1, :],
                shared_base[..., duration_idx, 1, :],
                shared_base[..., velocity_idx, 1, :],
            ],
            dim=-2,
        ),
        "shared_b": torch.stack(
            [
                shared_base[..., 0, 2, :],
                shared_base[..., duration_idx, 2, :],
                shared_base[..., velocity_idx, 2, :],
            ],
            dim=-2,
        ),
    }
    if pedal_representation == "binary_4":
        pedal_start = shared_base_dim
        params["pedal_binary_logits"] = raw_outputs[..., pedal_start : pedal_start + 4]
        return params
    if normalize_pedal_representation(pedal_representation) == "binary_4":
        return _attach_simple_pedal_params(config, params, raw_outputs, shared_base_dim)

    pedal_start = shared_base_dim
    pedal_end = pedal_start + pedal_base_dim
    pedal_base = raw_outputs[..., pedal_start:pedal_end].reshape(
        *raw_outputs.shape[:-1],
        4,
        per_component_dim,
        components,
    )
    params["pedal_logits"] = pedal_base[..., 0, :]
    params["pedal_a"] = pedal_base[..., 1, :]
    params["pedal_b"] = pedal_base[..., 2, :]
    return params

def _shared_scalar_params(config, params, index):
    distribution = getattr(config, "epr_distribution", "point").lower()
    if distribution in IACN_DISTRIBUTIONS:
        return (
            params["shared_mode_logits"][..., index, :],
            params["shared_a"][..., index],
            params["shared_b"][..., index],
            params["shared_c"][..., index],
        )
    if distribution in ACN_DISTRIBUTIONS:
        return (
            None,
            params["shared_a"][..., index],
            params["shared_b"][..., index],
            params["shared_c"][..., index],
        )
    return (
        params["shared_logits"][..., index, :],
        params["shared_a"][..., index, :],
        params["shared_b"][..., index, :],
        params["shared_c"][..., index, :] if distribution in ALN_DISTRIBUTIONS else None,
    )


def _pedal_scalar_params(config, params, index):
    distribution = str(getattr(config, "pedal_distribution", getattr(config, "epr_distribution", "point"))).lower()
    if distribution in ILN_DISTRIBUTIONS:
        return (
            params["pedal_mode_logits"][..., index, :],
            params["pedal_a"][..., index],
            params["pedal_b"][..., index],
            None,
        )
    if distribution in IACN_DISTRIBUTIONS:
        return (
            params["pedal_mode_logits"][..., index, :],
            params["pedal_a"][..., index],
            params["pedal_b"][..., index],
            params["pedal_c"][..., index],
        )
    if distribution in ACN_DISTRIBUTIONS:
        return (
            None,
            params["pedal_a"][..., index],
            params["pedal_b"][..., index],
            params["pedal_c"][..., index],
        )
    return (
        params["pedal_logits"][..., index, :],
        params["pedal_a"][..., index, :],
        params["pedal_b"][..., index, :],
        params.get("pedal_c", None)[..., index, :] if "pedal_c" in params else None,
    )

def _uses_inr_epr_targets(config):
    return (
        _config_value(config, "task_type", "epr") == "epr"
        and str(_config_value(config, "epr_timing_target", "absolute")).lower()
        in {"floor_log_deviation", "floor_log_dev"}
    )


def _uses_floor_log_deviation_targets(config):
    return str(_config_value(config, "epr_timing_target", "absolute")).lower() in {"floor_log_deviation", "floor_log_dev"}


def _timing_distribution_count(config):
    return 2 if str(_config_value(config, "raw_timing_head_type", "none")).lower() == "distribution" else 1


def _uses_raw_timing_regression_head(config):
    return str(_config_value(config, "raw_timing_head_type", "none")).lower() == "regression"


def _config_value(config, name, default):
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def _torch_floor_log_reconstruct(score_time_ms, dev):
    base = score_time_ms.float().clamp_min(1.0)
    return (base * torch.exp(dev.float())).clamp_min(0.0)


def _torch_timing_control_code(time_ms, timing_control_mode="dinr_floor_log", use_scale_bit=False, log_scale=50.0):
    resolve_timing_control_mode(
        timing_control_mode=timing_control_mode,
        use_timing_scale_bit=use_scale_bit,
    )
    value = time_ms.float().clamp(0.0, 8000.0)
    return torch.log(value.clamp_min(1.0)).unsqueeze(-1)


def _target7_to_raw7(score_shared_raw, target_predictions, config=None):
    score_shared_raw = score_shared_raw.float()
    target_predictions = target_predictions.float()
    pedal_representation = (
        _config_value(config, "pedal_representation", "binary_4")
        if config is not None
        else "binary_4"
    )
    pedal_dim = pedal_representation_dim(pedal_representation)
    if target_predictions.shape[-1] < 3 + pedal_dim:
        raise ValueError(f"Expected EPR target predictions, got shape {tuple(target_predictions.shape)}")

    def pedal_raw4(start):
        pedal_values = target_predictions[..., start : start + pedal_dim].clamp(0.0, 1.0)
        return pedal_values[..., :4] * 127.0

    if config is not None and not _uses_floor_log_deviation_targets(config):
        raise ValueError("target7 -> raw7 reconstruction expects floor_log_deviation timing targets")
    perf_ioi_ms = _torch_floor_log_reconstruct(score_shared_raw[..., 0], target_predictions[..., 0])
    perf_duration_ms = _torch_floor_log_reconstruct(score_shared_raw[..., 1], target_predictions[..., 1])
    if config is not None and str(_config_value(config, "epr_distribution", "point")).lower() in DINR_DISTRIBUTIONS:
        max_ms = float(_config_value(config, "dinr_absolute_max_ms", 8000.0))
        perf_ioi_ms = perf_ioi_ms.clamp(0.0, max_ms)
        perf_duration_ms = perf_duration_ms.clamp(0.0, max_ms)
    velocity = target_predictions[..., 2].clamp(0.0, 1.0) * 127.0
    pedal = pedal_raw4(3)
    return torch.cat([perf_ioi_ms.unsqueeze(-1), perf_duration_ms.unsqueeze(-1), velocity.unsqueeze(-1), pedal], dim=-1)


def _uses_epr_targets(config):
    return _uses_inr_epr_targets(config)


def _decoder_rows_require_score_shared_raw(config):
    return decoder_note_input_schema(config) == "integrated" and not slot_version_is_pt(
        getattr(config, "slot_version", None)
    )


def _build_epr_decoder_perf_target_rows(config, target_predictions):
    target_features = target_predictions.float()
    masks = target_features.new_ones(*target_features.shape[:-1], 3)
    return torch.cat([target_features, masks], dim=-1)


def _target_predictions_to_feedback7(config, target_predictions):
    target_predictions = target_predictions.float()
    pedal_dim = pedal_representation_dim(getattr(config, "pedal_representation", "binary_4"))
    if target_predictions.shape[-1] < 3 + pedal_dim:
        raise ValueError(f"Expected EPR target values, got shape {tuple(target_predictions.shape)}")
    if not _uses_floor_log_deviation_targets(config):
        raise ValueError("Only epr_timing_target=floor_log_deviation is supported")
    return torch.cat(
        [
            target_predictions[..., 0:2],
            target_predictions[..., 2:3].clamp(0.0, 1.0),
            target_predictions[..., 3 : 3 + pedal_dim].clamp(0.0, 1.0),
        ],
        dim=-1,
    )


def _extract_integrated_score_musical(config, score_input_continuous):
    if score_input_continuous is None:
        return None
    score_control_dim = int(getattr(config, "score_control_feature_dim", getattr(config, "control_feature_dim", 5)))
    performance_control_dim = int(
        getattr(config, "performance_control_feature_dim", getattr(config, "control_feature_dim", 5) + 4)
    )
    musical_dim = int(getattr(config, "musical_feature_dim", 12))
    start = score_control_dim + performance_control_dim
    end = start + musical_dim
    if score_input_continuous.shape[-1] < end:
        raise ValueError(
            "score_input_continuous is too small to carry integrated musical features: "
            f"got {score_input_continuous.shape[-1]}, need at least {end}"
        )
    return score_input_continuous[..., start:end]


def _build_epr_decoder_rows(config, score_shared_raw, target_predictions, score_input_continuous=None):
    if decoder_note_input_schema(config) == "perf_target":
        return _build_epr_decoder_perf_target_rows(config, target_predictions)
    timing_control_mode = getattr(config, "timing_control_mode", None)
    use_timing_scale_bit = getattr(config, "use_timing_scale_bit", True)
    resolve_timing_control_mode(timing_control_mode, use_timing_scale_bit)
    target7 = _target_predictions_to_feedback7(config, target_predictions)
    score_ioi = _torch_timing_control_code(
        score_shared_raw[..., 0],
        timing_control_mode=timing_control_mode,
        use_scale_bit=use_timing_scale_bit,
    )
    score_duration = _torch_timing_control_code(
        score_shared_raw[..., 1],
        timing_control_mode=timing_control_mode,
        use_scale_bit=use_timing_scale_bit,
    )
    score_velocity = (score_shared_raw[..., 2:3].float().clamp(0.0, 127.0) / 127.0)
    if _uses_floor_log_deviation_targets(config):
        ioi_dev = target7[..., 0:1]
        duration_dev = target7[..., 1:2]
        if str(_config_value(config, "epr_distribution", "point")).lower() in DINR_DISTRIBUTIONS:
            zero_score = score_shared_raw[..., 0:1].abs() <= float(
                _config_value(config, "zero_ioi_support_eps", 1e-6)
            )
            ioi_dev = torch.where(
                zero_score,
                ioi_dev.clamp(
                    float(_config_value(config, "dinr_zero_ioi_min", 0.0)),
                    float(_config_value(config, "dinr_zero_ioi_max", 5.0)),
                ),
                ioi_dev.clamp(
                    float(_config_value(config, "dinr_deviation_min", -2.0)),
                    float(_config_value(config, "dinr_deviation_max", 2.0)),
                ),
            )
            duration_dev = duration_dev.clamp(
                float(_config_value(config, "dinr_deviation_min", -2.0)),
                float(_config_value(config, "dinr_deviation_max", 2.0)),
            )
        perf_ioi_ms = _torch_floor_log_reconstruct(score_shared_raw[..., 0:1], ioi_dev)
        duration_ms = _torch_floor_log_reconstruct(score_shared_raw[..., 1:2], duration_dev)
        if str(_config_value(config, "epr_distribution", "point")).lower() in DINR_DISTRIBUTIONS:
            max_ms = float(_config_value(config, "dinr_absolute_max_ms", 8000.0))
            perf_ioi_ms = perf_ioi_ms.clamp(0.0, max_ms)
            duration_ms = duration_ms.clamp(0.0, max_ms)
        perf_ioi = _torch_timing_control_code(
                perf_ioi_ms.squeeze(-1),
                timing_control_mode=timing_control_mode,
                use_scale_bit=use_timing_scale_bit,
            )
        duration = _torch_timing_control_code(
                duration_ms.squeeze(-1),
                timing_control_mode=timing_control_mode,
                use_scale_bit=use_timing_scale_bit,
            )
    else:
        raise ValueError("Only epr_timing_target=floor_log_deviation is supported")
    pedal_dim = pedal_representation_dim(getattr(config, "pedal_representation", "binary_4"))
    velocity = target7[..., 2:3].float().clamp(0.0, 1.0)
    pedal = target7[..., 3 : 3 + pedal_dim].float().clamp(0.0, 1.0)
    score_control = torch.cat([score_ioi, score_duration, score_velocity], dim=-1)
    expected_score_dim = int(getattr(config, "score_control_feature_dim", getattr(config, "control_feature_dim", 5)))
    if score_control.shape[-1] != expected_score_dim:
        raise ValueError(f"score control dim mismatch: got {score_control.shape[-1]}, expected {expected_score_dim}")
    performance_control = torch.cat([perf_ioi, duration, velocity, pedal], dim=-1)
    musical_source = _extract_integrated_score_musical(config, score_input_continuous)
    if musical_source is None:
        musical = target_predictions.new_zeros(*target_predictions.shape[:-1], getattr(config, "musical_feature_dim", 12))
    else:
        musical = musical_source.to(dtype=target_predictions.dtype, device=target_predictions.device)
    score_visible = 0.0 if slot_version_is_pt(getattr(config, "slot_version", None)) else 1.0
    if int(getattr(config, "mask_feature_dim", 3)) == 2:
        masks = target_predictions.new_tensor([score_visible, 1.0]).expand(*target_predictions.shape[:-1], 2)
    else:
        musical_visible = 0.0
        if score_input_continuous is not None:
            musical_dim = int(getattr(config, "musical_feature_dim", 12))
            score_control_dim = int(getattr(config, "score_control_feature_dim", getattr(config, "control_feature_dim", 5)))
            performance_control_dim = int(
                getattr(config, "performance_control_feature_dim", getattr(config, "control_feature_dim", 5) + pedal_dim)
            )
            mask_start = score_control_dim + performance_control_dim + musical_dim
            if score_input_continuous.shape[-1] > mask_start + 2:
                musical_visible = score_input_continuous[..., mask_start + 2 : mask_start + 3]
        masks = torch.cat(
            [
                target_predictions.new_full((*target_predictions.shape[:-1], 1), score_visible),
                target_predictions.new_ones(*target_predictions.shape[:-1], 1),
                musical_visible
                if torch.is_tensor(musical_visible)
                else target_predictions.new_full((*target_predictions.shape[:-1], 1), float(musical_visible)),
            ],
            dim=-1,
        )
    return torch.cat([score_control, performance_control, musical, masks], dim=-1)


def _build_removed_task_decoder_rows(config, musical_predictions):
    musical = musical_predictions.float().clamp(0.0, 1.0)
    score_control_dim = int(getattr(config, "score_control_feature_dim", getattr(config, "control_feature_dim", 5)))
    performance_control_dim = int(
        getattr(config, "performance_control_feature_dim", getattr(config, "control_feature_dim", 5) + 4)
    )
    zeros = musical.new_zeros(*musical.shape[:-1], score_control_dim + performance_control_dim)
    masks = musical.new_tensor([0.0, 0.0, 1.0]).expand(*musical.shape[:-1], 3)
    return torch.cat([zeros, musical, masks], dim=-1)


def _removed_task_uses_grid_head(config):
    return str(getattr(config, "removed_task_grid_loss_type", "huber")).lower() in {
        "soft_ce",
        "soft_ce_huber",
        "ce",
        "hard_ce",
        "ordinal",
        "grid",
    }


def _removed_task_grid_bins(config, name):
    step = max(float(getattr(config, "removed_task_grid_step", 1.0 / 24.0)), 1e-12)
    max_value = float(getattr(config, f"removed_task_{name}_max"))
    return int(round(max_value / step)) + 1


def _removed_task_grid_raw_output_dim(config):
    return (
        _removed_task_grid_bins(config, "mo")
        + _removed_task_grid_bins(config, "md")
        + _removed_task_grid_bins(config, "ml")
        + 1
        + 8
    )


def _split_removed_task_grid_outputs(config, raw_outputs):
    start = 0
    outputs = {}
    for name in ("mo", "md", "ml"):
        bins = _removed_task_grid_bins(config, name)
        outputs[name] = raw_outputs[..., start : start + bins]
        start += bins
    outputs["tempo"] = raw_outputs[..., start]
    start += 1
    outputs["binary"] = raw_outputs[..., start : start + 8]
    return outputs


def _removed_task_grid_to_normalized(config, name, logits):
    bins = logits.shape[-1]
    values = torch.arange(bins, device=logits.device, dtype=torch.float32)
    step = float(getattr(config, "removed_task_grid_step", 1.0 / 24.0))
    max_value = max(float(getattr(config, f"removed_task_{name}_max")), step)
    indices = logits.float().argmax(dim=-1).to(dtype=torch.float32)
    return (indices * step / max_value).clamp(0.0, 1.0)


def _materialize_removed_task_prediction(config, raw_outputs):
    if not _removed_task_uses_grid_head(config):
        return torch.sigmoid(raw_outputs)
    parts = _split_removed_task_grid_outputs(config, raw_outputs)
    continuous = [
        _removed_task_grid_to_normalized(config, "mo", parts["mo"]),
        (torch.sigmoid(parts["binary"][..., 0]) >= 0.5).to(dtype=raw_outputs.dtype),
        _removed_task_grid_to_normalized(config, "md", parts["md"]),
        _removed_task_grid_to_normalized(config, "ml", parts["ml"]),
        torch.sigmoid(parts["tempo"]),
    ]
    binary = (torch.sigmoid(parts["binary"][..., 1:]) >= 0.5).to(dtype=raw_outputs.dtype)
    return torch.cat([torch.stack(continuous, dim=-1), binary], dim=-1)


def _logistic_normal_params(raw_mu, raw_log_sigma, sigma_min=1e-3, sigma_max=10.0):
    log_min = torch.log(raw_log_sigma.new_tensor(float(sigma_min)))
    log_max = torch.log(raw_log_sigma.new_tensor(float(sigma_max)))
    sigma = torch.exp(raw_log_sigma.float().clamp(min=log_min.item(), max=log_max.item()))
    return raw_mu.float(), sigma


def _mixture_logistic_normal_log_prob(logits, raw_mu, raw_log_sigma, target, eps, sigma_min, sigma_max):
    target = target.float().clamp(float(eps), 1.0 - float(eps))
    z = torch.logit(target, eps=float(eps)).unsqueeze(-1)
    mu, sigma = _logistic_normal_params(raw_mu, raw_log_sigma, sigma_min=sigma_min, sigma_max=sigma_max)
    log_pi = F.log_softmax(logits.float(), dim=-1)
    log_normal = torch.distributions.Normal(mu, sigma).log_prob(z)
    log_jacobian = -torch.log(target).unsqueeze(-1) - torch.log1p(-target).unsqueeze(-1)
    return torch.logsumexp(log_pi + log_normal + log_jacobian, dim=-1)


def _mixture_logistic_normal_nll(logits, raw_mu, raw_log_sigma, target, mask, eps, sigma_min, sigma_max):
    values = -_mixture_logistic_normal_log_prob(
        logits,
        raw_mu,
        raw_log_sigma,
        target,
        eps,
        sigma_min,
        sigma_max,
    )
    return _masked_mean(values, mask)


def _bounded_student_t_log_prob(raw_loc, raw_log_scale, raw_df, target, eps, sigma_min, sigma_max):
    if raw_loc.shape != target.shape:
        raw_loc = raw_loc[..., : target.shape[-1]]
        raw_log_scale = raw_log_scale[..., : target.shape[-1]]
        raw_df = raw_df[..., : target.shape[-1]]
    target = target.float().clamp(float(eps), 1.0 - float(eps))
    z = torch.atanh((2.0 * target - 1.0).clamp(-1.0 + float(eps), 1.0 - float(eps)))
    loc, scale = _logistic_normal_params(raw_loc, raw_log_scale, sigma_min, sigma_max)
    df = 2.0 + F.softplus(raw_df.float())
    log_base = torch.distributions.StudentT(df=df, loc=loc, scale=scale).log_prob(z)
    log_jacobian = -math.log(2.0) - torch.log(target) - torch.log1p(-target)
    return log_base + log_jacobian


def _bounded_student_t_nll(raw_loc, raw_log_scale, raw_df, target, mask, eps, sigma_min, sigma_max):
    return _masked_mean(
        -_bounded_student_t_log_prob(raw_loc, raw_log_scale, raw_df, target, eps, sigma_min, sigma_max),
        mask,
    )


def _bounded_student_t_mean_or_sample(config, raw_loc, raw_log_scale, raw_df, sampling_strategy="mean"):
    sigma_min = getattr(config, "logistic_normal_sigma_min", 1e-3)
    sigma_max = getattr(config, "logistic_normal_sigma_max", 10.0)
    loc, scale = _logistic_normal_params(raw_loc, raw_log_scale, sigma_min, sigma_max)
    mode = str(sampling_strategy).lower()
    if mode in {"mean", "deterministic", "mu", "greedy"}:
        z = loc
    elif mode in {"sample", "sampling", "stochastic"}:
        df = 2.0 + F.softplus(raw_df.float())
        z = torch.distributions.StudentT(df=df, loc=loc, scale=scale).sample()
    else:
        raise ValueError(f"Unsupported sampling_strategy: {sampling_strategy}")
    return 0.5 * (torch.tanh(z) + 1.0)


def _bounded_skew_normal_log_prob(raw_loc, raw_log_scale, raw_alpha, target, eps, sigma_min, sigma_max):
    if raw_loc.shape != target.shape:
        raw_loc = raw_loc[..., : target.shape[-1]]
        raw_log_scale = raw_log_scale[..., : target.shape[-1]]
        raw_alpha = raw_alpha[..., : target.shape[-1]]
    target = target.float().clamp(float(eps), 1.0 - float(eps))
    z = torch.logit(target, eps=float(eps))
    return _skew_normal_log_prob(raw_loc, raw_log_scale, raw_alpha, z, sigma_min, sigma_max) \
        - torch.log(target) - torch.log1p(-target)


def _bounded_skew_normal_nll(raw_loc, raw_log_scale, raw_alpha, target, mask, eps, sigma_min, sigma_max):
    return _masked_mean(
        -_bounded_skew_normal_log_prob(raw_loc, raw_log_scale, raw_alpha, target, eps, sigma_min, sigma_max),
        mask,
    )


def _bounded_skew_normal_mean_or_sample(config, raw_loc, raw_log_scale, raw_alpha, sampling_strategy="mean"):
    value = _skew_normal_mean_or_sample(
        raw_loc,
        raw_log_scale,
        raw_alpha,
        sampling_strategy=sampling_strategy,
        sigma_min=getattr(config, "skew_normal_sigma_min", 1e-4),
        sigma_max=getattr(config, "skew_normal_sigma_max", 1e4),
    )
    return torch.sigmoid(value)


def _logistic_asymmetric_normal_log_prob(
    logits,
    raw_mu,
    raw_log_sigma_left,
    raw_log_sigma_right,
    target,
    eps,
    sigma_min,
    sigma_max,
):
    target = target.float().clamp(float(eps), 1.0 - float(eps))
    z = torch.logit(target, eps=float(eps)).unsqueeze(-1)
    mu = raw_mu.float()
    _, sigma_left = _logistic_normal_params(
        raw_mu,
        raw_log_sigma_left,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
    )
    _, sigma_right = _logistic_normal_params(
        raw_mu,
        raw_log_sigma_right,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
    )
    left_mask = z < mu
    sigma = torch.where(left_mask, sigma_left, sigma_right)
    log_pi = F.log_softmax(logits.float(), dim=-1)
    # Canonical split-normal density: side mass is induced by scale,
    # P(left)=sigma_left/(sigma_left+sigma_right), not by an extra parameter.
    log_normal = (
        torch.distributions.Normal(mu, sigma).log_prob(z)
        + math.log(2.0)
        + torch.log(sigma)
        - torch.log((sigma_left + sigma_right).clamp_min(1e-12))
    )
    log_jacobian = -torch.log(target).unsqueeze(-1) - torch.log1p(-target).unsqueeze(-1)
    return torch.logsumexp(log_pi + log_normal + log_jacobian, dim=-1)


def _logistic_asymmetric_normal_nll(
    logits,
    raw_mu,
    raw_log_sigma_left,
    raw_log_sigma_right,
    target,
    mask,
    eps,
    sigma_min,
    sigma_max,
):
    values = -_logistic_asymmetric_normal_log_prob(
        logits,
        raw_mu,
        raw_log_sigma_left,
        raw_log_sigma_right,
        target,
        eps,
        sigma_min,
        sigma_max,
    )
    return _masked_mean(values, mask)


def _inflated_zero_one_mode_log_probs(raw_mode_logits, temperature=1.0):
    temperature = float(temperature)
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError(f"Inflated mode temperature must be finite and > 0, got {temperature}")
    center = raw_mode_logits.new_zeros(*raw_mode_logits.shape[:-1], 1)
    logits = torch.cat([raw_mode_logits.float(), center.float()], dim=-1) / temperature
    return F.log_softmax(logits, dim=-1)


def _inflated_zero_cont_mode_log_probs(raw_zero_logit, temperature=1.0):
    temperature = float(temperature)
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError(f"Inflated mode temperature must be finite and > 0, got {temperature}")
    center = torch.zeros_like(raw_zero_logit.float())
    logits = torch.stack([raw_zero_logit.float(), center], dim=-1) / temperature
    return F.log_softmax(logits, dim=-1)


def _split_normal_params(raw_mu, raw_log_sigma_left, raw_log_sigma_right, sigma_min=1e-4, sigma_max=1e4):
    log_min = torch.log(raw_log_sigma_left.new_tensor(float(sigma_min)))
    log_max = torch.log(raw_log_sigma_left.new_tensor(float(sigma_max)))
    sigma_left = torch.exp(raw_log_sigma_left.float().clamp(min=log_min.item(), max=log_max.item()))
    sigma_right = torch.exp(raw_log_sigma_right.float().clamp(min=log_min.item(), max=log_max.item()))
    return raw_mu.float(), sigma_left, sigma_right


def _split_normal_log_pdf(raw_mu, raw_log_sigma_left, raw_log_sigma_right, target, sigma_min, sigma_max):
    mu, sigma_left, sigma_right = _split_normal_params(
        raw_mu,
        raw_log_sigma_left,
        raw_log_sigma_right,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
    )
    target = target.float()
    sigma = torch.where(target < mu, sigma_left, sigma_right)
    log_normal = torch.distributions.Normal(mu, sigma).log_prob(target)
    return log_normal + math.log(2.0) + torch.log(sigma) - torch.log((sigma_left + sigma_right).clamp_min(1e-12))


def _split_normal_cdf(raw_mu, raw_log_sigma_left, raw_log_sigma_right, value, sigma_min, sigma_max):
    mu, sigma_left, sigma_right = _split_normal_params(
        raw_mu,
        raw_log_sigma_left,
        raw_log_sigma_right,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
    )
    value = torch.as_tensor(value, device=mu.device, dtype=mu.dtype)
    standard = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(mu))
    z_left = (value - mu) / sigma_left
    z_right = (value - mu) / sigma_right
    left_mass = sigma_left / (sigma_left + sigma_right).clamp_min(1e-12)
    cdf_left = 2.0 * left_mass * standard.cdf(z_left)
    cdf_right = left_mass + 2.0 * (1.0 - left_mass) * (standard.cdf(z_right) - 0.5)
    return torch.where(value < mu, cdf_left, cdf_right).clamp(0.0, 1.0)


def _split_normal_mean(raw_mu, raw_log_sigma_left, raw_log_sigma_right, sigma_min, sigma_max):
    mu, sigma_left, sigma_right = _split_normal_params(
        raw_mu,
        raw_log_sigma_left,
        raw_log_sigma_right,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
    )
    return mu + math.sqrt(2.0 / math.pi) * (sigma_right.square() - sigma_left.square()) / (
        sigma_left + sigma_right
    ).clamp_min(1e-12)


def _split_normal_sample(raw_mu, raw_log_sigma_left, raw_log_sigma_right, sigma_min, sigma_max):
    mu, sigma_left, sigma_right = _split_normal_params(
        raw_mu,
        raw_log_sigma_left,
        raw_log_sigma_right,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
    )
    p_right = sigma_right / (sigma_left + sigma_right).clamp_min(1e-12)
    right_side = torch.rand_like(mu) < p_right
    magnitude = torch.distributions.HalfNormal(torch.ones_like(mu)).sample()
    return torch.where(right_side, mu + sigma_right * magnitude, mu - sigma_left * magnitude)


def _clamped_asymmetric_normal_nll(raw_mu, raw_log_sigma_left, raw_log_sigma_right, target, mask, eps, sigma_min, sigma_max):
    target = target.float()
    zero_mask = target <= float(eps)
    one_mask = target >= 1.0 - float(eps)
    p_zero = _split_normal_cdf(raw_mu, raw_log_sigma_left, raw_log_sigma_right, 0.0, sigma_min, sigma_max)
    p_one = 1.0 - _split_normal_cdf(raw_mu, raw_log_sigma_left, raw_log_sigma_right, 1.0, sigma_min, sigma_max)
    log_pdf = _split_normal_log_pdf(raw_mu, raw_log_sigma_left, raw_log_sigma_right, target.clamp(0.0, 1.0), sigma_min, sigma_max)
    values = -log_pdf
    values = torch.where(zero_mask, -torch.log(p_zero.clamp_min(1e-12)), values)
    values = torch.where(one_mask, -torch.log(p_one.clamp_min(1e-12)), values)
    return _masked_mean(values, mask)


def _clamped_asymmetric_normal_mean_or_sample(config, raw_mu, raw_log_sigma_left, raw_log_sigma_right, sampling_strategy="mean"):
    mode = str(sampling_strategy).lower()
    sigma_min = getattr(config, "logistic_normal_sigma_min", 1e-4)
    sigma_max = getattr(config, "logistic_normal_sigma_max", 1e4)
    if mode in {"mean", "deterministic", "mu", "expected", "expectation"}:
        return _split_normal_mean(raw_mu, raw_log_sigma_left, raw_log_sigma_right, sigma_min, sigma_max).clamp(0.0, 1.0)
    if mode in {"argmax", "greedy"}:
        return raw_mu.float().clamp(0.0, 1.0)
    if mode in {"sample", "sampling", "stochastic"}:
        return _split_normal_sample(raw_mu, raw_log_sigma_left, raw_log_sigma_right, sigma_min, sigma_max).clamp(0.0, 1.0)
    raise ValueError(f"Unsupported sampling_strategy: {sampling_strategy}")


def _inflated_clamped_asymmetric_normal_nll(
    raw_mode_logits,
    raw_mu,
    raw_log_sigma_left,
    raw_log_sigma_right,
    target,
    mask,
    eps,
    sigma_min,
    sigma_max,
):
    target = target.float()
    log_probs = _inflated_zero_one_mode_log_probs(raw_mode_logits)
    zero_mask = target <= float(eps)
    one_mask = target >= 1.0 - float(eps)
    log_pdf = _split_normal_log_pdf(
        raw_mu,
        raw_log_sigma_left,
        raw_log_sigma_right,
        target.clamp(0.0, 1.0),
        sigma_min,
        sigma_max,
    )
    values = -(log_probs[..., 2] + log_pdf)
    values = torch.where(zero_mask, -log_probs[..., 0], values)
    values = torch.where(one_mask, -log_probs[..., 1], values)
    return _masked_mean(values, mask)


def _inflated_clamped_asymmetric_normal_mean_or_sample(
    config,
    raw_mode_logits,
    raw_mu,
    raw_log_sigma_left,
    raw_log_sigma_right,
    sampling_strategy="mean",
):
    mode = str(sampling_strategy).lower()
    probs = _inflated_zero_one_mode_log_probs(raw_mode_logits).exp()
    continuous = _clamped_asymmetric_normal_mean_or_sample(
        config,
        raw_mu,
        raw_log_sigma_left,
        raw_log_sigma_right,
        sampling_strategy=sampling_strategy,
    )
    if mode in {"mean", "deterministic", "mu", "expected", "expectation"}:
        return probs[..., 1] + probs[..., 2] * continuous
    if mode in {"argmax", "greedy"}:
        mode_idx = probs.argmax(dim=-1)
    elif mode in {"sample", "sampling", "stochastic"}:
        mode_idx = torch.distributions.Categorical(probs=probs).sample()
    else:
        raise ValueError(f"Unsupported sampling_strategy: {sampling_strategy}")
    return torch.where(
        mode_idx == 0,
        continuous.new_zeros(()),
        torch.where(mode_idx == 1, continuous.new_ones(()), continuous),
    )


def _inflated_logistic_normal_nll(
    raw_mode_logits,
    raw_mu,
    raw_log_sigma,
    target,
    mask,
    eps,
    sigma_min,
    sigma_max,
):
    target = target.float()
    log_probs = _inflated_zero_one_mode_log_probs(raw_mode_logits)
    zero_mask = target <= float(eps)
    one_mask = target >= 1.0 - float(eps)
    center_target = target.clamp(float(eps), 1.0 - float(eps))
    raw_mu_1 = raw_mu.unsqueeze(-1)
    raw_log_sigma_1 = raw_log_sigma.unsqueeze(-1)
    log_pdf = _mixture_logistic_normal_log_prob(
        raw_mu_1.new_zeros(*raw_mu_1.shape),
        raw_mu_1,
        raw_log_sigma_1,
        center_target,
        eps,
        sigma_min,
        sigma_max,
    )
    values = -(log_probs[..., 2] + log_pdf)
    values = torch.where(zero_mask, -log_probs[..., 0], values)
    values = torch.where(one_mask, -log_probs[..., 1], values)
    return _masked_mean(values, mask)


def _inflated_logistic_normal_mean_or_sample(
    config,
    raw_mode_logits,
    raw_mu,
    raw_log_sigma,
    sampling_strategy="mean",
):
    mode = str(sampling_strategy).lower()
    probs = _inflated_zero_one_mode_log_probs(raw_mode_logits).exp()
    raw_mu_1 = raw_mu.unsqueeze(-1)
    raw_log_sigma_1 = raw_log_sigma.unsqueeze(-1)
    center = _mixture_logistic_normal_mean_or_sample(
        config,
        raw_mu_1.new_zeros(*raw_mu_1.shape),
        raw_mu_1,
        raw_log_sigma_1,
        sampling_strategy=sampling_strategy,
    )
    if mode in {"mean", "deterministic", "mu", "expected", "expectation"}:
        return probs[..., 1] + probs[..., 2] * center
    if mode in {"argmax", "greedy"}:
        mode_idx = probs.argmax(dim=-1)
    elif mode in {"sample", "sampling", "stochastic"}:
        mode_idx = torch.distributions.Categorical(probs=probs).sample()
    else:
        raise ValueError(f"Unsupported sampling_strategy: {sampling_strategy}")
    return torch.where(
        mode_idx == 0,
        center.new_zeros(()),
        torch.where(mode_idx == 1, center.new_ones(()), center),
    )


def _mixture_beta_params(
    raw_alpha,
    raw_beta,
    alpha_min=1e-4,
    parameterization="alpha_beta",
    kappa_min=1e-3,
):
    if str(parameterization).lower() in {"mu_kappa", "mean_concentration"}:
        mean = torch.sigmoid(raw_alpha.float())
        kappa = F.softplus(raw_beta.float()) + float(kappa_min)
        alpha = mean * kappa + float(alpha_min)
        beta = (1.0 - mean) * kappa + float(alpha_min)
        return alpha, beta, mean
    alpha = F.softplus(raw_alpha.float()) + float(alpha_min)
    beta = F.softplus(raw_beta.float()) + float(alpha_min)
    mean = alpha / (alpha + beta).clamp_min(1e-12)
    return alpha, beta, mean


def _mixture_beta_log_prob(
    logits, raw_alpha, raw_beta, target, eps, alpha_min,
    parameterization="alpha_beta", kappa_min=1e-3,
):
    target = target.float().clamp(float(eps), 1.0 - float(eps)).unsqueeze(-1)
    alpha, beta, _ = _mixture_beta_params(
        raw_alpha, raw_beta, alpha_min=alpha_min,
        parameterization=parameterization, kappa_min=kappa_min,
    )
    log_pi = F.log_softmax(logits.float(), dim=-1)
    log_beta = torch.distributions.Beta(alpha, beta).log_prob(target)
    return torch.logsumexp(log_pi + log_beta, dim=-1)


def _mixture_beta_nll(
    logits, raw_alpha, raw_beta, target, mask, eps, alpha_min,
    parameterization="alpha_beta", kappa_min=1e-3,
):
    values = -_mixture_beta_log_prob(
        logits, raw_alpha, raw_beta, target, eps, alpha_min,
        parameterization=parameterization, kappa_min=kappa_min,
    )
    return _masked_mean(values, mask)


def _discrete_bounded_component_log_probs(config, distribution, raw_a, raw_b):
    """Return normalized component log-masses on a strict unit-interval grid."""
    bins = int(getattr(config, "dlm_timing_bins", 256))
    if bins < 2:
        raise ValueError(f"Discrete bounded distributions require at least two bins, got {bins}")
    dtype = torch.float32
    device = raw_a.device
    edges = torch.linspace(0.0, 1.0, bins + 1, device=device, dtype=dtype)
    centers = (edges[:-1] + edges[1:]) * 0.5
    distribution = str(distribution).lower()

    if distribution in DISCRETE_LN_DISTRIBUTIONS:
        sigma_min = float(getattr(config, "logistic_normal_sigma_min", 1e-5))
        mu = raw_a.float().unsqueeze(-2)
        sigma = (F.softplus(raw_b.float()) + sigma_min).unsqueeze(-2)
        z_edges = torch.logit(edges[1:-1].clamp(1e-12, 1.0 - 1e-12))
        interior = (
            z_edges.view(*([1] * (raw_a.ndim - 1)), -1, 1) - mu
        ) / sigma
        standardized = torch.cat(
            (
                torch.full_like(interior[..., :1, :], -1e6),
                interior,
                torch.full_like(interior[..., :1, :], 1e6),
            ),
            dim=-2,
        )
        lower, upper = standardized[..., :-1, :], standardized[..., 1:, :]

        def logdiffexp(log_hi, log_lo):
            delta = (log_lo - log_hi).clamp_max(-torch.finfo(log_hi.dtype).eps)
            return log_hi + torch.log(-torch.expm1(delta))

        def stable_log_ndtr(value):
            # PyTorch's log_ndtr backward becomes NaN deep in the negative
            # tail. Use its Mills-ratio asymptotic there, with a clamped safe
            # input for the unselected regular branch.
            regular = torch.special.log_ndtr(value.clamp_min(-20.0))
            tail = value.clamp_max(-20.0)
            inv_sq = tail.reciprocal().square()
            correction = torch.log1p(-inv_sq + 3.0 * inv_sq.square())
            asymptotic = (
                -0.5 * tail.square()
                - torch.log(-tail)
                - 0.5 * math.log(2.0 * math.pi)
                + correction
            )
            return torch.where(value < -20.0, asymptotic, regular)

        # Subtract log-CDFs in the nearer tail. This avoids float32 CDF
        # cancellation and retains gradients even for very narrow components.
        use_left_tail = upper <= 0.0
        near_hi = torch.where(use_left_tail, upper, -lower)
        near_lo = torch.where(use_left_tail, lower, -upper)
        component_log_probs = logdiffexp(
            stable_log_ndtr(near_hi), stable_log_ndtr(near_lo)
        )
    elif distribution in DISCRETE_BETA_DISTRIBUTIONS:
        alpha_min = float(getattr(config, "beta_alpha_min", 1e-5))
        kappa_min = float(getattr(config, "mixture_beta_kappa_min", 1e-5))
        mean = torch.sigmoid(raw_a.float()).unsqueeze(-2)
        kappa = (F.softplus(raw_b.float()) + kappa_min).unsqueeze(-2)
        alpha = mean * kappa + alpha_min
        beta = (1.0 - mean) * kappa + alpha_min
        # A Beta-binomial is the genuinely discrete analogue of Beta: each
        # integer bin receives an analytic probability, including endpoints.
        n = bins - 1
        k = torch.arange(bins, device=device, dtype=dtype).view(
            *([1] * (raw_a.ndim - 1)), -1, 1
        )
        log_choose = torch.lgamma(torch.tensor(float(n + 1), device=device))
        log_choose = log_choose - torch.lgamma(k + 1.0) - torch.lgamma(n - k + 1.0)
        log_beta_num = (
            torch.lgamma(k + alpha)
            + torch.lgamma(n - k + beta)
            - torch.lgamma(n + alpha + beta)
        )
        log_beta_den = torch.lgamma(alpha) + torch.lgamma(beta) - torch.lgamma(alpha + beta)
        component_log_probs = log_choose + log_beta_num - log_beta_den
        component_log_probs = component_log_probs - torch.logsumexp(component_log_probs, dim=-2, keepdim=True)
    elif distribution in TRUNCATED_LOGISTIC_DISTRIBUTIONS:
        scale_min = float(getattr(config, "dlm_scale_min", 1e-5))
        loc = raw_a.float().unsqueeze(-2)
        scale = (F.softplus(raw_b.float()) + scale_min).unsqueeze(-2)
        edge_view = edges.view(*([1] * (raw_a.ndim - 1)), -1, 1)
        standardized = (edge_view - loc) / scale
        lower, upper = standardized[..., :-1, :], standardized[..., 1:, :]
        component_log_probs = (
            F.logsigmoid(upper)
            + F.logsigmoid(-lower)
            + torch.log(-torch.expm1(lower - upper))
        )
        component_log_probs = component_log_probs - torch.logsumexp(
            component_log_probs, dim=-2, keepdim=True
        )
    else:
        raise ValueError(f"Unsupported discrete bounded distribution={distribution}")
    return component_log_probs


def _discrete_bounded_log_probs(config, distribution, logits, raw_a, raw_b):
    component_log_probs = _discrete_bounded_component_log_probs(
        config, distribution, raw_a, raw_b
    )
    log_mix = F.log_softmax(logits.float(), dim=-1).unsqueeze(-2)
    log_probs = torch.logsumexp(log_mix + component_log_probs, dim=-1)
    return log_probs - torch.logsumexp(log_probs, dim=-1, keepdim=True)


def _discrete_bounded_nll(config, distribution, logits, raw_a, raw_b, target, mask):
    log_probs = _discrete_bounded_log_probs(config, distribution, logits, raw_a, raw_b)
    bins = log_probs.shape[-1]
    target_bins = torch.floor(target.float().clamp(0.0, 1.0) * bins).long().clamp(0, bins - 1)
    values = -log_probs.gather(-1, target_bins.unsqueeze(-1)).squeeze(-1)
    return _masked_mean(values, mask)


def _discrete_bounded_mean_or_sample(
    config, distribution, logits, raw_a, raw_b, sampling_strategy="mean"
):
    log_probs = _discrete_bounded_log_probs(config, distribution, logits, raw_a, raw_b)
    temperature = float(getattr(config, "dlm_sampling_temperature", 1.0))
    if not math.isfinite(temperature):
        raise ValueError(f"sampling temperature must be finite, got {temperature}")
    bins = log_probs.shape[-1]
    centers = (torch.arange(bins, device=log_probs.device, dtype=log_probs.dtype) + 0.5) / bins
    mode = str(sampling_strategy).lower()
    if temperature <= 0.0:
        index = log_probs.argmax(dim=-1)
        return centers[index]
    probs = torch.softmax(log_probs / temperature, dim=-1)
    if mode in {"mean", "deterministic", "mu", "expected", "expectation"}:
        return torch.sum(probs * centers, dim=-1)
    if mode in {"argmax", "greedy"}:
        return centers[probs.argmax(dim=-1)]
    if mode in {"sample", "sampling", "stochastic"}:
        top_k = int(getattr(config, "dlm_sampling_top_k", getattr(config, "sampling_top_k", 0)))
        if top_k > 0 and top_k < probs.shape[-1]:
            threshold = torch.topk(probs, top_k, dim=-1).values[..., -1, None]
            probs = probs.masked_fill(probs < threshold, 0.0)
            probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        top_p = float(getattr(config, "dlm_sampling_top_p", 1.0))
        probs = _nucleus_probs(probs, top_p=top_p)
        index = torch.distributions.Categorical(
            probs=probs.reshape(-1, probs.shape[-1])
        ).sample().reshape(probs.shape[:-1])
        return centers[index]
    raise ValueError(f"Unsupported sampling_strategy={sampling_strategy}")


def _mixture_unit_variance(config, distribution, logits, raw_a, raw_b):
    probs = torch.softmax(logits.float(), dim=-1)
    distribution = str(distribution).lower()
    if distribution == "mixture_beta":
        alpha, beta, component_mean = _mixture_beta_params(
            raw_a,
            raw_b,
            alpha_min=getattr(config, "beta_alpha_min", 1e-4),
            parameterization=getattr(config, "mixture_beta_parameterization", "alpha_beta"),
            kappa_min=getattr(config, "mixture_beta_kappa_min", 1e-3),
        )
        concentration = (alpha + beta).clamp_min(1e-12)
        component_var = alpha * beta / (
            concentration.square() * (concentration + 1.0)
        ).clamp_min(1e-12)
    elif distribution in {"logistic_normal", "mixture_logistic_normal"}:
        loc, sigma = _logistic_normal_params(
            raw_a,
            raw_b,
            getattr(config, "logistic_normal_sigma_min", 1e-3),
            getattr(config, "logistic_normal_sigma_max", 10.0),
        )
        component_mean = torch.sigmoid(
            loc / torch.sqrt(1.0 + (math.pi / 8.0) * sigma.square())
        )
        slope = component_mean * (1.0 - component_mean)
        component_var = slope.square() * sigma.square()
    else:
        raise ValueError(f"Predictive variance is not implemented for {distribution}")
    mixture_mean = torch.sum(probs * component_mean, dim=-1)
    second_moment = torch.sum(
        probs * (component_var + component_mean.square()), dim=-1
    )
    return (second_moment - mixture_mean.square()).clamp_min(0.0)


def _predictive_variance_penalty(config, variance, mask, radius_unit):
    radius = torch.as_tensor(radius_unit, device=variance.device, dtype=variance.dtype)
    excess = F.relu(torch.sqrt(variance + 1e-12) - radius)
    return float(getattr(config, "predictive_variance_lambda", 0.0)) * _masked_mean(
        excess.square(), mask
    )


def _skew_normal_params(raw_loc, raw_log_scale, raw_alpha, sigma_min=1e-4, sigma_max=1e4):
    log_min = torch.log(raw_log_scale.new_tensor(float(sigma_min)))
    log_max = torch.log(raw_log_scale.new_tensor(float(sigma_max)))
    scale = torch.exp(raw_log_scale.float().clamp(min=log_min.item(), max=log_max.item()))
    return raw_loc.float(), scale, raw_alpha.float()


def _skew_normal_log_prob(raw_loc, raw_log_scale, raw_alpha, target, sigma_min=1e-4, sigma_max=1e4):
    loc, scale, alpha = _skew_normal_params(
        raw_loc,
        raw_log_scale,
        raw_alpha,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
    )
    z = (target.float() - loc) / scale.clamp_min(1e-12)
    standard = torch.distributions.Normal(torch.zeros_like(z), torch.ones_like(z))
    return (
        math.log(2.0)
        - torch.log(scale.clamp_min(1e-12))
        + standard.log_prob(z)
        + torch.log(standard.cdf(alpha * z).clamp_min(1e-12))
    )


def _skew_normal_nll(raw_loc, raw_log_scale, raw_alpha, target, mask, sigma_min=1e-4, sigma_max=1e4):
    values = -_skew_normal_log_prob(
        raw_loc,
        raw_log_scale,
        raw_alpha,
        target,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
    )
    return _masked_mean(values, mask)


def _skew_normal_log_prob_folded_abs(
    raw_loc,
    raw_log_scale,
    raw_alpha,
    target,
    sigma_min=1e-4,
    sigma_max=1e4,
    eps=1e-6,
):
    target = target.float()
    positive_target = target.clamp_min(float(eps))
    log_prob_pos = _skew_normal_log_prob(
        raw_loc,
        raw_log_scale,
        raw_alpha,
        positive_target,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
    )
    log_prob_neg = _skew_normal_log_prob(
        raw_loc,
        raw_log_scale,
        raw_alpha,
        -positive_target,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
    )
    folded_log_prob = torch.logsumexp(torch.stack([log_prob_pos, log_prob_neg], dim=0), dim=0)
    zero_bin_log_prob = (
        _skew_normal_log_prob(
            raw_loc,
            raw_log_scale,
            raw_alpha,
            torch.zeros_like(target),
            sigma_min=sigma_min,
            sigma_max=sigma_max,
        )
        + math.log(max(2.0 * float(eps), 1e-12))
    )
    return torch.where(target <= float(eps), zero_bin_log_prob, folded_log_prob)


def _dual_zero_ioi_timing_nll(
    params,
    feature_index,
    target,
    mask,
    zero_ioi_mask,
    zero_mode,
    sigma_min=1e-4,
    sigma_max=1e4,
    eps=1e-6,
):
    normal_log_prob = _skew_normal_log_prob(
        params["timing_log_loc"][..., feature_index],
        params["timing_log_log_scale"][..., feature_index],
        params["timing_log_alpha"][..., feature_index],
        target,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
    )
    zero_log_prob = _skew_normal_log_prob(
        params["timing_zero_log_loc"][..., feature_index],
        params["timing_zero_log_log_scale"][..., feature_index],
        params["timing_zero_log_alpha"][..., feature_index],
        target,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
    )
    if feature_index == 0 and str(zero_mode).lower() == "zero_folded":
        zero_log_prob = _skew_normal_log_prob_folded_abs(
            params["timing_zero_log_loc"][..., feature_index],
            params["timing_zero_log_log_scale"][..., feature_index],
            params["timing_zero_log_alpha"][..., feature_index],
            target,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            eps=eps,
        )
    use_zero = zero_ioi_mask.to(dtype=torch.bool, device=target.device)
    return _masked_mean(-torch.where(use_zero, zero_log_prob, normal_log_prob), mask)


def _inverse_softplus(value, eps=1e-6):
    value = value.float().clamp_min(float(eps))
    threshold = value.new_tensor(20.0)
    return torch.where(value > threshold, value, value + torch.log((-torch.expm1(-value)).clamp_min(1e-12)))


def _zero_score_ioi_mask(config, score_shared_raw, attention_mask=None):
    if score_shared_raw is None:
        return None
    eps = float(_config_value(config, "zero_ioi_support_eps", 1e-6))
    zero_mask = score_shared_raw[..., 0].float().abs() <= eps
    if attention_mask is not None:
        zero_mask = zero_mask & attention_mask.bool()
    return zero_mask


def _skew_normal_nll_zero_ioi_positive_support(
    raw_loc,
    raw_log_scale,
    raw_alpha,
    target,
    mask,
    zero_ioi_mask,
    sigma_min=1e-4,
    sigma_max=1e4,
    eps=1e-6,
    transform="softplus",
):
    target = target.float()
    use_transform = zero_ioi_mask.to(dtype=torch.bool, device=target.device)
    transform = str(transform or "softplus").lower()
    base_log_prob = _skew_normal_log_prob(
        raw_loc,
        raw_log_scale,
        raw_alpha,
        target,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
    )
    if transform == "softplus":
        latent_target = _inverse_softplus(target, eps=eps)
        transformed_log_prob = _skew_normal_log_prob(
            raw_loc,
            raw_log_scale,
            raw_alpha,
            latent_target,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
        ) - F.logsigmoid(latent_target)
    elif transform == "folded_abs":
        positive_target = target.clamp_min(float(eps))
        log_prob_pos = _skew_normal_log_prob(
            raw_loc,
            raw_log_scale,
            raw_alpha,
            positive_target,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
        )
        log_prob_neg = _skew_normal_log_prob(
            raw_loc,
            raw_log_scale,
            raw_alpha,
            -positive_target,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
        )
        folded_log_prob = torch.logsumexp(torch.stack([log_prob_pos, log_prob_neg], dim=0), dim=0)
        zero_bin_log_prob = (
            _skew_normal_log_prob(
                raw_loc,
                raw_log_scale,
                raw_alpha,
                torch.zeros_like(target),
                sigma_min=sigma_min,
                sigma_max=sigma_max,
            )
            + math.log(max(2.0 * float(eps), 1e-12))
        )
        transformed_log_prob = torch.where(target <= float(eps), zero_bin_log_prob, folded_log_prob)
    elif transform == "squared":
        positive_target = target.clamp_min(float(eps))
        sqrt_target = torch.sqrt(positive_target)
        log_prob_pos = _skew_normal_log_prob(
            raw_loc,
            raw_log_scale,
            raw_alpha,
            sqrt_target,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
        )
        log_prob_neg = _skew_normal_log_prob(
            raw_loc,
            raw_log_scale,
            raw_alpha,
            -sqrt_target,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
        )
        squared_log_prob = torch.logsumexp(torch.stack([log_prob_pos, log_prob_neg], dim=0), dim=0) - torch.log(
            (2.0 * sqrt_target).clamp_min(1e-12)
        )
        sqrt_eps = math.sqrt(max(float(eps), 1e-12))
        zero_bin_log_prob = (
            _skew_normal_log_prob(
                raw_loc,
                raw_log_scale,
                raw_alpha,
                torch.zeros_like(target),
                sigma_min=sigma_min,
                sigma_max=sigma_max,
            )
            + math.log(max(2.0 * sqrt_eps, 1e-12))
        )
        transformed_log_prob = torch.where(target <= float(eps), zero_bin_log_prob, squared_log_prob)
    else:
        raise ValueError(f"Unsupported zero_ioi_transform={transform}")
    nll_values = -torch.where(use_transform, transformed_log_prob, base_log_prob)
    return _masked_mean(nll_values, mask)


def _apply_zero_ioi_positive_support(config, logdev, score_shared_raw):
    transform = str(_config_value(config, "zero_ioi_transform", None) or "").lower()
    enabled = bool(_config_value(config, "zero_ioi_positive_support", False)) or transform in {
        "softplus",
        "folded_abs",
        "squared",
    }
    if not enabled or score_shared_raw is None:
        return logdev
    zero_mask = _zero_score_ioi_mask(config, score_shared_raw)
    if zero_mask is None:
        return logdev
    transform = transform or "softplus"
    if transform == "softplus":
        transformed_ioi = F.softplus(logdev[..., 0])
    elif transform == "folded_abs":
        transformed_ioi = logdev[..., 0].abs()
    elif transform == "squared":
        transformed_ioi = logdev[..., 0].square()
    else:
        raise ValueError(f"Unsupported zero_ioi_transform={transform}")
    ioi = torch.where(zero_mask.to(device=logdev.device), transformed_ioi, logdev[..., 0])
    return torch.cat([ioi.unsqueeze(-1), logdev[..., 1:]], dim=-1)


def _skew_normal_mean_or_sample(
    raw_loc,
    raw_log_scale,
    raw_alpha,
    sampling_strategy="mean",
    sigma_min=1e-4,
    sigma_max=1e4,
):
    loc, scale, alpha = _skew_normal_params(
        raw_loc,
        raw_log_scale,
        raw_alpha,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
    )
    delta = alpha / torch.sqrt(1.0 + alpha.square())
    mode = str(sampling_strategy).lower()
    if mode in {"mean", "deterministic", "mu", "expected", "expectation", "argmax", "greedy"}:
        return loc + scale * delta * math.sqrt(2.0 / math.pi)
    if mode in {"sample", "sampling", "stochastic"}:
        u0 = torch.randn_like(loc)
        u1 = torch.randn_like(loc)
        z = delta * u0.abs() + torch.sqrt((1.0 - delta.square()).clamp_min(1e-12)) * u1
        return loc + scale * z
    raise ValueError(f"Unsupported sampling_strategy: {sampling_strategy}")


def _timing_truncate_radius(config):
    radius = getattr(config, "timing_sample_truncate_radius", None)
    if radius is None:
        radius = getattr(config, "dlm_timing_sample_truncate_radius", 0.0)
    return float(radius or 0.0)


def _timing_truncate_center_mode(config):
    center = getattr(config, "timing_sample_truncate_center", None)
    if center is None:
        center = getattr(config, "dlm_timing_sample_truncate_center", "mean")
    return str(center or "mean").lower()


def _apply_timing_sample_shrink(config, sample, center):
    mode = str(getattr(config, "timing_sample_shrink_mode", "none") or "none").lower()
    if mode in {"none", "off", "false", "0"}:
        return sample
    delta = sample - center
    if mode in {"linear", "scale", "shrink"}:
        factor = float(getattr(config, "timing_sample_shrink_factor", 1.0))
        return center + factor * delta
    if mode in {"tanh", "softcap", "soft_cap"}:
        radius = float(getattr(config, "timing_sample_shrink_radius", 0.0) or 0.0)
        if radius <= 0.0:
            return sample
        return center + radius * torch.tanh(delta / radius)
    raise ValueError(f"Unsupported timing_sample_shrink_mode={mode}")


def _apply_dlm_attribute_sample_shrink(config, sample, center, feature, zero_mask=None):
    """Shrink stochastic DLM draws around their exact discrete mean at inference only."""
    feature = str(feature)
    if feature == "ioi":
        nonzero = getattr(config, "dlm_ioi_nonzero_sample_shrink_factor", None)
        zero = getattr(config, "dlm_ioi_zero_sample_shrink_factor", None)
        if nonzero is None and zero is None:
            return _apply_timing_sample_shrink(config, sample, center)
        nonzero = 1.0 if nonzero is None else float(nonzero)
        zero = nonzero if zero is None else float(zero)
        factor = sample.new_full(sample.shape, nonzero)
        if zero_mask is not None:
            factor = torch.where(zero_mask.bool(), factor.new_full((), zero), factor)
    elif feature == "duration":
        factor = getattr(config, "dlm_duration_sample_shrink_factor", None)
        if factor is None:
            return _apply_timing_sample_shrink(config, sample, center)
        factor = float(factor)
    elif feature == "velocity":
        factor = getattr(config, "dlm_velocity_sample_shrink_factor", None)
        if factor is None:
            return sample
        factor = float(factor)
    else:
        return sample
    if torch.is_tensor(factor):
        if not torch.isfinite(factor).all() or (factor < 0.0).any():
            raise ValueError(f"Invalid DLM {feature} sample shrink factor")
    elif not math.isfinite(factor) or factor < 0.0:
        raise ValueError(f"Invalid DLM {feature} sample shrink factor={factor}")
    return center + factor * (sample - center)


def _skew_normal_mean_or_sample_timing(
    config,
    raw_loc,
    raw_log_scale,
    raw_alpha,
    sampling_strategy="mean",
    sigma_min=1e-4,
    sigma_max=1e4,
):
    mode = str(sampling_strategy).lower()
    radius = _timing_truncate_radius(config)
    if mode not in {"sample", "sampling", "stochastic"}:
        return _skew_normal_mean_or_sample(
            raw_loc,
            raw_log_scale,
            raw_alpha,
            sampling_strategy=sampling_strategy,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
        )
    center_mode = _timing_truncate_center_mode(config)
    center_strategy = "argmax" if center_mode in {"mode", "argmax", "greedy"} else "mean"
    center = _skew_normal_mean_or_sample(
        raw_loc,
        raw_log_scale,
        raw_alpha,
        sampling_strategy=center_strategy,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
    )
    if radius <= 0.0:
        sample = _skew_normal_mean_or_sample(
            raw_loc,
            raw_log_scale,
            raw_alpha,
            sampling_strategy="sample",
            sigma_min=sigma_min,
            sigma_max=sigma_max,
        )
        return _apply_timing_sample_shrink(config, sample, center)
    sample = center
    accepted = torch.zeros_like(center, dtype=torch.bool)
    for _ in range(32):
        candidate = _skew_normal_mean_or_sample(
            raw_loc,
            raw_log_scale,
            raw_alpha,
            sampling_strategy="sample",
            sigma_min=sigma_min,
            sigma_max=sigma_max,
        )
        ok = (candidate >= center - radius) & (candidate <= center + radius)
        sample = torch.where((~accepted) & ok, candidate, sample)
        accepted = accepted | ok
        if bool(accepted.all()):
            break
    return _apply_timing_sample_shrink(config, sample.clamp(center - radius, center + radius), center)


def _mixture_logistic_normal_mean_or_sample(config, logits, raw_mu, raw_log_sigma, sampling_strategy="mean"):
    mode = str(sampling_strategy).lower()
    mu, sigma = _logistic_normal_params(
        raw_mu,
        raw_log_sigma,
        sigma_min=getattr(config, "logistic_normal_sigma_min", 1e-3),
        sigma_max=getattr(config, "logistic_normal_sigma_max", 10.0),
    )
    probs = torch.softmax(logits.float(), dim=-1)
    if mode in {"mean", "deterministic", "mu", "expected", "expectation"}:
        return torch.sum(probs * torch.sigmoid(mu), dim=-1)
    if mode in {"argmax", "greedy"}:
        index = probs.argmax(dim=-1, keepdim=True)
        return torch.sigmoid(mu.gather(dim=-1, index=index).squeeze(-1))
    if mode in {"sample", "sampling", "stochastic"}:
        index = torch.distributions.Categorical(probs=probs).sample().unsqueeze(-1)
        sampled_mu = mu.gather(dim=-1, index=index).squeeze(-1)
        sampled_sigma = sigma.gather(dim=-1, index=index).squeeze(-1)
        return torch.sigmoid(torch.distributions.Normal(sampled_mu, sampled_sigma).sample())
    raise ValueError(f"Unsupported sampling_strategy: {sampling_strategy}")


def _mixture_logistic_normal_mean_or_sample_timing(config, logits, raw_mu, raw_log_sigma, sampling_strategy="mean"):
    mode = str(sampling_strategy).lower()
    radius = _timing_truncate_radius(config)
    if mode not in {"sample", "sampling", "stochastic"}:
        return _mixture_logistic_normal_mean_or_sample(
            config,
            logits,
            raw_mu,
            raw_log_sigma,
            sampling_strategy=sampling_strategy,
        )
    center_mode = _timing_truncate_center_mode(config)
    center_strategy = "argmax" if center_mode in {"mode", "argmax", "greedy"} else "mean"
    center = _mixture_logistic_normal_mean_or_sample(
        config,
        logits,
        raw_mu,
        raw_log_sigma,
        sampling_strategy=center_strategy,
    )
    if radius <= 0.0:
        sample = _mixture_logistic_normal_mean_or_sample(
            config,
            logits,
            raw_mu,
            raw_log_sigma,
            sampling_strategy="sample",
        )
        return _apply_timing_sample_shrink(config, sample, center)
    sample = center
    accepted = torch.zeros_like(center, dtype=torch.bool)
    for _ in range(32):
        candidate = _mixture_logistic_normal_mean_or_sample(
            config,
            logits,
            raw_mu,
            raw_log_sigma,
            sampling_strategy="sample",
        )
        ok = (candidate >= center - radius) & (candidate <= center + radius)
        sample = torch.where((~accepted) & ok, candidate, sample)
        accepted = accepted | ok
        if bool(accepted.all()):
            break
    return _apply_timing_sample_shrink(config, sample.clamp(center - radius, center + radius), center)


def _logistic_asymmetric_normal_mean_or_sample(
    config,
    logits,
    raw_mu,
    raw_log_sigma_left,
    raw_log_sigma_right,
    sampling_strategy="mean",
):
    mode = str(sampling_strategy).lower()
    mu, sigma_left = _logistic_normal_params(
        raw_mu,
        raw_log_sigma_left,
        sigma_min=getattr(config, "logistic_normal_sigma_min", 1e-3),
        sigma_max=getattr(config, "logistic_normal_sigma_max", 10.0),
    )
    _, sigma_right = _logistic_normal_params(
        raw_mu,
        raw_log_sigma_right,
        sigma_min=getattr(config, "logistic_normal_sigma_min", 1e-3),
        sigma_max=getattr(config, "logistic_normal_sigma_max", 10.0),
    )
    probs = torch.softmax(logits.float(), dim=-1)
    denom = (sigma_left + sigma_right).clamp_min(1e-12)
    p_left = sigma_left / denom
    p_right = sigma_right / denom
    if mode in {"mean", "deterministic", "mu", "expected", "expectation"}:
        component_mean_z = mu + math.sqrt(2.0 / math.pi) * (p_right * sigma_right - p_left * sigma_left)
        return torch.sum(probs * torch.sigmoid(component_mean_z), dim=-1)
    if mode in {"argmax", "greedy"}:
        index = probs.argmax(dim=-1, keepdim=True)
        return torch.sigmoid(mu.gather(dim=-1, index=index).squeeze(-1))
    if mode in {"sample", "sampling", "stochastic"}:
        index = torch.distributions.Categorical(probs=probs).sample().unsqueeze(-1)
        sampled_mu = mu.gather(dim=-1, index=index).squeeze(-1)
        sampled_left = sigma_left.gather(dim=-1, index=index).squeeze(-1)
        sampled_right = sigma_right.gather(dim=-1, index=index).squeeze(-1)
        sampled_p_right = p_right.gather(dim=-1, index=index).squeeze(-1)
        side = torch.rand_like(sampled_mu) >= sampled_p_right
        magnitude = torch.distributions.HalfNormal(torch.ones_like(sampled_mu)).sample()
        z = torch.where(
            side,
            sampled_mu - sampled_left * magnitude,
            sampled_mu + sampled_right * magnitude,
        )
        return torch.sigmoid(z)
    raise ValueError(f"Unsupported sampling_strategy: {sampling_strategy}")


def _mixture_beta_mean_or_sample(config, logits, raw_alpha, raw_beta, sampling_strategy="mean"):
    mode = str(sampling_strategy).lower()
    alpha, beta, mean = _mixture_beta_params(
        raw_alpha,
        raw_beta,
        alpha_min=getattr(config, "beta_alpha_min", 1e-4),
        parameterization=getattr(config, "mixture_beta_parameterization", "alpha_beta"),
        kappa_min=getattr(config, "mixture_beta_kappa_min", 1e-3),
    )
    probs = torch.softmax(logits.float(), dim=-1)
    if mode in {"mean", "deterministic", "mu", "expected", "expectation"}:
        return torch.sum(probs * mean, dim=-1)
    if mode in {"argmax", "greedy"}:
        index = probs.argmax(dim=-1, keepdim=True)
        return mean.gather(dim=-1, index=index).squeeze(-1)
    if mode in {"sample", "sampling", "stochastic"}:
        index = torch.distributions.Categorical(probs=probs).sample().unsqueeze(-1)
        sampled_alpha = alpha.gather(dim=-1, index=index).squeeze(-1)
        sampled_beta = beta.gather(dim=-1, index=index).squeeze(-1)
        return torch.distributions.Beta(sampled_alpha, sampled_beta).sample()
    raise ValueError(f"Unsupported sampling_strategy: {sampling_strategy}")


def _decode_mixture_value(config, logits, a, b, c=None, sampling_strategy="mean", distribution=None):
    distribution = (distribution or getattr(config, "epr_distribution", "point")).lower()
    if distribution in DISCRETE_BOUNDED_DISTRIBUTIONS:
        return _discrete_bounded_mean_or_sample(
            config, distribution, logits, a, b, sampling_strategy=sampling_strategy
        )
    if distribution in TANH_T_DISTRIBUTIONS:
        return _bounded_student_t_mean_or_sample(
            config, a.squeeze(-1), b.squeeze(-1), logits.squeeze(-1), sampling_strategy=sampling_strategy
        )
    if distribution in BOUNDED_SN_DISTRIBUTIONS:
        return _bounded_skew_normal_mean_or_sample(
            config, a.squeeze(-1), b.squeeze(-1), logits.squeeze(-1), sampling_strategy=sampling_strategy
        )
    if distribution in ACN_DISTRIBUTIONS:
        if c is None:
            raise ValueError("ACN decoding requires right-scale parameter c")
        return _clamped_asymmetric_normal_mean_or_sample(
            config,
            a,
            b,
            c,
            sampling_strategy=sampling_strategy,
        )
    if distribution in IACN_DISTRIBUTIONS:
        if c is None:
            raise ValueError("IACN decoding requires right-scale parameter c")
        return _inflated_clamped_asymmetric_normal_mean_or_sample(
            config,
            logits,
            a,
            b,
            c,
            sampling_strategy=sampling_strategy,
        )
    if distribution in ILN_DISTRIBUTIONS:
        return _inflated_logistic_normal_mean_or_sample(
            config,
            logits,
            a,
            b,
            sampling_strategy=sampling_strategy,
        )
    if distribution == "mixture_beta":
        return _mixture_beta_mean_or_sample(config, logits, a, b, sampling_strategy=sampling_strategy)
    if distribution in ALN_DISTRIBUTIONS:
        if c is None:
            return _mixture_logistic_normal_mean_or_sample(config, logits, a, b, sampling_strategy=sampling_strategy)
        return _logistic_asymmetric_normal_mean_or_sample(
            config,
            logits,
            a,
            b,
            c,
            sampling_strategy=sampling_strategy,
        )
    return _mixture_logistic_normal_mean_or_sample(config, logits, a, b, sampling_strategy=sampling_strategy)


def _decode_scalar_mln_value(config, logits, raw_mu, raw_log_sigma, sampling_strategy="mean"):
    return _mixture_logistic_normal_mean_or_sample(config, logits, raw_mu, raw_log_sigma, sampling_strategy=sampling_strategy)


def _scalar_mln_loss_value(config, logits, raw_mu, raw_log_sigma, target, mask, eps, sigma_min, sigma_max):
    return _mixture_logistic_normal_nll(logits, raw_mu, raw_log_sigma, target, mask, eps, sigma_min, sigma_max)


def _mixture_loss_value(config, logits, a, b, target, mask, eps, sigma_min, sigma_max, alpha_min, c=None, distribution=None):
    distribution = (distribution or getattr(config, "epr_distribution", "point")).lower()
    if distribution in DISCRETE_BOUNDED_DISTRIBUTIONS:
        return _discrete_bounded_nll(
            config, distribution, logits, a, b, target, mask
        )
    if distribution in TANH_T_DISTRIBUTIONS:
        return _bounded_student_t_nll(
            a.squeeze(-1), b.squeeze(-1), logits.squeeze(-1), target, mask, eps, sigma_min, sigma_max
        )
    if distribution in BOUNDED_SN_DISTRIBUTIONS:
        return _bounded_skew_normal_nll(
            a.squeeze(-1), b.squeeze(-1), logits.squeeze(-1), target, mask, eps,
            getattr(config, "skew_normal_sigma_min", sigma_min),
            getattr(config, "skew_normal_sigma_max", sigma_max),
        )
    if distribution in ACN_DISTRIBUTIONS:
        if c is None:
            raise ValueError("ACN loss requires right-scale parameter c")
        return _clamped_asymmetric_normal_nll(
            a,
            b,
            c,
            target,
            mask,
            eps,
            sigma_min,
            sigma_max,
        )
    if distribution in IACN_DISTRIBUTIONS:
        if c is None:
            raise ValueError("IACN loss requires right-scale parameter c")
        return _inflated_clamped_asymmetric_normal_nll(
            logits,
            a,
            b,
            c,
            target,
            mask,
            eps,
            sigma_min,
            sigma_max,
        )
    if distribution in ILN_DISTRIBUTIONS:
        return _inflated_logistic_normal_nll(
            logits,
            a,
            b,
            target,
            mask,
            eps,
            sigma_min,
            sigma_max,
        )
    if distribution == "mixture_beta":
        return _mixture_beta_nll(
            logits, a, b, target, mask, eps, alpha_min,
            parameterization=getattr(config, "mixture_beta_parameterization", "alpha_beta"),
            kappa_min=getattr(config, "mixture_beta_kappa_min", 1e-3),
        )
    if distribution in ALN_DISTRIBUTIONS:
        if c is None:
            return _mixture_logistic_normal_nll(logits, a, b, target, mask, eps, sigma_min, sigma_max)
        return _logistic_asymmetric_normal_nll(logits, a, b, c, target, mask, eps, sigma_min, sigma_max)
    return _mixture_logistic_normal_nll(logits, a, b, target, mask, eps, sigma_min, sigma_max)


def _categorical_sample_or_argmax(logits, sampling_strategy="mean"):
    mode = str(sampling_strategy).lower()
    if mode in {"mean", "deterministic", "mu", "argmax", "greedy"}:
        return logits.float().argmax(dim=-1)
    if mode in {"sample", "sampling", "stochastic"}:
        probs = torch.softmax(logits.float(), dim=-1)
        return torch.distributions.Categorical(probs=probs).sample()
    raise ValueError(f"Unsupported sampling_strategy: {sampling_strategy}")


def _nucleus_probs(probs, top_p=1.0):
    """Keep the smallest highest-probability set whose cumulative mass reaches top_p."""
    top_p = float(top_p)
    if not math.isfinite(top_p) or top_p <= 0.0 or top_p > 1.0:
        raise ValueError(f"sampling top_p must be in (0, 1], got {top_p}")
    if top_p >= 1.0:
        return probs
    sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
    remove = torch.cumsum(sorted_probs, dim=-1) - sorted_probs >= top_p
    sorted_probs = sorted_probs.masked_fill(remove, 0.0)
    filtered = torch.zeros_like(probs).scatter(-1, sorted_indices, sorted_probs)
    return filtered / filtered.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def _categorical_sample_with_temperature_top_p(
    logits, sampling_strategy, temperature=1.0, top_p=1.0, top_k=0
):
    scaled = logits.float() / float(temperature)
    mode = str(sampling_strategy).lower()
    if mode in {"sample", "sampling", "stochastic"}:
        top_k = int(top_k)
        if top_k > 0 and top_k < scaled.shape[-1]:
            threshold = torch.topk(scaled, top_k, dim=-1).values[..., -1, None]
            scaled = scaled.masked_fill(scaled < threshold, float("-inf"))
        probs = _nucleus_probs(torch.softmax(scaled, dim=-1), top_p=top_p)
        return torch.distributions.Categorical(probs=probs).sample()
    return _categorical_sample_or_argmax(scaled, sampling_strategy)


def _soft_categorical_sample_or_expected(logits, sampling_strategy="mean"):
    mode = str(sampling_strategy).lower()
    if mode in {"mean", "deterministic", "mu", "expected", "expectation"}:
        probs = torch.softmax(logits.float(), dim=-1)
        values = torch.arange(logits.shape[-1], device=logits.device, dtype=probs.dtype)
        return torch.sum(probs * values, dim=-1)
    if mode in {"argmax", "greedy"}:
        return logits.float().argmax(dim=-1)
    if mode in {"sample", "sampling", "stochastic"}:
        probs = torch.softmax(logits.float(), dim=-1)
        return torch.distributions.Categorical(probs=probs).sample()
    raise ValueError(f"Unsupported sampling_strategy: {sampling_strategy}")


def _epr_bins_to_normalized(config, ioi_bins, duration_bins, velocity_bins, pedal_bins):
    timing_bins = max(1, int(config.epr_timing_bins))
    value_bins = max(1, int(config.epr_value_bins))
    value_scale = float(value_bins - 1) if value_bins > 1 else 1.0
    timing_norm = str(getattr(config, "timing_input_normalization", "linear_5000")).lower()
    ioi_ms = ioi_bins.to(dtype=torch.float32).clamp(0.0, float(timing_bins - 1))
    duration_ms = duration_bins.to(dtype=torch.float32).clamp(0.0, float(timing_bins - 1))
    if timing_norm in {"linear_5000", "linear_5000"}:
        ioi_norm = ioi_ms.clamp(max=5000.0) / 5000.0
        duration_norm = duration_ms.clamp(max=5000.0) / 5000.0
    else:
        raise ValueError(f"Unsupported timing normalization: {timing_norm}")
    return torch.cat(
        [
            ioi_norm.unsqueeze(-1),
            duration_norm.unsqueeze(-1),
            velocity_bins.to(dtype=torch.float32).unsqueeze(-1) / value_scale,
            pedal_bins.to(dtype=torch.float32) / value_scale,
        ],
        dim=-1,
    ).to(dtype=torch.float32)


def _materialize_binary4_logits(logits, sampling_strategy="mean"):
    probs = torch.sigmoid(logits.float())
    mode_name = str(sampling_strategy).lower()
    if mode_name in {"soft", "prob", "probs", "probability", "probabilities"}:
        return probs
    if mode_name in {"sample", "sampling", "stochastic"}:
        return torch.bernoulli(probs)
    return (probs >= 0.5).to(dtype=probs.dtype)


def _materialize_pedal_params(config, params, sampling_strategy="mean"):
    representation = normalize_pedal_representation(getattr(config, "pedal_representation", "binary_4"))
    if representation == "binary_4":
        return _materialize_binary4_logits(params["pedal_binary_logits"], sampling_strategy=sampling_strategy)
    raise ValueError("Only pedal_representation=binary_4 is supported")


def _materialize_pedal_raw(config, raw_pedal, sampling_strategy="mean"):
    representation = normalize_pedal_representation(getattr(config, "pedal_representation", "binary_4"))
    if representation == "binary_4":
        return _materialize_binary4_logits(raw_pedal, sampling_strategy=sampling_strategy)
    raise ValueError("Only pedal_representation=binary_4 is supported")


def _materialize_epr_prediction(config, raw_outputs, sampling_strategy="mean", score_shared_raw=None):
    if not _uses_epr_targets(config):
        raise ValueError("EPR materialization only supports configured EPR timing targets")

    strategy_name = str(sampling_strategy).lower()
    shared_strategy = "mean" if strategy_name in {"soft", "prob", "probs", "probability", "probabilities"} else sampling_strategy
    distribution = getattr(config, "epr_distribution", "point").lower()
    if distribution in DINR_DISTRIBUTIONS:
        logits = _split_dinr_logits(config, raw_outputs)
        ioi_logits = _dinr_support_mask(config, logits["ioi"], "ioi", score_shared_raw)
        duration_logits = _dinr_support_mask(config, logits["duration"], "duration")
        temperature = float(getattr(config, "dinr_sampling_temperature", 1.0))
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ValueError("dinr_sampling_temperature must be finite and positive")
        top_p = float(getattr(config, "dinr_sampling_top_p", getattr(config, "sampling_top_p", 1.0)))
        top_k = int(getattr(config, "dinr_sampling_top_k", getattr(config, "sampling_top_k", 0)))
        ioi_bins = _categorical_sample_with_temperature_top_p(ioi_logits, shared_strategy, temperature, top_p, top_k)
        duration_bins = _categorical_sample_with_temperature_top_p(duration_logits, shared_strategy, temperature, top_p, top_k)
        velocity_bins = _categorical_sample_with_temperature_top_p(logits["velocity"], shared_strategy, temperature, top_p, top_k)
        dev_step = float(config.dinr_output_timing_step)
        dev_zero_bin = int(config.dinr_output_zero_bin)
        ioi = (ioi_bins.float() - dev_zero_bin) * dev_step
        if str(getattr(config, "dinr_vocabulary_mode", "unified")).lower() == "separated":
            if score_shared_raw is None:
                raise ValueError("Separated DINR materialization requires score_shared_raw")
            absolute_ioi = (
                ioi_bins.float() - int(config.dinr_zero_bin)
            ) * float(config.dinr_timing_step)
            zero_score = score_shared_raw[..., 0].abs() <= float(
                getattr(config, "zero_ioi_support_eps", 1e-6)
            )
            ioi = torch.where(zero_score, absolute_ioi, ioi)
        duration = (duration_bins.float() - dev_zero_bin) * dev_step
        velocity = velocity_bins.float().clamp(0.0, 127.0) / 127.0
        if logits["pedal"].ndim == raw_outputs.ndim + 1 and logits["pedal"].shape[-1] == 2:
            pedal = _categorical_sample_or_argmax(logits["pedal"] / temperature, sampling_strategy).float()
        else:
            pedal = _materialize_binary4_logits(logits["pedal"], sampling_strategy=sampling_strategy)
        return torch.cat([torch.stack([ioi, duration, velocity], dim=-1), pedal], dim=-1)
    if distribution in DLM_DISTRIBUTIONS:
        params = _split_epr_mixture_params(config, raw_outputs)
        if score_shared_raw is None:
            raise ValueError("DLM materialization requires score_shared_raw for score-IOI group ranges")
        zero_mask = _zero_score_ioi_mask(config, score_shared_raw)
        if bool(getattr(config, "dlm_ioi_zero_inflated", False)):
            ioi = _dlm_zero_inflated_mean_or_sample(
                config,
                params["ioi_zero_logit"],
                params["ioi_logits"],
                params["ioi_loc"],
                params["ioi_log_scale"],
                "ioi",
                sampling_strategy=shared_strategy,
                zero_mask=zero_mask,
            )
        else:
            ioi = _dlm_mean_or_sample(
                config,
                params["ioi_logits"],
                params["ioi_loc"],
                params["ioi_log_scale"],
                "ioi",
                sampling_strategy=shared_strategy,
                zero_mask=zero_mask,
            )
        duration = _dlm_mean_or_sample(
            config,
            params["duration_logits"],
            params["duration_loc"],
            params["duration_log_scale"],
            "duration",
            sampling_strategy=shared_strategy,
        )
        velocity_distribution = str(getattr(config, "velocity_distribution", "skew_normal")).lower()
        if velocity_distribution in DLM_DISTRIBUTIONS:
            velocity_raw = _dlm_mean_or_sample(
                config,
                params["velocity_logits"],
                params["velocity_loc"],
                params["velocity_log_scale"],
                "velocity",
                sampling_strategy=shared_strategy,
            )
            velocity = (velocity_raw / 127.0).clamp(0.0, 1.0)
        else:
            sigma_min = getattr(config, "skew_normal_sigma_min", getattr(config, "logistic_normal_sigma_min", 1e-4))
            sigma_max = getattr(config, "skew_normal_sigma_max", getattr(config, "logistic_normal_sigma_max", 1e4))
            velocity = _skew_normal_mean_or_sample(
                params["velocity_loc"],
                params["velocity_log_scale"],
                params["velocity_alpha"],
                sampling_strategy=shared_strategy,
                sigma_min=sigma_min,
                sigma_max=sigma_max,
            ).clamp(0.0, 1.0)
        pedal = _materialize_pedal_params(config, params, sampling_strategy=sampling_strategy)
        return torch.cat([torch.stack([ioi, duration, velocity], dim=-1), pedal], dim=-1)

    if distribution in SN_DISTRIBUTIONS:
        params = _split_epr_mixture_params(config, raw_outputs)
        sigma_min = getattr(config, "skew_normal_sigma_min", getattr(config, "logistic_normal_sigma_min", 1e-4))
        sigma_max = getattr(config, "skew_normal_sigma_max", getattr(config, "logistic_normal_sigma_max", 1e4))
        logdev = _skew_normal_mean_or_sample_timing(
            config,
            params["timing_log_loc"],
            params["timing_log_log_scale"],
            params["timing_log_alpha"],
            sampling_strategy=shared_strategy,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
        )
        zero_dual_mode = str(_config_value(config, "zero_ioi_dual_distribution_mode", "none") or "none").lower()
        if zero_dual_mode == "folded_abs":
            zero_dual_mode = "zero_folded"
        if zero_dual_mode not in {"none", "off", "false", "0"}:
            if score_shared_raw is None:
                raise ValueError("zero-IOI dual distribution materialization requires score_shared_raw")
            zero_logdev = _skew_normal_mean_or_sample_timing(
                config,
                params["timing_zero_log_loc"],
                params["timing_zero_log_log_scale"],
                params["timing_zero_log_alpha"],
                sampling_strategy=shared_strategy,
                sigma_min=sigma_min,
                sigma_max=sigma_max,
            )
            if zero_dual_mode == "zero_folded":
                zero_logdev = torch.cat([zero_logdev[..., 0:1].abs(), zero_logdev[..., 1:]], dim=-1)
            zero_mask = _zero_score_ioi_mask(config, score_shared_raw)
            logdev = torch.where(zero_mask.to(device=logdev.device).unsqueeze(-1), zero_logdev, logdev)
        else:
            logdev = _apply_zero_ioi_positive_support(config, logdev, score_shared_raw)
        velocity = _skew_normal_mean_or_sample(
            params["velocity_loc"],
            params["velocity_log_scale"],
            params["velocity_alpha"],
            sampling_strategy=shared_strategy,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
        ).clamp(0.0, 1.0)
        pedal = _materialize_pedal_params(config, params, sampling_strategy=sampling_strategy)
        return torch.cat([logdev, velocity.unsqueeze(-1), pedal], dim=-1)

    if distribution == "beta_mu_kappa":
        params = _split_epr_distribution_params(raw_outputs)
        shared_mu, _, shared_alpha, shared_beta = _beta_params(
            params["shared_mu"],
            params["shared_kappa"],
            eps=getattr(config, "beta_eps", 1e-5),
            kappa_min=getattr(config, "beta_kappa_min", 1e-3),
        )
        mode_name = str(shared_strategy).lower()
        shared = (
            torch.distributions.Beta(shared_alpha, shared_beta).sample()
            if mode_name in {"sample", "sampling", "stochastic"}
            else shared_mu
        )
        pedal = _materialize_pedal_raw(
            config,
            raw_outputs[..., -pedal_representation_dim(getattr(config, "pedal_representation", "binary_4")) :],
            sampling_strategy=sampling_strategy,
        )
        return torch.cat([shared, pedal], dim=-1)

    if _is_scalar_distribution(distribution):
        params = _split_epr_mixture_params(config, raw_outputs)
        bounded_floorlog = bool(getattr(config, "bounded_floorlog_support", False))
        zero_mask = (
            _zero_score_ioi_mask(config, score_shared_raw)
            if bounded_floorlog and score_shared_raw is not None
            else None
        )

        def decode_shared(index):
            logits, a, b, c = _shared_scalar_params(config, params, index)
            if (
                distribution in {"logistic_normal", "mixture_logistic_normal"}
                and index in {0, 1}
                and not bounded_floorlog
            ):
                return _mixture_logistic_normal_mean_or_sample_timing(
                    config,
                    logits,
                    a,
                    b,
                    sampling_strategy=shared_strategy,
                )
            value = _decode_mixture_value(
                config,
                logits,
                a,
                b,
                c,
                sampling_strategy=shared_strategy,
            )
            if bounded_floorlog and index < 3:
                bounded_value = _bounded_feature_from_unit(
                    config,
                    value,
                    "ioi" if index == 0 else "duration" if index == 1 else "velocity",
                    zero_mask=zero_mask if index == 0 else None,
                )
                return bounded_value / 127.0 if index == 2 else bounded_value
            return value

        shared = torch.stack([decode_shared(0), decode_shared(1), decode_shared(2)], dim=-1)
        pedal = _materialize_pedal_params(config, params, sampling_strategy=sampling_strategy)
        return torch.cat([shared, pedal], dim=-1)

    shared = raw_outputs[..., :3].clamp(0.0, 1.0)
    pedal = _materialize_pedal_raw(
        config,
        raw_outputs[..., -pedal_representation_dim(getattr(config, "pedal_representation", "binary_4")) :],
        sampling_strategy=sampling_strategy,
    )
    return torch.cat([shared, pedal], dim=-1)


def _shift_continuous_right(continuous, attention_mask):
    shifted = torch.zeros_like(continuous)
    if continuous.shape[1] > 1:
        prev_values = continuous[:, :-1]
        prev_mask = attention_mask[:, :-1].to(dtype=continuous.dtype).unsqueeze(-1)
        shifted[:, 1:] = prev_values * prev_mask
    shifted = shifted * attention_mask.to(dtype=continuous.dtype).unsqueeze(-1)
    return shifted


def _shift_feedback_mask_right(feedback_mask, attention_mask):
    if feedback_mask is None:
        return None
    shifted = torch.zeros_like(feedback_mask)
    if feedback_mask.shape[1] > 1:
        prev_values = feedback_mask[:, :-1]
        prev_mask = attention_mask[:, :-1].to(dtype=feedback_mask.dtype).unsqueeze(-1)
        shifted[:, 1:] = prev_values * prev_mask
    shifted = shifted * attention_mask.to(dtype=feedback_mask.dtype).unsqueeze(-1)
    return shifted


def _target_feedback_mask_to_decoder_performance_mask(config, feedback_mask):
    if feedback_mask is None:
        return None
    if decoder_note_input_schema(config) == "perf_target":
        return None
    pedal_dim = pedal_representation_dim(getattr(config, "pedal_representation", "binary_4"))
    performance_dim = int(
        getattr(config, "performance_control_feature_dim", getattr(config, "control_feature_dim", 3) + pedal_dim)
    )
    if feedback_mask.shape[-1] < 3 + pedal_dim:
        raise ValueError(f"decoder_feedback_mask expects EPR target or performance-control mask, got {tuple(feedback_mask.shape)}")
    decoder_mask = feedback_mask.new_zeros(*feedback_mask.shape[:-1], performance_dim)
    timing_dim = max(
        1,
        (
            int(getattr(config, "score_control_feature_dim", getattr(config, "control_feature_dim", 3)))
            - 1
        )
        // 2,
    )
    if timing_dim == 2:
        decoder_mask[..., 0:2] = feedback_mask[..., 0:1]
        decoder_mask[..., 2:4] = feedback_mask[..., 1:2]
        decoder_mask[..., 4:5] = feedback_mask[..., 2:3]
        pedal_start = 3
        decoder_mask[..., 5 : 5 + pedal_dim] = feedback_mask[..., pedal_start : pedal_start + pedal_dim]
    else:
        decoder_mask[..., 0:1] = feedback_mask[..., 0:1]
        decoder_mask[..., 1:2] = feedback_mask[..., 1:2]
        decoder_mask[..., 2:3] = feedback_mask[..., 2:3]
        decoder_mask[..., 3 : 3 + pedal_dim] = feedback_mask[..., 3 : 3 + pedal_dim]
    return decoder_mask


def _shift_pitch_right(config, pitch_ids, attention_mask):
    shifted = pitch_ids.new_full(pitch_ids.shape, int(config.pitch_pad_id))
    if pitch_ids.shape[1] > 1:
        shifted[:, 1:] = pitch_ids[:, :-1]
        prev_mask = attention_mask[:, :-1].bool()
        shifted[:, 1:] = torch.where(
            prev_mask,
            shifted[:, 1:],
            shifted[:, 1:].new_full(shifted[:, 1:].shape, int(config.pitch_pad_id)),
        )
    shifted = torch.where(
        attention_mask.bool(),
        shifted,
        shifted.new_full(shifted.shape, int(config.pitch_pad_id)),
    )
    return shifted


def _apply_prior_note_dropout(
    config,
    decoder_input_continuous,
    special_note_ids,
    attention_mask,
    protected_feedback_mask=None,
):
    property_missing_mask = None
    property_dropout_prob = getattr(config, "prior_property_dropout_prob", None)
    slot_decoder_mask_mode = str(getattr(config, "slot_decoder_mask_mode", "property") or "property").lower()
    if (
        str(getattr(config, "note_embedding_mode", "")).lower() == "slot_attribute"
        and slot_decoder_mask_mode != "property"
    ):
        property_dropout_prob = None
    if (
        property_dropout_prob is None
        and str(getattr(config, "note_embedding_mode", "")).lower() == "slot_attribute"
        and bool(getattr(config, "tf_embedding_mask_decoder", False))
        and slot_decoder_mask_mode == "property"
    ):
        property_dropout_prob = 1.0 - float(getattr(config, "tf_embedding_mask_keep_prob", 1.0))
    if property_dropout_prob is not None and decoder_input_continuous.shape[1] > 1:
        dropout_prob = float(property_dropout_prob)
        if dropout_prob > 0.0:
            valid = attention_mask[:, 1:].bool()
            score_control_dim = int(getattr(config, "score_control_feature_dim", getattr(config, "control_feature_dim", 3)))
            performance_dim = int(
                getattr(config, "performance_control_feature_dim", getattr(config, "control_feature_dim", 3) + 4)
            )
            if performance_dim > 0:
                slot_count = 4
                timing_dim = max(1, (score_control_dim - 1) // 2)
                slot_slices = [
                    slice(0, timing_dim),
                    slice(timing_dim, timing_dim * 2),
                    slice(timing_dim * 2, timing_dim * 2 + 1),
                    slice(timing_dim * 2 + 1, min(timing_dim * 2 + 5, performance_dim)),
                ]
                pattern = str(
                    getattr(config, "prior_property_dropout_pattern", "independent") or "independent"
                ).lower()
                batch_shape = (
                    decoder_input_continuous.shape[0],
                    decoder_input_continuous.shape[1] - 1,
                )
                if pattern == "independent":
                    slot_drop = torch.rand(
                        *batch_shape,
                        slot_count,
                        device=decoder_input_continuous.device,
                    ) < dropout_prob
                elif pattern == "correlated":
                    drop_all = torch.rand(*batch_shape, device=decoder_input_continuous.device) < dropout_prob
                    slot_drop = drop_all.unsqueeze(-1).expand(*batch_shape, slot_count)
                elif pattern == "mixed":
                    visible_prob = float(getattr(config, "prior_property_visible_prob", 0.50))
                    all_dropout_prob = float(getattr(config, "prior_property_all_dropout_prob", 0.25))
                    draw = torch.rand(*batch_shape, device=decoder_input_continuous.device)
                    slot_drop = torch.zeros(*batch_shape, slot_count, dtype=torch.bool, device=draw.device)
                    all_drop = (draw >= visible_prob) & (draw < visible_prob + all_dropout_prob)
                    slot_drop[all_drop] = True
                    partial = draw >= visible_prob + all_dropout_prob
                    if partial.any():
                        subset_ids = torch.randint(1, 2**slot_count - 1, (int(partial.sum().item()),), device=draw.device)
                        bit_ids = torch.arange(slot_count, device=draw.device)
                        slot_drop[partial] = ((subset_ids.unsqueeze(-1) >> bit_ids) & 1).bool()
                else:
                    raise ValueError(f"Unsupported prior_property_dropout_pattern: {pattern}")
                slot_drop = slot_drop & valid.unsqueeze(-1)
                property_missing_mask = decoder_input_continuous.new_zeros(
                    decoder_input_continuous.shape[0],
                    decoder_input_continuous.shape[1],
                    performance_dim,
                )
                for slot_idx, column_slice in enumerate(slot_slices):
                    if column_slice.start >= performance_dim:
                        continue
                    slot_mask = slot_drop[..., slot_idx].unsqueeze(-1)
                    property_missing_mask[:, 1:, column_slice] = torch.where(
                        slot_mask,
                        torch.ones_like(property_missing_mask[:, 1:, column_slice]),
                        property_missing_mask[:, 1:, column_slice],
                    )
                if protected_feedback_mask is not None:
                    protected = _shift_feedback_mask_right(
                        _target_feedback_mask_to_decoder_performance_mask(config, protected_feedback_mask),
                        attention_mask,
                    )
                    if bool(getattr(config, "stable_force_all_properties_visible", False)):
                        protected = protected.bool().any(dim=-1, keepdim=True).expand_as(property_missing_mask)
                    property_missing_mask = property_missing_mask.masked_fill(protected.bool(), 0.0)

    keep_prob = float(getattr(config, "prior_token_keep_prob", 1.0))
    if keep_prob >= 1.0 or decoder_input_continuous.shape[1] <= 1:
        return decoder_input_continuous, special_note_ids, property_missing_mask

    valid_mask = attention_mask[:, 1:].bool()
    if not valid_mask.any():
        return decoder_input_continuous, special_note_ids, property_missing_mask

    keep_mask = torch.rand(
        decoder_input_continuous.shape[0],
        decoder_input_continuous.shape[1] - 1,
        device=decoder_input_continuous.device,
    ) < keep_prob
    drop_mask = (~keep_mask) & valid_mask

    dropout_mode = str(getattr(config, "prior_token_dropout_mode", "mask")).lower()
    if dropout_mode == "mask":
        masked_special_note_ids = special_note_ids.clone()
        mask_id = int(config.special_note_ids.get("mask", 1))
        masked_special_note_ids[:, 1:] = torch.where(
            drop_mask,
            masked_special_note_ids[:, 1:].new_full(masked_special_note_ids[:, 1:].shape, mask_id),
            masked_special_note_ids[:, 1:],
        )
        return decoder_input_continuous, masked_special_note_ids, property_missing_mask
    if dropout_mode in {"zero", "feature_zero"}:
        dropped = decoder_input_continuous.clone()
        keep_mask = keep_mask & valid_mask
        if dropped.shape[-1] > 2:
            dropped[:, 1:, 2:] = dropped[:, 1:, 2:] * keep_mask.unsqueeze(-1).to(dtype=dropped.dtype)
        else:
            dropped[:, 1:] = dropped[:, 1:] * keep_mask.unsqueeze(-1).to(dtype=dropped.dtype)
        return dropped, special_note_ids, property_missing_mask
    if dropout_mode in {"attribute_zero", "attribute_noise", "attribute_uniform"}:
        dropped = decoder_input_continuous.clone()
        if _uses_epr_targets(config):
            attr_start = int(getattr(config, "score_control_feature_dim", getattr(config, "control_feature_dim", 3)))
            attr_dim = int(
                getattr(
                    config,
                    "performance_control_feature_dim",
                    getattr(config, "control_feature_dim", 3) + 4,
                )
            )
        elif str(getattr(config, "task_type", "epr")).lower() == "epr":
            attr_start = 2
            attr_dim = int(getattr(config, "output_continuous_dim", getattr(config, "continuous_dim", 5)))
        else:
            attr_start = 0
            attr_dim = min(
                int(getattr(config, "output_continuous_dim", dropped.shape[-1])),
                dropped.shape[-1],
            )
        attr_end = min(attr_start + attr_dim, dropped.shape[-1])
        if attr_start >= attr_end:
            return dropped, special_note_ids, property_missing_mask

        attr_values = dropped[:, 1:, attr_start:attr_end]
        attr_dim = attr_values.shape[-1]
        raw_keep_probs = getattr(config, "prior_attribute_keep_probs", None)
        if raw_keep_probs is None:
            keep_probs = attr_values.new_full((attr_dim,), keep_prob)
        else:
            keep_probs = torch.as_tensor(raw_keep_probs, dtype=attr_values.dtype, device=attr_values.device).flatten()
            if keep_probs.numel() == 0:
                keep_probs = attr_values.new_full((attr_dim,), keep_prob)
            elif keep_probs.numel() < attr_dim:
                pad = keep_probs.new_full((attr_dim - keep_probs.numel(),), float(keep_probs[-1]))
                keep_probs = torch.cat([keep_probs, pad], dim=0)
            keep_probs = keep_probs[:attr_dim].clamp(0.0, 1.0)

        attr_keep_mask = torch.rand(
            *attr_values.shape,
            device=attr_values.device,
            dtype=attr_values.dtype,
        ) < keep_probs.view(1, 1, -1)
        attr_drop_mask = (~attr_keep_mask) & valid_mask.unsqueeze(-1)
        if dropout_mode == "attribute_zero":
            replacement = torch.zeros_like(attr_values)
        elif dropout_mode == "attribute_uniform":
            replacement = torch.rand_like(attr_values)
        else:
            raw_noise_std = getattr(config, "prior_attribute_noise_std", 0.05)
            noise_std = torch.as_tensor(raw_noise_std, dtype=attr_values.dtype, device=attr_values.device).flatten()
            if noise_std.numel() == 0:
                noise_std = attr_values.new_full((attr_dim,), 0.05)
            elif noise_std.numel() < attr_dim:
                pad = noise_std.new_full((attr_dim - noise_std.numel(),), float(noise_std[-1]))
                noise_std = torch.cat([noise_std, pad], dim=0)
            noise_std = noise_std[:attr_dim].clamp_min(0.0)
            replacement = (attr_values + torch.randn_like(attr_values) * noise_std.view(1, 1, -1)).clamp(0.0, 1.0)
        dropped[:, 1:, attr_start:attr_end] = torch.where(attr_drop_mask, replacement, attr_values)
        return dropped, special_note_ids, property_missing_mask
    if dropout_mode in {"none", "off"}:
        return decoder_input_continuous, special_note_ids, property_missing_mask
    else:
        raise ValueError(f"Unsupported prior_token_dropout_mode: {dropout_mode}")


def _apply_tf_embedding_mask(note_encoder, embeddings, attention_mask, keep_prob=1.0, skip_first_token=False):
    keep_prob = float(keep_prob)
    if keep_prob >= 1.0:
        return embeddings
    valid_mask = attention_mask.bool()
    if skip_first_token and valid_mask.shape[1] > 0:
        valid_mask = valid_mask.clone()
        valid_mask[:, 0] = False
    if not valid_mask.any():
        return embeddings
    keep_mask = torch.rand(valid_mask.shape, device=embeddings.device) < keep_prob
    drop_mask = valid_mask & (~keep_mask)
    if not drop_mask.any():
        return embeddings
    mask_embed = note_encoder.mask_embedding().to(dtype=embeddings.dtype, device=embeddings.device)
    return torch.where(drop_mask.unsqueeze(-1), mask_embed.view(1, 1, -1), embeddings)


def _share_slot_attribute_encoders(score_encoder, decoder_encoder):
    decoder_encoder.slot_pitch_embedding = score_encoder.slot_pitch_embedding
    decoder_encoder.slot_score_ioi_projection = score_encoder.slot_score_ioi_projection
    decoder_encoder.slot_score_duration_projection = score_encoder.slot_score_duration_projection
    decoder_encoder.slot_score_velocity_projection = score_encoder.slot_score_velocity_projection
    decoder_encoder.slot_perf_ioi_projection = score_encoder.slot_perf_ioi_projection
    decoder_encoder.slot_perf_duration_projection = score_encoder.slot_perf_duration_projection
    decoder_encoder.slot_perf_velocity_projection = score_encoder.slot_perf_velocity_projection
    decoder_encoder.slot_perf_pedal_projection = score_encoder.slot_perf_pedal_projection
    decoder_encoder.slot_musical_onset_embedding = score_encoder.slot_musical_onset_embedding
    decoder_encoder.slot_musical_duration_embedding = score_encoder.slot_musical_duration_embedding
    decoder_encoder.slot_musical_length_embedding = score_encoder.slot_musical_length_embedding
    decoder_encoder.slot_musical_onset_scalar_projection = score_encoder.slot_musical_onset_scalar_projection
    decoder_encoder.slot_musical_duration_scalar_projection = score_encoder.slot_musical_duration_scalar_projection
    decoder_encoder.slot_musical_length_scalar_projection = score_encoder.slot_musical_length_scalar_projection
    decoder_encoder.slot_musical_binary_projection = score_encoder.slot_musical_binary_projection
    decoder_encoder.slot_musical_compact_norm = score_encoder.slot_musical_compact_norm
    decoder_encoder.slot_musical_fusion = score_encoder.slot_musical_fusion
    decoder_encoder.slot_null_embeddings = score_encoder.slot_null_embeddings
    decoder_encoder.slot_mask_embeddings = score_encoder.slot_mask_embeddings
    decoder_encoder.slot_pad_embeddings = score_encoder.slot_pad_embeddings
    decoder_encoder.slot_zero_score_ioi_embedding = score_encoder.slot_zero_score_ioi_embedding
    decoder_encoder.slot_gate_logits = score_encoder.slot_gate_logits
    decoder_encoder.slot_gate_mask = score_encoder.slot_gate_mask
    decoder_encoder.slot_musical_component_gate_logits = score_encoder.slot_musical_component_gate_logits
    decoder_encoder.slot_fusion = score_encoder.slot_fusion
    decoder_encoder.dinr_timing_table = score_encoder.dinr_timing_table
    decoder_encoder.dinr_velocity_table = score_encoder.dinr_velocity_table
    decoder_encoder.dinr_field_embedding = score_encoder.dinr_field_embedding
    decoder_encoder.dinr_role_embedding = score_encoder.dinr_role_embedding


def _share_dinr_value_tables(note_encoder, decoder_note_encoder, output_decoder):
    if not bool(getattr(note_encoder, "dinr_enabled", False)):
        return
    if getattr(output_decoder, "dinr_timing_table", None) is None:
        raise ValueError("DINR note encoding requires DINR categorical output heads")
    share_timing = str(getattr(note_encoder.config, "dinr_vocabulary_mode", "unified")).lower() == "unified"
    if share_timing:
        note_encoder.dinr_timing_table = output_decoder.dinr_timing_table
    elif getattr(output_decoder, "dinr_absolute_timing_table", None) is not None:
        note_encoder.dinr_timing_table = output_decoder.dinr_absolute_timing_table
    note_encoder.dinr_velocity_table = output_decoder.dinr_velocity_table
    if share_timing:
        decoder_note_encoder.dinr_timing_table = output_decoder.dinr_timing_table
    elif getattr(output_decoder, "dinr_absolute_timing_table", None) is not None:
        decoder_note_encoder.dinr_timing_table = output_decoder.dinr_absolute_timing_table
    decoder_note_encoder.dinr_velocity_table = output_decoder.dinr_velocity_table


def _build_ar_special_note_ids(config, attention_mask):
    special_note_ids = attention_mask.new_full(attention_mask.shape, -1)
    if special_note_ids.shape[1] > 0:
        bos_id = int(config.special_note_ids.get("bos", 2))
        special_note_ids[:, 0] = bos_id
    return special_note_ids


def _build_prefilled_ar_note_inputs(
    config,
    attention_mask,
    output_dim,
    prefix_predictions=None,
    score_shared_raw=None,
    score_input_continuous=None,
):
    batch_size, seq_len = attention_mask.shape
    if _uses_epr_targets(config) or getattr(config, "task_type", "epr") == "removed_task":
        decoder_dim = int(getattr(config, "decoder_input_continuous_dim", getattr(config, "input_continuous_dim")))
    else:
        decoder_dim = output_dim + 2
    decoder_input_continuous = attention_mask.new_zeros((batch_size, seq_len, decoder_dim), dtype=torch.float32)
    special_note_ids = attention_mask.new_full((batch_size, seq_len), -1)
    if seq_len > 0:
        special_note_ids[:, 0] = int(config.special_note_ids.get("bos", 2))

    prefix_len = 0
    if prefix_predictions is not None:
        prefix_len = int(prefix_predictions.shape[1])
        if prefix_len > 0:
            if _uses_epr_targets(config):
                if _decoder_rows_require_score_shared_raw(config) and score_shared_raw is None:
                    raise ValueError("score_shared_raw is required for INR floor_log_deviation AR prefix inputs")
                decoder_input_continuous[:, 1 : prefix_len + 1] = _build_epr_decoder_rows(
                    config,
                    (
                        score_shared_raw[:, :prefix_len].to(
                        dtype=decoder_input_continuous.dtype,
                        device=decoder_input_continuous.device,
                        )
                        if score_shared_raw is not None
                        else prefix_predictions.new_zeros(
                            prefix_predictions.shape[0],
                            prefix_len,
                            3,
                            dtype=decoder_input_continuous.dtype,
                            device=decoder_input_continuous.device,
                        )
                    ),
                    prefix_predictions[:, :prefix_len].to(
                        dtype=decoder_input_continuous.dtype,
                        device=decoder_input_continuous.device,
                    ),
                    (
                        score_input_continuous[:, :prefix_len].to(
                            dtype=decoder_input_continuous.dtype,
                            device=decoder_input_continuous.device,
                        )
                        if score_input_continuous is not None
                        else None
                    ),
                )
            elif config.task_type == "removed_task":
                decoder_input_continuous[:, 1 : prefix_len + 1] = _build_removed_task_decoder_rows(
                    config,
                    prefix_predictions[:, :prefix_len].to(
                        dtype=decoder_input_continuous.dtype,
                        device=decoder_input_continuous.device,
                    ),
                )
            else:
                if config.task_type == "epr":
                    decoder_input_continuous[:, 1 : prefix_len + 1, 1] = 1.0
                elif config.task_type == "removed_task":
                    decoder_input_continuous[:, 1 : prefix_len + 1, 0] = 1.0
                decoder_input_continuous[:, 1 : prefix_len + 1, 2:] = prefix_predictions[:, :prefix_len].to(
                    dtype=decoder_input_continuous.dtype,
                    device=decoder_input_continuous.device,
                )
    return decoder_input_continuous, special_note_ids, prefix_len


def _build_ar_note_continuous(
    config,
    labels_continuous,
    score_shared_raw=None,
    score_input_continuous=None,
    task_type="epr",
):
    if _uses_epr_targets(config):
        if _decoder_rows_require_score_shared_raw(config) and score_shared_raw is None:
            raise ValueError("score_shared_raw is required for INR floor_log_deviation AR note construction")
        if score_shared_raw is None:
            score_shared_raw = labels_continuous.new_zeros(*labels_continuous.shape[:-1], 3)
        return _build_epr_decoder_rows(
            config,
            score_shared_raw,
            labels_continuous,
            score_input_continuous=score_input_continuous,
        )
    if task_type == "removed_task":
        return _build_removed_task_decoder_rows(config, labels_continuous)
    batch_size, seq_len, _ = labels_continuous.shape
    if task_type == "epr":
        type_bits = labels_continuous.new_zeros(batch_size, seq_len, 2)
        type_bits[..., 1] = 1.0
    else:
        raise ValueError(f"Unsupported task_type for AR note build: {task_type}")
    return torch.cat([type_bits, labels_continuous], dim=-1)


class IntegratedT5GemmaEncoder(T5GemmaPreTrainedModel):
    _can_record_outputs = {
        "attentions": T5GemmaSelfAttention,
        "hidden_states": T5GemmaEncoderLayer,
    }

    def __init__(self, config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.norm = T5GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = T5GemmaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.layers = nn.ModuleList(
            [T5GemmaEncoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.dropout = nn.Dropout(config.dropout_rate)
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutput:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            raise ValueError("IntegratedT5GemmaEncoder expects note-level inputs_embeds")

        cache_position = torch.arange(0, inputs_embeds.shape[1], device=inputs_embeds.device)

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        if attention_mask is None:
            attention_mask = make_default_2d_attention_mask(input_ids, inputs_embeds, self.config.pad_token_id)

        if not isinstance(self_attn_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": None,
                "position_ids": position_ids,
            }
            self_attn_mask_mapping = {
                "full_attention": create_causal_mask(
                    **mask_kwargs,
                    or_mask_function=bidirectional_mask_function(attention_mask),
                ),
                "sliding_attention": create_sliding_window_causal_mask(
                    **mask_kwargs,
                    or_mask_function=sliding_window_bidirectional_mask_function(self.config.sliding_window),
                    and_mask_function=bidirectional_mask_function(attention_mask),
                ),
            }

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        normalizer = torch.tensor(self.config.hidden_size**0.5, dtype=hidden_states.dtype, device=hidden_states.device)
        hidden_states = hidden_states * normalizer
        hidden_states = self.dropout(hidden_states)

        for layer_module in self.layers[: self.config.num_hidden_layers]:
            hidden_states = layer_module(
                hidden_states,
                position_embeddings,
                self_attn_mask_mapping[layer_module.attention_type],
                position_ids,
                **kwargs,
            )
        hidden_states = self.norm(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return BaseModelOutput(last_hidden_state=hidden_states)


class IntegratedPianoT5GemmaModel(T5GemmaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        if not config.is_encoder_decoder:
            raise ValueError("IntegratedPianoT5GemmaModel requires encoder-decoder config.")
        self.encoder = IntegratedT5GemmaEncoder(config.encoder)
        self.decoder = T5GemmaDecoder(config.decoder)
        self.post_init()

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def forward(
        self,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        decoder_position_ids: Optional[torch.LongTensor] = None,
        encoder_outputs: Optional[BaseModelOutput] = None,
        past_key_values: Optional[EncoderDecoderCache] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        decoder_inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Seq2SeqModelOutput:
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                attention_mask=attention_mask,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                **kwargs,
            )

        decoder_outputs = self.decoder(
            attention_mask=decoder_attention_mask,
            position_ids=decoder_position_ids,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=encoder_outputs.last_hidden_state,
            encoder_attention_mask=attention_mask,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states
            if kwargs.get("output_hidden_states", False)
            else (decoder_outputs.last_hidden_state,),
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


def _masked_mean(values, mask):
    mask = mask.to(dtype=values.dtype, device=values.device)
    while mask.ndim < values.ndim:
        mask = mask.unsqueeze(-1)
    denom = mask.sum().clamp_min(1.0)
    return (values * mask).sum() / denom


def _regression_loss(pred, target, mask, loss_type, huber_delta):
    pred = pred.float()
    target = target.float()
    if loss_type == "huber":
        values = F.huber_loss(pred, target, reduction="none", delta=huber_delta)
    elif loss_type == "mse":
        values = F.mse_loss(pred, target, reduction="none")
    elif loss_type == "l1":
        values = F.l1_loss(pred, target, reduction="none")
    else:
        raise ValueError(f"Unsupported regression loss type: {loss_type}")
    return _masked_mean(values, mask)


def _bce_logits_loss(logits, target, mask, pos_weight=None):
    logits = logits.float()
    target = target.float()
    kwargs = {"reduction": "none"}
    if pos_weight is not None:
        kwargs["pos_weight"] = logits.new_tensor(float(pos_weight))
    values = F.binary_cross_entropy_with_logits(logits, target, **kwargs)
    return _masked_mean(values, mask)


def _pedal_loss_components(config, pedal_pred, pedal_target, mask, detail_components, prefix=""):
    representation = normalize_pedal_representation(getattr(config, "pedal_representation", "binary_4"))
    if representation == "binary_4":
        loss = _bce_logits_loss(
            pedal_pred,
            pedal_target,
            mask.unsqueeze(-1).expand_as(pedal_target),
        )
        return loss, pedal_pred, pedal_target, ("pedal_0", "pedal_25", "pedal_50", "pedal_75"), mask
    raise ValueError("Only pedal_representation=binary_4 is supported")


def _beta_nll_loss(raw_mu, raw_kappa, target, mask, eps, kappa_min):
    target = target.float().clamp(eps, 1.0 - eps)
    _, _, alpha, beta = _beta_params(raw_mu.float(), raw_kappa.float(), eps=eps, kappa_min=kappa_min)
    values = -torch.distributions.Beta(alpha, beta).log_prob(target)
    return _masked_mean(values, mask)


def _hard_categorical_loss(logits, target, mask):
    values = F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]),
        target.long().reshape(-1),
        reduction="none",
    ).view_as(target)
    return _masked_mean(values, mask)


def _soft_categorical_loss(logits, target, mask, tau, radius=None):
    if radius is None:
        radius = max(1, int(round(float(tau) * 4.0)))
    offsets = torch.arange(-radius, radius + 1, device=logits.device, dtype=torch.long)
    target = target.long()
    candidate = (target.unsqueeze(-1) + offsets).clamp(0, logits.shape[-1] - 1)
    distances = (candidate - target.unsqueeze(-1)).abs().to(dtype=torch.float32)
    weights = torch.exp(-distances / max(float(tau), 1e-6))
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    log_probs = F.log_softmax(logits.float(), dim=-1)
    gathered = log_probs.gather(dim=-1, index=candidate)
    values = -(weights * gathered).sum(dim=-1)
    return _masked_mean(values, mask)


def _compute_epr_categorical_loss_components(config, raw_outputs, labels_epr_bins, mask):
    logits = _split_epr_categorical_logits(config, raw_outputs)
    distribution = getattr(config, "epr_distribution", "point").lower()
    soft = distribution == "soft_categorical"
    tau = getattr(config, "soft_ce_tau", {}) or {}

    def loss_one(name, feature_logits, target):
        if soft:
            return _soft_categorical_loss(
                feature_logits,
                target,
                mask,
                tau=float(tau.get(name, 1.0)),
            )
        return _hard_categorical_loss(feature_logits, target, mask)

    pedal_losses = []
    for idx in range(4):
        pedal_losses.append(
            loss_one(
                "pedal",
                logits["pedal"][..., idx, :],
                labels_epr_bins[..., 3 + idx],
            )
        )
    return {
        "ioi": loss_one("ioi", logits["ioi"], labels_epr_bins[..., 0]),
        "duration": loss_one("duration", logits["duration"], labels_epr_bins[..., 1]),
        "velocity": loss_one("velocity", logits["velocity"], labels_epr_bins[..., 2]),
        "pedal": torch.stack(pedal_losses).mean(),
    }


def _dinr_quantize(config, values, vocabulary="deviation"):
    if vocabulary == "absolute":
        step = float(config.dinr_timing_step)
        zero_bin = int(config.dinr_zero_bin)
        bins = int(config.dinr_timing_bins)
    else:
        step = float(config.dinr_output_timing_step)
        zero_bin = int(config.dinr_output_zero_bin)
        bins = int(config.dinr_output_timing_bins)
    return torch.round(
        values.float() / step + zero_bin
    ).long().clamp(0, bins - 1)


def _dinr_support_bin(config, value, upper=False, vocabulary="deviation"):
    if vocabulary == "absolute":
        step = float(config.dinr_timing_step)
        zero_bin = int(config.dinr_zero_bin)
        bins = int(config.dinr_timing_bins)
    else:
        step = float(config.dinr_output_timing_step)
        zero_bin = int(config.dinr_output_zero_bin)
        bins = int(config.dinr_output_timing_bins)
    raw = float(value) / step + zero_bin
    index = math.floor(raw + 1e-9) if upper else math.ceil(raw - 1e-9)
    return max(0, min(bins - 1, int(index)))


def _compute_dinr_loss_components(
    config,
    raw_outputs,
    labels_continuous,
    mask,
    score_shared_raw,
    label_valid_mask=None,
):
    logits = _split_dinr_logits(config, raw_outputs)
    ioi_logits = _dinr_support_mask(config, logits["ioi"], "ioi", score_shared_raw)
    duration_logits = _dinr_support_mask(config, logits["duration"], "duration")
    zero_score = score_shared_raw[..., 0].abs() <= float(getattr(config, "zero_ioi_support_eps", 1e-6))
    ioi_min = torch.where(
        zero_score,
        labels_continuous.new_full((), float(config.dinr_zero_ioi_min)),
        labels_continuous.new_full((), float(config.dinr_deviation_min)),
    )
    ioi_max = torch.where(
        zero_score,
        labels_continuous.new_full((), float(config.dinr_zero_ioi_max)),
        labels_continuous.new_full((), float(config.dinr_deviation_max)),
    )
    duration_min, duration_max = float(config.dinr_deviation_min), float(config.dinr_deviation_max)
    ioi_target_value = labels_continuous[..., 0]
    duration_target_value = labels_continuous[..., 1]
    ioi_mask = mask.bool() & (ioi_target_value >= ioi_min) & (ioi_target_value <= ioi_max)
    duration_mask = mask.bool() & (
        duration_target_value >= duration_min
    ) & (duration_target_value <= duration_max)
    velocity_mask = mask.bool()
    if label_valid_mask is not None:
        ioi_mask &= label_valid_mask[..., 0].bool()
        duration_mask &= label_valid_mask[..., 1].bool()
        if label_valid_mask.shape[-1] > 2:
            velocity_mask &= label_valid_mask[..., 2].bool()
    separated_vocabulary = str(getattr(config, "dinr_vocabulary_mode", "unified")).lower() == "separated"
    ioi_target = _dinr_quantize(config, ioi_target_value)
    if separated_vocabulary:
        absolute_ioi_target = _dinr_quantize(config, ioi_target_value, vocabulary="absolute")
        ioi_target = torch.where(zero_score, absolute_ioi_target, ioi_target)
    nonzero_lo = _dinr_support_bin(config, config.dinr_deviation_min)
    nonzero_hi = _dinr_support_bin(config, config.dinr_deviation_max, upper=True)
    zero_vocabulary = "absolute" if separated_vocabulary else "deviation"
    zero_lo = _dinr_support_bin(config, config.dinr_zero_ioi_min, vocabulary=zero_vocabulary)
    zero_hi = _dinr_support_bin(config, config.dinr_zero_ioi_max, upper=True, vocabulary=zero_vocabulary)
    ioi_target = torch.maximum(
        torch.minimum(ioi_target, torch.where(zero_score, ioi_target.new_full((), zero_hi), ioi_target.new_full((), nonzero_hi))),
        torch.where(zero_score, ioi_target.new_full((), zero_lo), ioi_target.new_full((), nonzero_lo)),
    )
    duration_target = _dinr_quantize(config, duration_target_value).clamp(nonzero_lo, nonzero_hi)
    velocity_target = torch.round(labels_continuous[..., 2].float().clamp(0.0, 1.0) * 127.0).long()
    losses = {
        "ioi": _hard_categorical_loss(ioi_logits, ioi_target, ioi_mask),
        "duration": _hard_categorical_loss(duration_logits, duration_target, duration_mask),
        "velocity": _hard_categorical_loss(logits["velocity"], velocity_target, velocity_mask),
    }
    pedal_dim = pedal_representation_dim(getattr(config, "pedal_representation", "binary_4"))
    if pedal_dim == 4 and logits["pedal"].shape[-1] == 2:
        pedal_losses = []
        for idx in range(4):
            target = (labels_continuous[..., 3 + idx] >= 0.5).long()
            pedal_losses.append(_hard_categorical_loss(logits["pedal"][..., idx, :], target, mask))
        losses["pedal"] = torch.stack(pedal_losses).mean()
    else:
        losses["pedal"] = F.binary_cross_entropy_with_logits(
            logits["pedal"].float(),
            labels_continuous[..., 3 : 3 + pedal_dim].float(),
            reduction="none",
        ).mean(dim=-1)
        losses["pedal"] = _masked_mean(losses["pedal"], mask)
    return losses


def _dlm_scale(config, raw_log_scale, feature=None, zero_mask=None):
    sigma_min = float(getattr(config, "dlm_scale_min", 1e-3))
    sigma_max = float(getattr(config, "dlm_scale_max", 10.0))
    parameterization = str(
        getattr(config, "dlm_timing_scale_parameterization", "legacy_clamp")
    ).lower()
    feature = str(feature)
    if feature in {"ioi", "duration"} and parameterization in {
        "bounded_sigmoid",
        "sigmoid",
    }:
        sigma_min = float(getattr(config, "dlm_timing_scale_min", sigma_min))
        if feature == "duration":
            sigma_max = float(getattr(config, "dlm_duration_scale_max", getattr(config, "dlm_timing_scale_max", sigma_max)))
        else:
            nonzero_max = float(getattr(config, "dlm_ioi_nonzero_scale_max", getattr(config, "dlm_timing_scale_max", sigma_max)))
            zero_max = float(getattr(config, "dlm_ioi_zero_scale_max", nonzero_max))
            sigma_max = nonzero_max
        if sigma_max <= sigma_min:
            raise ValueError(
                f"DLM bounded scale requires max > min, got {sigma_min}, {sigma_max}"
            )
        unit = torch.sigmoid(raw_log_scale.float())
        if feature == "ioi" and zero_mask is not None and zero_max != nonzero_max:
            max_value = torch.where(
                zero_mask.bool().unsqueeze(-1),
                unit.new_tensor(zero_max),
                unit.new_tensor(nonzero_max),
            )
            return sigma_min + (max_value - sigma_min) * unit
        return sigma_min + (sigma_max - sigma_min) * unit
    velocity_parameterization = str(
        getattr(config, "dlm_velocity_scale_parameterization", "legacy_clamp")
    ).lower()
    if feature == "velocity" and velocity_parameterization in {"bounded_sigmoid", "sigmoid"}:
        sigma_min = float(getattr(config, "dlm_velocity_scale_min", sigma_min))
        sigma_max = float(getattr(config, "dlm_velocity_scale_max", sigma_max))
        if sigma_max <= sigma_min:
            raise ValueError(
                f"DLM bounded velocity scale requires max > min, got {sigma_min}, {sigma_max}"
            )
        return sigma_min + (sigma_max - sigma_min) * torch.sigmoid(raw_log_scale.float())
    scale = F.softplus(raw_log_scale.float()) + sigma_min
    if str(feature) in {"velocity", "pedal"}:
        lo = float(getattr(config, f"dlm_{feature}_min", getattr(config, "dlm_velocity_min", -0.5)))
        hi = float(getattr(config, f"dlm_{feature}_max", getattr(config, "dlm_velocity_max", 127.5)))
        scale = scale + (hi - lo) / 16.0
    active_parameterization = (
        velocity_parameterization if feature == "velocity" else parameterization
    )
    if active_parameterization in {
        "softplus_unbounded",
        "unbounded_softplus",
        "softplus_no_clamp",
        "no_clamp",
    }:
        return scale
    return scale.clamp(max=sigma_max)


def _dlm_loc(config, raw_loc, feature):
    loc = raw_loc.float()
    if str(feature) in {"velocity", "pedal"}:
        lo = float(getattr(config, f"dlm_{feature}_min", getattr(config, "dlm_velocity_min", -0.5)))
        hi = float(getattr(config, f"dlm_{feature}_max", getattr(config, "dlm_velocity_max", 127.5)))
        return loc + (lo + hi) * 0.5
    return loc


def _dlm_feature_range(config, feature, zero_mask=None):
    feature = str(feature)
    if feature == "ioi":
        if zero_mask is None:
            return (
                float(getattr(config, "dlm_ioi_nonzero_min", -2.5)),
                float(getattr(config, "dlm_ioi_nonzero_max", 1.5)),
            )
        lo = torch.where(
            zero_mask,
            zero_mask.new_full((), float(getattr(config, "dlm_ioi_zero_min", 0.0)), dtype=torch.float32),
            zero_mask.new_full((), float(getattr(config, "dlm_ioi_nonzero_min", -2.5)), dtype=torch.float32),
        )
        hi = torch.where(
            zero_mask,
            zero_mask.new_full((), float(getattr(config, "dlm_ioi_zero_max", 5.0)), dtype=torch.float32),
            zero_mask.new_full((), float(getattr(config, "dlm_ioi_nonzero_max", 1.5)), dtype=torch.float32),
        )
        return lo, hi
    if feature == "duration":
        return (
            float(getattr(config, "dlm_duration_min", -3.0)),
            float(getattr(config, "dlm_duration_max", 2.0)),
        )
    if feature in {"velocity", "pedal"}:
        return (
            float(getattr(config, f"dlm_{feature}_min", getattr(config, "dlm_velocity_min", -0.5))),
            float(getattr(config, f"dlm_{feature}_max", getattr(config, "dlm_velocity_max", 127.5))),
        )
    raise ValueError(f"Unsupported DLM feature={feature}")


def _bounded_feature_to_unit(config, value, feature, zero_mask=None, eps=1e-5):
    lo, hi = _dlm_feature_range(config, feature, zero_mask=zero_mask)
    lo = torch.as_tensor(lo, device=value.device, dtype=value.dtype)
    hi = torch.as_tensor(hi, device=value.device, dtype=value.dtype)
    return ((value - lo) / (hi - lo)).clamp(float(eps), 1.0 - float(eps))


def _bounded_feature_from_unit(config, value, feature, zero_mask=None):
    lo, hi = _dlm_feature_range(config, feature, zero_mask=zero_mask)
    lo = torch.as_tensor(lo, device=value.device, dtype=value.dtype)
    hi = torch.as_tensor(hi, device=value.device, dtype=value.dtype)
    return lo + value * (hi - lo)


def _dlm_bins(config, feature):
    if feature in {"velocity", "pedal"}:
        return int(getattr(config, f"dlm_{feature}_bins", getattr(config, "dlm_velocity_bins", 128)))
    return int(getattr(config, "dlm_timing_bins", 256))


def _dlm_log_bin_probs(config, logits, raw_loc, raw_log_scale, feature, zero_mask=None):
    bins = _dlm_bins(config, feature)
    if bins < 2:
        raise ValueError(f"DLM bins must be >= 2 for {feature}, got {bins}")
    loc = _dlm_loc(config, raw_loc, feature)
    scale = _dlm_scale(config, raw_log_scale, feature=feature, zero_mask=zero_mask)
    lo, hi = _dlm_feature_range(config, feature, zero_mask=zero_mask)
    if torch.is_tensor(lo):
        lo = lo.to(device=loc.device, dtype=loc.dtype)
        hi = hi.to(device=loc.device, dtype=loc.dtype)
        t = torch.linspace(0.0, 1.0, bins + 1, device=loc.device, dtype=loc.dtype)
        edges = lo.unsqueeze(-1) + (hi - lo).unsqueeze(-1) * t
    else:
        edges = torch.linspace(float(lo), float(hi), bins + 1, device=loc.device, dtype=loc.dtype)
        view_shape = [1] * loc.ndim
        view_shape[-1] = bins + 1
        edges = edges.view(*view_shape)
    left = edges[..., :-1].unsqueeze(-1)
    right = edges[..., 1:].unsqueeze(-1)
    z_left = (left - loc.unsqueeze(-2)) / scale.unsqueeze(-2)
    z_right = (right - loc.unsqueeze(-2)) / scale.unsqueeze(-2)
    cdf_left = torch.sigmoid(z_left)
    cdf_right = torch.sigmoid(z_right)
    first = cdf_right[..., 0:1, :]
    middle = (cdf_right[..., 1:-1, :] - cdf_left[..., 1:-1, :]).clamp_min(1e-12)
    last = (1.0 - cdf_left[..., -1:, :]).clamp_min(1e-12)
    comp_probs = torch.cat([first.clamp_min(1e-12), middle, last], dim=-2)
    log_comp = torch.log(comp_probs)
    log_mix = torch.log_softmax(logits.float(), dim=-1).unsqueeze(-2)
    return torch.logsumexp(log_mix + log_comp, dim=-1)


def _dlm_target_bins(config, target, feature, zero_mask=None):
    bins = _dlm_bins(config, feature)
    lo, hi = _dlm_feature_range(config, feature, zero_mask=zero_mask)
    target = target.float()
    if torch.is_tensor(lo):
        lo = lo.to(device=target.device, dtype=target.dtype)
        hi = hi.to(device=target.device, dtype=target.dtype)
        scaled = (target - lo) / (hi - lo).clamp_min(1e-12)
    else:
        scaled = (target - float(lo)) / max(float(hi) - float(lo), 1e-12)
    return torch.floor(scaled * bins).long().clamp(0, bins - 1)


def _dlm_nll(config, logits, raw_loc, raw_log_scale, target, mask, feature, zero_mask=None, weights=None):
    log_probs = _dlm_log_bin_probs(config, logits, raw_loc, raw_log_scale, feature, zero_mask=zero_mask)
    return _dlm_nll_from_log_probs(
        config, log_probs, target, mask, feature, zero_mask=zero_mask, weights=weights
    )


def _dlm_nll_from_log_probs(config, log_probs, target, mask, feature, zero_mask=None, weights=None):
    target_bins = _dlm_target_bins(config, target, feature, zero_mask=zero_mask)
    nll = -log_probs.gather(-1, target_bins.unsqueeze(-1)).squeeze(-1)
    if weights is not None:
        weighted_mask = mask.to(dtype=nll.dtype, device=nll.device) * weights.to(dtype=nll.dtype, device=nll.device)
        return (nll * weighted_mask).sum() / weighted_mask.sum().clamp_min(1.0)
    return _masked_mean(nll, mask)


def _weighted_or_masked_mean(values, mask, weights=None):
    if weights is not None:
        weighted_mask = mask.to(dtype=values.dtype, device=values.device) * weights.to(
            dtype=values.dtype,
            device=values.device,
        )
        return (values * weighted_mask).sum() / weighted_mask.sum().clamp_min(1.0)
    return _masked_mean(values, mask)


def _dlm_zero_inflated_nll_from_log_probs(
    config,
    raw_zero_logit,
    log_probs,
    target,
    mask,
    feature,
    zero_mask,
    weights=None,
):
    if zero_mask is None:
        return _dlm_nll_from_log_probs(
            config, log_probs, target, mask, feature, zero_mask=zero_mask, weights=weights
        )
    target_bins = _dlm_target_bins(config, target, feature, zero_mask=zero_mask)
    bin_log_prob = log_probs.gather(-1, target_bins.unsqueeze(-1)).squeeze(-1)
    mode_log_probs = _inflated_zero_cont_mode_log_probs(raw_zero_logit)
    eps = float(getattr(config, "zero_ioi_support_eps", 1e-6))
    atom_target = zero_mask.to(device=target.device).bool() & (target.float().abs() <= eps)
    inflated_rows = zero_mask.to(device=target.device).bool()
    regular_nll = -bin_log_prob
    inflated_nll = -(mode_log_probs[..., 1] + bin_log_prob)
    inflated_nll = torch.where(atom_target, -mode_log_probs[..., 0], inflated_nll)
    values = torch.where(inflated_rows, inflated_nll, regular_nll)
    return _weighted_or_masked_mean(values, mask, weights=weights)


def _dlm_zero_inflated_nll(
    config,
    raw_zero_logit,
    logits,
    raw_loc,
    raw_log_scale,
    target,
    mask,
    feature,
    zero_mask=None,
    weights=None,
):
    log_probs = _dlm_log_bin_probs(config, logits, raw_loc, raw_log_scale, feature, zero_mask=zero_mask)
    return _dlm_zero_inflated_nll_from_log_probs(
        config,
        raw_zero_logit,
        log_probs,
        target,
        mask,
        feature,
        zero_mask,
        weights=weights,
    )


def _dlm_zero_one_inflated_nll(
    config,
    raw_mode_logits,
    logits,
    raw_loc,
    raw_log_scale,
    target,
    mask,
    feature,
    zero_value=0.0,
    one_value=127.0,
    eps=None,
    weights=None,
):
    log_probs = _dlm_log_bin_probs(config, logits, raw_loc, raw_log_scale, feature)
    target_bins = _dlm_target_bins(config, target, feature)
    bin_log_prob = log_probs.gather(-1, target_bins.unsqueeze(-1)).squeeze(-1)
    mode_log_probs = _inflated_zero_one_mode_log_probs(raw_mode_logits)
    target = target.float()
    eps = float(getattr(config, "dlm_pedal_inflated_eps", 0.5) if eps is None else eps)
    zero_target = target <= float(zero_value) + eps
    one_target = target >= float(one_value) - eps
    values = -(mode_log_probs[..., 2] + bin_log_prob)
    values = torch.where(zero_target, -mode_log_probs[..., 0], values)
    values = torch.where(one_target, -mode_log_probs[..., 1], values)
    return _weighted_or_masked_mean(values, mask, weights=weights)


def _dlm_bin_centers(config, feature, like, zero_mask=None):
    bins = _dlm_bins(config, feature)
    lo, hi = _dlm_feature_range(config, feature, zero_mask=zero_mask)
    if torch.is_tensor(lo):
        lo = lo.to(device=like.device, dtype=like.dtype)
        hi = hi.to(device=like.device, dtype=like.dtype)
        t = (torch.arange(bins, device=like.device, dtype=like.dtype) + 0.5) / float(bins)
        return lo.unsqueeze(-1) + (hi - lo).unsqueeze(-1) * t
    centers = torch.linspace(
        float(lo) + (float(hi) - float(lo)) / (2 * bins),
        float(hi) - (float(hi) - float(lo)) / (2 * bins),
        bins,
        device=like.device,
        dtype=like.dtype,
    )
    view_shape = [1] * like.ndim
    view_shape[-1] = bins
    return centers.view(*view_shape)


def _dlm_tail_mass(config, logits, raw_loc, raw_log_scale, mask, feature, zero_mask=None):
    log_probs = _dlm_log_bin_probs(
        config, logits, raw_loc, raw_log_scale, feature, zero_mask=zero_mask
    )
    return _dlm_tail_mass_from_log_probs(config, log_probs, mask, feature, zero_mask=zero_mask)


def _dlm_tail_mass_from_log_probs(config, log_probs, mask, feature, zero_mask=None):
    probs = torch.softmax(log_probs, dim=-1)
    centers = _dlm_bin_centers(config, feature, log_probs, zero_mask=zero_mask).expand_as(probs)
    mean = torch.sum(probs * centers, dim=-1, keepdim=True).detach()
    radius = float(getattr(config, "dlm_tail_radius", 0.05))
    outside = (centers - mean).abs() > radius
    tail_mass = (probs * outside.to(dtype=probs.dtype)).sum(dim=-1)
    return _masked_mean(tail_mass, mask)


def _dlm_target_tail_penalty_from_log_probs(config, log_probs, target, mask, feature, zero_mask=None):
    probs = torch.softmax(log_probs, dim=-1)
    centers = _dlm_bin_centers(config, feature, log_probs, zero_mask=zero_mask).expand_as(probs)
    target = target.float().unsqueeze(-1)
    explicit_radius = getattr(config, f"dlm_target_tail_{feature}_radius", None)
    if explicit_radius is None:
        lo, hi = _dlm_feature_range(config, feature, zero_mask=zero_mask)
        if torch.is_tensor(lo):
            lo = lo.to(device=log_probs.device, dtype=log_probs.dtype)
            hi = hi.to(device=log_probs.device, dtype=log_probs.dtype)
            radius = (hi - lo).abs().unsqueeze(-1) * float(
                getattr(config, "dlm_target_tail_radius_frac", 0.0)
            )
        else:
            radius = log_probs.new_tensor(
                abs(float(hi) - float(lo)) * float(getattr(config, "dlm_target_tail_radius_frac", 0.0))
            )
    else:
        radius = log_probs.new_tensor(float(explicit_radius))
    excess = F.relu((centers - target).abs() - radius)
    penalty = torch.sum(probs * excess.square(), dim=-1)
    return _masked_mean(penalty, mask)


def _dlm_timing_nll_weights(config, score_time_ms, target_dev, mask):
    alpha = float(getattr(config, "dlm_timing_weighted_nll_alpha", 0.0))
    if alpha <= 0.0:
        return None
    target_time_ms = _torch_floor_log_reconstruct(score_time_ms, target_dev).detach()
    valid = target_time_ms[mask.bool()]
    reference = valid.median().clamp_min(1.0) if valid.numel() else target_time_ms.new_tensor(1.0)
    weights = (target_time_ms / reference).clamp_min(1e-12).pow(alpha)
    weight_min = float(getattr(config, "dlm_timing_weight_min", 0.5))
    weight_max = float(getattr(config, "dlm_timing_weight_max", 4.0))
    return weights.clamp(weight_min, weight_max)


def _dlm_raw_ms_crps_from_log_probs(
    config, log_probs, score_time_ms, target_dev, mask, feature, zero_mask=None
):
    probs = torch.softmax(log_probs, dim=-1)
    dev_centers = _dlm_bin_centers(
        config, feature, log_probs, zero_mask=zero_mask
    ).expand_as(probs)
    base = score_time_ms.float().clamp_min(1.0).unsqueeze(-1)
    raw_centers = base * torch.exp(dev_centers)
    target_ms = _torch_floor_log_reconstruct(score_time_ms, target_dev).unsqueeze(-1)

    expected_abs = torch.sum(probs * (raw_centers - target_ms).abs(), dim=-1)
    weighted_raw = probs * raw_centers
    cumulative_prob_before = torch.cumsum(probs, dim=-1) - probs
    cumulative_raw_before = torch.cumsum(weighted_raw, dim=-1) - weighted_raw
    half_pairwise_abs = torch.sum(
        probs * (raw_centers * cumulative_prob_before - cumulative_raw_before), dim=-1
    )
    crps_ms = (expected_abs - half_pairwise_abs).clamp_min(0.0)
    scale_ms = float(getattr(config, "dlm_raw_ms_crps_scale_ms", 1000.0))
    return _masked_mean(crps_ms / scale_ms, mask)


def _dlm_mean_or_sample(config, logits, raw_loc, raw_log_scale, feature, sampling_strategy="mean", zero_mask=None):
    log_probs = _dlm_log_bin_probs(config, logits, raw_loc, raw_log_scale, feature, zero_mask=zero_mask)
    temperature = float(getattr(config, "dlm_sampling_temperature", 1.0))
    if not math.isfinite(temperature):
        raise ValueError(f"dlm_sampling_temperature must be finite, got {temperature}")
    centers = _dlm_bin_centers(config, feature, log_probs, zero_mask=zero_mask)
    mode = str(sampling_strategy).lower()
    if temperature <= 0.0:
        index = log_probs.argmax(dim=-1, keepdim=True)
        return centers.expand_as(log_probs).gather(-1, index).squeeze(-1)
    probs = torch.softmax(log_probs / temperature, dim=-1)
    if mode in {"mean", "deterministic", "mu", "expected", "expectation"}:
        return torch.sum(probs * centers, dim=-1)
    centers = centers.expand_as(probs)
    if mode in {"argmax", "greedy"}:
        index = probs.argmax(dim=-1, keepdim=True)
        return centers.gather(-1, index).squeeze(-1)
    if mode in {"sample", "sampling", "stochastic"}:
        top_k = int(getattr(config, "dlm_sampling_top_k", getattr(config, "sampling_top_k", 0)))
        if top_k > 0 and top_k < probs.shape[-1]:
            threshold = torch.topk(probs, top_k, dim=-1).values[..., -1, None]
            probs = probs.masked_fill(probs < threshold, 0.0)
            probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        top_p = float(getattr(config, "dlm_sampling_top_p", getattr(config, "sampling_top_p", 1.0)))
        probs = _nucleus_probs(probs, top_p=top_p)
        sample_center = torch.sum(probs * centers, dim=-1, keepdim=True)
        radius = _timing_truncate_radius(config)
        if str(feature) in {"ioi", "duration"}:
            center_mode = _timing_truncate_center_mode(config)
            if center_mode in {"mode", "argmax", "greedy"}:
                center_index = probs.argmax(dim=-1, keepdim=True)
                sample_center = centers.gather(-1, center_index)
        if radius > 0.0 and str(feature) in {"ioi", "duration"}:
            local_mask = (centers >= sample_center - radius) & (centers <= sample_center + radius)
            local_probs = probs.masked_fill(~local_mask, 0.0)
            local_mass = local_probs.sum(dim=-1, keepdim=True)
            probs = torch.where(local_mass > 0.0, local_probs / local_mass.clamp_min(1e-12), probs)
        index = torch.distributions.Categorical(probs=probs.reshape(-1, probs.shape[-1])).sample()
        index = index.reshape(probs.shape[:-1]).unsqueeze(-1)
        sample = centers.gather(-1, index)
        sample = _apply_dlm_attribute_sample_shrink(
            config, sample, sample_center, feature, zero_mask=zero_mask
        )
        return sample.squeeze(-1)
    raise ValueError(f"Unsupported sampling_strategy: {sampling_strategy}")


def _categorical_sample_from_probs(probs):
    index = torch.distributions.Categorical(probs=probs.reshape(-1, probs.shape[-1])).sample()
    return index.reshape(probs.shape[:-1])


def _dlm_zero_inflated_mean_or_sample(
    config,
    raw_zero_logit,
    logits,
    raw_loc,
    raw_log_scale,
    feature,
    sampling_strategy="mean",
    zero_mask=None,
):
    continuous = _dlm_mean_or_sample(
        config,
        logits,
        raw_loc,
        raw_log_scale,
        feature,
        sampling_strategy=sampling_strategy,
        zero_mask=zero_mask,
    )
    if zero_mask is None:
        return continuous
    inflated_rows = zero_mask.to(device=continuous.device).bool()
    temperature = float(getattr(config, "dlm_sampling_temperature", 1.0))
    probs = _inflated_zero_cont_mode_log_probs(raw_zero_logit, temperature=temperature).exp()
    mode = str(sampling_strategy).lower()
    if mode in {"mean", "deterministic", "mu", "expected", "expectation"}:
        inflated = probs[..., 1] * continuous
    elif mode in {"argmax", "greedy"}:
        mode_idx = probs.argmax(dim=-1)
        inflated = torch.where(mode_idx == 0, continuous.new_zeros(()), continuous)
    elif mode in {"sample", "sampling", "stochastic"}:
        mode_idx = _categorical_sample_from_probs(probs)
        inflated = torch.where(mode_idx == 0, continuous.new_zeros(()), continuous)
    else:
        raise ValueError(f"Unsupported sampling_strategy: {sampling_strategy}")
    return torch.where(inflated_rows, inflated, continuous)


def _dlm_zero_one_inflated_mean_or_sample(
    config,
    raw_mode_logits,
    logits,
    raw_loc,
    raw_log_scale,
    feature,
    sampling_strategy="mean",
    zero_value=0.0,
    one_value=127.0,
):
    continuous = _dlm_mean_or_sample(
        config,
        logits,
        raw_loc,
        raw_log_scale,
        feature,
        sampling_strategy=sampling_strategy,
    )
    temperature = float(getattr(config, "dlm_sampling_temperature", 1.0))
    probs = _inflated_zero_one_mode_log_probs(raw_mode_logits, temperature=temperature).exp()
    zero = continuous.new_full((), float(zero_value))
    one = continuous.new_full((), float(one_value))
    mode = str(sampling_strategy).lower()
    if mode in {"mean", "deterministic", "mu", "expected", "expectation"}:
        return probs[..., 1] * one + probs[..., 2] * continuous
    if mode in {"argmax", "greedy"}:
        mode_idx = probs.argmax(dim=-1)
    elif mode in {"sample", "sampling", "stochastic"}:
        mode_idx = _categorical_sample_from_probs(probs)
    else:
        raise ValueError(f"Unsupported sampling_strategy: {sampling_strategy}")
    return torch.where(mode_idx == 0, zero, torch.where(mode_idx == 1, one, continuous))


def _compute_integrated_loss_components(
    config,
    continuous_pred,
    labels_continuous,
    attention_mask,
    labels_epr_bins=None,
    score_shared_raw=None,
    label_valid_mask=None,
):
    del labels_epr_bins
    if getattr(config, "task_type", "epr") == "removed_task":
        return _compute_removed_task_loss_components(config, continuous_pred, labels_continuous, attention_mask)
    if not _uses_epr_targets(config):
        raise ValueError("EPR loss only supports INR floor_log_deviation targets")

    mask = attention_mask.bool()
    feature_mask = label_valid_mask.bool() if label_valid_mask is not None else None
    distribution = getattr(config, "epr_distribution", "point").lower()
    detail_components = {}
    if distribution in DINR_DISTRIBUTIONS:
        if score_shared_raw is None:
            raise ValueError("DINR loss requires score_shared_raw")
        return _compute_dinr_loss_components(
            config,
            continuous_pred,
            labels_continuous,
            mask,
            score_shared_raw,
            label_valid_mask=feature_mask,
        )
    if distribution in DLM_DISTRIBUTIONS:
        params = _split_epr_mixture_params(config, continuous_pred)
        if score_shared_raw is None:
            raise ValueError("DLM timing loss requires score_shared_raw for score-IOI group ranges")
        zero_mask = _zero_score_ioi_mask(config, score_shared_raw, attention_mask=mask)
        tail_lambda = float(getattr(config, "dlm_tail_loss_lambda", 0.0))
        target_tail_lambda = float(getattr(config, "dlm_target_tail_loss_lambda", 0.0))
        crps_lambda = float(getattr(config, "dlm_raw_ms_crps_lambda", 0.0))
        ioi_mask = mask if feature_mask is None else mask & feature_mask[..., 0]
        duration_mask = mask if feature_mask is None else mask & feature_mask[..., 1]
        ioi_nll_weights = _dlm_timing_nll_weights(
            config, score_shared_raw[..., 0], labels_continuous[..., 0], ioi_mask
        )
        duration_nll_weights = _dlm_timing_nll_weights(
            config, score_shared_raw[..., 1], labels_continuous[..., 1], duration_mask
        )
        if tail_lambda > 0.0 or target_tail_lambda > 0.0 or crps_lambda > 0.0:
            ioi_log_probs = _dlm_log_bin_probs(
                config, params["ioi_logits"], params["ioi_loc"], params["ioi_log_scale"],
                "ioi", zero_mask=zero_mask,
            )
            duration_log_probs = _dlm_log_bin_probs(
                config, params["duration_logits"], params["duration_loc"],
                params["duration_log_scale"], "duration",
            )
            if bool(getattr(config, "dlm_ioi_zero_inflated", False)):
                loss_ioi = _dlm_zero_inflated_nll_from_log_probs(
                    config,
                    params["ioi_zero_logit"],
                    ioi_log_probs,
                    labels_continuous[..., 0],
                    ioi_mask,
                    "ioi",
                    zero_mask,
                    weights=ioi_nll_weights,
                )
            else:
                loss_ioi = _dlm_nll_from_log_probs(
                    config, ioi_log_probs, labels_continuous[..., 0], ioi_mask, "ioi",
                    zero_mask=zero_mask, weights=ioi_nll_weights,
                )
            loss_duration = _dlm_nll_from_log_probs(
                config, duration_log_probs, labels_continuous[..., 1], duration_mask, "duration",
                weights=duration_nll_weights,
            )
        else:
            if bool(getattr(config, "dlm_ioi_zero_inflated", False)):
                loss_ioi = _dlm_zero_inflated_nll(
                    config,
                    params["ioi_zero_logit"],
                    params["ioi_logits"],
                    params["ioi_loc"],
                    params["ioi_log_scale"],
                    labels_continuous[..., 0],
                    ioi_mask,
                    "ioi",
                    zero_mask=zero_mask,
                    weights=ioi_nll_weights,
                )
            else:
                loss_ioi = _dlm_nll(
                    config, params["ioi_logits"], params["ioi_loc"], params["ioi_log_scale"],
                    labels_continuous[..., 0], ioi_mask, "ioi", zero_mask=zero_mask,
                    weights=ioi_nll_weights,
                )
            loss_duration = _dlm_nll(
                config, params["duration_logits"], params["duration_loc"],
                params["duration_log_scale"], labels_continuous[..., 1], duration_mask, "duration",
                weights=duration_nll_weights,
            )
        velocity_distribution = str(getattr(config, "velocity_distribution", "skew_normal")).lower()
        if velocity_distribution in DLM_DISTRIBUTIONS:
            loss_velocity = _dlm_nll(
                config,
                params["velocity_logits"],
                params["velocity_loc"],
                params["velocity_log_scale"],
                labels_continuous[..., 2] * 127.0,
                mask,
                "velocity",
            )
            detail_components["velocity_dlm_nll"] = loss_velocity
        else:
            sigma_min = getattr(config, "skew_normal_sigma_min", getattr(config, "logistic_normal_sigma_min", 1e-4))
            sigma_max = getattr(config, "skew_normal_sigma_max", getattr(config, "logistic_normal_sigma_max", 1e4))
            loss_velocity = _skew_normal_nll(
                params["velocity_loc"],
                params["velocity_log_scale"],
                params["velocity_alpha"],
                labels_continuous[..., 2],
                mask,
                sigma_min,
                sigma_max,
            )
            detail_components["velocity_sn_nll"] = loss_velocity
        pedal_representation = normalize_pedal_representation(getattr(config, "pedal_representation", "binary_4"))
        pedal_dim = pedal_representation_dim(pedal_representation)
        loss_pedal, pedal_logits_for_detail, pedal_target_for_detail, pedal_names, pedal_detail_mask = _pedal_loss_components(
            config,
            params["pedal_binary_logits"],
            labels_continuous[..., 3 : 3 + pedal_dim],
            mask,
            detail_components,
        )
        detail_components.update(
            {
                "ioi_dlm_nll": loss_ioi,
                "duration_dlm_nll": loss_duration,
            }
        )
        if tail_lambda > 0.0:
            ioi_tail_mass = _dlm_tail_mass_from_log_probs(
                config, ioi_log_probs, mask, "ioi", zero_mask=zero_mask,
            )
            duration_tail_mass = _dlm_tail_mass_from_log_probs(
                config, duration_log_probs, mask, "duration",
            )
            detail_components.update(
                {
                    "ioi_dlm_tail_mass": ioi_tail_mass,
                    "duration_dlm_tail_mass": duration_tail_mass,
                    "dlm_tail": tail_lambda * (ioi_tail_mass + duration_tail_mass),
                }
            )
        if target_tail_lambda > 0.0:
            ioi_target_tail = _dlm_target_tail_penalty_from_log_probs(
                config, ioi_log_probs, labels_continuous[..., 0], ioi_mask, "ioi",
                zero_mask=zero_mask,
            )
            duration_target_tail = _dlm_target_tail_penalty_from_log_probs(
                config, duration_log_probs, labels_continuous[..., 1], duration_mask, "duration",
            )
            detail_components.update(
                {
                    "ioi_dlm_target_tail": ioi_target_tail,
                    "duration_dlm_target_tail": duration_target_tail,
                    "dlm_target_tail": target_tail_lambda * (ioi_target_tail + duration_target_tail),
                }
            )
        if crps_lambda > 0.0:
            ioi_raw_ms_crps = _dlm_raw_ms_crps_from_log_probs(
                config, ioi_log_probs, score_shared_raw[..., 0], labels_continuous[..., 0],
                ioi_mask, "ioi", zero_mask=zero_mask,
            )
            duration_raw_ms_crps = _dlm_raw_ms_crps_from_log_probs(
                config, duration_log_probs, score_shared_raw[..., 1], labels_continuous[..., 1],
                duration_mask, "duration",
            )
            detail_components.update(
                {
                    "ioi_raw_ms_crps": ioi_raw_ms_crps,
                    "duration_raw_ms_crps": duration_raw_ms_crps,
                    "dlm_raw_ms_crps": crps_lambda * (ioi_raw_ms_crps + duration_raw_ms_crps),
                }
            )
    elif distribution in SN_DISTRIBUTIONS:
        params = _split_epr_mixture_params(config, continuous_pred)
        sigma_min = getattr(config, "skew_normal_sigma_min", getattr(config, "logistic_normal_sigma_min", 1e-4))
        sigma_max = getattr(config, "skew_normal_sigma_max", getattr(config, "logistic_normal_sigma_max", 1e4))
        zero_ioi_transform = str(getattr(config, "zero_ioi_transform", None) or "").lower()
        zero_ioi_enabled = bool(getattr(config, "zero_ioi_positive_support", False)) or zero_ioi_transform in {
            "softplus",
            "folded_abs",
            "squared",
        }
        zero_dual_mode = str(getattr(config, "zero_ioi_dual_distribution_mode", "none") or "none").lower()
        if zero_dual_mode == "folded_abs":
            zero_dual_mode = "zero_folded"
        zero_dual_enabled = zero_dual_mode not in {"none", "off", "false", "0"}
        if zero_dual_enabled:
            if score_shared_raw is None:
                raise ValueError("zero-IOI dual distribution requires score_shared_raw")
            zero_mask = _zero_score_ioi_mask(config, score_shared_raw, attention_mask=mask)
            loss_ioi_log = _dual_zero_ioi_timing_nll(
                params,
                0,
                labels_continuous[..., 0],
                mask,
                zero_mask,
                zero_dual_mode,
                sigma_min,
                sigma_max,
                eps=getattr(config, "zero_ioi_support_eps", 1e-6),
            )
            loss_duration_log = _dual_zero_ioi_timing_nll(
                params,
                1,
                labels_continuous[..., 1],
                mask,
                zero_mask,
                "skew_normal",
                sigma_min,
                sigma_max,
                eps=getattr(config, "zero_ioi_support_eps", 1e-6),
            )
        elif zero_ioi_enabled:
            if score_shared_raw is None:
                raise ValueError("zero-IOI transform requires score_shared_raw")
            loss_ioi_log = _skew_normal_nll_zero_ioi_positive_support(
                params["timing_log_loc"][..., 0],
                params["timing_log_log_scale"][..., 0],
                params["timing_log_alpha"][..., 0],
                labels_continuous[..., 0],
                mask,
                _zero_score_ioi_mask(config, score_shared_raw, attention_mask=mask),
                sigma_min,
                sigma_max,
                eps=getattr(config, "zero_ioi_support_eps", 1e-6),
                transform=zero_ioi_transform or "softplus",
            )
            loss_duration_log = _skew_normal_nll(
                params["timing_log_loc"][..., 1],
                params["timing_log_log_scale"][..., 1],
                params["timing_log_alpha"][..., 1],
                labels_continuous[..., 1],
                mask,
                sigma_min,
                sigma_max,
            )
        else:
            loss_ioi_log = _skew_normal_nll(
                params["timing_log_loc"][..., 0],
                params["timing_log_log_scale"][..., 0],
                params["timing_log_alpha"][..., 0],
                labels_continuous[..., 0],
                mask,
                sigma_min,
                sigma_max,
            )
            loss_duration_log = _skew_normal_nll(
                params["timing_log_loc"][..., 1],
                params["timing_log_log_scale"][..., 1],
                params["timing_log_alpha"][..., 1],
                labels_continuous[..., 1],
                mask,
                sigma_min,
                sigma_max,
            )
        velocity_col = 2
        pedal_start = 3
        loss_ioi = loss_ioi_log
        loss_duration = loss_duration_log
        detail_components.update(
            {
                "ioi_log_nll": loss_ioi_log,
                "duration_log_nll": loss_duration_log,
            }
        )
        loss_velocity = _skew_normal_nll(
            params["velocity_loc"],
            params["velocity_log_scale"],
            params["velocity_alpha"],
            labels_continuous[..., velocity_col],
            mask,
            sigma_min,
            sigma_max,
        )
        pedal_representation = normalize_pedal_representation(getattr(config, "pedal_representation", "binary_4"))
        pedal_dim = pedal_representation_dim(pedal_representation)
        loss_pedal, pedal_logits_for_detail, pedal_target_for_detail, pedal_names, pedal_detail_mask = _pedal_loss_components(
            config,
            params["pedal_binary_logits"],
            labels_continuous[..., pedal_start : pedal_start + pedal_dim],
            mask,
            detail_components,
        )
    elif _is_scalar_distribution(distribution):
        params = _split_epr_mixture_params(config, continuous_pred)
        eps = getattr(config, "epr_distribution_eps", getattr(config, "beta_eps", 1e-5))
        sigma_min = getattr(config, "logistic_normal_sigma_min", 1e-3)
        sigma_max = getattr(config, "logistic_normal_sigma_max", 10.0)
        alpha_min = getattr(config, "beta_alpha_min", 1e-4)

        def loss_one(logits, raw_a, raw_b, target, raw_c=None):
            return _mixture_loss_value(
                config,
                logits,
                raw_a,
                raw_b,
                target,
                mask,
                eps,
                sigma_min,
                sigma_max,
                alpha_min,
                raw_c,
            )

        bounded_floorlog = bool(getattr(config, "bounded_floorlog_support", False))
        zero_mask = (
            _zero_score_ioi_mask(config, score_shared_raw, attention_mask=mask)
            if bounded_floorlog and score_shared_raw is not None
            else None
        )
        ioi_target = labels_continuous[..., 0]
        duration_target = labels_continuous[..., 1]
        velocity_target = labels_continuous[..., 2]
        if bounded_floorlog:
            if score_shared_raw is None:
                raise ValueError("bounded floor-log distributions require score_shared_raw")
            ioi_target = _bounded_feature_to_unit(config, ioi_target, "ioi", zero_mask=zero_mask, eps=eps)
            duration_target = _bounded_feature_to_unit(config, duration_target, "duration", eps=eps)
            velocity_target = _bounded_feature_to_unit(
                config, velocity_target * 127.0, "velocity", eps=eps
            )

        logits, a, b, c = _shared_scalar_params(config, params, 0)
        loss_ioi = loss_one(logits, a, b, ioi_target, c)
        ioi_variance_args = (logits, a, b)
        logits, a, b, c = _shared_scalar_params(config, params, 1)
        loss_duration = loss_one(logits, a, b, duration_target, c)
        duration_variance_args = (logits, a, b)
        logits, a, b, c = _shared_scalar_params(config, params, 2)
        loss_velocity = loss_one(logits, a, b, velocity_target, c)
        velocity_variance_args = (logits, a, b)
        pedal_representation = normalize_pedal_representation(getattr(config, "pedal_representation", "binary_4"))
        pedal_dim = pedal_representation_dim(pedal_representation)
        loss_pedal, pedal_logits_for_detail, pedal_target_for_detail, pedal_names, pedal_detail_mask = _pedal_loss_components(
            config,
            params["pedal_binary_logits"],
            labels_continuous[..., 3 : 3 + pedal_dim],
            mask,
            detail_components,
        )
        variance_lambda = float(getattr(config, "predictive_variance_lambda", 0.0))
        if variance_lambda > 0.0:
            timing_radius = float(getattr(config, "predictive_timing_radius", 0.05))
            velocity_radius = float(getattr(config, "predictive_velocity_radius", 0.05))
            if zero_mask is None:
                ioi_width = float(getattr(config, "dlm_ioi_nonzero_max", 1.5)) - float(
                    getattr(config, "dlm_ioi_nonzero_min", -2.5)
                )
                ioi_radius_unit = timing_radius / ioi_width
            else:
                zero_width = float(getattr(config, "dlm_ioi_zero_max", 5.0)) - float(
                    getattr(config, "dlm_ioi_zero_min", 0.0)
                )
                nonzero_width = float(getattr(config, "dlm_ioi_nonzero_max", 1.5)) - float(
                    getattr(config, "dlm_ioi_nonzero_min", -2.5)
                )
                ioi_radius_unit = torch.where(
                    zero_mask,
                    zero_mask.new_full((), timing_radius / zero_width, dtype=torch.float32),
                    zero_mask.new_full((), timing_radius / nonzero_width, dtype=torch.float32),
                )
            duration_width = float(getattr(config, "dlm_duration_max", 2.0)) - float(
                getattr(config, "dlm_duration_min", -3.0)
            )
            variance_penalties = {
                "ioi": _predictive_variance_penalty(
                    config,
                    _mixture_unit_variance(config, distribution, *ioi_variance_args),
                    mask,
                    ioi_radius_unit,
                ),
                "duration": _predictive_variance_penalty(
                    config,
                    _mixture_unit_variance(config, distribution, *duration_variance_args),
                    mask,
                    timing_radius / duration_width,
                ),
                "velocity": _predictive_variance_penalty(
                    config,
                    _mixture_unit_variance(config, distribution, *velocity_variance_args),
                    mask,
                    velocity_radius,
                ),
            }
            detail_components["predictive_variance"] = sum(variance_penalties.values())
            detail_components.update(
                {f"{name}_variance_penalty": value for name, value in variance_penalties.items()}
            )
    elif distribution == "beta_mu_kappa":
        params = _split_epr_distribution_params(continuous_pred)
        eps = getattr(config, "beta_eps", 1e-5)
        kappa_min = getattr(config, "beta_kappa_min", 1e-3)
        loss_ioi = _beta_nll_loss(
            params["shared_mu"][..., 0],
            params["shared_kappa"][..., 0],
            labels_continuous[..., 0],
            mask,
            eps,
            kappa_min,
        )
        loss_duration = _beta_nll_loss(
            params["shared_mu"][..., 1],
            params["shared_kappa"][..., 1],
            labels_continuous[..., 1],
            mask,
            eps,
            kappa_min,
        )
        loss_velocity = _beta_nll_loss(
            params["shared_mu"][..., 2],
            params["shared_kappa"][..., 2],
            labels_continuous[..., 2],
            mask,
            eps,
            kappa_min,
        )
        pedal_dim = pedal_representation_dim(getattr(config, "pedal_representation", "binary_4"))
        loss_pedal, pedal_logits_for_detail, pedal_target_for_detail, pedal_names, pedal_detail_mask = _pedal_loss_components(
            config,
            continuous_pred[..., -pedal_dim:],
            labels_continuous[..., 3 : 3 + pedal_dim],
            mask,
            detail_components,
        )
    else:
        pred = continuous_pred
        pedal_dim = pedal_representation_dim(getattr(config, "pedal_representation", "binary_4"))
        velocity_col = 2
        pedal_start = 3
        pedal_pred_start = pred.shape[-1] - pedal_dim
        loss_ioi = _regression_loss(
            pred[..., 0],
            labels_continuous[..., 0],
            mask,
            config.time_loss_type,
            config.huber_delta,
        )
        loss_duration = _regression_loss(
            pred[..., 1],
            labels_continuous[..., 1],
            mask,
            config.time_loss_type,
            config.huber_delta,
        )
        loss_velocity = _regression_loss(
            pred[..., 2],
            labels_continuous[..., 2],
            mask,
            config.value_loss_type,
            config.huber_delta,
        )
        loss_pedal, pedal_logits_for_detail, pedal_target_for_detail, pedal_names, pedal_detail_mask = _pedal_loss_components(
            config,
            pred[..., pedal_pred_start : pedal_pred_start + pedal_dim],
            labels_continuous[..., pedal_start : pedal_start + pedal_dim],
            mask,
            detail_components,
        )
    for index, name in enumerate(pedal_names):
        if name in detail_components:
            continue
        detail_components[name] = _bce_loss(
            pedal_logits_for_detail[..., index],
            pedal_target_for_detail[..., index],
            mask,
        )
    components = {
        "ioi": loss_ioi,
        "duration": loss_duration,
        "velocity": loss_velocity,
        "pedal": loss_pedal,
        **detail_components,
    }
    return components


def _bce_loss(logits, target, mask):
    values = F.binary_cross_entropy_with_logits(logits.float(), target.float(), reduction="none")
    return _masked_mean(values, mask)


def _normalized_to_removed_task_grid_target(config, name, target):
    step = max(float(getattr(config, "removed_task_grid_step", 1.0 / 24.0)), 1e-12)
    max_value = float(getattr(config, f"removed_task_{name}_max"))
    bins = _removed_task_grid_bins(config, name)
    return torch.round(target.float().clamp(0.0, 1.0) * max_value / step).long().clamp(0, bins - 1)


def _removed_task_grid_loss(config, name, logits, target, mask):
    target_bin = _normalized_to_removed_task_grid_target(config, name, target)
    grid_loss_type = str(getattr(config, "removed_task_grid_loss_type", "huber")).lower()
    if grid_loss_type in {"ce", "hard_ce", "ordinal", "grid"}:
        return _hard_categorical_loss(logits, target_bin, mask)
    if grid_loss_type in {"soft_ce", "soft_ce_huber"}:
        return _soft_categorical_loss(
            logits,
            target_bin,
            mask,
            tau=float(getattr(config, "removed_task_grid_soft_ce_tau", 1.5)),
        )
    raise ValueError(f"Unsupported removed_task grid loss type: {grid_loss_type}")


def _compute_removed_task_loss_components(config, musical_logits, labels_musical, musical_mask):
    score_mask = musical_mask.bool()
    first_target = labels_musical[..., 5].float()
    ml_mask = score_mask & (first_target >= 0.5)
    grid_loss_type = getattr(config, "removed_task_grid_loss_type", "huber")

    if _removed_task_uses_grid_head(config):
        parts = _split_removed_task_grid_outputs(config, musical_logits)
        binary = parts["binary"]
        return {
            "mo": _removed_task_grid_loss(config, "mo", parts["mo"], labels_musical[..., 0], score_mask),
            "ioi_zero": _bce_loss(binary[..., 0], labels_musical[..., 1], score_mask),
            "md": _removed_task_grid_loss(config, "md", parts["md"], labels_musical[..., 2], score_mask),
            "ml": _removed_task_grid_loss(config, "ml", parts["ml"], labels_musical[..., 3], ml_mask),
            "tempo": _regression_loss(
                torch.sigmoid(parts["tempo"]),
                labels_musical[..., 4],
                score_mask,
                "huber",
                config.huber_delta,
            ),
            "first": _bce_loss(binary[..., 1], labels_musical[..., 5], score_mask),
            "grace": _bce_loss(binary[..., 2], labels_musical[..., 6], score_mask),
            "hand": _bce_loss(binary[..., 3], labels_musical[..., 7], score_mask),
            "trill": _bce_loss(binary[..., 4], labels_musical[..., 8], score_mask),
            "stacc": _bce_loss(binary[..., 5], labels_musical[..., 9], score_mask),
            "stem": 0.5 * (
                _bce_loss(binary[..., 6], labels_musical[..., 10], score_mask)
                + _bce_loss(binary[..., 7], labels_musical[..., 11], score_mask)
            ),
        }

    return {
        "mo": _regression_loss(
            torch.sigmoid(musical_logits[..., 0]),
            labels_musical[..., 0],
            score_mask,
            grid_loss_type,
            config.huber_delta,
        ),
        "ioi_zero": _bce_loss(musical_logits[..., 1], labels_musical[..., 1], score_mask),
        "md": _regression_loss(
            torch.sigmoid(musical_logits[..., 2]),
            labels_musical[..., 2],
            score_mask,
            grid_loss_type,
            config.huber_delta,
        ),
        "ml": _regression_loss(
            torch.sigmoid(musical_logits[..., 3]),
            labels_musical[..., 3],
            ml_mask,
            grid_loss_type,
            config.huber_delta,
        ),
        "tempo": _regression_loss(
            torch.sigmoid(musical_logits[..., 4]),
            labels_musical[..., 4],
            score_mask,
            grid_loss_type,
            config.huber_delta,
        ),
        "first": _bce_loss(musical_logits[..., 5], labels_musical[..., 5], score_mask),
        "grace": _bce_loss(musical_logits[..., 6], labels_musical[..., 6], score_mask),
        "hand": _bce_loss(musical_logits[..., 7], labels_musical[..., 7], score_mask),
        "trill": _bce_loss(musical_logits[..., 8], labels_musical[..., 8], score_mask),
        "stacc": _bce_loss(musical_logits[..., 9], labels_musical[..., 9], score_mask),
        "stem": 0.5 * (
            _bce_loss(musical_logits[..., 10], labels_musical[..., 10], score_mask)
            + _bce_loss(musical_logits[..., 11], labels_musical[..., 11], score_mask)
        ),
    }


def _compute_integrated_loss(
    config,
    continuous_pred,
    labels_continuous,
    attention_mask,
    labels_epr_bins=None,
    score_shared_raw=None,
    label_valid_mask=None,
):
    components = _compute_integrated_loss_components(
        config,
        continuous_pred,
        labels_continuous,
        attention_mask,
        labels_epr_bins=labels_epr_bins,
        score_shared_raw=score_shared_raw,
        label_valid_mask=label_valid_mask,
    )
    if getattr(config, "task_type", "epr") == "removed_task":
        weights = config.removed_task_loss_weights
        alias = {
            "hand": ("hand", "staff"),
            "stacc": ("stacc", "staccato"),
            "stem": ("stem",),
        }
        total = 0.0
        for name, value in components.items():
            candidates = alias.get(name, (name,))
            weight = None
            for key in candidates:
                if key in weights:
                    weight = weights[key]
                    break
            total = total + (1.0 if weight is None else weight) * value
        return total

    weights = config.loss_weights
    use_loss_normalization = bool(
        getattr(config, "loss_normalization", False)
    )
    timing_norm = math.log(256.0) if use_loss_normalization else 1.0
    velocity_norm = math.log(128.0) if use_loss_normalization else 1.0
    total = (
        weights.get("ioi", 1.0) * components["ioi"] / timing_norm
        + weights.get("duration", 1.0) * components["duration"] / timing_norm
        + weights.get("velocity", 1.0) * components["velocity"] / velocity_norm
        + weights.get("pedal", 1.0) * components["pedal"]
        + components.get("predictive_variance", 0.0)
        + components.get("dlm_tail", 0.0)
        + components.get("dlm_target_tail", 0.0)
        + components.get("dlm_raw_ms_crps", 0.0)
    )
    return total


def _compute_stable_contract_loss(
    config,
    continuous_pred,
    labels_continuous,
    decoder_feedback_continuous,
    stable_feedback_mask,
    attention_mask,
):
    if continuous_pred is None or labels_continuous is None or decoder_feedback_continuous is None:
        return None
    if stable_feedback_mask is None or labels_continuous.shape[1] <= 1:
        return None
    if getattr(config, "task_type", "epr") != "epr" or not _uses_inr_epr_targets(config):
        return None

    pred_mean = _materialize_epr_prediction(
        config,
        continuous_pred,
        sampling_strategy="mean",
    )
    mask = attention_mask[:, 1:].bool() & attention_mask[:, :-1].bool()
    eps = float(getattr(config, "stable_contract_eps", 1e-6))
    ioi_lambda = getattr(config, "stable_contract_ioi_lambda", None)
    duration_lambda = getattr(config, "stable_contract_duration_lambda", None)
    has_channel_weights = ioi_lambda is not None or duration_lambda is not None

    if not has_channel_weights:
        timing = slice(0, 2)
        e_current = decoder_feedback_continuous[:, :-1, timing].float() - labels_continuous[:, :-1, timing].float()
        r_next = pred_mean[:, 1:, timing].float() - labels_continuous[:, 1:, timing].float()
        norm_current = torch.linalg.vector_norm(e_current, dim=-1)
        norm_next = torch.linalg.vector_norm(r_next, dim=-1)
        corrupted = stable_feedback_mask[:, :-1, timing].to(dtype=torch.bool).any(dim=-1)
        valid = mask & corrupted & (norm_current > eps)
        if not valid.any():
            return norm_next.new_zeros(())
        alpha = float(getattr(config, "stable_contract_alpha", 1.0))
        values = F.relu(norm_next - alpha * norm_current)
        return float(getattr(config, "stable_contract_lambda", 0.0)) * values.masked_select(valid).mean()

    total = pred_mean.new_zeros(())
    for column, name in ((0, "ioi"), (1, "duration")):
        weight = getattr(config, f"stable_contract_{name}_lambda", None)
        if weight is None:
            weight = 0.0
        weight = float(weight)
        if weight <= 0.0:
            continue
        alpha = getattr(config, f"stable_contract_{name}_alpha", None)
        alpha = float(getattr(config, "stable_contract_alpha", 1.0) if alpha is None else alpha)
        current = (
            decoder_feedback_continuous[:, :-1, column].float()
            - labels_continuous[:, :-1, column].float()
        ).abs()
        next_residual = (
            pred_mean[:, 1:, column].float()
            - labels_continuous[:, 1:, column].float()
        ).abs()
        corrupted = stable_feedback_mask[:, :-1, column].to(dtype=torch.bool)
        valid = mask & corrupted & (current > eps)
        if not valid.any():
            continue
        values = F.relu(next_residual - alpha * current)
        total = total + weight * values.masked_select(valid).mean()
    return total


def _stable_contract_enabled(config):
    if not bool(getattr(config, "stable_contract_loss", False)):
        return False
    channel_lambdas = [
        getattr(config, "stable_contract_ioi_lambda", None),
        getattr(config, "stable_contract_duration_lambda", None),
    ]
    if any(value is not None and float(value) > 0.0 for value in channel_lambdas):
        return True
    return float(getattr(config, "stable_contract_lambda", 0.0)) > 0.0


class IntegratedGQAAttention(nn.Module):
    def __init__(self, config, causal=False):
        super().__init__()
        if config.num_attention_heads % config.num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.causal = causal
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)
        self.dropout = nn.Dropout(getattr(config, "attention_dropout", 0.0))

    def _shape(self, tensor, heads):
        batch_size, seq_len, _ = tensor.shape
        return tensor.view(batch_size, seq_len, heads, self.head_dim).transpose(1, 2)

    def forward(self, hidden_states, attention_mask=None):
        batch_size, seq_len, _ = hidden_states.shape
        query = self._shape(self.q_proj(hidden_states), self.num_heads)
        key = self._shape(self.k_proj(hidden_states), self.num_key_value_heads)
        value = self._shape(self.v_proj(hidden_states), self.num_key_value_heads)

        repeat = self.num_heads // self.num_key_value_heads
        if repeat != 1:
            key = key.repeat_interleave(repeat, dim=1)
            value = value.repeat_interleave(repeat, dim=1)

        scores = torch.matmul(query, key.transpose(-2, -1)) * (self.head_dim ** -0.5)
        if attention_mask is not None:
            key_mask = attention_mask[:, None, None, :].to(dtype=torch.bool, device=scores.device)
            scores = scores.masked_fill(~key_mask, torch.finfo(scores.dtype).min)
        if self.causal:
            causal_mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=scores.device).tril()
            scores = scores.masked_fill(~causal_mask[None, None, :, :], torch.finfo(scores.dtype).min)

        attn_weights = torch.softmax(scores.float(), dim=-1).to(dtype=query.dtype)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, value)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size,
            seq_len,
            self.num_heads * self.head_dim,
        )
        return self.o_proj(attn_output)


class IntegratedTransformerBlock(nn.Module):
    def __init__(self, config, causal=False):
        super().__init__()
        self.self_attn = IntegratedGQAAttention(config, causal=causal)
        self.input_norm = nn.LayerNorm(config.hidden_size)
        self.post_attn_norm = nn.LayerNorm(config.hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Linear(config.intermediate_size, config.hidden_size),
        )
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states, attention_mask=None):
        attn_input = self.input_norm(hidden_states)
        hidden_states = hidden_states + self.dropout(self.self_attn(attn_input, attention_mask=attention_mask))
        mlp_input = self.post_attn_norm(hidden_states)
        hidden_states = hidden_states + self.dropout(self.mlp(mlp_input))
        return hidden_states


class IntegratedBertBackbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        layer_count = config.bert_layers_num or config.encoder.num_hidden_layers
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.layers = nn.ModuleList([IntegratedTransformerBlock(config, causal=False) for _ in range(layer_count)])
        self.norm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, score_note_embeds, attention_mask):
        batch_size, seq_len, _ = score_note_embeds.shape
        if seq_len > self.position_embeddings.num_embeddings:
            raise ValueError(f"Sequence length {seq_len} exceeds max_position_embeddings")
        position_ids = torch.arange(seq_len, device=score_note_embeds.device).unsqueeze(0)
        hidden_states = score_note_embeds + self.position_embeddings(position_ids)
        hidden_states = self.dropout(hidden_states)
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask=attention_mask)
        return self.norm(hidden_states)


class IntegratedGptBackbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        layer_count = config.gpt_layers_num or config.encoder.num_hidden_layers + config.decoder.num_hidden_layers
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.performance_query = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.layers = nn.ModuleList([IntegratedTransformerBlock(config, causal=True) for _ in range(layer_count)])
        self.norm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, score_note_embeds, attention_mask, performance_embeds=None, performance_attention_mask=None):
        batch_size, seq_len, _ = score_note_embeds.shape
        if performance_embeds is None:
            performance_embeds = self.performance_query.expand(batch_size, seq_len, -1)
            performance_attention_mask = attention_mask
        perf_len = performance_embeds.shape[1]
        total_len = seq_len + perf_len
        if total_len > self.position_embeddings.num_embeddings:
            raise ValueError(f"GPT sequence length {total_len} exceeds max_position_embeddings")

        hidden_states = torch.cat([score_note_embeds, performance_embeds], dim=1)
        position_ids = torch.arange(total_len, device=score_note_embeds.device).unsqueeze(0)
        hidden_states = hidden_states + self.position_embeddings(position_ids)
        hidden_states = self.dropout(hidden_states)

        if performance_attention_mask is None:
            performance_attention_mask = attention_mask[:, :perf_len]
        full_attention_mask = torch.cat([attention_mask, performance_attention_mask], dim=1)
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask=full_attention_mask)
        hidden_states = self.norm(hidden_states)
        return hidden_states[:, seq_len:, :]


class IntegratedPianoTransformer(IntegratedStyleTokenMixin, nn.Module):
    def __init__(self, config: IntegratedPianoT5GemmaConfig):
        super().__init__()
        self.config = config
        self.note_encoder = IntegratedNoteEncoder(
            config,
            continuous_dim=getattr(config, "score_input_continuous_dim", config.input_continuous_dim),
            role="score",
        )
        share_note_encoder = (
            str(getattr(config, "note_embedding_mode", "")).lower() == "slot_attribute"
            and bool(getattr(config, "slot_share_role_encoders", True))
        )
        self._shared_note_encoder = share_note_encoder
        if share_note_encoder:
            if score_note_input_schema(config) != decoder_note_input_schema(config):
                raise ValueError("A shared slot note encoder requires matching score and decoder note-input schemas")
            if int(getattr(config, "score_input_continuous_dim", config.input_continuous_dim)) != int(
                getattr(config, "decoder_input_continuous_dim", config.input_continuous_dim)
            ):
                raise ValueError("A shared slot note encoder requires matching score and decoder continuous dimensions")
        else:
            self._decoder_note_encoder = IntegratedNoteEncoder(
                config,
                continuous_dim=getattr(config, "decoder_input_continuous_dim", config.input_continuous_dim),
                role="decoder",
            )
        if (
            not share_note_encoder
            and
            str(getattr(config, "note_embedding_mode", "")).lower() == "slot_attribute"
            and bool(getattr(config, "slot_share_role_encoders", True))
        ):
            _share_slot_attribute_encoders(self.note_encoder, self.decoder_note_encoder)
        self.style_token_encoder = IntegratedStyleTokenEncoder(config) if config.use_style_tokens else None
        self.continuous_decoder = IntegratedContinuousDecoder(config)
        _share_dinr_value_tables(self.note_encoder, self.decoder_note_encoder, self.continuous_decoder)
        backbone_type = config.backbone_type.lower()
        self.backbone_type = backbone_type
        if backbone_type == "bert":
            self.backbone = IntegratedBertBackbone(config)
        elif backbone_type == "gpt":
            self.backbone = IntegratedGptBackbone(config)
        else:
            raise ValueError(f"IntegratedPianoTransformer supports bert/gpt, got {config.backbone_type}")

    @property
    def decoder_note_encoder(self):
        return self.note_encoder if self._shared_note_encoder else self._decoder_note_encoder

    def forward(
        self,
        pitch_ids: Optional[torch.LongTensor] = None,
        continuous: Optional[torch.FloatTensor] = None,
        score_shared_raw: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        example_index: Optional[torch.LongTensor] = None,
        labels_continuous: Optional[torch.FloatTensor] = None,
        decoder_feedback_continuous: Optional[torch.FloatTensor] = None,
        decoder_feedback_mask: Optional[torch.FloatTensor] = None,
        stable_feedback_mask: Optional[torch.FloatTensor] = None,
        labels_epr_bins: Optional[torch.LongTensor] = None,
        label_mask: Optional[torch.LongTensor] = None,
        label_valid_mask: Optional[torch.BoolTensor] = None,
        interpolated: Optional[torch.BoolTensor] = None,
        style_creator_ids: Optional[torch.LongTensor] = None,
        style_source_ids: Optional[torch.LongTensor] = None,
        style_score_stats: Optional[torch.FloatTensor] = None,
        style_perf_stats: Optional[torch.FloatTensor] = None,
        style_perf_is_pad: Optional[torch.BoolTensor] = None,
        continuous_sampling_strategy: str = "mean",
        **kwargs,
    ) -> Seq2SeqLMOutput:
        del interpolated, example_index, kwargs
        if pitch_ids is None or continuous is None:
            raise ValueError("pitch_ids and continuous are required")
        if _uses_epr_targets(self.config) and score_shared_raw is None:
            raise ValueError("score_shared_raw is required for deviation EPR")
        if attention_mask is None:
            attention_mask = (pitch_ids != self.config.pitch_pad_id).long()

        style_kwargs = {
            "style_creator_ids": style_creator_ids,
            "style_source_ids": style_source_ids,
            "style_score_stats": style_score_stats,
            "style_perf_stats": style_perf_stats,
            "style_perf_is_pad": style_perf_is_pad,
        }
        score_note_embeds = self.note_encoder(pitch_ids, continuous)
        score_note_embeds = self._apply_style_to_note_embeds(score_note_embeds, **style_kwargs)
        if self.training and bool(getattr(self.config, "tf_embedding_mask_score", False)):
            score_note_embeds = _apply_tf_embedding_mask(
                self.note_encoder,
                score_note_embeds,
                attention_mask,
                keep_prob=getattr(self.config, "tf_embedding_mask_keep_prob", 1.0),
                skip_first_token=False,
            )
        score_context_embeds, context_attention_mask, _ = self._prepend_style_tokens(
            score_note_embeds,
            attention_mask,
            **style_kwargs,
        )
        decoder_mode = self.config.decoder_input_mode.lower()
        if decoder_mode == "score":
            hidden_states = self.backbone(score_context_embeds, context_attention_mask)
            if self._style_tokens_enabled():
                hidden_states = hidden_states[:, -score_note_embeds.shape[1]:, :]
            continuous_pred = self.continuous_decoder(hidden_states, score_shared_raw=score_shared_raw)
            continuous_pred = self._apply_zero_ioi_residual(continuous_pred, score_shared_raw)
        elif decoder_mode == "ar":
            if self.backbone_type != "gpt":
                raise ValueError("decoder_input_mode='ar' is supported for gpt in IntegratedPianoTransformer")
            if labels_continuous is not None:
                feedback_continuous = decoder_feedback_continuous if decoder_feedback_continuous is not None else labels_continuous
                decoder_target_continuous = _build_ar_note_continuous(
                    self.config,
                    feedback_continuous,
                    score_shared_raw=score_shared_raw,
                    score_input_continuous=continuous,
                    task_type=self.config.task_type,
                )
                decoder_input_continuous = _shift_continuous_right(decoder_target_continuous, attention_mask)
                decoder_missing_mask = _shift_feedback_mask_right(
                    _target_feedback_mask_to_decoder_performance_mask(self.config, decoder_feedback_mask),
                    attention_mask,
                )
                special_note_ids = _build_ar_special_note_ids(self.config, attention_mask)
                if self.training:
                    decoder_input_continuous, special_note_ids, property_missing_mask = _apply_prior_note_dropout(
                        self.config,
                        decoder_input_continuous,
                        special_note_ids,
                        attention_mask,
                        protected_feedback_mask=stable_feedback_mask,
                    )
                    if property_missing_mask is not None:
                        decoder_missing_mask = (
                            property_missing_mask
                            if decoder_missing_mask is None
                            else torch.maximum(decoder_missing_mask, property_missing_mask)
                        )
                decoder_pitch_ids = _shift_pitch_right(self.config, pitch_ids, attention_mask)
                performance_embeds = self.decoder_note_encoder(
                    decoder_pitch_ids,
                    decoder_input_continuous,
                    special_note_ids=special_note_ids,
                    performance_missing_mask=decoder_missing_mask,
                    role="decoder",
                )
                performance_embeds = self._apply_style_to_decoder_inputs(
                    performance_embeds,
                    **style_kwargs,
                )
                if (
                    self.training
                    and bool(getattr(self.config, "tf_embedding_mask_decoder", False))
                    and (
                        str(getattr(self.config, "note_embedding_mode", "")).lower() != "slot_attribute"
                        or str(getattr(self.config, "slot_decoder_mask_mode", "property")).lower() == "whole_token"
                    )
                ):
                    performance_embeds = _apply_tf_embedding_mask(
                        self.decoder_note_encoder,
                        performance_embeds,
                        attention_mask,
                        keep_prob=getattr(self.config, "tf_embedding_mask_keep_prob", 1.0),
                        skip_first_token=True,
                    )
                hidden_states = self.backbone(
                    score_context_embeds,
                    context_attention_mask,
                    performance_embeds=performance_embeds,
                    performance_attention_mask=attention_mask,
                )
                hidden_states = self._apply_style_to_decoder_hidden(hidden_states, **style_kwargs)
                continuous_pred = self.continuous_decoder(hidden_states, score_shared_raw=score_shared_raw)
                continuous_pred = self._apply_zero_ioi_residual(continuous_pred, score_shared_raw)
            else:
                continuous_pred = self._autoregressive_rollout_gpt(
                    pitch_ids=pitch_ids,
                    continuous=continuous,
                    score_shared_raw=score_shared_raw,
                    attention_mask=attention_mask,
                    score_context_embeds=score_context_embeds,
                    context_attention_mask=context_attention_mask,
                    sampling_strategy=continuous_sampling_strategy,
                    style_creator_ids=style_creator_ids,
                    style_source_ids=style_source_ids,
                    style_score_stats=style_score_stats,
                    style_perf_stats=style_perf_stats,
                    style_perf_is_pad=style_perf_is_pad,
                )
        else:
            raise ValueError(f"Unsupported decoder_input_mode: {self.config.decoder_input_mode}")

        if labels_continuous is None:
            if self.config.task_type == "epr":
                continuous_pred = _materialize_epr_prediction(
                    self.config,
                    continuous_pred,
                    sampling_strategy=continuous_sampling_strategy,
                    score_shared_raw=score_shared_raw,
                )
            elif self.config.task_type == "removed_task":
                continuous_pred = _materialize_removed_task_prediction(self.config, continuous_pred)

        loss = None
        if labels_continuous is not None:
            loss_mask = label_mask if (self.config.task_type == "removed_task" and label_mask is not None) else attention_mask
            loss = _compute_integrated_loss(
                self.config,
                continuous_pred,
                labels_continuous,
                loss_mask,
                labels_epr_bins=labels_epr_bins,
                score_shared_raw=score_shared_raw,
                label_valid_mask=label_valid_mask,
            )
            if _stable_contract_enabled(self.config):
                contract_loss = _compute_stable_contract_loss(
                    self.config,
                    continuous_pred,
                    labels_continuous,
                    decoder_feedback_continuous,
                    stable_feedback_mask,
                    loss_mask,
                )
                if contract_loss is not None:
                    loss = loss + contract_loss

        return Seq2SeqLMOutput(loss=loss, logits=continuous_pred)

    def predict_performance_continuous(
        self,
        pitch_ids,
        continuous,
        score_shared_raw=None,
        attention_mask=None,
        prefix_predictions=None,
        sampling_strategy="mean",
        style_creator_ids=None,
        style_source_ids=None,
        style_score_stats=None,
        style_perf_stats=None,
        style_perf_is_pad=None,
    ):
        if attention_mask is None:
            attention_mask = (pitch_ids != self.config.pitch_pad_id).long()
        decoder_mode = self.config.decoder_input_mode.lower()
        if decoder_mode != "ar":
            outputs = self(
                pitch_ids=pitch_ids,
                continuous=continuous,
                score_shared_raw=score_shared_raw,
                attention_mask=attention_mask,
                style_creator_ids=style_creator_ids,
                style_source_ids=style_source_ids,
                style_score_stats=style_score_stats,
                style_perf_stats=style_perf_stats,
                style_perf_is_pad=style_perf_is_pad,
                continuous_sampling_strategy=sampling_strategy,
            )
            return outputs.logits
        if self.backbone_type != "gpt":
            raise ValueError("Prefix continuation is only implemented for AR GPT in IntegratedPianoTransformer")
        return self._autoregressive_rollout_gpt(
            pitch_ids=pitch_ids,
            continuous=continuous,
            score_shared_raw=score_shared_raw,
            attention_mask=attention_mask,
            sampling_strategy=sampling_strategy,
            prefix_predictions=prefix_predictions,
            style_creator_ids=style_creator_ids,
            style_source_ids=style_source_ids,
            style_score_stats=style_score_stats,
            style_perf_stats=style_perf_stats,
            style_perf_is_pad=style_perf_is_pad,
        )

    def _autoregressive_rollout_gpt(
        self,
        pitch_ids,
        continuous,
        score_shared_raw,
        attention_mask,
        score_context_embeds=None,
        context_attention_mask=None,
        sampling_strategy="mean",
        prefix_predictions=None,
        style_creator_ids=None,
        style_source_ids=None,
        style_score_stats=None,
        style_perf_stats=None,
        style_perf_is_pad=None,
    ):
        batch_size, seq_len = pitch_ids.shape
        style_kwargs = {
            "style_creator_ids": style_creator_ids,
            "style_source_ids": style_source_ids,
            "style_score_stats": style_score_stats,
            "style_perf_stats": style_perf_stats,
            "style_perf_is_pad": style_perf_is_pad,
        }
        score_note_embeds = self.note_encoder(pitch_ids, continuous)
        score_note_embeds = self._apply_style_to_note_embeds(score_note_embeds, **style_kwargs)
        if score_context_embeds is None or context_attention_mask is None:
            score_context_embeds, context_attention_mask, _ = self._prepend_style_tokens(
                score_note_embeds,
                attention_mask,
                **style_kwargs,
            )
        decoder_input_continuous, special_note_ids, prefix_len = _build_prefilled_ar_note_inputs(
            self.config,
            attention_mask,
            self.config.output_continuous_dim,
            prefix_predictions=prefix_predictions,
            score_shared_raw=score_shared_raw,
            score_input_continuous=continuous,
        )
        decoder_pitch_ids = _shift_pitch_right(self.config, pitch_ids, attention_mask)
        predictions = []
        if prefix_predictions is not None and prefix_len > 0:
            predictions.extend(prefix_predictions[:, idx : idx + 1] for idx in range(prefix_len))

        for step in range(prefix_len, seq_len):
            perf_prefix_embeds = self.decoder_note_encoder(
                decoder_pitch_ids[:, : step + 1],
                decoder_input_continuous[:, : step + 1],
                special_note_ids=special_note_ids[:, : step + 1],
                role="decoder",
            )
            perf_prefix_embeds = self._apply_style_to_decoder_inputs(
                perf_prefix_embeds,
                **style_kwargs,
            )
            perf_prefix_mask = attention_mask[:, : step + 1]
            hidden_states = self.backbone(
                score_context_embeds,
                context_attention_mask,
                performance_embeds=perf_prefix_embeds,
                performance_attention_mask=perf_prefix_mask,
            )
            hidden_states = self._apply_style_to_decoder_hidden(hidden_states, **style_kwargs)
            step_raw = self.continuous_decoder(
                hidden_states[:, -1:, :],
                score_shared_raw=score_shared_raw[:, step : step + 1],
            )
            step_raw = self._apply_zero_ioi_residual(step_raw, score_shared_raw[:, step : step + 1])
            if self.config.task_type == "removed_task":
                step_pred = _materialize_removed_task_prediction(self.config, step_raw)
            else:
                step_pred = _materialize_epr_prediction(
                    self.config,
                    step_raw,
                    sampling_strategy=sampling_strategy,
                    score_shared_raw=score_shared_raw[:, step : step + 1],
                )
            predictions.append(step_pred)
            if step + 1 < seq_len:
                if _uses_epr_targets(self.config):
                    decoder_input_continuous[:, step + 1] = _build_epr_decoder_rows(
                        self.config,
                        score_shared_raw[:, step : step + 1],
                        step_pred,
                        score_input_continuous=continuous[:, step : step + 1],
                    )[:, 0]
                elif self.config.task_type == "epr":
                    decoder_input_continuous[:, step + 1, 1] = 1.0
                elif self.config.task_type == "removed_task":
                    decoder_input_continuous[:, step + 1] = _build_removed_task_decoder_rows(
                        self.config,
                        step_pred,
                    )[:, 0]
                if not _uses_epr_targets(self.config) and self.config.task_type != "removed_task":
                    decoder_input_continuous[:, step + 1, 2:] = step_pred[:, 0, :]

        output = torch.cat(predictions, dim=1) if predictions else continuous.new_zeros((batch_size, 0, self.config.output_continuous_dim))
        return output


class IntegratedPianoT5Gemma(IntegratedStyleTokenMixin, T5GemmaPreTrainedModel, GenerationMixin):
    config_class = IntegratedPianoT5GemmaConfig
    _tp_plan = {
        "continuous_decoder.shared_head": "colwise_rep",
        "continuous_decoder.pedal_head": "colwise_rep",
    }
    _pp_plan = {"continuous_decoder": (["hidden_states"], ["continuous_pred"])}

    def __init__(self, config: IntegratedPianoT5GemmaConfig):
        config.is_encoder_decoder = True
        super().__init__(config)
        self.note_encoder = IntegratedNoteEncoder(
            config,
            continuous_dim=getattr(config, "score_input_continuous_dim", config.input_continuous_dim),
            role="score",
        )
        share_note_encoder = (
            str(getattr(config, "note_embedding_mode", "")).lower() == "slot_attribute"
            and bool(getattr(config, "slot_share_role_encoders", True))
        )
        self._shared_note_encoder = share_note_encoder
        if share_note_encoder:
            if score_note_input_schema(config) != decoder_note_input_schema(config):
                raise ValueError("A shared slot note encoder requires matching score and decoder note-input schemas")
            if int(getattr(config, "score_input_continuous_dim", config.input_continuous_dim)) != int(
                getattr(config, "decoder_input_continuous_dim", config.input_continuous_dim)
            ):
                raise ValueError("A shared slot note encoder requires matching score and decoder continuous dimensions")
        else:
            self._decoder_note_encoder = IntegratedNoteEncoder(
                config,
                continuous_dim=getattr(config, "decoder_input_continuous_dim", config.input_continuous_dim),
                role="decoder",
            )
        if (
            not share_note_encoder
            and
            str(getattr(config, "note_embedding_mode", "")).lower() == "slot_attribute"
            and bool(getattr(config, "slot_share_role_encoders", True))
        ):
            _share_slot_attribute_encoders(self.note_encoder, self.decoder_note_encoder)
        self.style_token_encoder = IntegratedStyleTokenEncoder(config) if config.use_style_tokens else None
        self.model = IntegratedPianoT5GemmaModel(config)
        self.continuous_decoder = IntegratedContinuousDecoder(config)
        _share_dinr_value_tables(self.note_encoder, self.decoder_note_encoder, self.continuous_decoder)
        self.post_init()

    @property
    def decoder_note_encoder(self):
        return self.note_encoder if self._shared_note_encoder else self._decoder_note_encoder

    def get_encoder(self):
        return self.model.encoder

    def get_decoder(self):
        return self.model.decoder

    def load_pianoformer_backbone(self, pretrained_model_path, torch_dtype=None):
        source_model = PianoT5Gemma.from_pretrained(pretrained_model_path, torch_dtype=torch_dtype)
        incompatible = self.load_state_dict(source_model.state_dict(), strict=False)
        return incompatible

    def _build_decoder_inputs(
        self,
        pitch_ids,
        continuous,
        score_shared_raw,
        score_note_embeds,
        attention_mask,
        labels_continuous=None,
        decoder_feedback_continuous=None,
        decoder_feedback_mask=None,
        stable_feedback_mask=None,
    ):
        decoder_mode = self.config.decoder_input_mode.lower()
        if decoder_mode == "score":
            return score_note_embeds, attention_mask
        if decoder_mode == "ar":
            if labels_continuous is None:
                return None, None
            feedback_continuous = decoder_feedback_continuous if decoder_feedback_continuous is not None else labels_continuous
            decoder_target_continuous = _build_ar_note_continuous(
                self.config,
                feedback_continuous,
                score_shared_raw=score_shared_raw,
                score_input_continuous=continuous,
                task_type=self.config.task_type,
            )
            decoder_input_continuous = _shift_continuous_right(decoder_target_continuous, attention_mask)
            decoder_missing_mask = _shift_feedback_mask_right(
                _target_feedback_mask_to_decoder_performance_mask(self.config, decoder_feedback_mask),
                attention_mask,
            )
            special_note_ids = _build_ar_special_note_ids(self.config, attention_mask)
            if self.training:
                decoder_input_continuous, special_note_ids, property_missing_mask = _apply_prior_note_dropout(
                    self.config,
                    decoder_input_continuous,
                    special_note_ids,
                    attention_mask,
                    protected_feedback_mask=stable_feedback_mask,
                )
                if property_missing_mask is not None:
                    decoder_missing_mask = (
                        property_missing_mask
                        if decoder_missing_mask is None
                        else torch.maximum(decoder_missing_mask, property_missing_mask)
                    )
            decoder_pitch_ids = _shift_pitch_right(self.config, pitch_ids, attention_mask)
            decoder_inputs_embeds = self.decoder_note_encoder(
                decoder_pitch_ids,
                decoder_input_continuous,
                special_note_ids=special_note_ids,
                performance_missing_mask=decoder_missing_mask,
                role="decoder",
            )
            return decoder_inputs_embeds, attention_mask
        raise ValueError(f"Unsupported decoder_input_mode: {self.config.decoder_input_mode}")

    def forward(
        self,
        pitch_ids: Optional[torch.LongTensor] = None,
        continuous: Optional[torch.FloatTensor] = None,
        score_shared_raw: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        example_index: Optional[torch.LongTensor] = None,
        labels_continuous: Optional[torch.FloatTensor] = None,
        decoder_feedback_continuous: Optional[torch.FloatTensor] = None,
        decoder_feedback_mask: Optional[torch.FloatTensor] = None,
        stable_feedback_mask: Optional[torch.FloatTensor] = None,
        labels_epr_bins: Optional[torch.LongTensor] = None,
        label_mask: Optional[torch.LongTensor] = None,
        label_valid_mask: Optional[torch.BoolTensor] = None,
        interpolated: Optional[torch.BoolTensor] = None,
        style_creator_ids: Optional[torch.LongTensor] = None,
        style_source_ids: Optional[torch.LongTensor] = None,
        style_score_stats: Optional[torch.FloatTensor] = None,
        style_perf_stats: Optional[torch.FloatTensor] = None,
        style_perf_is_pad: Optional[torch.BoolTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        decoder_position_ids: Optional[torch.LongTensor] = None,
        encoder_outputs: Optional[BaseModelOutput] = None,
        past_key_values: Optional[EncoderDecoderCache] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        continuous_sampling_strategy: str = "mean",
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        del interpolated, example_index

        if self.training and self.config._attn_implementation != "eager":
            msg = (
                "It is strongly recommended to train T5Gemma models with the `eager` attention implementation "
                f"instead of `{self.config._attn_implementation}`."
            )
            if is_torchdynamo_compiling():
                raise ValueError(msg)
            logger.warning_once(msg)

        if pitch_ids is None or continuous is None:
            raise ValueError("pitch_ids and continuous are required")
        if _uses_epr_targets(self.config) and score_shared_raw is None:
            raise ValueError("score_shared_raw is required for INR floor_log_deviation EPR")

        if attention_mask is None:
            attention_mask = (pitch_ids != self.config.pitch_pad_id).long()

        style_kwargs = {
            "style_creator_ids": style_creator_ids,
            "style_source_ids": style_source_ids,
            "style_score_stats": style_score_stats,
            "style_perf_stats": style_perf_stats,
            "style_perf_is_pad": style_perf_is_pad,
        }
        score_note_embeds = self.note_encoder(pitch_ids, continuous)
        score_note_embeds = self._apply_style_to_note_embeds(score_note_embeds, **style_kwargs)
        score_context_embeds, context_attention_mask, _ = self._prepend_style_tokens(
            score_note_embeds,
            attention_mask,
            **style_kwargs,
        )
        decoder_inputs_embeds, decoder_input_mask = self._build_decoder_inputs(
            pitch_ids,
            continuous,
            score_shared_raw,
            score_note_embeds,
            attention_mask,
            labels_continuous=labels_continuous,
            decoder_feedback_continuous=decoder_feedback_continuous,
            decoder_feedback_mask=decoder_feedback_mask,
            stable_feedback_mask=stable_feedback_mask,
        )
        if decoder_inputs_embeds is not None:
            decoder_inputs_embeds = self._apply_style_to_decoder_inputs(
                decoder_inputs_embeds,
                **style_kwargs,
            )
            if (
                self.training
                and bool(getattr(self.config, "tf_embedding_mask_decoder", False))
                and (
                    str(getattr(self.config, "note_embedding_mode", "")).lower() != "slot_attribute"
                    or str(getattr(self.config, "slot_decoder_mask_mode", "property")).lower() == "whole_token"
                )
            ):
                decoder_inputs_embeds = _apply_tf_embedding_mask(
                    self.decoder_note_encoder,
                    decoder_inputs_embeds,
                    decoder_input_mask,
                    keep_prob=getattr(self.config, "tf_embedding_mask_keep_prob", 1.0),
                    skip_first_token=True,
                )
        if decoder_inputs_embeds is None:
            continuous_pred = self._autoregressive_rollout(
                pitch_ids=pitch_ids,
                continuous=continuous,
                score_shared_raw=score_shared_raw,
                attention_mask=attention_mask,
                score_context_embeds=score_context_embeds,
                context_attention_mask=context_attention_mask,
                sampling_strategy=continuous_sampling_strategy,
                position_ids=position_ids,
                encoder_outputs=encoder_outputs,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )
            loss = None
            if labels_continuous is not None:
                loss_mask = label_mask if (self.config.task_type == "removed_task" and label_mask is not None) else attention_mask
                loss = _compute_integrated_loss(
                    self.config,
                    continuous_pred,
                    labels_continuous,
                    loss_mask,
                    labels_epr_bins=labels_epr_bins,
                    score_shared_raw=score_shared_raw,
                    label_valid_mask=label_valid_mask,
                )
                if _stable_contract_enabled(self.config):
                    contract_loss = _compute_stable_contract_loss(
                        self.config,
                        continuous_pred,
                        labels_continuous,
                        decoder_feedback_continuous,
                        stable_feedback_mask,
                        loss_mask,
                    )
                    if contract_loss is not None:
                        loss = loss + contract_loss
            return Seq2SeqLMOutput(loss=loss, logits=continuous_pred)
        if decoder_attention_mask is None:
            decoder_attention_mask = decoder_input_mask

        decoder_outputs = self.model(
            attention_mask=context_attention_mask,
            position_ids=position_ids,
            decoder_attention_mask=decoder_attention_mask,
            decoder_position_ids=decoder_position_ids,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            inputs_embeds=score_context_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        decoder_hidden = self._apply_style_to_decoder_hidden(
            decoder_outputs.last_hidden_state,
            **style_kwargs,
        )
        continuous_pred = self.continuous_decoder(decoder_hidden, score_shared_raw=score_shared_raw)
        continuous_pred = self._apply_zero_ioi_residual(continuous_pred, score_shared_raw)
        if labels_continuous is None:
            if self.config.task_type == "epr":
                continuous_pred = _materialize_epr_prediction(
                    self.config,
                    continuous_pred,
                    sampling_strategy=continuous_sampling_strategy,
                    score_shared_raw=score_shared_raw,
                )
            elif self.config.task_type == "removed_task":
                continuous_pred = _materialize_removed_task_prediction(self.config, continuous_pred)

        loss = None
        if labels_continuous is not None:
            loss_mask = label_mask if (self.config.task_type == "removed_task" and label_mask is not None) else attention_mask
            loss = _compute_integrated_loss(
                self.config,
                continuous_pred,
                labels_continuous,
                loss_mask,
                labels_epr_bins=labels_epr_bins,
                score_shared_raw=score_shared_raw,
                label_valid_mask=label_valid_mask,
            )
            if _stable_contract_enabled(self.config):
                contract_loss = _compute_stable_contract_loss(
                    self.config,
                    continuous_pred,
                    labels_continuous,
                    decoder_feedback_continuous,
                    stable_feedback_mask,
                    loss_mask,
                )
                if contract_loss is not None:
                    loss = loss + contract_loss

        return Seq2SeqLMOutput(
            loss=loss,
            logits=continuous_pred,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.decoder_hidden_states,
            decoder_attentions=decoder_outputs.decoder_attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=decoder_outputs.encoder_last_hidden_state,
            encoder_hidden_states=decoder_outputs.encoder_hidden_states,
            encoder_attentions=decoder_outputs.encoder_attentions,
        )

    def predict_performance_continuous(
        self,
        pitch_ids,
        continuous,
        score_shared_raw=None,
        attention_mask=None,
        prefix_predictions=None,
        sampling_strategy="mean",
        style_creator_ids=None,
        style_source_ids=None,
        style_score_stats=None,
        style_perf_stats=None,
        style_perf_is_pad=None,
    ):
        if attention_mask is None:
            attention_mask = (pitch_ids != self.config.pitch_pad_id).long()
        decoder_mode = self.config.decoder_input_mode.lower()
        if decoder_mode == "ar":
            return self._autoregressive_rollout(
                pitch_ids=pitch_ids,
                continuous=continuous,
                score_shared_raw=score_shared_raw,
                attention_mask=attention_mask,
                sampling_strategy=sampling_strategy,
                prefix_predictions=prefix_predictions,
                style_creator_ids=style_creator_ids,
                style_source_ids=style_source_ids,
                style_score_stats=style_score_stats,
                style_perf_stats=style_perf_stats,
                style_perf_is_pad=style_perf_is_pad,
            )
        outputs = self(
            pitch_ids=pitch_ids,
            continuous=continuous,
            score_shared_raw=score_shared_raw,
            attention_mask=attention_mask,
            style_creator_ids=style_creator_ids,
            style_source_ids=style_source_ids,
            style_score_stats=style_score_stats,
            style_perf_stats=style_perf_stats,
            style_perf_is_pad=style_perf_is_pad,
            continuous_sampling_strategy=sampling_strategy,
        )
        return outputs.logits

    def _autoregressive_rollout(
        self,
        pitch_ids,
        continuous,
        score_shared_raw,
        attention_mask,
        score_context_embeds=None,
        context_attention_mask=None,
        sampling_strategy="mean",
        prefix_predictions=None,
        style_creator_ids=None,
        style_source_ids=None,
        style_score_stats=None,
        style_perf_stats=None,
        style_perf_is_pad=None,
        position_ids=None,
        encoder_outputs=None,
        past_key_values=None,
        use_cache=None,
        cache_position=None,
        **kwargs,
    ):
        del kwargs
        batch_size, seq_len = pitch_ids.shape
        style_kwargs = {
            "style_creator_ids": style_creator_ids,
            "style_source_ids": style_source_ids,
            "style_score_stats": style_score_stats,
            "style_perf_stats": style_perf_stats,
            "style_perf_is_pad": style_perf_is_pad,
        }
        score_note_embeds = self.note_encoder(pitch_ids, continuous)
        score_note_embeds = self._apply_style_to_note_embeds(score_note_embeds, **style_kwargs)
        if score_context_embeds is None or context_attention_mask is None:
            score_context_embeds, context_attention_mask, _ = self._prepend_style_tokens(
                score_note_embeds,
                attention_mask,
                **style_kwargs,
            )
        if encoder_outputs is None:
            encoder_outputs = self.model.encoder(
                attention_mask=context_attention_mask,
                position_ids=position_ids,
                inputs_embeds=score_context_embeds,
            )

        decoder_input_continuous, special_note_ids, prefix_len = _build_prefilled_ar_note_inputs(
            self.config,
            attention_mask,
            self.config.output_continuous_dim,
            prefix_predictions=prefix_predictions,
            score_shared_raw=score_shared_raw,
            score_input_continuous=continuous,
        )
        decoder_pitch_ids = _shift_pitch_right(self.config, pitch_ids, attention_mask)
        if prefix_len >= seq_len:
            return prefix_predictions[:, :seq_len]
        predictions = []
        if prefix_predictions is not None and prefix_len > 0:
            predictions.extend(prefix_predictions[:, idx : idx + 1] for idx in range(prefix_len))

        # Use KV cache for efficient autoregressive decoding
        # Step 0: process first token with full prefix
        # Steps 1+: process only the new token, reuse cached KV

        cached_past_key_values = past_key_values
        current_decoder_outputs = None

        if prefix_len > 0:
            prime_len = prefix_len + 1
            decoder_inputs_embeds = self.decoder_note_encoder(
                decoder_pitch_ids[:, :prime_len],
                decoder_input_continuous[:, :prime_len],
                special_note_ids=special_note_ids[:, :prime_len],
                role="decoder",
            )
            decoder_inputs_embeds = self._apply_style_to_decoder_inputs(
                decoder_inputs_embeds,
                **style_kwargs,
            )
            decoder_attention_mask = attention_mask[:, :prime_len]
            current_decoder_outputs = self.model(
                attention_mask=context_attention_mask,
                position_ids=position_ids,
                decoder_attention_mask=decoder_attention_mask,
                encoder_outputs=encoder_outputs,
                decoder_inputs_embeds=decoder_inputs_embeds,
                use_cache=True,
                past_key_values=cached_past_key_values,
            )
            cached_past_key_values = current_decoder_outputs.past_key_values

        for step in range(prefix_len, seq_len):
            if prefix_len == 0 and step == 0:
                # First step: process prefix of length 1
                decoder_inputs_embeds = self.decoder_note_encoder(
                    decoder_pitch_ids[:, :1],
                    decoder_input_continuous[:, :1],
                    special_note_ids=special_note_ids[:, :1],
                    role="decoder",
                )
                decoder_inputs_embeds = self._apply_style_to_decoder_inputs(
                    decoder_inputs_embeds,
                    **style_kwargs,
                )
                decoder_attention_mask = attention_mask[:, :1]
                decoder_outputs = self.model(
                    attention_mask=context_attention_mask,
                    position_ids=position_ids,
                    decoder_attention_mask=decoder_attention_mask,
                    encoder_outputs=encoder_outputs,
                    decoder_inputs_embeds=decoder_inputs_embeds,
                    use_cache=True,
                    past_key_values=cached_past_key_values,
                )
                cached_past_key_values = decoder_outputs.past_key_values
                current_decoder_outputs = decoder_outputs
            elif step == prefix_len and current_decoder_outputs is not None:
                decoder_outputs = current_decoder_outputs
            else:
                # Subsequent steps: process only the new token
                step_idx = step
                decoder_inputs_embeds = self.decoder_note_encoder(
                    decoder_pitch_ids[:, step_idx:step_idx+1],
                    decoder_input_continuous[:, step_idx:step_idx+1],
                    special_note_ids=special_note_ids[:, step_idx:step_idx+1],
                    role="decoder",
                )
                decoder_inputs_embeds = self._apply_style_to_decoder_inputs(
                    decoder_inputs_embeds,
                    **style_kwargs,
                )
                # For incremental decoding, we need the full decoder attention mask
                # but only the new token's embeddings
                decoder_attention_mask = attention_mask[:, :step_idx+1]
                cache_position_tensor = torch.tensor(
                    [step_idx], device=pitch_ids.device, dtype=torch.long
                )
                decoder_outputs = self.model(
                    attention_mask=context_attention_mask,
                    position_ids=position_ids,
                    decoder_attention_mask=decoder_attention_mask,
                    encoder_outputs=encoder_outputs,
                    decoder_inputs_embeds=decoder_inputs_embeds,
                    use_cache=True,
                    past_key_values=cached_past_key_values,
                    cache_position=cache_position_tensor,
                )
                cached_past_key_values = decoder_outputs.past_key_values

            decoder_hidden = self._apply_style_to_decoder_hidden(
                decoder_outputs.last_hidden_state,
                **style_kwargs,
            )
            step_raw = self.continuous_decoder(
                decoder_hidden[:, -1:, :],
                score_shared_raw=score_shared_raw[:, step : step + 1],
            )
            step_raw = self._apply_zero_ioi_residual(step_raw, score_shared_raw[:, step : step + 1])
            if self.config.task_type == "removed_task":
                step_pred = _materialize_removed_task_prediction(self.config, step_raw)
            else:
                step_pred = _materialize_epr_prediction(
                    self.config,
                    step_raw,
                    sampling_strategy=sampling_strategy,
                    score_shared_raw=score_shared_raw[:, step : step + 1],
                )
            predictions.append(step_pred)

            if step + 1 < seq_len:
                if _uses_epr_targets(self.config):
                    decoder_input_continuous[:, step + 1] = _build_epr_decoder_rows(
                        self.config,
                        score_shared_raw[:, step : step + 1],
                        step_pred,
                        score_input_continuous=continuous[:, step : step + 1],
                    )[:, 0]
                elif self.config.task_type == "epr":
                    decoder_input_continuous[:, step + 1, 1] = 1.0
                elif self.config.task_type == "removed_task":
                    decoder_input_continuous[:, step + 1] = _build_removed_task_decoder_rows(
                        self.config,
                        step_pred,
                    )[:, 0]
                if not _uses_epr_targets(self.config) and self.config.task_type != "removed_task":
                    decoder_input_continuous[:, step + 1, 2:] = step_pred[:, 0, :]

        output = torch.cat(predictions, dim=1) if predictions else continuous.new_zeros((batch_size, 0, self.config.output_continuous_dim))
        return output
