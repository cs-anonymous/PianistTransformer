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


def _is_scalar_distribution(distribution):
    return distribution in {
        "logistic_normal",
        "mixture_logistic_normal",
        "mixture_beta",
        *SN_DISTRIBUTIONS,
        *ALN_DISTRIBUTIONS,
        *ACN_DISTRIBUTIONS,
        *IACN_DISTRIBUTIONS,
        *ILN_DISTRIBUTIONS,
    }


def _scalar_distribution_dim(distribution):
    if distribution in {*ACN_DISTRIBUTIONS, "logistic_normal", "mixture_logistic_normal", "mixture_beta", *SN_DISTRIBUTIONS}:
        return 3
    if distribution in IACN_DISTRIBUTIONS:
        return 5
    if distribution in ILN_DISTRIBUTIONS:
        return 4
    if distribution in ALN_DISTRIBUTIONS:
        return 4
    return 1


def _scalar_distribution_components(config, distribution):
    if distribution in {*SN_DISTRIBUTIONS, *ALN_DISTRIBUTIONS, *ACN_DISTRIBUTIONS, *IACN_DISTRIBUTIONS, *ILN_DISTRIBUTIONS, "logistic_normal"}:
        return 1
    return int(getattr(config, "epr_mixture_components", 1))


def resolve_timing_control_mode(timing_control_mode="log_scaled", use_timing_scale_bit=False):
    if timing_control_mode is None:
        return "log_scaled"
    mode = str(timing_control_mode).lower()
    valid_modes = {
        "piecewise_scale_bit",
        "piecewise_single",
        "dual_log_linear",
        "dual_clip_linear",
        "log_scaled",
        "raw_log",
    }
    if mode not in valid_modes:
        raise ValueError(f"Unsupported timing_control_mode={timing_control_mode}")
    return mode


def timing_control_feature_dim(timing_control_mode="log_scaled", use_timing_scale_bit=False):
    mode = resolve_timing_control_mode(
        timing_control_mode=timing_control_mode,
        use_timing_scale_bit=use_timing_scale_bit,
    )
    if mode == "raw_log":
        return 5
    return 3 if mode in {"piecewise_single", "log_scaled"} else 5


def musical_feature_dim(musical_feature_mode="categorical"):
    mode = str(musical_feature_mode).lower()
    if mode == "continuous":
        return 12
    if mode in {"categorical", "categorical51", "musical51"}:
        return 51
    if mode in {"categorical62", "musical62"}:
        return 62
    raise ValueError(f"Unsupported musical_feature_mode={musical_feature_mode}")


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
        continuous_dim=7,
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
        csr_grid_loss_type="huber",
        csr_grid_step=1.0 / 24.0,
        csr_grid_soft_ce_tau=1.5,
        csr_mo_max=6.0,
        csr_md_max=6.0,
        csr_ml_max=6.0,
        huber_delta=0.05,
        loss_weights=None,
        csr_loss_weights=None,
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
        skew_normal_sigma_min=1e-4,
        skew_normal_sigma_max=1e4,
        raw_timing_loss_lambda=0.5,
        epr_inflated_features=None,
        epr_timing_bins=5000,
        epr_value_bins=128,
        epr_timing_target="log_deviation",
        timing_control_mode="log_scaled",
        timing_log_scale=50.0,
        use_timing_scale_bit=False,
        soft_ce_tau=None,
        timing_input_normalization="scaled_log_5000_s10",
        musical_feature_mode="categorical",
        prior_token_keep_prob=1.0,
        prior_token_dropout_mode="mask",
        prior_attribute_keep_probs=None,
        prior_attribute_noise_std=0.05,
        tf_embedding_mask_keep_prob=1.0,
        tf_embedding_mask_score=False,
        tf_embedding_mask_decoder=False,
        stable_contract_loss=False,
        stable_contract_alpha=1.0,
        stable_contract_lambda=0.0,
        stable_contract_ioi_alpha=None,
        stable_contract_duration_alpha=None,
        stable_contract_ioi_lambda=None,
        stable_contract_duration_lambda=None,
        stable_contract_eps=1e-6,
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
        self.csr_grid_loss_type = csr_grid_loss_type
        self.csr_grid_step = float(csr_grid_step)
        self.csr_grid_soft_ce_tau = float(csr_grid_soft_ce_tau)
        self.csr_mo_max = float(csr_mo_max)
        self.csr_md_max = float(csr_md_max)
        self.csr_ml_max = float(csr_ml_max)
        self.huber_delta = huber_delta
        self.loss_weights = loss_weights or {
            "ioi": 1.0,
            "duration": 1.0,
            "velocity": 1.0,
            "pedal": 1.0,
        }
        self.csr_loss_weights = csr_loss_weights or {
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
        self.skew_normal_sigma_min = skew_normal_sigma_min
        self.skew_normal_sigma_max = skew_normal_sigma_max
        self.raw_timing_loss_lambda = raw_timing_loss_lambda
        self.epr_inflated_features = epr_inflated_features or {
            "ioi": "zero",
            "pedal": "zero_one",
        }
        self.epr_timing_bins = int(epr_timing_bins)
        self.epr_value_bins = int(epr_value_bins)
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
        self.performance_control_feature_dim = self.control_feature_dim + 4
        self.musical_feature_mode = str(musical_feature_mode).lower()
        self.musical_feature_dim = musical_feature_dim(self.musical_feature_mode)
        self.mask_feature_dim = 3
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
        self.tf_embedding_mask_keep_prob = float(tf_embedding_mask_keep_prob)
        self.tf_embedding_mask_score = bool(tf_embedding_mask_score)
        self.tf_embedding_mask_decoder = bool(tf_embedding_mask_decoder)
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
        self.piano_pitch_min = int(piano_pitch_min)
        self.pedal_representation = str(pedal_representation).lower()
        if self.pedal_representation != "binary_4":
            raise ValueError(f"Unsupported pedal_representation={self.pedal_representation}; use binary_4")
        if bool(use_style_tokens):
            raise ValueError("use_style_tokens is disabled for the simplified EPR/CSR pipelines")
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
        self.score_control_dim = int(
            getattr(config, "score_control_feature_dim", getattr(config, "control_feature_dim", 5))
        )
        self.performance_control_dim = int(
            getattr(
                config,
                "performance_control_feature_dim",
                getattr(config, "control_feature_dim", 5) + 4,
            )
        )
        self.musical_dim = int(getattr(config, "musical_feature_dim", 12))
        self.mask_dim = int(getattr(config, "mask_feature_dim", 3))
        self.decoder_target_dim = (
            max(0, int(self.continuous_dim) - self.mask_dim)
            if self.role == "decoder" and self.schema == "perf_target"
            else 7
        )
        self.decoder_target_mask_dim = 3
        if self.role == "decoder" and self.schema == "integrated":
            self.performance_missing_embeddings = nn.Parameter(
                torch.zeros(self.performance_control_dim, config.hidden_size)
            )
            nn.init.normal_(self.performance_missing_embeddings, mean=0.0, std=0.02)
        else:
            self.register_parameter("performance_missing_embeddings", None)

        self.pitch_projection = _make_mlp(
            self.pitch_factor_dim,
            config.hidden_size,
            config.hidden_size,
            self.embedding_depth,
            self.activation,
        )
        if self.mode not in {"sine", "cine", "split_score_perf"}:
            raise ValueError(
                f"Unsupported note_embedding_mode: {self.mode}. Expected one of: sine, cine, split_score_perf"
            )

        self.score_control_projection = None
        self.performance_control_projection = None
        self.musical_projection = None
        self.mask_projection = None
        self.continuous_mlp = None
        self.decoder_target_projection = None

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
            if self.mode == "split_score_perf":
                self.score_control_projection = _make_mlp(
                    self.score_control_dim,
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
            self.musical_projection = _make_mlp(
                self.musical_dim,
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
            self.score_control_projection = _make_mlp(
                self.score_control_dim,
                config.hidden_size,
                config.hidden_size,
                self.embedding_depth,
                self.activation,
            )
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

    def forward(self, pitch_ids, continuous, special_note_ids=None, performance_missing_mask=None):
        projection_dtype = next(self.parameters()).dtype
        continuous = continuous.to(dtype=projection_dtype)
        pitch_factors = self._pitch_factors(pitch_ids).to(dtype=projection_dtype)
        pitch_embeds = self.pitch_projection(pitch_factors)

        if self.role == "decoder" and self.schema == "perf_target":
            target_features, masks = self._split_perf_target(continuous)
            target_embeds = self.decoder_target_projection(target_features)
            if self.mode == "split_score_perf":
                embeddings = target_embeds
            else:
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
        elif self.mode == "split_score_perf":
            embeddings = pitch_embeds + self.score_control_projection(score_control)
        else:
            m_score_control = masks[..., 0:1]
            m_performance_control = masks[..., 1:2]
            m_m = masks[..., 2:3]
            score_control_embeds = self.score_control_projection(score_control) * m_score_control
            performance_control_embeds = self.performance_control_projection(performance_control) * m_performance_control
            musical_embeds = self.musical_projection(musical) * m_m
            mask_embeds = self.mask_projection(masks)
            embeddings = (
                pitch_embeds
                + score_control_embeds
                + performance_control_embeds
                + musical_embeds
                + mask_embeds
            )
        if performance_missing_mask is not None:
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


class IntegratedContinuousDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.output_dim = config.output_continuous_dim
        self.epr_distribution = getattr(config, "epr_distribution", "point").lower()
        self.pedal_distribution = getattr(config, "pedal_distribution", self.epr_distribution).lower()
        self.pedal_representation = str(getattr(config, "pedal_representation", "binary_4")).lower()
        if self.pedal_representation != "binary_4":
            raise ValueError(f"Unsupported pedal_representation={self.pedal_representation}; use binary_4")
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
            and self.epr_distribution in {"categorical", "hard_categorical", "soft_categorical"}
        ):
            ioi_output_dim = int(config.epr_timing_bins)
            duration_output_dim = int(config.epr_timing_bins)
            velocity_output_dim = int(config.epr_value_bins)
            pedal_output_dim = int(config.epr_value_bins) * 4
        elif getattr(config, "task_type", "epr") == "epr" and self.epr_distribution == "beta_mu_kappa":
            ioi_output_dim = duration_output_dim = velocity_output_dim = 2
            shared_pack_mode = "beta_mu_kappa"
            pedal_output_dim = 8
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
                ioi_output_dim = duration_output_dim = per_feature_dim * 2
                velocity_output_dim = per_feature_dim
            else:
                ioi_output_dim = duration_output_dim = velocity_output_dim = per_feature_dim
            if self.epr_distribution in ALN_DISTRIBUTIONS:
                # ALN is timing-only split-normal with one component.
                velocity_output_dim = components * 3
            pedal_components = _scalar_distribution_components(config, self.pedal_distribution)
            pedal_output_dim = pedal_components * _scalar_distribution_dim(self.pedal_distribution) * 4
            if self.pedal_representation == "binary_4":
                pedal_output_dim = 4
        else:
            ioi_output_dim = duration_output_dim = velocity_output_dim = 1
            pedal_output_dim = 4

        generic_output_dim = self.output_dim
        if getattr(config, "task_type", "epr") == "csr" and _csr_uses_grid_head(config):
            generic_output_dim = _csr_grid_raw_output_dim(config)

        self.split_shared_heads = getattr(config, "task_type", "epr") == "epr"
        self.shared_pack_mode = shared_pack_mode
        if self.split_shared_heads:
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

    def _shared_outputs(self, hidden_states):
        shared_hidden = hidden_states[..., self.shared_slice]
        if not self.split_shared_heads:
            return self.shared_head(shared_hidden)

        ioi = self.ioi_head(shared_hidden)
        duration = (
            self.duration_head(shared_hidden)
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

    def forward(self, hidden_states):
        if _uses_inr_epr_targets(self.config):
            shared = self._shared_outputs(hidden_states)
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

        shared = self._shared_outputs(hidden_states)
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
    return str(getattr(config, "pedal_representation", "binary_4")).lower() == "binary_4"


def _mask_count(mask):
    return mask.to(dtype=torch.float32).sum().clamp_min(1.0)


def _split_epr_mixture_params(config, raw_outputs):
    components = _epr_mixture_components(config)
    distribution = getattr(config, "epr_distribution", "point").lower()
    if distribution in SN_DISTRIBUTIONS:
        feature_dim = 3
        ioi_base = raw_outputs[..., : feature_dim * 2].reshape(*raw_outputs.shape[:-1], 2, feature_dim)
        cursor = feature_dim * 2
        duration_base = raw_outputs[..., cursor : cursor + feature_dim * 2].reshape(
            *raw_outputs.shape[:-1],
            2,
            feature_dim,
        )
        cursor += feature_dim * 2
        velocity_base = raw_outputs[..., cursor : cursor + feature_dim]
        cursor += feature_dim
        return {
            "timing_log_loc": torch.stack([ioi_base[..., 0, 0], duration_base[..., 0, 0]], dim=-1),
            "timing_log_log_scale": torch.stack([ioi_base[..., 0, 1], duration_base[..., 0, 1]], dim=-1),
            "timing_log_alpha": torch.stack([ioi_base[..., 0, 2], duration_base[..., 0, 2]], dim=-1),
            "timing_raw_loc": torch.stack([ioi_base[..., 1, 0], duration_base[..., 1, 0]], dim=-1),
            "timing_raw_log_scale": torch.stack([ioi_base[..., 1, 1], duration_base[..., 1, 1]], dim=-1),
            "timing_raw_alpha": torch.stack([ioi_base[..., 1, 2], duration_base[..., 1, 2]], dim=-1),
            "velocity_loc": velocity_base[..., 0],
            "velocity_log_scale": velocity_base[..., 1],
            "velocity_alpha": velocity_base[..., 2],
            "pedal_binary_logits": raw_outputs[..., cursor : cursor + 4],
        }
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
        in {"log_deviation", "log_dev", "raw_log_deviation", "raw_log_dev"}
    )


def _uses_log_deviation_targets(config):
    return str(_config_value(config, "epr_timing_target", "absolute")).lower() in {
        "log_deviation",
        "log_dev",
        "raw_log_deviation",
        "raw_log_dev",
    }


def _uses_raw_log_deviation_targets(config):
    return str(_config_value(config, "epr_timing_target", "absolute")).lower() in {
        "raw_log_deviation",
        "raw_log_dev",
    }


def _config_value(config, name, default):
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def _torch_log_timing_code(time_ms, scale=50.0, max_time_ms=5000.0):
    value = time_ms.float().clamp(0.0, float(max_time_ms))
    scale_value = value.new_tensor(max(float(scale), 1e-12))
    denom = torch.log1p(value.new_tensor(float(max_time_ms)) / scale_value)
    return torch.log1p(value / scale_value) / denom


def _torch_raw_log_timing_code(time_ms, scale=50.0, max_time_ms=5000.0):
    value = time_ms.float().clamp(0.0, float(max_time_ms))
    seconds = value / 1000.0
    factor = value.new_tensor(1000.0 / max(float(scale), 1e-12))
    return torch.log1p(factor * seconds)


def _torch_log_timing_decode(time_norm, scale=50.0, max_time_ms=5000.0):
    clipped = time_norm.float().clamp(0.0, 1.0)
    scale_value = clipped.new_tensor(max(float(scale), 1e-12))
    denom = torch.log1p(clipped.new_tensor(float(max_time_ms)) / scale_value)
    return scale_value * torch.expm1(clipped * denom)


def _torch_raw_log_timing_decode(log_value, scale=50.0, max_time_ms=5000.0):
    factor = log_value.new_tensor(1000.0 / max(float(scale), 1e-12))
    decoded = 1000.0 * torch.expm1(log_value.float()) / factor
    return decoded.clamp(0.0, float(max_time_ms))


def _torch_timing_control_code(time_ms, timing_control_mode="log_scaled", use_scale_bit=False, log_scale=50.0):
    value = time_ms.float().clamp(0.0, 5000.0)
    mode = resolve_timing_control_mode(
        timing_control_mode=timing_control_mode,
        use_timing_scale_bit=use_scale_bit,
    )
    if mode == "piecewise_scale_bit":
        scale_bit = (value > 500.0).to(dtype=value.dtype)
        cont = torch.where(value > 500.0, value / 5000.0, value / 500.0)
        return torch.stack([scale_bit, cont], dim=-1)
    if mode == "piecewise_single":
        return torch.stack(
            [
                torch.where(value > 500.0, value / 5000.0, value / 500.0),
            ],
            dim=-1,
        )
    if mode == "dual_log_linear":
        return torch.stack(
            [
                torch.log1p(value) / torch.log1p(value.new_tensor(5000.0)),
                value / 5000.0,
            ],
            dim=-1,
        )
    if mode == "log_scaled":
        return _torch_log_timing_code(value, scale=log_scale, max_time_ms=5000.0).unsqueeze(-1)
    if mode == "raw_log":
        return torch.stack(
            [
                value / 1000.0,
                _torch_raw_log_timing_code(value, scale=log_scale, max_time_ms=5000.0),
            ],
            dim=-1,
        )
    if mode == "dual_clip_linear":
        return torch.stack(
            [
                torch.clamp(value / 500.0, max=1.0),
                value / 5000.0,
            ],
            dim=-1,
        )
    raise ValueError(f"Unsupported timing_control_mode={mode}")


def _target7_to_raw7(score_shared_raw, target_predictions, config=None):
    score_shared_raw = score_shared_raw.float()
    target_predictions = target_predictions.float()
    if target_predictions.shape[-1] < 7:
        raise ValueError(f"Expected target7 predictions, got shape {tuple(target_predictions.shape)}")
    if config is not None and not _uses_log_deviation_targets(config):
        raise ValueError("target7 -> raw7 reconstruction expects log_deviation timing targets")
    log_scale = float(_config_value(config, "timing_log_scale", 50.0)) if config is not None else 50.0
    if _uses_raw_log_deviation_targets(config):
        score_ioi_log = _torch_raw_log_timing_code(score_shared_raw[..., 0], scale=log_scale, max_time_ms=5000.0)
        score_duration_log = _torch_raw_log_timing_code(score_shared_raw[..., 1], scale=log_scale, max_time_ms=5000.0)
        perf_ioi_ms = _torch_raw_log_timing_decode(score_ioi_log + target_predictions[..., 0], scale=log_scale)
        perf_duration_ms = _torch_raw_log_timing_decode(score_duration_log + target_predictions[..., 1], scale=log_scale)
        if target_predictions.shape[-1] >= 9:
            velocity = target_predictions[..., 4].clamp(0.0, 1.0) * 127.0
            pedal = target_predictions[..., 5:9].clamp(0.0, 1.0) * 127.0
        else:
            velocity = target_predictions[..., 2].clamp(0.0, 1.0) * 127.0
            pedal = target_predictions[..., 3:7].clamp(0.0, 1.0) * 127.0
        return torch.cat([perf_ioi_ms.unsqueeze(-1), perf_duration_ms.unsqueeze(-1), velocity.unsqueeze(-1), pedal], dim=-1)

    target_predictions = target_predictions.clamp(0.0, 1.0)
    score_ioi_norm = _torch_log_timing_code(score_shared_raw[..., 0], scale=log_scale, max_time_ms=5000.0)
    score_duration_norm = _torch_log_timing_code(score_shared_raw[..., 1], scale=log_scale, max_time_ms=5000.0)
    perf_ioi_norm = (score_ioi_norm + (target_predictions[..., 0] - 0.5)).clamp(0.0, 1.0)
    perf_duration_norm = (score_duration_norm + (target_predictions[..., 1] - 0.5)).clamp(0.0, 1.0)
    perf_ioi_ms = _torch_log_timing_decode(perf_ioi_norm, scale=log_scale, max_time_ms=5000.0)
    perf_duration_ms = _torch_log_timing_decode(perf_duration_norm, scale=log_scale, max_time_ms=5000.0)
    velocity = target_predictions[..., 2] * 127.0
    pedal = target_predictions[..., 3:7] * 127.0
    return torch.cat([perf_ioi_ms.unsqueeze(-1), perf_duration_ms.unsqueeze(-1), velocity.unsqueeze(-1), pedal], dim=-1)


def _decoder_rows_require_score_shared_raw(config):
    return decoder_note_input_schema(config) == "integrated"


def _build_epr_decoder_perf_target_rows(config, target_predictions):
    target_features = target_predictions.float()
    masks = target_features.new_ones(*target_features.shape[:-1], 3)
    return torch.cat([target_features, masks], dim=-1)


def _target_predictions_to_feedback7(config, target_predictions):
    target_predictions = target_predictions.float()
    if _uses_raw_log_deviation_targets(config) and target_predictions.shape[-1] >= 9:
        return torch.cat(
            [
                target_predictions[..., 0:2],
                target_predictions[..., 4:5].clamp(0.0, 1.0),
                target_predictions[..., 5:9].clamp(0.0, 1.0),
            ],
            dim=-1,
        )
    if target_predictions.shape[-1] < 7:
        raise ValueError(f"Expected at least 7 target values, got shape {tuple(target_predictions.shape)}")
    if _uses_raw_log_deviation_targets(config):
        return torch.cat(
            [
                target_predictions[..., 0:2],
                target_predictions[..., 2:3].clamp(0.0, 1.0),
                target_predictions[..., 3:7].clamp(0.0, 1.0),
            ],
            dim=-1,
        )
    return target_predictions[..., :7].clamp(0.0, 1.0)


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
    log_scale = getattr(config, "timing_log_scale", 50.0)
    if resolve_timing_control_mode(timing_control_mode, use_timing_scale_bit) not in {"log_scaled", "raw_log"}:
        raise ValueError("EPR decoder feedback requires timing_control_mode=log_scaled or raw_log")
    target7 = _target_predictions_to_feedback7(config, target_predictions)
    score_ioi = _torch_timing_control_code(
        score_shared_raw[..., 0],
        timing_control_mode=timing_control_mode,
        use_scale_bit=use_timing_scale_bit,
        log_scale=log_scale,
    )
    score_duration = _torch_timing_control_code(
        score_shared_raw[..., 1],
        timing_control_mode=timing_control_mode,
        use_scale_bit=use_timing_scale_bit,
        log_scale=log_scale,
    )
    score_velocity = (score_shared_raw[..., 2:3].float().clamp(0.0, 127.0) / 127.0)
    if _uses_raw_log_deviation_targets(config):
        score_ioi_log = _torch_raw_log_timing_code(score_shared_raw[..., 0], scale=log_scale, max_time_ms=5000.0).unsqueeze(-1)
        score_duration_log = _torch_raw_log_timing_code(score_shared_raw[..., 1], scale=log_scale, max_time_ms=5000.0).unsqueeze(-1)
        perf_ioi_ms = _torch_raw_log_timing_decode(score_ioi_log + target7[..., 0:1], scale=log_scale)
        duration_ms = _torch_raw_log_timing_decode(score_duration_log + target7[..., 1:2], scale=log_scale)
        perf_ioi = _torch_timing_control_code(
            perf_ioi_ms.squeeze(-1),
            timing_control_mode=timing_control_mode,
            use_scale_bit=use_timing_scale_bit,
            log_scale=log_scale,
        )
        duration = _torch_timing_control_code(
            duration_ms.squeeze(-1),
            timing_control_mode=timing_control_mode,
            use_scale_bit=use_timing_scale_bit,
            log_scale=log_scale,
        )
    else:
        perf_ioi = (score_ioi[..., :1] + (target7[..., 0:1].float().clamp(0.0, 1.0) - 0.5)).clamp(0.0, 1.0)
        duration = (score_duration[..., :1] + (target7[..., 1:2].float().clamp(0.0, 1.0) - 0.5)).clamp(0.0, 1.0)
    velocity = target7[..., 2:3].float().clamp(0.0, 1.0)
    pedal = target7[..., 3:7].float().clamp(0.0, 1.0)
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
    masks = target_predictions.new_tensor([0.0, 1.0, 0.0]).expand(*target_predictions.shape[:-1], 3)
    return torch.cat([score_control, performance_control, musical, masks], dim=-1)


def _build_csr_decoder_rows(config, musical_predictions):
    musical = musical_predictions.float().clamp(0.0, 1.0)
    score_control_dim = int(getattr(config, "score_control_feature_dim", getattr(config, "control_feature_dim", 5)))
    performance_control_dim = int(
        getattr(config, "performance_control_feature_dim", getattr(config, "control_feature_dim", 5) + 4)
    )
    zeros = musical.new_zeros(*musical.shape[:-1], score_control_dim + performance_control_dim)
    masks = musical.new_tensor([0.0, 0.0, 1.0]).expand(*musical.shape[:-1], 3)
    return torch.cat([zeros, musical, masks], dim=-1)


def _csr_uses_grid_head(config):
    return str(getattr(config, "csr_grid_loss_type", "huber")).lower() in {
        "soft_ce",
        "soft_ce_huber",
        "ce",
        "hard_ce",
        "ordinal",
        "grid",
    }


def _csr_grid_bins(config, name):
    step = max(float(getattr(config, "csr_grid_step", 1.0 / 24.0)), 1e-12)
    max_value = float(getattr(config, f"csr_{name}_max"))
    return int(round(max_value / step)) + 1


def _csr_grid_raw_output_dim(config):
    return (
        _csr_grid_bins(config, "mo")
        + _csr_grid_bins(config, "md")
        + _csr_grid_bins(config, "ml")
        + 1
        + 8
    )


def _split_csr_grid_outputs(config, raw_outputs):
    start = 0
    outputs = {}
    for name in ("mo", "md", "ml"):
        bins = _csr_grid_bins(config, name)
        outputs[name] = raw_outputs[..., start : start + bins]
        start += bins
    outputs["tempo"] = raw_outputs[..., start]
    start += 1
    outputs["binary"] = raw_outputs[..., start : start + 8]
    return outputs


def _csr_grid_to_normalized(config, name, logits):
    bins = logits.shape[-1]
    values = torch.arange(bins, device=logits.device, dtype=torch.float32)
    step = float(getattr(config, "csr_grid_step", 1.0 / 24.0))
    max_value = max(float(getattr(config, f"csr_{name}_max")), step)
    indices = logits.float().argmax(dim=-1).to(dtype=torch.float32)
    return (indices * step / max_value).clamp(0.0, 1.0)


def _materialize_csr_prediction(config, raw_outputs):
    if not _csr_uses_grid_head(config):
        return torch.sigmoid(raw_outputs)
    parts = _split_csr_grid_outputs(config, raw_outputs)
    continuous = [
        _csr_grid_to_normalized(config, "mo", parts["mo"]),
        (torch.sigmoid(parts["binary"][..., 0]) >= 0.5).to(dtype=raw_outputs.dtype),
        _csr_grid_to_normalized(config, "md", parts["md"]),
        _csr_grid_to_normalized(config, "ml", parts["ml"]),
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


def _inflated_zero_one_mode_log_probs(raw_mode_logits):
    center = raw_mode_logits.new_zeros(*raw_mode_logits.shape[:-1], 1)
    logits = torch.cat([raw_mode_logits.float(), center.float()], dim=-1)
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


def _mixture_beta_params(raw_alpha, raw_beta, alpha_min=1e-4):
    alpha = F.softplus(raw_alpha.float()) + float(alpha_min)
    beta = F.softplus(raw_beta.float()) + float(alpha_min)
    mean = alpha / (alpha + beta).clamp_min(1e-12)
    return alpha, beta, mean


def _mixture_beta_log_prob(logits, raw_alpha, raw_beta, target, eps, alpha_min):
    target = target.float().clamp(float(eps), 1.0 - float(eps)).unsqueeze(-1)
    alpha, beta, _ = _mixture_beta_params(raw_alpha, raw_beta, alpha_min=alpha_min)
    log_pi = F.log_softmax(logits.float(), dim=-1)
    log_beta = torch.distributions.Beta(alpha, beta).log_prob(target)
    return torch.logsumexp(log_pi + log_beta, dim=-1)


def _mixture_beta_nll(logits, raw_alpha, raw_beta, target, mask, eps, alpha_min):
    values = -_mixture_beta_log_prob(logits, raw_alpha, raw_beta, target, eps, alpha_min)
    return _masked_mean(values, mask)


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
        return _mixture_beta_nll(logits, a, b, target, mask, eps, alpha_min)
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
    timing_norm = str(getattr(config, "timing_input_normalization", "scaled_log_5000_s10")).lower()
    ioi_ms = ioi_bins.to(dtype=torch.float32).clamp(0.0, float(timing_bins - 1))
    duration_ms = duration_bins.to(dtype=torch.float32).clamp(0.0, float(timing_bins - 1))
    if timing_norm in {"scaled_log_5000_s10", "log1p_t_over_10_5000", "log1p_x_over_10_5000"}:
        denom = torch.log1p(ioi_ms.new_tensor(500.0))
        ioi_norm = torch.log1p(ioi_ms.clamp(max=5000.0) / 10.0) / denom
        duration_norm = torch.log1p(duration_ms.clamp(max=5000.0) / 10.0) / denom
    elif timing_norm in {"log1p_t_over_50_5000", "log1p_x_over_50_5000"}:
        denom = torch.log1p(ioi_ms.new_tensor(100.0))
        ioi_norm = torch.log1p(ioi_ms.clamp(max=5000.0) / 50.0) / denom
        duration_norm = torch.log1p(duration_ms.clamp(max=5000.0) / 50.0) / denom
    elif timing_norm in {"log1p_t_over_100_5000", "log1p_x_over_100_5000"}:
        denom = torch.log1p(ioi_ms.new_tensor(50.0))
        ioi_norm = torch.log1p(ioi_ms.clamp(max=5000.0) / 100.0) / denom
        duration_norm = torch.log1p(duration_ms.clamp(max=5000.0) / 100.0) / denom
    elif timing_norm in {"legacy_log1p", "log1p", "log1p_10000"}:
        max_time = float(getattr(config, "max_time_ms", 10000.0))
        denom = torch.log1p(ioi_ms.new_tensor(max_time))
        ioi_norm = torch.log1p(ioi_ms.clamp(max=max_time)) / denom
        duration_norm = torch.log1p(duration_ms.clamp(max=max_time)) / denom
    elif timing_norm in {"linear_5000", "raw_linear_5000"}:
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


def _materialize_epr_prediction(config, raw_outputs, sampling_strategy="mean", score_shared_raw=None):
    if not _uses_inr_epr_targets(config):
        raise ValueError("EPR materialization only supports INR log_deviation targets")

    strategy_name = str(sampling_strategy).lower()
    shared_strategy = "mean" if strategy_name in {"soft", "prob", "probs", "probability", "probabilities"} else sampling_strategy
    distribution = getattr(config, "epr_distribution", "point").lower()
    if distribution in SN_DISTRIBUTIONS:
        params = _split_epr_mixture_params(config, raw_outputs)
        sigma_min = getattr(config, "skew_normal_sigma_min", getattr(config, "logistic_normal_sigma_min", 1e-4))
        sigma_max = getattr(config, "skew_normal_sigma_max", getattr(config, "logistic_normal_sigma_max", 1e4))
        logdev = _skew_normal_mean_or_sample(
            params["timing_log_loc"],
            params["timing_log_log_scale"],
            params["timing_log_alpha"],
            sampling_strategy=shared_strategy,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
        )
        velocity = _skew_normal_mean_or_sample(
            params["velocity_loc"],
            params["velocity_log_scale"],
            params["velocity_alpha"],
            sampling_strategy=shared_strategy,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
        ).clamp(0.0, 1.0)
        pedal = _materialize_binary4_logits(
            params["pedal_binary_logits"],
            sampling_strategy=sampling_strategy,
        )
        if _uses_raw_log_deviation_targets(config) and score_shared_raw is not None:
            score_shared_raw = score_shared_raw.float()
            log_scale = float(_config_value(config, "timing_log_scale", 50.0))
            score_ioi_log = _torch_raw_log_timing_code(score_shared_raw[..., 0], scale=log_scale, max_time_ms=5000.0)
            score_duration_log = _torch_raw_log_timing_code(
                score_shared_raw[..., 1],
                scale=log_scale,
                max_time_ms=5000.0,
            )
            perf_ioi_ms = _torch_raw_log_timing_decode(score_ioi_log + logdev[..., 0], scale=log_scale)
            perf_duration_ms = _torch_raw_log_timing_decode(score_duration_log + logdev[..., 1], scale=log_scale)
            rawdev = torch.stack(
                [
                    (perf_ioi_ms - score_shared_raw[..., 0]) / 1000.0,
                    (perf_duration_ms - score_shared_raw[..., 1]) / 1000.0,
                ],
                dim=-1,
            )
            return torch.cat([logdev, rawdev, velocity.unsqueeze(-1), pedal], dim=-1)
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
        pedal = _materialize_binary4_logits(raw_outputs[..., -4:], sampling_strategy=sampling_strategy)
        return torch.cat([shared, pedal], dim=-1)

    if _is_scalar_distribution(distribution):
        params = _split_epr_mixture_params(config, raw_outputs)

        def decode_shared(index):
            logits, a, b, c = _shared_scalar_params(config, params, index)
            return _decode_mixture_value(
                config,
                logits,
                a,
                b,
                c,
                sampling_strategy=shared_strategy,
            )

        shared = torch.stack([decode_shared(0), decode_shared(1), decode_shared(2)], dim=-1)
        pedal = _materialize_binary4_logits(
            params["pedal_binary_logits"],
            sampling_strategy=sampling_strategy,
        )
        return torch.cat([shared, pedal], dim=-1)

    shared = raw_outputs[..., :3].clamp(0.0, 1.0)
    pedal = _materialize_binary4_logits(raw_outputs[..., -4:], sampling_strategy=sampling_strategy)
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
    performance_dim = int(
        getattr(config, "performance_control_feature_dim", getattr(config, "control_feature_dim", 3) + 4)
    )
    if feedback_mask.shape[-1] == performance_dim:
        return feedback_mask
    if feedback_mask.shape[-1] < 7:
        raise ValueError(f"decoder_feedback_mask expects target7 or performance-control mask, got {tuple(feedback_mask.shape)}")
    decoder_mask = feedback_mask.new_zeros(*feedback_mask.shape[:-1], performance_dim)
    decoder_mask[..., 0:1] = feedback_mask[..., 0:1]
    decoder_mask[..., 1:2] = feedback_mask[..., 1:2]
    decoder_mask[..., 2:3] = feedback_mask[..., 2:3]
    decoder_mask[..., 3:7] = feedback_mask[..., 3:7]
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


def _apply_prior_note_dropout(config, decoder_input_continuous, special_note_ids, attention_mask):
    keep_prob = float(getattr(config, "prior_token_keep_prob", 1.0))
    if keep_prob >= 1.0 or decoder_input_continuous.shape[1] <= 1:
        return decoder_input_continuous, special_note_ids

    valid_mask = attention_mask[:, 1:].bool()
    if not valid_mask.any():
        return decoder_input_continuous, special_note_ids

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
        return decoder_input_continuous, masked_special_note_ids
    if dropout_mode in {"zero", "feature_zero"}:
        dropped = decoder_input_continuous.clone()
        keep_mask = keep_mask & valid_mask
        if dropped.shape[-1] > 2:
            dropped[:, 1:, 2:] = dropped[:, 1:, 2:] * keep_mask.unsqueeze(-1).to(dtype=dropped.dtype)
        else:
            dropped[:, 1:] = dropped[:, 1:] * keep_mask.unsqueeze(-1).to(dtype=dropped.dtype)
        return dropped, special_note_ids
    if dropout_mode in {"attribute_zero", "attribute_noise", "attribute_uniform"}:
        dropped = decoder_input_continuous.clone()
        if _uses_inr_epr_targets(config):
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
            return dropped, special_note_ids

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
        return dropped, special_note_ids
    if dropout_mode in {"none", "off"}:
        return decoder_input_continuous, special_note_ids
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
    pad_embed = note_encoder.pad_embedding().to(dtype=embeddings.dtype, device=embeddings.device)
    return torch.where(drop_mask.unsqueeze(-1), pad_embed.view(1, 1, -1), embeddings)


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
    if _uses_inr_epr_targets(config) or getattr(config, "task_type", "epr") == "csr":
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
            if _uses_inr_epr_targets(config):
                if _decoder_rows_require_score_shared_raw(config) and score_shared_raw is None:
                    raise ValueError("score_shared_raw is required for INR log_deviation AR prefix inputs")
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
            elif config.task_type == "csr":
                decoder_input_continuous[:, 1 : prefix_len + 1] = _build_csr_decoder_rows(
                    config,
                    prefix_predictions[:, :prefix_len].to(
                        dtype=decoder_input_continuous.dtype,
                        device=decoder_input_continuous.device,
                    ),
                )
            else:
                if config.task_type == "epr":
                    decoder_input_continuous[:, 1 : prefix_len + 1, 1] = 1.0
                elif config.task_type == "csr":
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
    if _uses_inr_epr_targets(config):
        if _decoder_rows_require_score_shared_raw(config) and score_shared_raw is None:
            raise ValueError("score_shared_raw is required for INR log_deviation AR note construction")
        if score_shared_raw is None:
            score_shared_raw = labels_continuous.new_zeros(*labels_continuous.shape[:-1], 3)
        return _build_epr_decoder_rows(
            config,
            score_shared_raw,
            labels_continuous,
            score_input_continuous=score_input_continuous,
        )
    if task_type == "csr":
        return _build_csr_decoder_rows(config, labels_continuous)
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


def _compute_integrated_loss_components(
    config,
    continuous_pred,
    labels_continuous,
    attention_mask,
    labels_epr_bins=None,
    score_shared_raw=None,
):
    del labels_epr_bins, score_shared_raw
    if getattr(config, "task_type", "epr") == "csr":
        return _compute_csr_loss_components(config, continuous_pred, labels_continuous, attention_mask)
    if not _uses_inr_epr_targets(config):
        raise ValueError("EPR loss only supports INR log_deviation targets")

    mask = attention_mask.bool()
    distribution = getattr(config, "epr_distribution", "point").lower()
    if distribution in SN_DISTRIBUTIONS:
        params = _split_epr_mixture_params(config, continuous_pred)
        sigma_min = getattr(config, "skew_normal_sigma_min", getattr(config, "logistic_normal_sigma_min", 1e-4))
        sigma_max = getattr(config, "skew_normal_sigma_max", getattr(config, "logistic_normal_sigma_max", 1e4))
        raw_lambda = float(getattr(config, "raw_timing_loss_lambda", 0.5))
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
        raw_offset = 2 if labels_continuous.shape[-1] >= 9 else None
        if raw_offset is not None:
            loss_ioi_raw = _skew_normal_nll(
                params["timing_raw_loc"][..., 0],
                params["timing_raw_log_scale"][..., 0],
                params["timing_raw_alpha"][..., 0],
                labels_continuous[..., 2],
                mask,
                sigma_min,
                sigma_max,
            )
            loss_duration_raw = _skew_normal_nll(
                params["timing_raw_loc"][..., 1],
                params["timing_raw_log_scale"][..., 1],
                params["timing_raw_alpha"][..., 1],
                labels_continuous[..., 3],
                mask,
                sigma_min,
                sigma_max,
            )
        else:
            loss_ioi_raw = loss_ioi_log.new_zeros(())
            loss_duration_raw = loss_duration_log.new_zeros(())
            raw_lambda = 0.0
        velocity_col = 4 if labels_continuous.shape[-1] >= 9 else 2
        pedal_start = 5 if labels_continuous.shape[-1] >= 9 else 3
        loss_ioi = loss_ioi_log + raw_lambda * loss_ioi_raw
        loss_duration = loss_duration_log + raw_lambda * loss_duration_raw
        loss_velocity = _skew_normal_nll(
            params["velocity_loc"],
            params["velocity_log_scale"],
            params["velocity_alpha"],
            labels_continuous[..., velocity_col],
            mask,
            sigma_min,
            sigma_max,
        )
        loss_pedal = _bce_loss(
            params["pedal_binary_logits"],
            labels_continuous[..., pedal_start : pedal_start + 4],
            mask.unsqueeze(-1).expand_as(labels_continuous[..., pedal_start : pedal_start + 4]),
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

        logits, a, b, c = _shared_scalar_params(config, params, 0)
        loss_ioi = loss_one(logits, a, b, labels_continuous[..., 0], c)
        logits, a, b, c = _shared_scalar_params(config, params, 1)
        loss_duration = loss_one(logits, a, b, labels_continuous[..., 1], c)
        logits, a, b, c = _shared_scalar_params(config, params, 2)
        loss_velocity = loss_one(logits, a, b, labels_continuous[..., 2], c)
        loss_pedal = _bce_loss(
            params["pedal_binary_logits"],
            labels_continuous[..., 3:7],
            mask.unsqueeze(-1).expand_as(labels_continuous[..., 3:7]),
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
        loss_pedal = _bce_loss(
            continuous_pred[..., -4:],
            labels_continuous[..., 3:7],
            mask.unsqueeze(-1).expand_as(labels_continuous[..., 3:7]),
        )
    else:
        pred = continuous_pred
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
        loss_pedal = _bce_loss(
            pred[..., -4:],
            labels_continuous[..., 3:7],
            mask.unsqueeze(-1).expand_as(labels_continuous[..., 3:7]),
        )
    return {
        "ioi": loss_ioi,
        "duration": loss_duration,
        "velocity": loss_velocity,
        "pedal": loss_pedal,
    }


def _bce_loss(logits, target, mask):
    values = F.binary_cross_entropy_with_logits(logits.float(), target.float(), reduction="none")
    return _masked_mean(values, mask)


def _normalized_to_csr_grid_target(config, name, target):
    step = max(float(getattr(config, "csr_grid_step", 1.0 / 24.0)), 1e-12)
    max_value = float(getattr(config, f"csr_{name}_max"))
    bins = _csr_grid_bins(config, name)
    return torch.round(target.float().clamp(0.0, 1.0) * max_value / step).long().clamp(0, bins - 1)


def _csr_grid_loss(config, name, logits, target, mask):
    target_bin = _normalized_to_csr_grid_target(config, name, target)
    grid_loss_type = str(getattr(config, "csr_grid_loss_type", "huber")).lower()
    if grid_loss_type in {"ce", "hard_ce", "ordinal", "grid"}:
        return _hard_categorical_loss(logits, target_bin, mask)
    if grid_loss_type in {"soft_ce", "soft_ce_huber"}:
        return _soft_categorical_loss(
            logits,
            target_bin,
            mask,
            tau=float(getattr(config, "csr_grid_soft_ce_tau", 1.5)),
        )
    raise ValueError(f"Unsupported CSR grid loss type: {grid_loss_type}")


def _compute_csr_loss_components(config, musical_logits, labels_musical, musical_mask):
    score_mask = musical_mask.bool()
    first_target = labels_musical[..., 5].float()
    ml_mask = score_mask & (first_target >= 0.5)
    grid_loss_type = getattr(config, "csr_grid_loss_type", "huber")

    if _csr_uses_grid_head(config):
        parts = _split_csr_grid_outputs(config, musical_logits)
        binary = parts["binary"]
        return {
            "mo": _csr_grid_loss(config, "mo", parts["mo"], labels_musical[..., 0], score_mask),
            "ioi_zero": _bce_loss(binary[..., 0], labels_musical[..., 1], score_mask),
            "md": _csr_grid_loss(config, "md", parts["md"], labels_musical[..., 2], score_mask),
            "ml": _csr_grid_loss(config, "ml", parts["ml"], labels_musical[..., 3], ml_mask),
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
):
    components = _compute_integrated_loss_components(
        config,
        continuous_pred,
        labels_continuous,
        attention_mask,
        labels_epr_bins=labels_epr_bins,
        score_shared_raw=score_shared_raw,
    )
    if getattr(config, "task_type", "epr") == "csr":
        weights = config.csr_loss_weights
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
    return (
        weights.get("ioi", 1.0) * components["ioi"]
        + weights.get("duration", 1.0) * components["duration"]
        + weights.get("velocity", 1.0) * components["velocity"]
        + weights.get("pedal", 1.0) * components["pedal"]
    )


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
        self._decoder_note_encoder = IntegratedNoteEncoder(
            config,
            continuous_dim=getattr(config, "decoder_input_continuous_dim", config.input_continuous_dim),
            role="decoder",
        )
        self.style_token_encoder = IntegratedStyleTokenEncoder(config) if config.use_style_tokens else None
        self.continuous_decoder = IntegratedContinuousDecoder(config)
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
        return self._decoder_note_encoder

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
        if _uses_inr_epr_targets(self.config) and score_shared_raw is None:
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
            continuous_pred = self.continuous_decoder(hidden_states)
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
                    decoder_input_continuous, special_note_ids = _apply_prior_note_dropout(
                        self.config,
                        decoder_input_continuous,
                        special_note_ids,
                        attention_mask,
                    )
                decoder_pitch_ids = _shift_pitch_right(self.config, pitch_ids, attention_mask)
                performance_embeds = self.decoder_note_encoder(
                    decoder_pitch_ids,
                    decoder_input_continuous,
                    special_note_ids=special_note_ids,
                    performance_missing_mask=decoder_missing_mask,
                )
                performance_embeds = self._apply_style_to_decoder_inputs(
                    performance_embeds,
                    **style_kwargs,
                )
                hidden_states = self.backbone(
                    score_context_embeds,
                    context_attention_mask,
                    performance_embeds=performance_embeds,
                    performance_attention_mask=attention_mask,
                )
                hidden_states = self._apply_style_to_decoder_hidden(hidden_states, **style_kwargs)
                continuous_pred = self.continuous_decoder(hidden_states)
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
            elif self.config.task_type == "csr":
                continuous_pred = _materialize_csr_prediction(self.config, continuous_pred)

        loss = None
        if labels_continuous is not None:
            loss_mask = label_mask if (self.config.task_type == "csr" and label_mask is not None) else attention_mask
            loss = _compute_integrated_loss(
                self.config,
                continuous_pred,
                labels_continuous,
                loss_mask,
                labels_epr_bins=labels_epr_bins,
                score_shared_raw=score_shared_raw,
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
            step_raw = self.continuous_decoder(hidden_states[:, -1:, :])
            if self.config.task_type == "csr":
                step_pred = _materialize_csr_prediction(self.config, step_raw)
            else:
                step_pred = _materialize_epr_prediction(
                    self.config,
                    step_raw,
                    sampling_strategy=sampling_strategy,
                    score_shared_raw=score_shared_raw[:, step : step + 1],
                )
            predictions.append(step_pred)
            if step + 1 < seq_len:
                if _uses_inr_epr_targets(self.config):
                    decoder_input_continuous[:, step + 1] = _build_epr_decoder_rows(
                        self.config,
                        score_shared_raw[:, step : step + 1],
                        step_pred,
                        score_input_continuous=continuous[:, step : step + 1],
                    )[:, 0]
                elif self.config.task_type == "epr":
                    decoder_input_continuous[:, step + 1, 1] = 1.0
                elif self.config.task_type == "csr":
                    decoder_input_continuous[:, step + 1] = _build_csr_decoder_rows(
                        self.config,
                        step_pred,
                    )[:, 0]
                if not _uses_inr_epr_targets(self.config) and self.config.task_type != "csr":
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
        self._decoder_note_encoder = IntegratedNoteEncoder(
            config,
            continuous_dim=getattr(config, "decoder_input_continuous_dim", config.input_continuous_dim),
            role="decoder",
        )
        self.style_token_encoder = IntegratedStyleTokenEncoder(config) if config.use_style_tokens else None
        self.model = IntegratedPianoT5GemmaModel(config)
        self.continuous_decoder = IntegratedContinuousDecoder(config)
        self.post_init()

    @property
    def decoder_note_encoder(self):
        return self._decoder_note_encoder

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
                decoder_input_continuous, special_note_ids = _apply_prior_note_dropout(
                    self.config,
                    decoder_input_continuous,
                    special_note_ids,
                    attention_mask,
                )
            decoder_pitch_ids = _shift_pitch_right(self.config, pitch_ids, attention_mask)
            decoder_inputs_embeds = self.decoder_note_encoder(
                decoder_pitch_ids,
                decoder_input_continuous,
                special_note_ids=special_note_ids,
                performance_missing_mask=decoder_missing_mask,
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
        if _uses_inr_epr_targets(self.config) and score_shared_raw is None:
            raise ValueError("score_shared_raw is required for INR log_deviation EPR")

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
        )
        if decoder_inputs_embeds is not None:
            decoder_inputs_embeds = self._apply_style_to_decoder_inputs(
                decoder_inputs_embeds,
                **style_kwargs,
            )
            if self.training and bool(getattr(self.config, "tf_embedding_mask_decoder", False)):
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
                loss_mask = label_mask if (self.config.task_type == "csr" and label_mask is not None) else attention_mask
                loss = _compute_integrated_loss(
                    self.config,
                    continuous_pred,
                    labels_continuous,
                    loss_mask,
                    labels_epr_bins=labels_epr_bins,
                    score_shared_raw=score_shared_raw,
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
        continuous_pred = self.continuous_decoder(decoder_hidden)
        if labels_continuous is None:
            if self.config.task_type == "epr":
                continuous_pred = _materialize_epr_prediction(
                    self.config,
                    continuous_pred,
                    sampling_strategy=continuous_sampling_strategy,
                    score_shared_raw=score_shared_raw,
                )
            elif self.config.task_type == "csr":
                continuous_pred = _materialize_csr_prediction(self.config, continuous_pred)

        loss = None
        if labels_continuous is not None:
            loss_mask = label_mask if (self.config.task_type == "csr" and label_mask is not None) else attention_mask
            loss = _compute_integrated_loss(
                self.config,
                continuous_pred,
                labels_continuous,
                loss_mask,
                labels_epr_bins=labels_epr_bins,
                score_shared_raw=score_shared_raw,
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
            step_raw = self.continuous_decoder(decoder_hidden[:, -1:, :])
            if self.config.task_type == "csr":
                step_pred = _materialize_csr_prediction(self.config, step_raw)
            else:
                step_pred = _materialize_epr_prediction(
                    self.config,
                    step_raw,
                    sampling_strategy=sampling_strategy,
                    score_shared_raw=score_shared_raw[:, step : step + 1],
                )
            predictions.append(step_pred)

            if step + 1 < seq_len:
                if _uses_inr_epr_targets(self.config):
                    decoder_input_continuous[:, step + 1] = _build_epr_decoder_rows(
                        self.config,
                        score_shared_raw[:, step : step + 1],
                        step_pred,
                        score_input_continuous=continuous[:, step : step + 1],
                    )[:, 0]
                elif self.config.task_type == "epr":
                    decoder_input_continuous[:, step + 1, 1] = 1.0
                elif self.config.task_type == "csr":
                    decoder_input_continuous[:, step + 1] = _build_csr_decoder_rows(
                        self.config,
                        step_pred,
                    )[:, 0]
                if not _uses_inr_epr_targets(self.config) and self.config.task_type != "csr":
                    decoder_input_continuous[:, step + 1, 2:] = step_pred[:, 0, :]

        output = torch.cat(predictions, dim=1) if predictions else continuous.new_zeros((batch_size, 0, self.config.output_continuous_dim))
        return output
