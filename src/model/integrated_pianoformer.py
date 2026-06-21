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


class IntegratedPianoT5GemmaConfig(PianoT5GemmaConfig):
    def __init__(
        self,
        backbone_type="t5",
        continuous_dim=7,
        input_continuous_dim=None,
        output_continuous_dim=None,
        pitch_vocab_size=128,
        pitch_pad_id=128,
        max_time_ms=10000.0,
        pedal_output_activation="sigmoid",
        task_type="epr",
        input_feature_mode="legacy",
        score_feature_dim=8,
        time_loss_type="huber",
        value_loss_type="mse",
        csr_grid_loss_type="huber",
        huber_delta=0.05,
        loss_weights=None,
        csr_loss_weights=None,
        decoder_input_mode="score",
        note_embedding_mode="fine",
        special_note_vocab_size=5,
        special_note_ids=None,
        pine_partition_dims=None,
        use_full_type_embedding=True,
        use_group_presence_mask=True,
        head_input_mode="full",
        embedding_depth=2,
        head_depth=2,
        head_activation="gelu",
        gpt_layers_num=None,
        bert_layers_num=None,
        max_position_embeddings=4096,
        attention_dropout=0.0,
        epr_distribution="point",
        epr_mixture_components=1,
        epr_distribution_eps=None,
        logistic_normal_sigma_min=1e-3,
        logistic_normal_sigma_max=10.0,
        beta_eps=1e-5,
        beta_kappa_min=1e-3,
        beta_alpha_min=1e-4,
        epr_inflated_features=None,
        epr_timing_bins=5000,
        epr_value_bins=128,
        soft_ce_tau=None,
        timing_input_normalization="legacy_log1p",
        prior_token_keep_prob=1.0,
        prior_token_dropout_mode="mask",
        pitch_onehot_dim=88,
        feature_embedding_dim=680,
        piano_pitch_min=21,
        pedal_representation="continuous_4",
        pedal_start_loss_weight=1.0,
        pedal_ctrl_loss_weight=1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.backbone_type = backbone_type
        self.continuous_dim = continuous_dim
        self.input_continuous_dim = input_continuous_dim or continuous_dim
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
        self.huber_delta = huber_delta
        self.loss_weights = loss_weights or {
            "ioi": 1.0,
            "duration": 1.0,
            "velocity": 1.0,
            "pedal": 1.0,
        }
        self.csr_loss_weights = csr_loss_weights or {
            "mo": 1.0,
            "md": 1.0,
            "first": 1.0,
            "ml": 1.0,
            "staff": 0.5,
            "trill": 0.4,
            "grace": 0.4,
            "staccato": 0.3,
        }
        self.decoder_input_mode = decoder_input_mode
        self.note_embedding_mode = note_embedding_mode
        self.special_note_vocab_size = special_note_vocab_size
        self.special_note_ids = special_note_ids or {
            "pad": 0,
            "mask": 1,
            "bos": 2,
            "eos": 3,
            "play": 4,
        }
        self.pine_partition_dims = pine_partition_dims or {
            "pitch": 128,
            "shared": 256,
            "score": 256,
            "perf": 128,
        }
        self.use_full_type_embedding = use_full_type_embedding
        self.use_group_presence_mask = use_group_presence_mask
        self.head_input_mode = head_input_mode
        self.embedding_depth = embedding_depth
        self.head_depth = head_depth
        self.head_activation = head_activation
        self.gpt_layers_num = gpt_layers_num
        self.bert_layers_num = bert_layers_num
        self.max_position_embeddings = max_position_embeddings
        self.attention_dropout = attention_dropout
        self.epr_distribution = epr_distribution
        self.epr_mixture_components = int(epr_mixture_components)
        self.epr_distribution_eps = beta_eps if epr_distribution_eps is None else epr_distribution_eps
        self.logistic_normal_sigma_min = logistic_normal_sigma_min
        self.logistic_normal_sigma_max = logistic_normal_sigma_max
        self.beta_eps = beta_eps
        self.beta_kappa_min = beta_kappa_min
        self.beta_alpha_min = beta_alpha_min
        self.epr_inflated_features = epr_inflated_features or {
            "ioi": "zero",
            "pedal": "zero_one",
        }
        self.epr_timing_bins = int(epr_timing_bins)
        self.epr_value_bins = int(epr_value_bins)
        self.soft_ce_tau = soft_ce_tau or {
            "ioi": 10.0,
            "duration": 30.0,
            "velocity": 6.0,
            "pedal": 2.0,
        }
        self.timing_input_normalization = timing_input_normalization
        self.prior_token_keep_prob = prior_token_keep_prob
        self.prior_token_dropout_mode = prior_token_dropout_mode
        self.pitch_onehot_dim = int(pitch_onehot_dim)
        self.feature_embedding_dim = int(feature_embedding_dim)
        self.piano_pitch_min = int(piano_pitch_min)
        self.pedal_representation = str(pedal_representation).lower()
        self.pedal_start_loss_weight = float(pedal_start_loss_weight)
        self.pedal_ctrl_loss_weight = float(pedal_ctrl_loss_weight)


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
    depth = int(depth)
    if depth <= 1:
        return nn.Linear(input_dim, output_dim)
    layers = [nn.Linear(input_dim, hidden_dim), _activation(activation)]
    for _ in range(depth - 2):
        layers.extend([nn.Linear(hidden_dim, hidden_dim), _activation(activation)])
    layers.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*layers)


class IntegratedNoteEncoder(nn.Module):
    def __init__(self, config, continuous_dim=None, role="score"):
        super().__init__()
        self.config = config
        continuous_dim = continuous_dim or config.input_continuous_dim
        self.continuous_dim = continuous_dim
        self.role = str(role).lower()
        self.mode = getattr(config, "note_embedding_mode", "fine").lower()
        self.special_note_embeddings = nn.Embedding(
            config.special_note_vocab_size,
            config.hidden_size,
        )
        self.embedding_depth = getattr(config, "embedding_depth", 2)
        self.activation = getattr(config, "head_activation", "gelu")

        if self.mode in {"score_perf", "score_perf_split"}:
            self.pitch_dim = int(getattr(config, "pitch_onehot_dim", 88))
            self.feature_dim = int(getattr(config, "feature_embedding_dim", config.hidden_size - self.pitch_dim))
            if self.pitch_dim + self.feature_dim != config.hidden_size:
                raise ValueError(
                    "score_perf embedding requires pitch_onehot_dim + feature_embedding_dim "
                    f"to equal hidden_size={config.hidden_size}, got {self.pitch_dim}+{self.feature_dim}"
                )
            if self.pitch_dim != 88:
                raise ValueError(f"score_perf embedding currently expects pitch_onehot_dim=88, got {self.pitch_dim}")
            if self.role == "score":
                feature_input_dim = 11
            elif self.role in {"perf", "performance", "decoder"}:
                feature_input_dim = 7
            else:
                raise ValueError(f"Unsupported score_perf encoder role: {self.role}")
            self.feature_projection = _make_mlp(
                feature_input_dim,
                self.feature_dim,
                self.feature_dim,
                self.embedding_depth,
                self.activation,
            )
            self.feature_norm = nn.LayerNorm(self.feature_dim)
        elif self.mode == "pine":
            dims = config.pine_partition_dims
            self.pitch_dim = int(dims["pitch"])
            self.shared_dim = int(dims["shared"])
            self.score_dim = int(dims["score"])
            self.perf_dim = int(dims["perf"])
            partition_total = self.pitch_dim + self.shared_dim + self.score_dim + self.perf_dim
            if partition_total != config.hidden_size:
                raise ValueError(
                    f"PINE partition dims must sum to hidden_size={config.hidden_size}, got {partition_total}"
                )
            self.pitch_embedding = nn.Embedding(
                config.pitch_vocab_size + 1,
                self.pitch_dim,
                padding_idx=config.pitch_pad_id,
            )
            self.shared_projection = _make_mlp(3, self.shared_dim, self.shared_dim, self.embedding_depth, self.activation)
            self.score_projection = _make_mlp(8, self.score_dim, self.score_dim, self.embedding_depth, self.activation)
            self.pedal_projection = _make_mlp(4, self.perf_dim, self.perf_dim, self.embedding_depth, self.activation)
        elif self.mode == "fine":
            self.pitch_embedding = nn.Embedding(
                config.pitch_vocab_size + 1,
                config.hidden_size,
                padding_idx=config.pitch_pad_id,
            )
            self.shared_projection = _make_mlp(3, config.hidden_size, config.hidden_size, self.embedding_depth, self.activation)
            self.score_projection = _make_mlp(8, config.hidden_size, config.hidden_size, self.embedding_depth, self.activation)
            self.pedal_projection = _make_mlp(4, config.hidden_size, config.hidden_size, self.embedding_depth, self.activation)
            if getattr(config, "use_full_type_embedding", True):
                self.type_projection = _make_mlp(2, config.hidden_size, config.hidden_size, self.embedding_depth, self.activation)
            else:
                self.type_projection = None
        elif self.mode == "legacy":
            self.pitch_embedding = nn.Embedding(
                config.pitch_vocab_size + 1,
                config.hidden_size,
                padding_idx=config.pitch_pad_id,
            )
            self.continuous_mlp = _make_mlp(
                continuous_dim,
                config.hidden_size,
                config.hidden_size,
                self.embedding_depth,
                self.activation,
            )
        else:
            raise ValueError(f"Unsupported note_embedding_mode: {self.mode}")
        self.norm = nn.LayerNorm(config.hidden_size)

    def _pitch_one_hot(self, pitch_ids):
        pitch_min = int(getattr(self.config, "piano_pitch_min", 21))
        pitch_index = pitch_ids.long() - pitch_min
        valid = (pitch_index >= 0) & (pitch_index < self.pitch_dim)
        safe_index = pitch_index.clamp(0, self.pitch_dim - 1)
        one_hot = F.one_hot(safe_index, num_classes=self.pitch_dim).to(
            dtype=self.special_note_embeddings.weight.dtype,
            device=pitch_ids.device,
        )
        return one_hot * valid.unsqueeze(-1).to(dtype=one_hot.dtype)

    def _split_groups(self, continuous):
        batch_size, seq_len, _ = continuous.shape
        continuous_dim = continuous.shape[-1]
        type_bits = continuous.new_zeros(batch_size, seq_len, 2)
        shared = continuous.new_zeros(batch_size, seq_len, 3)
        score = continuous.new_zeros(batch_size, seq_len, 8)
        pedal = continuous.new_zeros(batch_size, seq_len, 4)

        if continuous_dim >= 13:
            type_bits = continuous[..., 0:2]
            shared = continuous[..., 2:5]
            score = continuous[..., 5:13]
        elif continuous_dim >= 9:
            type_bits = continuous[..., 0:2]
            shared = continuous[..., 2:5]
            pedal = continuous[..., 5:9]
        elif continuous_dim >= 7:
            type_bits[..., 1] = 1.0
            shared = continuous[..., 0:3]
            pedal = continuous[..., 3:7]
        elif continuous_dim >= 3:
            type_bits[..., 0] = 1.0
            shared = continuous[..., 0:3]
        else:
            raise ValueError(f"Unsupported continuous_dim={continuous_dim}")

        if getattr(self.config, "use_group_presence_mask", True):
            score = score * type_bits[..., 0:1]
            pedal = pedal * type_bits[..., 1:2]
        return type_bits, shared, score, pedal

    def forward(self, pitch_ids, continuous, special_note_ids=None):
        if self.mode in {"score_perf", "score_perf_split"}:
            projection_dtype = self.feature_norm.weight.dtype
            continuous = continuous.to(dtype=projection_dtype)
            if self.role == "score":
                _, shared, score, _ = self._split_groups(continuous)
                features = torch.cat([shared, score], dim=-1)
            else:
                if continuous.shape[-1] >= 9:
                    _, shared, _, pedal = self._split_groups(continuous)
                elif continuous.shape[-1] >= 7:
                    shared = continuous[..., 0:3]
                    pedal = continuous[..., 3:7]
                else:
                    raise ValueError(f"Perf score_perf encoder requires at least 7 continuous dims, got {continuous.shape[-1]}")
                features = torch.cat([shared, pedal], dim=-1)
            pitch_embeds = self._pitch_one_hot(pitch_ids).to(dtype=projection_dtype)
            feature_embeds = self.feature_norm(self.feature_projection(features))
            embeddings = torch.cat([pitch_embeds, feature_embeds], dim=-1)
            return self._apply_special_embeddings(embeddings, special_note_ids)

        if self.mode == "legacy":
            pitch_embeds = self.pitch_embedding(pitch_ids)
            continuous = continuous.to(dtype=self.continuous_mlp[0].weight.dtype if isinstance(self.continuous_mlp, nn.Sequential) else self.continuous_mlp.weight.dtype)
            continuous_embeds = self.continuous_mlp(continuous)
            embeddings = self.norm(pitch_embeds + continuous_embeds)
            return self._apply_special_embeddings(embeddings, special_note_ids)

        projection_dtype = next(self.parameters()).dtype
        continuous = continuous.to(dtype=projection_dtype)
        type_bits, shared, score, pedal = self._split_groups(continuous)
        pitch_embeds = self.pitch_embedding(pitch_ids)
        shared_embeds = self.shared_projection(shared)
        score_embeds = self.score_projection(score)
        pedal_embeds = self.pedal_projection(pedal)
        if getattr(self.config, "use_group_presence_mask", True):
            score_embeds = score_embeds * type_bits[..., 0:1]
            pedal_embeds = pedal_embeds * type_bits[..., 1:2]

        if self.mode == "pine":
            embeddings = torch.cat([pitch_embeds, shared_embeds, score_embeds, pedal_embeds], dim=-1)
        else:
            embeddings = pitch_embeds + shared_embeds + score_embeds + pedal_embeds
            if self.type_projection is not None:
                embeddings = embeddings + self.type_projection(type_bits)
        embeddings = self.norm(embeddings)
        return self._apply_special_embeddings(embeddings, special_note_ids)

    def _apply_special_embeddings(self, embeddings, special_note_ids):
        if special_note_ids is None:
            return embeddings
        special_mask = special_note_ids >= 0
        if not special_mask.any():
            return embeddings
        safe_ids = special_note_ids.clamp_min(0)
        special_embeds = self.special_note_embeddings(safe_ids).to(dtype=embeddings.dtype)
        return torch.where(special_mask.unsqueeze(-1), special_embeds, embeddings)


class IntegratedContinuousDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.output_dim = config.output_continuous_dim
        self.mode = getattr(config, "note_embedding_mode", "fine").lower()
        self.head_input_mode = getattr(config, "head_input_mode", "full").lower()
        self.epr_distribution = getattr(config, "epr_distribution", "point").lower()
        head_depth = getattr(config, "head_depth", 2)
        activation = getattr(config, "head_activation", "gelu")

        full_dim = config.hidden_size
        if self.mode in {"score_perf", "score_perf_split"} and self.head_input_mode in {"feature", "partitioned"}:
            pitch_dim = int(getattr(config, "pitch_onehot_dim", 88))
            feature_dim = int(getattr(config, "feature_embedding_dim", config.hidden_size - pitch_dim))
            if pitch_dim + feature_dim != config.hidden_size:
                raise ValueError(
                    "score_perf head requires pitch_onehot_dim + feature_embedding_dim "
                    f"to equal hidden_size={config.hidden_size}, got {pitch_dim}+{feature_dim}"
                )
            self.shared_slice = self.score_slice = self.perf_slice = slice(pitch_dim, pitch_dim + feature_dim)
            shared_dim = score_dim = perf_dim = feature_dim
        elif self.mode == "pine" and self.head_input_mode == "partitioned":
            dims = config.pine_partition_dims
            self.shared_slice = slice(int(dims["pitch"]), int(dims["pitch"]) + int(dims["shared"]))
            self.score_slice = slice(self.shared_slice.stop, self.shared_slice.stop + int(dims["score"]))
            self.perf_slice = slice(self.score_slice.stop, self.score_slice.stop + int(dims["perf"]))
            shared_dim = int(dims["shared"])
            score_dim = int(dims["score"])
            perf_dim = int(dims["perf"])
        else:
            self.shared_slice = self.score_slice = self.perf_slice = slice(None)
            shared_dim = score_dim = perf_dim = full_dim

        if (
            getattr(config, "task_type", "epr") == "epr"
            and self.epr_distribution in {"categorical", "hard_categorical", "soft_categorical"}
        ):
            shared_output_dim = int(config.epr_timing_bins) * 2 + int(config.epr_value_bins)
            pedal_output_dim = int(config.epr_value_bins) * 4
        elif getattr(config, "task_type", "epr") == "epr" and self.epr_distribution == "beta_mu_kappa":
            shared_output_dim = 6
            pedal_output_dim = 8
        elif (
            getattr(config, "task_type", "epr") == "epr"
            and self.epr_distribution in {
                "logistic_normal",
                "mixture_logistic_normal",
                "inflated_mixture_logistic_normal",
                "mixture_beta",
            }
        ):
            components = int(getattr(config, "epr_mixture_components", 1))
            if components < 1:
                raise ValueError(f"epr_mixture_components must be >= 1, got {components}")
            per_feature_dim = components * 3
            shared_output_dim = per_feature_dim * 3
            pedal_output_dim = per_feature_dim * 4
            if self.epr_distribution == "inflated_mixture_logistic_normal":
                shared_output_dim += 2
                pedal_output_dim += 3 * 4
        else:
            shared_output_dim = 3
            pedal_output_dim = 4

        self.pedal_representation = str(getattr(config, "pedal_representation", "continuous_4")).lower()
        if getattr(config, "task_type", "epr") == "epr" and self.pedal_representation == "start_ctrl":
            if self.epr_distribution in {"categorical", "hard_categorical", "soft_categorical"}:
                raise ValueError("pedal_representation=start_ctrl is not implemented for categorical EPR heads")
            if self.epr_distribution == "beta_mu_kappa":
                pedal_output_dim = 4
            elif self.epr_distribution in {
                "logistic_normal",
                "mixture_logistic_normal",
                "inflated_mixture_logistic_normal",
                "mixture_beta",
            }:
                pedal_output_dim = int(getattr(config, "epr_mixture_components", 1)) * 3 * 2
            else:
                pedal_output_dim = 2

        self.shared_head = _make_mlp(shared_dim, shared_output_dim, shared_dim, head_depth, activation)
        self.pedal_head = _make_mlp(perf_dim, pedal_output_dim, perf_dim, head_depth, activation)
        self.generic_head = _make_mlp(score_dim, self.output_dim, score_dim, head_depth, activation)

    def forward(self, hidden_states):
        if self.output_dim != 7:
            return self.generic_head(hidden_states[..., self.score_slice])

        shared = self.shared_head(hidden_states[..., self.shared_slice])
        pedal = self.pedal_head(hidden_states[..., self.perf_slice])
        if self.pedal_representation == "start_ctrl":
            if self.epr_distribution in {
                "beta_mu_kappa",
                "categorical",
                "hard_categorical",
                "soft_categorical",
                "logistic_normal",
                "mixture_logistic_normal",
                "inflated_mixture_logistic_normal",
                "mixture_beta",
            }:
                return torch.cat([shared, pedal], dim=-1)
            return torch.cat([torch.sigmoid(shared), pedal], dim=-1)
        if self.epr_distribution in {
            "beta_mu_kappa",
            "categorical",
            "hard_categorical",
            "soft_categorical",
            "logistic_normal",
            "mixture_logistic_normal",
            "inflated_mixture_logistic_normal",
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


def _split_epr_mixture_params(config, raw_outputs):
    components = _epr_mixture_components(config)
    per_feature_dim = components * 3
    shared_base_dim = per_feature_dim * 3
    pedal_base_dim = per_feature_dim * 4
    shared_base = raw_outputs[..., :shared_base_dim].reshape(*raw_outputs.shape[:-1], 3, 3, components)
    distribution = getattr(config, "epr_distribution", "point").lower()
    pedal_representation = str(getattr(config, "pedal_representation", "continuous_4")).lower()
    params = {
        "shared_logits": shared_base[..., 0, :],
        "shared_a": shared_base[..., 1, :],
        "shared_b": shared_base[..., 2, :],
    }
    if pedal_representation == "start_ctrl":
        if distribution == "inflated_mixture_logistic_normal":
            params["ioi_mode_logits"] = raw_outputs[..., shared_base_dim : shared_base_dim + 2]
        return params

    pedal_start = shared_base_dim
    if distribution == "inflated_mixture_logistic_normal":
        pedal_start += 2
    pedal_end = pedal_start + pedal_base_dim
    pedal_base = raw_outputs[..., pedal_start:pedal_end].reshape(*raw_outputs.shape[:-1], 4, 3, components)
    params["pedal_logits"] = pedal_base[..., 0, :]
    params["pedal_a"] = pedal_base[..., 1, :]
    params["pedal_b"] = pedal_base[..., 2, :]
    if distribution == "inflated_mixture_logistic_normal":
        params["ioi_mode_logits"] = raw_outputs[..., shared_base_dim : shared_base_dim + 2]
        params["pedal_mode_logits"] = raw_outputs[..., pedal_end : pedal_end + 12].reshape(*raw_outputs.shape[:-1], 4, 3)
    return params


def _pedal_start_ctrl_scalar_dim(config):
    distribution = getattr(config, "epr_distribution", "point").lower()
    if distribution == "beta_mu_kappa":
        return 2
    if distribution in {
        "logistic_normal",
        "mixture_logistic_normal",
        "inflated_mixture_logistic_normal",
        "mixture_beta",
    }:
        return int(getattr(config, "epr_mixture_components", 1)) * 3
    return 1


def _split_start_ctrl_from_outputs(config, raw_outputs):
    scalar_dim = _pedal_start_ctrl_scalar_dim(config)
    pedal_raw = raw_outputs[..., -(2 * scalar_dim):]
    return {
        "start_raw": pedal_raw[..., :scalar_dim],
        "ctrl_raw": pedal_raw[..., scalar_dim:],
    }


def _pedal_start_ctrl_targets(pedal_values, attention_mask=None):
    values = pedal_values.float().clamp(0.0, 1.0)
    start = values[..., 0]
    if start.shape[-1] > 1:
        raw_next_start = torch.cat([start[..., 1:], start[..., -1:]], dim=-1)
        if attention_mask is not None:
            next_valid = torch.cat(
                [
                    attention_mask[..., 1:].bool(),
                    attention_mask[..., -1:].new_zeros(attention_mask[..., -1:].shape, dtype=torch.bool),
                ],
                dim=-1,
            )
            next_start = torch.where(next_valid, raw_next_start, start)
        else:
            next_start = raw_next_start
    else:
        next_start = start
    candidates = values[..., 1:4]
    lower = torch.minimum(start, next_start).unsqueeze(-1)
    upper = torch.maximum(start, next_start).unsqueeze(-1)
    outside_distance = (lower - candidates).clamp_min(0.0) + (candidates - upper).clamp_min(0.0)
    outside_idx = outside_distance.argmax(dim=-1, keepdim=True)
    middle_idx = outside_idx.new_full(outside_idx.shape, 1)
    ctrl_idx = torch.where(outside_distance.max(dim=-1, keepdim=True).values > 0.0, outside_idx, middle_idx)
    ctrl = candidates.gather(dim=-1, index=ctrl_idx).squeeze(-1)
    return start, ctrl


def materialize_start_ctrl_sequence(predictions, attention_mask=None):
    predictions = predictions.float().clamp(0.0, 1.0)
    if predictions.shape[-1] < 7:
        raise ValueError(f"Expected predictions with 7 continuous dims, got {predictions.shape[-1]}")
    shared = predictions[..., :3]
    start = predictions[..., 3].clamp(0.0, 1.0)
    ctrl = predictions[..., 4].clamp(0.0, 1.0)
    if start.shape[-1] > 1:
        raw_next_start = torch.cat([start[..., 1:], start[..., -1:]], dim=-1)
        if attention_mask is not None:
            next_valid = torch.cat(
                [
                    attention_mask[..., 1:].bool(),
                    attention_mask[..., -1:].new_zeros(attention_mask[..., -1:].shape, dtype=torch.bool),
                ],
                dim=-1,
            )
            next_start = torch.where(next_valid, raw_next_start, start)
        else:
            next_start = raw_next_start
    else:
        next_start = start
    pedal = torch.stack(
        [
            start,
            (start + ctrl) * 0.5,
            ctrl,
            (ctrl + next_start) * 0.5,
        ],
        dim=-1,
    )
    return torch.cat([shared, pedal.clamp(0.0, 1.0)], dim=-1)


def canonicalize_start_ctrl_sequence(predictions):
    predictions = predictions.float().clamp(0.0, 1.0)
    if predictions.shape[-1] < 7:
        raise ValueError(f"Expected predictions with 7 continuous dims, got {predictions.shape[-1]}")
    shared = predictions[..., :3]
    start = predictions[..., 3]
    ctrl = predictions[..., 5]
    return _pack_start_ctrl_prediction(shared, start, ctrl)


def _pack_start_ctrl_prediction(shared, start, ctrl):
    pedal = torch.stack([start, ctrl, ctrl, start], dim=-1)
    return torch.cat([shared, pedal], dim=-1).clamp_(0.0, 1.0)


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


def _inflated_logistic_normal_nll(config, logits, raw_mu, raw_log_sigma, mode_logits, target, mask, inflation):
    eps = float(getattr(config, "epr_distribution_eps", getattr(config, "beta_eps", 1e-5)))
    continuous_log_prob = _mixture_logistic_normal_log_prob(
        logits,
        raw_mu,
        raw_log_sigma,
        target,
        eps,
        getattr(config, "logistic_normal_sigma_min", 1e-3),
        getattr(config, "logistic_normal_sigma_max", 10.0),
    )
    mode_log_probs = F.log_softmax(mode_logits.float(), dim=-1)
    target = target.float()
    if inflation == "zero":
        zero_mask = target <= eps
        values = torch.where(zero_mask, -mode_log_probs[..., 0], -(mode_log_probs[..., 1] + continuous_log_prob))
    elif inflation == "zero_one":
        zero_mask = target <= eps
        one_mask = target >= 1.0 - eps
        cont_values = -(mode_log_probs[..., 2] + continuous_log_prob)
        values = torch.where(zero_mask, -mode_log_probs[..., 0], cont_values)
        values = torch.where(one_mask, -mode_log_probs[..., 1], values)
    else:
        raise ValueError(f"Unsupported inflation mode: {inflation}")
    return _masked_mean(values, mask)


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
    timing_norm = str(getattr(config, "timing_input_normalization", "legacy_log1p")).lower()
    ioi_ms = ioi_bins.to(dtype=torch.float32).clamp(0.0, float(timing_bins - 1))
    duration_ms = duration_bins.to(dtype=torch.float32).clamp(0.0, float(timing_bins - 1))
    if timing_norm in {"scaled_log_5000_s10", "log1p_x_over_10_5000"}:
        denom = torch.log1p(ioi_ms.new_tensor(500.0))
        ioi_norm = torch.log1p(ioi_ms.clamp(max=5000.0) / 10.0) / denom
        duration_norm = torch.log1p(duration_ms.clamp(max=5000.0) / 10.0) / denom
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


def _materialize_epr_prediction(config, raw_outputs, sampling_strategy="mean"):
    distribution = getattr(config, "epr_distribution", "point").lower()
    pedal_representation = str(getattr(config, "pedal_representation", "continuous_4")).lower()
    if distribution in {"categorical", "hard_categorical", "soft_categorical"}:
        logits = _split_epr_categorical_logits(config, raw_outputs)
        decode = (
            _soft_categorical_sample_or_expected
            if distribution == "soft_categorical"
            else _categorical_sample_or_argmax
        )
        ioi = decode(logits["ioi"], sampling_strategy)
        duration = decode(logits["duration"], sampling_strategy)
        velocity = decode(logits["velocity"], sampling_strategy)
        pedal = decode(logits["pedal"], sampling_strategy)
        return _epr_bins_to_normalized(config, ioi, duration, velocity, pedal).to(device=raw_outputs.device)

    if pedal_representation == "start_ctrl":
        pedal_params = _split_start_ctrl_from_outputs(config, raw_outputs)
        if distribution == "beta_mu_kappa":
            def decode_beta(raw):
                mu, _, alpha, beta = _beta_params(
                    raw[..., 0],
                    raw[..., 1],
                    eps=getattr(config, "beta_eps", 1e-5),
                    kappa_min=getattr(config, "beta_kappa_min", 1e-3),
                )
                mode_name = str(sampling_strategy).lower()
                if mode_name in {"sample", "sampling", "stochastic"}:
                    return torch.distributions.Beta(alpha, beta).sample()
                return mu

            start = decode_beta(pedal_params["start_raw"])
            ctrl = decode_beta(pedal_params["ctrl_raw"])
        elif distribution in {
            "logistic_normal",
            "mixture_logistic_normal",
            "inflated_mixture_logistic_normal",
            "mixture_beta",
        }:
            components = int(getattr(config, "epr_mixture_components", 1))

            def decode_mixture(raw):
                base = raw.reshape(*raw.shape[:-1], 3, components)
                if distribution == "mixture_beta":
                    return _mixture_beta_mean_or_sample(
                        config,
                        base[..., 0, :],
                        base[..., 1, :],
                        base[..., 2, :],
                        sampling_strategy=sampling_strategy,
                    )
                return _mixture_logistic_normal_mean_or_sample(
                    config,
                    base[..., 0, :],
                    base[..., 1, :],
                    base[..., 2, :],
                    sampling_strategy=sampling_strategy,
                )

            start = decode_mixture(pedal_params["start_raw"])
            ctrl = decode_mixture(pedal_params["ctrl_raw"])
        else:
            start = torch.sigmoid(pedal_params["start_raw"].squeeze(-1))
            ctrl = torch.sigmoid(pedal_params["ctrl_raw"].squeeze(-1))

        if distribution not in {
            "logistic_normal",
            "mixture_logistic_normal",
            "inflated_mixture_logistic_normal",
            "mixture_beta",
            "beta_mu_kappa",
        }:
            shared = raw_outputs[..., :3].clamp(0.0, 1.0)
        elif distribution == "beta_mu_kappa":
            params = _split_epr_distribution_params(raw_outputs)
            shared_mu, _, shared_alpha, shared_beta = _beta_params(
                params["shared_mu"],
                params["shared_kappa"],
                eps=getattr(config, "beta_eps", 1e-5),
                kappa_min=getattr(config, "beta_kappa_min", 1e-3),
            )
            mode_name = str(sampling_strategy).lower()
            shared = (
                torch.distributions.Beta(shared_alpha, shared_beta).sample()
                if mode_name in {"sample", "sampling", "stochastic"}
                else shared_mu
            )
        else:
            params = _split_epr_mixture_params(config, raw_outputs)
            decode = _mixture_beta_mean_or_sample if distribution == "mixture_beta" else _mixture_logistic_normal_mean_or_sample
            shared = decode(
                config,
                params["shared_logits"],
                params["shared_a"],
                params["shared_b"],
                sampling_strategy=sampling_strategy,
            )
        return _pack_start_ctrl_prediction(shared, start, ctrl)

    if distribution != "beta_mu_kappa":
        if distribution not in {
            "logistic_normal",
            "mixture_logistic_normal",
            "inflated_mixture_logistic_normal",
            "mixture_beta",
        }:
            return raw_outputs

        params = _split_epr_mixture_params(config, raw_outputs)
        if distribution in {"logistic_normal", "mixture_logistic_normal", "inflated_mixture_logistic_normal"}:
            decode = _mixture_logistic_normal_mean_or_sample
        else:
            decode = _mixture_beta_mean_or_sample

        shared = decode(
            config,
            params["shared_logits"],
            params["shared_a"],
            params["shared_b"],
            sampling_strategy=sampling_strategy,
        )
        pedal = decode(
            config,
            params["pedal_logits"],
            params["pedal_a"],
            params["pedal_b"],
            sampling_strategy=sampling_strategy,
        )

        if distribution == "inflated_mixture_logistic_normal":
            mode = str(sampling_strategy).lower()
            ioi_mode_probs = torch.softmax(params["ioi_mode_logits"].float(), dim=-1)
            pedal_mode_probs = torch.softmax(params["pedal_mode_logits"].float(), dim=-1)
            if mode in {"mean", "deterministic", "mu", "expected", "expectation"}:
                shared_ioi = ioi_mode_probs[..., 1] * shared[..., 0]
                pedal = pedal_mode_probs[..., 1] + pedal_mode_probs[..., 2] * pedal
            elif mode in {"argmax", "greedy"}:
                ioi_mode = ioi_mode_probs.argmax(dim=-1)
                shared_ioi = torch.where(ioi_mode == 0, shared[..., 0].new_zeros(()), shared[..., 0])
                pedal_mode = pedal_mode_probs.argmax(dim=-1)
                pedal = torch.where(pedal_mode == 0, pedal.new_zeros(()), pedal)
                pedal = torch.where(pedal_mode == 1, pedal.new_ones(()), pedal)
            elif mode in {"sample", "sampling", "stochastic"}:
                ioi_mode = torch.distributions.Categorical(probs=ioi_mode_probs).sample()
                shared_ioi = torch.where(ioi_mode == 0, shared[..., 0].new_zeros(()), shared[..., 0])
                pedal_mode = torch.distributions.Categorical(probs=pedal_mode_probs).sample()
                pedal = torch.where(pedal_mode == 0, pedal.new_zeros(()), pedal)
                pedal = torch.where(pedal_mode == 1, pedal.new_ones(()), pedal)
            else:
                raise ValueError(f"Unsupported sampling_strategy: {sampling_strategy}")
            shared = torch.cat([shared_ioi.unsqueeze(-1), shared[..., 1:3]], dim=-1)

        return torch.cat([shared, pedal], dim=-1).clamp_(0.0, 1.0)

    params = _split_epr_distribution_params(raw_outputs)
    shared_mu, _, shared_alpha, shared_beta = _beta_params(
        params["shared_mu"],
        params["shared_kappa"],
        eps=getattr(config, "beta_eps", 1e-5),
        kappa_min=getattr(config, "beta_kappa_min", 1e-3),
    )
    pedal_mu, _, pedal_alpha, pedal_beta = _beta_params(
        params["pedal_mu"],
        params["pedal_kappa"],
        eps=getattr(config, "beta_eps", 1e-5),
        kappa_min=getattr(config, "beta_kappa_min", 1e-3),
    )

    mode = str(sampling_strategy).lower()
    if mode in {"mean", "deterministic", "mu"}:
        shared = shared_mu
        pedal = pedal_mu
    elif mode in {"sample", "sampling", "stochastic"}:
        shared = torch.distributions.Beta(shared_alpha, shared_beta).sample()
        pedal = torch.distributions.Beta(pedal_alpha, pedal_beta).sample()
    else:
        raise ValueError(f"Unsupported sampling_strategy: {sampling_strategy}")

    return torch.cat([shared, pedal], dim=-1).clamp_(0.0, 1.0)


def _shift_continuous_right(continuous, attention_mask):
    shifted = torch.zeros_like(continuous)
    if continuous.shape[1] > 1:
        prev_values = continuous[:, :-1]
        prev_mask = attention_mask[:, :-1].to(dtype=continuous.dtype).unsqueeze(-1)
        shifted[:, 1:] = prev_values * prev_mask
    shifted = shifted * attention_mask.to(dtype=continuous.dtype).unsqueeze(-1)
    return shifted


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
    if dropout_mode in {"none", "off"}:
        return decoder_input_continuous, special_note_ids
    else:
        raise ValueError(f"Unsupported prior_token_dropout_mode: {dropout_mode}")


def _build_ar_special_note_ids(config, attention_mask):
    special_note_ids = attention_mask.new_full(attention_mask.shape, -1)
    if special_note_ids.shape[1] > 0:
        bos_id = int(config.special_note_ids.get("bos", 2))
        special_note_ids[:, 0] = bos_id
    return special_note_ids


def _build_prefilled_ar_note_inputs(config, attention_mask, output_dim, prefix_predictions=None):
    batch_size, seq_len = attention_mask.shape
    decoder_input_continuous = attention_mask.new_zeros((batch_size, seq_len, output_dim + 2), dtype=torch.float32)
    special_note_ids = attention_mask.new_full((batch_size, seq_len), -1)
    if seq_len > 0:
        special_note_ids[:, 0] = int(config.special_note_ids.get("bos", 2))

    prefix_len = 0
    if prefix_predictions is not None:
        prefix_len = int(prefix_predictions.shape[1])
        if prefix_len > 0:
            if config.task_type == "epr":
                decoder_input_continuous[:, 1 : prefix_len + 1, 1] = 1.0
            elif config.task_type == "csr":
                decoder_input_continuous[:, 1 : prefix_len + 1, 0] = 1.0
            decoder_input_continuous[:, 1 : prefix_len + 1, 2:] = prefix_predictions[:, :prefix_len].to(
                dtype=decoder_input_continuous.dtype,
                device=decoder_input_continuous.device,
            )
    return decoder_input_continuous, special_note_ids, prefix_len


def _build_ar_note_continuous(labels_continuous, task_type, input_feature_mode="integrated"):
    if input_feature_mode == "legacy":
        return labels_continuous
    batch_size, seq_len, _ = labels_continuous.shape
    if task_type == "epr":
        type_bits = labels_continuous.new_zeros(batch_size, seq_len, 2)
        type_bits[..., 1] = 1.0
    elif task_type == "csr":
        type_bits = labels_continuous.new_zeros(batch_size, seq_len, 2)
        type_bits[..., 0] = 1.0
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


def _compute_integrated_loss_components(config, continuous_pred, labels_continuous, attention_mask, labels_epr_bins=None):
    if getattr(config, "task_type", "epr") == "csr":
        return _compute_csr_loss_components(config, continuous_pred, labels_continuous, attention_mask)

    mask = attention_mask.bool()
    distribution = getattr(config, "epr_distribution", "point").lower()
    if distribution in {"categorical", "hard_categorical", "soft_categorical"}:
        if labels_epr_bins is None:
            raise ValueError("Categorical EPR loss requires labels_epr_bins")
        return _compute_epr_categorical_loss_components(config, continuous_pred, labels_epr_bins, mask)

    pedal_representation = str(getattr(config, "pedal_representation", "continuous_4")).lower()
    if pedal_representation == "start_ctrl":
        start_ctrl = _split_start_ctrl_from_outputs(config, continuous_pred)
        start_target, ctrl_target = _pedal_start_ctrl_targets(labels_continuous[..., 3:7], mask)

        if distribution in {
            "logistic_normal",
            "mixture_logistic_normal",
            "inflated_mixture_logistic_normal",
            "mixture_beta",
        }:
            params = _split_epr_mixture_params(config, continuous_pred)
            eps = getattr(config, "epr_distribution_eps", getattr(config, "beta_eps", 1e-5))
            sigma_min = getattr(config, "logistic_normal_sigma_min", 1e-3)
            sigma_max = getattr(config, "logistic_normal_sigma_max", 10.0)
            alpha_min = getattr(config, "beta_alpha_min", 1e-4)

            if distribution == "mixture_beta":
                def loss_one(logits, raw_a, raw_b, target):
                    return _mixture_beta_nll(logits, raw_a, raw_b, target, mask, eps, alpha_min)
            else:
                def loss_one(logits, raw_a, raw_b, target):
                    return _mixture_logistic_normal_nll(
                        logits,
                        raw_a,
                        raw_b,
                        target,
                        mask,
                        eps,
                        sigma_min,
                        sigma_max,
                    )

            loss_ioi = loss_one(
                params["shared_logits"][..., 0, :],
                params["shared_a"][..., 0, :],
                params["shared_b"][..., 0, :],
                labels_continuous[..., 0],
            )
            loss_duration = loss_one(
                params["shared_logits"][..., 1, :],
                params["shared_a"][..., 1, :],
                params["shared_b"][..., 1, :],
                labels_continuous[..., 1],
            )
            loss_velocity = loss_one(
                params["shared_logits"][..., 2, :],
                params["shared_a"][..., 2, :],
                params["shared_b"][..., 2, :],
                labels_continuous[..., 2],
            )
            components = int(getattr(config, "epr_mixture_components", 1))
            start_base = start_ctrl["start_raw"].reshape(*start_ctrl["start_raw"].shape[:-1], 3, components)
            ctrl_base = start_ctrl["ctrl_raw"].reshape(*start_ctrl["ctrl_raw"].shape[:-1], 3, components)
            loss_pedal_start = loss_one(
                start_base[..., 0, :],
                start_base[..., 1, :],
                start_base[..., 2, :],
                start_target,
            )
            loss_pedal_ctrl = loss_one(
                ctrl_base[..., 0, :],
                ctrl_base[..., 1, :],
                ctrl_base[..., 2, :],
                ctrl_target,
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
            loss_pedal_start = _beta_nll_loss(
                start_ctrl["start_raw"][..., 0],
                start_ctrl["start_raw"][..., 1],
                start_target,
                mask,
                eps,
                kappa_min,
            )
            loss_pedal_ctrl = _beta_nll_loss(
                start_ctrl["ctrl_raw"][..., 0],
                start_ctrl["ctrl_raw"][..., 1],
                ctrl_target,
                mask,
                eps,
                kappa_min,
            )
        else:
            loss_ioi = _regression_loss(
                continuous_pred[..., 0],
                labels_continuous[..., 0],
                mask,
                config.time_loss_type,
                config.huber_delta,
            )
            loss_duration = _regression_loss(
                continuous_pred[..., 1],
                labels_continuous[..., 1],
                mask,
                config.time_loss_type,
                config.huber_delta,
            )
            loss_velocity = _regression_loss(
                continuous_pred[..., 2],
                labels_continuous[..., 2],
                mask,
                config.value_loss_type,
                config.huber_delta,
            )
            loss_pedal_start = _regression_loss(
                torch.sigmoid(start_ctrl["start_raw"].squeeze(-1)),
                start_target,
                mask,
                config.value_loss_type,
                config.huber_delta,
            )
            loss_pedal_ctrl = _regression_loss(
                torch.sigmoid(start_ctrl["ctrl_raw"].squeeze(-1)),
                ctrl_target,
                mask,
                config.value_loss_type,
                config.huber_delta,
            )
        start_weight = float(getattr(config, "pedal_start_loss_weight", 1.0))
        ctrl_weight = float(getattr(config, "pedal_ctrl_loss_weight", 1.0))
        loss_pedal = (start_weight * loss_pedal_start + ctrl_weight * loss_pedal_ctrl) / max(
            start_weight + ctrl_weight,
            1e-12,
        )
        return {
            "ioi": loss_ioi,
            "duration": loss_duration,
            "velocity": loss_velocity,
            "pedal": loss_pedal,
        }

    if distribution in {
        "logistic_normal",
        "mixture_logistic_normal",
        "inflated_mixture_logistic_normal",
        "mixture_beta",
    }:
        params = _split_epr_mixture_params(config, continuous_pred)
        eps = getattr(config, "epr_distribution_eps", getattr(config, "beta_eps", 1e-5))
        sigma_min = getattr(config, "logistic_normal_sigma_min", 1e-3)
        sigma_max = getattr(config, "logistic_normal_sigma_max", 10.0)
        alpha_min = getattr(config, "beta_alpha_min", 1e-4)

        if distribution == "mixture_beta":
            def loss_one(logits, raw_a, raw_b, target):
                return _mixture_beta_nll(logits, raw_a, raw_b, target, mask, eps, alpha_min)
        else:
            def loss_one(logits, raw_a, raw_b, target):
                return _mixture_logistic_normal_nll(
                    logits,
                    raw_a,
                    raw_b,
                    target,
                    mask,
                    eps,
                    sigma_min,
                    sigma_max,
                )

        if distribution == "inflated_mixture_logistic_normal":
            loss_ioi = _inflated_logistic_normal_nll(
                config,
                params["shared_logits"][..., 0, :],
                params["shared_a"][..., 0, :],
                params["shared_b"][..., 0, :],
                params["ioi_mode_logits"],
                labels_continuous[..., 0],
                mask,
                "zero",
            )
        else:
            loss_ioi = loss_one(
                params["shared_logits"][..., 0, :],
                params["shared_a"][..., 0, :],
                params["shared_b"][..., 0, :],
                labels_continuous[..., 0],
            )

        loss_duration = loss_one(
            params["shared_logits"][..., 1, :],
            params["shared_a"][..., 1, :],
            params["shared_b"][..., 1, :],
            labels_continuous[..., 1],
        )
        loss_velocity = loss_one(
            params["shared_logits"][..., 2, :],
            params["shared_a"][..., 2, :],
            params["shared_b"][..., 2, :],
            labels_continuous[..., 2],
        )
        pedal_losses = []
        for idx in range(4):
            if distribution == "inflated_mixture_logistic_normal":
                pedal_losses.append(
                    _inflated_logistic_normal_nll(
                        config,
                        params["pedal_logits"][..., idx, :],
                        params["pedal_a"][..., idx, :],
                        params["pedal_b"][..., idx, :],
                        params["pedal_mode_logits"][..., idx, :],
                        labels_continuous[..., 3 + idx],
                        mask,
                        "zero_one",
                    )
                )
            else:
                pedal_losses.append(
                    loss_one(
                        params["pedal_logits"][..., idx, :],
                        params["pedal_a"][..., idx, :],
                        params["pedal_b"][..., idx, :],
                        labels_continuous[..., 3 + idx],
                    )
                )
        return {
            "ioi": loss_ioi,
            "duration": loss_duration,
            "velocity": loss_velocity,
            "pedal": torch.stack(pedal_losses).mean(),
        }

    if getattr(config, "epr_distribution", "point").lower() == "beta_mu_kappa":
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
        pedal_losses = []
        for idx in range(4):
            pedal_losses.append(
                _beta_nll_loss(
                    params["pedal_mu"][..., idx],
                    params["pedal_kappa"][..., idx],
                    labels_continuous[..., 3 + idx],
                    mask,
                    eps,
                    kappa_min,
                )
            )
        loss_pedal = torch.stack(pedal_losses).mean()
        return {
            "ioi": loss_ioi,
            "duration": loss_duration,
            "velocity": loss_velocity,
            "pedal": loss_pedal,
        }

    loss_ioi = _regression_loss(
        continuous_pred[..., 0],
        labels_continuous[..., 0],
        mask,
        config.time_loss_type,
        config.huber_delta,
    )
    loss_duration = _regression_loss(
        continuous_pred[..., 1],
        labels_continuous[..., 1],
        mask,
        config.time_loss_type,
        config.huber_delta,
    )
    loss_velocity = _regression_loss(
        continuous_pred[..., 2],
        labels_continuous[..., 2],
        mask,
        config.value_loss_type,
        config.huber_delta,
    )
    loss_pedal = _regression_loss(
        continuous_pred[..., 3:7],
        labels_continuous[..., 3:7],
        mask.unsqueeze(-1).expand_as(continuous_pred[..., 3:7]),
        config.value_loss_type,
        config.huber_delta,
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


def _compute_csr_loss_components(config, score_feature_logits, labels_score_feature, score_feature_mask):
    score_mask = score_feature_mask.bool()
    first_target = labels_score_feature[..., 3].float()
    ml_mask = score_mask & (first_target >= 0.5)
    grid_loss_type = getattr(config, "csr_grid_loss_type", "huber")

    return {
        "mo": _regression_loss(
            torch.sigmoid(score_feature_logits[..., 0]),
            labels_score_feature[..., 0],
            score_mask,
            grid_loss_type,
            config.huber_delta,
        ),
        "md": _regression_loss(
            torch.sigmoid(score_feature_logits[..., 1]),
            labels_score_feature[..., 1],
            score_mask,
            grid_loss_type,
            config.huber_delta,
        ),
        "first": _bce_loss(score_feature_logits[..., 3], labels_score_feature[..., 3], score_mask),
        "ml": _regression_loss(
            torch.sigmoid(score_feature_logits[..., 2]),
            labels_score_feature[..., 2],
            ml_mask,
            grid_loss_type,
            config.huber_delta,
        ),
        "staff": _bce_loss(score_feature_logits[..., 4], labels_score_feature[..., 4], score_mask),
        "trill": _bce_loss(score_feature_logits[..., 5], labels_score_feature[..., 5], score_mask),
        "grace": _bce_loss(score_feature_logits[..., 6], labels_score_feature[..., 6], score_mask),
        "staccato": _bce_loss(score_feature_logits[..., 7], labels_score_feature[..., 7], score_mask),
    }


def _compute_integrated_loss(config, continuous_pred, labels_continuous, attention_mask, labels_epr_bins=None):
    components = _compute_integrated_loss_components(
        config,
        continuous_pred,
        labels_continuous,
        attention_mask,
        labels_epr_bins=labels_epr_bins,
    )
    if getattr(config, "task_type", "epr") == "csr":
        weights = config.csr_loss_weights
        return sum(weights.get(name, 1.0) * value for name, value in components.items())

    weights = config.loss_weights
    return (
        weights.get("ioi", 1.0) * components["ioi"]
        + weights.get("duration", 1.0) * components["duration"]
        + weights.get("velocity", 1.0) * components["velocity"]
        + weights.get("pedal", 1.0) * components["pedal"]
    )


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


class IntegratedPianoTransformer(nn.Module):
    def __init__(self, config: IntegratedPianoT5GemmaConfig):
        super().__init__()
        self.config = config
        self.note_encoder = IntegratedNoteEncoder(config, role="score")
        embedding_mode = getattr(config, "note_embedding_mode", "fine").lower()
        if embedding_mode in {"legacy", "score_perf", "score_perf_split"}:
            self._decoder_note_encoder = IntegratedNoteEncoder(
                config,
                continuous_dim=config.output_continuous_dim + 2,
                role="perf",
            )
        else:
            self._decoder_note_encoder = None
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
        return self._decoder_note_encoder if self._decoder_note_encoder is not None else self.note_encoder

    def forward(
        self,
        pitch_ids: Optional[torch.LongTensor] = None,
        continuous: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        labels_continuous: Optional[torch.FloatTensor] = None,
        labels_epr_bins: Optional[torch.LongTensor] = None,
        interpolated: Optional[torch.BoolTensor] = None,
        continuous_sampling_strategy: str = "mean",
        **kwargs,
    ) -> Seq2SeqLMOutput:
        del interpolated, kwargs
        if pitch_ids is None or continuous is None:
            raise ValueError("pitch_ids and continuous are required")
        if attention_mask is None:
            attention_mask = (pitch_ids != self.config.pitch_pad_id).long()

        score_note_embeds = self.note_encoder(pitch_ids, continuous)
        decoder_mode = self.config.decoder_input_mode.lower()
        if decoder_mode == "score":
            hidden_states = self.backbone(score_note_embeds, attention_mask)
            continuous_pred = self.continuous_decoder(hidden_states)
        elif decoder_mode == "ar":
            if self.backbone_type != "gpt":
                raise ValueError("decoder_input_mode='ar' is supported for gpt in IntegratedPianoTransformer")
            if labels_continuous is not None:
                decoder_target_continuous = _build_ar_note_continuous(
                    labels_continuous,
                    self.config.task_type,
                    getattr(self.config, "input_feature_mode", "integrated"),
                )
                decoder_input_continuous = _shift_continuous_right(decoder_target_continuous, attention_mask)
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
                )
                hidden_states = self.backbone(
                    score_note_embeds,
                    attention_mask,
                    performance_embeds=performance_embeds,
                    performance_attention_mask=attention_mask,
                )
                continuous_pred = self.continuous_decoder(hidden_states)
            else:
                continuous_pred = self._autoregressive_rollout_gpt(
                    pitch_ids=pitch_ids,
                    continuous=continuous,
                    attention_mask=attention_mask,
                    sampling_strategy=continuous_sampling_strategy,
                )
        else:
            raise ValueError(f"Unsupported decoder_input_mode: {self.config.decoder_input_mode}")

        if labels_continuous is None and self.config.task_type == "epr":
            continuous_pred = _materialize_epr_prediction(
                self.config,
                continuous_pred,
                sampling_strategy=continuous_sampling_strategy,
            )
            if str(getattr(self.config, "pedal_representation", "continuous_4")).lower() == "start_ctrl":
                continuous_pred = materialize_start_ctrl_sequence(continuous_pred, attention_mask)

        loss = None
        if labels_continuous is not None:
            loss = _compute_integrated_loss(
                self.config,
                continuous_pred,
                labels_continuous,
                attention_mask,
                labels_epr_bins=labels_epr_bins,
            )

        return Seq2SeqLMOutput(loss=loss, logits=continuous_pred)

    def predict_performance_continuous(
        self,
        pitch_ids,
        continuous,
        attention_mask=None,
        prefix_predictions=None,
        sampling_strategy="mean",
    ):
        if attention_mask is None:
            attention_mask = (pitch_ids != self.config.pitch_pad_id).long()
        decoder_mode = self.config.decoder_input_mode.lower()
        if decoder_mode != "ar":
            outputs = self(
                pitch_ids=pitch_ids,
                continuous=continuous,
                attention_mask=attention_mask,
                continuous_sampling_strategy=sampling_strategy,
            )
            return outputs.logits
        if self.backbone_type != "gpt":
            raise ValueError("Prefix continuation is only implemented for AR GPT in IntegratedPianoTransformer")
        return self._autoregressive_rollout_gpt(
            pitch_ids=pitch_ids,
            continuous=continuous,
            attention_mask=attention_mask,
            sampling_strategy=sampling_strategy,
            prefix_predictions=prefix_predictions,
        )

    def _autoregressive_rollout_gpt(self, pitch_ids, continuous, attention_mask, sampling_strategy="mean", prefix_predictions=None):
        batch_size, seq_len = pitch_ids.shape
        score_note_embeds = self.note_encoder(pitch_ids, continuous)
        decoder_input_continuous, special_note_ids, prefix_len = _build_prefilled_ar_note_inputs(
            self.config,
            attention_mask,
            self.config.output_continuous_dim,
            prefix_predictions=canonicalize_start_ctrl_sequence(prefix_predictions)
            if (
                prefix_predictions is not None
                and self.config.task_type == "epr"
                and str(getattr(self.config, "pedal_representation", "continuous_4")).lower() == "start_ctrl"
            )
            else prefix_predictions,
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
            perf_prefix_mask = attention_mask[:, : step + 1]
            hidden_states = self.backbone(
                score_note_embeds,
                attention_mask,
                performance_embeds=perf_prefix_embeds,
                performance_attention_mask=perf_prefix_mask,
            )
            step_raw = self.continuous_decoder(hidden_states[:, -1:, :])
            step_pred = _materialize_epr_prediction(self.config, step_raw, sampling_strategy=sampling_strategy)
            predictions.append(step_pred)
            if step + 1 < seq_len:
                if self.config.task_type == "epr":
                    decoder_input_continuous[:, step + 1, 1] = 1.0
                elif self.config.task_type == "csr":
                    decoder_input_continuous[:, step + 1, 0] = 1.0
                decoder_input_continuous[:, step + 1, 2:] = step_pred[:, 0, :]

        output = torch.cat(predictions, dim=1) if predictions else torch.zeros_like(continuous)
        if str(getattr(self.config, "pedal_representation", "continuous_4")).lower() == "start_ctrl":
            output = materialize_start_ctrl_sequence(output, attention_mask)
        return output


class IntegratedPianoT5Gemma(T5GemmaPreTrainedModel, GenerationMixin):
    config_class = IntegratedPianoT5GemmaConfig
    _tp_plan = {
        "continuous_decoder.shared_head": "colwise_rep",
        "continuous_decoder.pedal_head": "colwise_rep",
    }
    _pp_plan = {"continuous_decoder": (["hidden_states"], ["continuous_pred"])}

    def __init__(self, config: IntegratedPianoT5GemmaConfig):
        config.is_encoder_decoder = True
        super().__init__(config)
        self.note_encoder = IntegratedNoteEncoder(config, role="score")
        embedding_mode = getattr(config, "note_embedding_mode", "fine").lower()
        if embedding_mode in {"legacy", "score_perf", "score_perf_split"}:
            self._decoder_note_encoder = IntegratedNoteEncoder(
                config,
                continuous_dim=config.output_continuous_dim + 2,
                role="perf",
            )
        else:
            self._decoder_note_encoder = None
        self.model = IntegratedPianoT5GemmaModel(config)
        self.continuous_decoder = IntegratedContinuousDecoder(config)
        self.post_init()

    @property
    def decoder_note_encoder(self):
        return self._decoder_note_encoder if self._decoder_note_encoder is not None else self.note_encoder

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
        score_note_embeds,
        attention_mask,
        labels_continuous=None,
    ):
        decoder_mode = self.config.decoder_input_mode.lower()
        if decoder_mode == "score":
            return score_note_embeds, attention_mask
        if decoder_mode == "ar":
            if labels_continuous is None:
                return None, None
            decoder_target_continuous = _build_ar_note_continuous(
                labels_continuous,
                self.config.task_type,
                getattr(self.config, "input_feature_mode", "integrated"),
            )
            decoder_input_continuous = _shift_continuous_right(decoder_target_continuous, attention_mask)
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
            )
            return decoder_inputs_embeds, attention_mask
        raise ValueError(f"Unsupported decoder_input_mode: {self.config.decoder_input_mode}")

    def forward(
        self,
        pitch_ids: Optional[torch.LongTensor] = None,
        continuous: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        labels_continuous: Optional[torch.FloatTensor] = None,
        labels_epr_bins: Optional[torch.LongTensor] = None,
        interpolated: Optional[torch.BoolTensor] = None,
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
        del interpolated

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

        if attention_mask is None:
            attention_mask = (pitch_ids != self.config.pitch_pad_id).long()

        score_note_embeds = self.note_encoder(pitch_ids, continuous)
        decoder_inputs_embeds, decoder_input_mask = self._build_decoder_inputs(
            pitch_ids,
            score_note_embeds,
            attention_mask,
            labels_continuous=labels_continuous,
        )
        if decoder_inputs_embeds is None:
            continuous_pred = self._autoregressive_rollout(
                pitch_ids=pitch_ids,
                continuous=continuous,
                attention_mask=attention_mask,
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
                loss = _compute_integrated_loss(
                    self.config,
                    continuous_pred,
                    labels_continuous,
                    attention_mask,
                    labels_epr_bins=labels_epr_bins,
                )
            return Seq2SeqLMOutput(loss=loss, logits=continuous_pred)
        if decoder_attention_mask is None:
            decoder_attention_mask = decoder_input_mask

        decoder_outputs = self.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            decoder_attention_mask=decoder_attention_mask,
            decoder_position_ids=decoder_position_ids,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            inputs_embeds=score_note_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        continuous_pred = self.continuous_decoder(decoder_outputs.last_hidden_state)
        if labels_continuous is None and self.config.task_type == "epr":
            continuous_pred = _materialize_epr_prediction(
                self.config,
                continuous_pred,
                sampling_strategy=continuous_sampling_strategy,
            )
            if str(getattr(self.config, "pedal_representation", "continuous_4")).lower() == "start_ctrl":
                continuous_pred = materialize_start_ctrl_sequence(continuous_pred, attention_mask)

        loss = None
        if labels_continuous is not None:
            loss = _compute_integrated_loss(
                self.config,
                continuous_pred,
                labels_continuous,
                attention_mask,
                labels_epr_bins=labels_epr_bins,
            )

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
        attention_mask=None,
        prefix_predictions=None,
        sampling_strategy="mean",
    ):
        if attention_mask is None:
            attention_mask = (pitch_ids != self.config.pitch_pad_id).long()
        decoder_mode = self.config.decoder_input_mode.lower()
        if decoder_mode == "ar":
            return self._autoregressive_rollout(
                pitch_ids=pitch_ids,
                continuous=continuous,
                attention_mask=attention_mask,
                sampling_strategy=sampling_strategy,
                prefix_predictions=prefix_predictions,
            )
        outputs = self(
            pitch_ids=pitch_ids,
            continuous=continuous,
            attention_mask=attention_mask,
            continuous_sampling_strategy=sampling_strategy,
        )
        return outputs.logits

    def _autoregressive_rollout(
        self,
        pitch_ids,
        continuous,
        attention_mask,
        sampling_strategy="mean",
        prefix_predictions=None,
        position_ids=None,
        encoder_outputs=None,
        past_key_values=None,
        use_cache=None,
        cache_position=None,
        **kwargs,
    ):
        del kwargs
        batch_size, seq_len = pitch_ids.shape
        score_note_embeds = self.note_encoder(pitch_ids, continuous)
        if encoder_outputs is None:
            encoder_outputs = self.model.encoder(
                attention_mask=attention_mask,
                position_ids=position_ids,
                inputs_embeds=score_note_embeds,
            )

        decoder_input_continuous, special_note_ids, prefix_len = _build_prefilled_ar_note_inputs(
            self.config,
            attention_mask,
            self.config.output_continuous_dim,
            prefix_predictions=canonicalize_start_ctrl_sequence(prefix_predictions)
            if (
                prefix_predictions is not None
                and self.config.task_type == "epr"
                and str(getattr(self.config, "pedal_representation", "continuous_4")).lower() == "start_ctrl"
            )
            else prefix_predictions,
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
            decoder_attention_mask = attention_mask[:, :prime_len]
            current_decoder_outputs = self.model(
                attention_mask=attention_mask,
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
                decoder_attention_mask = attention_mask[:, :1]
                decoder_outputs = self.model(
                    attention_mask=attention_mask,
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
                # For incremental decoding, we need the full decoder attention mask
                # but only the new token's embeddings
                decoder_attention_mask = attention_mask[:, :step_idx+1]
                cache_position_tensor = torch.tensor(
                    [step_idx], device=pitch_ids.device, dtype=torch.long
                )
                decoder_outputs = self.model(
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    decoder_attention_mask=decoder_attention_mask,
                    encoder_outputs=encoder_outputs,
                    decoder_inputs_embeds=decoder_inputs_embeds,
                    use_cache=True,
                    past_key_values=cached_past_key_values,
                    cache_position=cache_position_tensor,
                )
                cached_past_key_values = decoder_outputs.past_key_values

            step_raw = self.continuous_decoder(decoder_outputs.last_hidden_state[:, -1:, :])
            step_pred = _materialize_epr_prediction(self.config, step_raw, sampling_strategy=sampling_strategy)
            predictions.append(step_pred)

            if step + 1 < seq_len:
                if self.config.task_type == "epr":
                    decoder_input_continuous[:, step + 1, 1] = 1.0
                elif self.config.task_type == "csr":
                    decoder_input_continuous[:, step + 1, 0] = 1.0
                decoder_input_continuous[:, step + 1, 2:] = step_pred[:, 0, :]

        output = torch.cat(predictions, dim=1) if predictions else torch.zeros_like(continuous)
        if str(getattr(self.config, "pedal_representation", "continuous_4")).lower() == "start_ctrl":
            output = materialize_start_ctrl_sequence(output, attention_mask)
        return output
