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
    def __init__(self, config, continuous_dim=None):
        super().__init__()
        self.config = config
        continuous_dim = continuous_dim or config.input_continuous_dim
        self.continuous_dim = continuous_dim
        self.mode = getattr(config, "note_embedding_mode", "fine").lower()
        self.special_note_embeddings = nn.Embedding(
            config.special_note_vocab_size,
            config.hidden_size,
        )
        self.embedding_depth = getattr(config, "embedding_depth", 2)
        self.activation = getattr(config, "head_activation", "gelu")

        if self.mode == "pine":
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
        head_depth = getattr(config, "head_depth", 2)
        activation = getattr(config, "head_activation", "gelu")

        full_dim = config.hidden_size
        if self.mode == "pine" and self.head_input_mode == "partitioned":
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

        self.shared_head = _make_mlp(shared_dim, 3, shared_dim, head_depth, activation)
        self.pedal_head = _make_mlp(perf_dim, 4, perf_dim, head_depth, activation)
        self.generic_head = _make_mlp(score_dim, self.output_dim, score_dim, head_depth, activation)

    def forward(self, hidden_states):
        if self.output_dim != 7:
            return self.generic_head(hidden_states[..., self.score_slice])

        shared = self.shared_head(hidden_states[..., self.shared_slice])
        pedal = self.pedal_head(hidden_states[..., self.perf_slice])
        shared = torch.sigmoid(shared)
        if self.config.pedal_output_activation == "sigmoid":
            pedal = torch.sigmoid(pedal)
        elif self.config.pedal_output_activation != "linear":
            raise ValueError(f"Unsupported pedal_output_activation: {self.config.pedal_output_activation}")
        return torch.cat([shared, pedal], dim=-1)


def _shift_continuous_right(continuous, attention_mask):
    shifted = torch.zeros_like(continuous)
    if continuous.shape[1] > 1:
        prev_values = continuous[:, :-1]
        prev_mask = attention_mask[:, :-1].to(dtype=continuous.dtype).unsqueeze(-1)
        shifted[:, 1:] = prev_values * prev_mask
    shifted = shifted * attention_mask.to(dtype=continuous.dtype).unsqueeze(-1)
    return shifted


def _build_ar_special_note_ids(config, attention_mask):
    special_note_ids = attention_mask.new_full(attention_mask.shape, -1)
    if special_note_ids.shape[1] > 0:
        bos_id = int(config.special_note_ids.get("bos", 2))
        special_note_ids[:, 0] = bos_id
    return special_note_ids


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


def _compute_integrated_loss_components(config, continuous_pred, labels_continuous, attention_mask):
    if getattr(config, "task_type", "epr") == "csr":
        return _compute_csr_loss_components(config, continuous_pred, labels_continuous, attention_mask)

    mask = attention_mask.bool()
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


def _compute_integrated_loss(config, continuous_pred, labels_continuous, attention_mask):
    components = _compute_integrated_loss_components(
        config,
        continuous_pred,
        labels_continuous,
        attention_mask,
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
        self.note_encoder = IntegratedNoteEncoder(config)
        if getattr(config, "note_embedding_mode", "fine").lower() == "legacy":
            self._decoder_note_encoder = IntegratedNoteEncoder(config, continuous_dim=config.output_continuous_dim + 2)
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
        interpolated: Optional[torch.BoolTensor] = None,
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
                performance_embeds = self.decoder_note_encoder(
                    pitch_ids,
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
                )
        else:
            raise ValueError(f"Unsupported decoder_input_mode: {self.config.decoder_input_mode}")

        loss = None
        if labels_continuous is not None:
            loss = _compute_integrated_loss(self.config, continuous_pred, labels_continuous, attention_mask)

        return Seq2SeqLMOutput(loss=loss, logits=continuous_pred)

    def _autoregressive_rollout_gpt(self, pitch_ids, continuous, attention_mask):
        batch_size, seq_len = pitch_ids.shape
        score_note_embeds = self.note_encoder(pitch_ids, continuous)
        decoder_input_continuous = continuous.new_zeros(batch_size, seq_len, self.config.output_continuous_dim + 2)
        special_note_ids = attention_mask.new_full((batch_size, seq_len), -1)
        if seq_len > 0:
            special_note_ids[:, 0] = int(self.config.special_note_ids.get("bos", 2))
        predictions = []

        for step in range(seq_len):
            perf_prefix_embeds = self.decoder_note_encoder(
                pitch_ids[:, : step + 1],
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
            step_pred = self.continuous_decoder(hidden_states[:, -1:, :])
            predictions.append(step_pred)
            if step + 1 < seq_len:
                if self.config.task_type == "epr":
                    decoder_input_continuous[:, step + 1, 1] = 1.0
                elif self.config.task_type == "csr":
                    decoder_input_continuous[:, step + 1, 0] = 1.0
                decoder_input_continuous[:, step + 1, 2:] = step_pred[:, 0, :]

        return torch.cat(predictions, dim=1) if predictions else torch.zeros_like(continuous)


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
        self.note_encoder = IntegratedNoteEncoder(config)
        if getattr(config, "note_embedding_mode", "fine").lower() == "legacy":
            self._decoder_note_encoder = IntegratedNoteEncoder(config, continuous_dim=config.output_continuous_dim + 2)
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
            decoder_inputs_embeds = self.decoder_note_encoder(
                pitch_ids,
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
        interpolated: Optional[torch.BoolTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        decoder_position_ids: Optional[torch.LongTensor] = None,
        encoder_outputs: Optional[BaseModelOutput] = None,
        past_key_values: Optional[EncoderDecoderCache] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
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
                position_ids=position_ids,
                encoder_outputs=encoder_outputs,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )
            loss = None
            if labels_continuous is not None:
                loss = _compute_integrated_loss(self.config, continuous_pred, labels_continuous, attention_mask)
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

        loss = None
        if labels_continuous is not None:
            loss = _compute_integrated_loss(self.config, continuous_pred, labels_continuous, attention_mask)

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

    def _autoregressive_rollout(
        self,
        pitch_ids,
        continuous,
        attention_mask,
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

        decoder_input_continuous = continuous.new_zeros(
            continuous.shape[0],
            continuous.shape[1],
            self.config.output_continuous_dim + 2,
        )
        special_note_ids = attention_mask.new_full((batch_size, seq_len), -1)
        if seq_len > 0:
            special_note_ids[:, 0] = int(self.config.special_note_ids.get("bos", 2))
        predictions = []

        # Use KV cache for efficient autoregressive decoding
        # Step 0: process first token with full prefix
        # Steps 1+: process only the new token, reuse cached KV

        cached_past_key_values = past_key_values

        for step in range(seq_len):
            if step == 0:
                # First step: process prefix of length 1
                decoder_inputs_embeds = self.decoder_note_encoder(
                    pitch_ids[:, :1],
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
            else:
                # Subsequent steps: process only the new token
                step_idx = step
                decoder_inputs_embeds = self.decoder_note_encoder(
                    pitch_ids[:, step_idx:step_idx+1],
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

            step_pred = self.continuous_decoder(decoder_outputs.last_hidden_state[:, -1:, :])
            predictions.append(step_pred)

            if step + 1 < seq_len:
                if self.config.task_type == "epr":
                    decoder_input_continuous[:, step + 1, 1] = 1.0
                elif self.config.task_type == "csr":
                    decoder_input_continuous[:, step + 1, 0] = 1.0
                decoder_input_continuous[:, step + 1, 2:] = step_pred[:, 0, :]

        return torch.cat(predictions, dim=1) if predictions else torch.zeros_like(continuous)
