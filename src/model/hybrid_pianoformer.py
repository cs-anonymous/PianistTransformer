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


class HybridPianoT5GemmaConfig(PianoT5GemmaConfig):
    def __init__(
        self,
        continuous_dim=7,
        pitch_vocab_size=128,
        pitch_pad_id=128,
        max_time_ms=10000.0,
        time_loss_type="huber",
        value_loss_type="mse",
        huber_delta=0.05,
        loss_weights=None,
        decoder_input_mode="score",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.continuous_dim = continuous_dim
        self.pitch_vocab_size = pitch_vocab_size
        self.pitch_pad_id = pitch_pad_id
        self.max_time_ms = max_time_ms
        self.time_loss_type = time_loss_type
        self.value_loss_type = value_loss_type
        self.huber_delta = huber_delta
        self.loss_weights = loss_weights or {
            "ioi": 1.0,
            "duration": 1.0,
            "velocity": 1.0,
            "pedal": 1.0,
        }
        self.decoder_input_mode = decoder_input_mode


class HybridNoteEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.pitch_embedding = nn.Embedding(
            config.pitch_vocab_size + 1,
            config.hidden_size,
            padding_idx=config.pitch_pad_id,
        )
        self.continuous_mlp = nn.Sequential(
            nn.Linear(config.continuous_dim, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size),
        )
        self.norm = nn.LayerNorm(config.hidden_size)

    def forward(self, pitch_ids, continuous):
        pitch_embeds = self.pitch_embedding(pitch_ids)
        continuous = continuous.to(dtype=self.continuous_mlp[0].weight.dtype)
        continuous_embeds = self.continuous_mlp(continuous)
        return self.norm(pitch_embeds + continuous_embeds)


class HybridContinuousDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.continuous_dim),
        )

    def forward(self, hidden_states):
        return torch.sigmoid(self.head(hidden_states))


class HybridT5GemmaEncoder(T5GemmaPreTrainedModel):
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
            raise ValueError("HybridT5GemmaEncoder expects note-level inputs_embeds")

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


class HybridPianoT5GemmaModel(T5GemmaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        if not config.is_encoder_decoder:
            raise ValueError("HybridPianoT5GemmaModel requires encoder-decoder config.")
        self.encoder = HybridT5GemmaEncoder(config.encoder)
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


class HybridPianoT5Gemma(T5GemmaPreTrainedModel, GenerationMixin):
    config_class = HybridPianoT5GemmaConfig
    _tp_plan = {"continuous_decoder.head.2": "colwise_rep"}
    _pp_plan = {"continuous_decoder.head": (["hidden_states"], ["continuous_pred"])}

    def __init__(self, config: HybridPianoT5GemmaConfig):
        config.is_encoder_decoder = True
        super().__init__(config)
        self.note_encoder = HybridNoteEncoder(config)
        self.model = HybridPianoT5GemmaModel(config)
        self.continuous_decoder = HybridContinuousDecoder(config)
        self.post_init()

    def get_encoder(self):
        return self.model.encoder

    def get_decoder(self):
        return self.model.decoder

    def load_pianoformer_backbone(self, pretrained_model_path, torch_dtype=None):
        source_model = PianoT5Gemma.from_pretrained(pretrained_model_path, torch_dtype=torch_dtype)
        incompatible = self.load_state_dict(source_model.state_dict(), strict=False)
        return incompatible

    def _build_decoder_inputs(self, score_note_embeds, attention_mask):
        if self.config.decoder_input_mode != "score":
            raise ValueError(f"Unsupported decoder_input_mode: {self.config.decoder_input_mode}")
        return score_note_embeds, attention_mask

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
        decoder_inputs_embeds, decoder_input_mask = self._build_decoder_inputs(score_note_embeds, attention_mask)
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
            mask = attention_mask.bool()
            weights = self.config.loss_weights
            loss_ioi = _regression_loss(
                continuous_pred[..., 0],
                labels_continuous[..., 0],
                mask,
                self.config.time_loss_type,
                self.config.huber_delta,
            )
            loss_duration = _regression_loss(
                continuous_pred[..., 1],
                labels_continuous[..., 1],
                mask,
                self.config.time_loss_type,
                self.config.huber_delta,
            )
            loss_velocity = _regression_loss(
                continuous_pred[..., 2],
                labels_continuous[..., 2],
                mask,
                self.config.value_loss_type,
                self.config.huber_delta,
            )
            loss_pedal = _regression_loss(
                continuous_pred[..., 3:7],
                labels_continuous[..., 3:7],
                mask.unsqueeze(-1).expand_as(continuous_pred[..., 3:7]),
                self.config.value_loss_type,
                self.config.huber_delta,
            )
            loss = (
                weights.get("ioi", 1.0) * loss_ioi
                + weights.get("duration", 1.0) * loss_duration
                + weights.get("velocity", 1.0) * loss_velocity
                + weights.get("pedal", 1.0) * loss_pedal
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
