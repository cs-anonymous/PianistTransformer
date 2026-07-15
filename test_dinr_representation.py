import torch

from src.model.integrated_pianoformer import (
    IntegratedContinuousDecoder,
    IntegratedNoteEncoder,
    IntegratedPianoTransformer,
    IntegratedPianoT5GemmaConfig,
    _compute_integrated_loss_components,
    _build_epr_decoder_rows,
    _apply_tf_embedding_mask,
    _dinr_support_mask,
    _share_dinr_value_tables,
    _split_dinr_logits,
)


def _config(**overrides):
    values = dict(
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        task_type="epr",
        epr_distribution="dinr",
        epr_timing_target="floor_log_deviation",
        timing_control_mode="dinr_floor_log",
        note_embedding_mode="slot_attribute",
        slot_version="slot5",
        slot_dim=32,
        slot_fusion="mlp",
        musical_feature_mode="none",
        pedal_representation="binary_4",
        input_continuous_dim=12,
        score_input_continuous_dim=12,
        decoder_input_continuous_dim=12,
        output_continuous_dim=7,
        pitch_vocab_size=129,
        pitch_pad_id=128,
        decoder_head_layout="mlp",
        backbone_type="bert",
        bert_layers_num=1,
    )
    values.update(overrides)
    return IntegratedPianoT5GemmaConfig(**values)


def test_dinr_dynamic_ioi_support_uses_one_head():
    config = _config()
    decoder = IntegratedContinuousDecoder(config)
    hidden = torch.randn(1, 2, config.hidden_size)
    score_raw = torch.tensor([[[0.0, 100.0, 64.0], [100.0, 100.0, 64.0]]])
    logits = _split_dinr_logits(config, decoder(hidden, score_shared_raw=score_raw))["ioi"]
    masked = _dinr_support_mask(config, logits, "ioi", score_raw)
    coordinates = (
        torch.arange(config.dinr_timing_bins) - config.dinr_zero_bin
    ) * config.dinr_timing_step
    zero_values = coordinates[torch.isfinite(masked[0, 0])]
    nonzero_values = coordinates[torch.isfinite(masked[0, 1])]
    assert zero_values.min() >= 0.0
    assert zero_values.max() <= 5.0
    assert nonzero_values.min() >= -2.0
    assert nonzero_values.max() <= 2.0


def test_dinr_ignores_out_of_support_targets_without_nan():
    config = _config()
    decoder = IntegratedContinuousDecoder(config)
    score_raw = torch.tensor([[[0.0, 100.0, 64.0], [100.0, 100.0, 64.0]]])
    raw = decoder(torch.randn(1, 2, config.hidden_size), score_shared_raw=score_raw)
    labels = torch.zeros(1, 2, 7)
    labels[..., 0] = torch.tensor([[6.0, -3.0]])
    labels[..., 1] = torch.tensor([[1.0, 3.0]])
    labels[..., 2] = 0.5
    losses = _compute_integrated_loss_components(
        config,
        raw,
        labels,
        torch.ones(1, 2, dtype=torch.long),
        score_shared_raw=score_raw,
    )
    assert all(torch.isfinite(value) for value in losses.values())


def test_dinr_note_encoders_share_numerical_value_tables_with_heads():
    config = _config()
    model = IntegratedPianoTransformer(config)
    note_encoder = model.note_encoder
    output_decoder = IntegratedContinuousDecoder(config)
    _share_dinr_value_tables(note_encoder, note_encoder, output_decoder)
    continuous = torch.zeros(1, 3, 12)
    continuous[..., 0] = 4.6
    continuous[..., 1] = 5.0
    continuous[..., 2] = 0.5
    continuous[..., 10:] = 1.0
    pitch = torch.full((1, 3), 60)
    assert model.note_encoder is model.decoder_note_encoder
    assert note_encoder(pitch, continuous, role="score").shape == (1, 3, config.hidden_size)
    assert note_encoder(pitch, continuous, role="decoder").shape == (1, 3, config.hidden_size)
    assert note_encoder.dinr_timing_table is output_decoder.dinr_timing_table
    assert note_encoder.dinr_velocity_table is output_decoder.dinr_velocity_table


def test_teacher_forcing_replaces_the_whole_note_with_mask_not_pad():
    config = _config()
    note_encoder = IntegratedNoteEncoder(config, 12, "score")
    embeddings = torch.randn(1, 3, config.hidden_size)
    attention_mask = torch.ones(1, 3, dtype=torch.long)
    masked = _apply_tf_embedding_mask(
        note_encoder,
        embeddings,
        attention_mask,
        keep_prob=0.0,
        skip_first_token=True,
    )
    assert torch.equal(masked[:, 0], embeddings[:, 0])
    expected = note_encoder.mask_embedding().view(1, 1, -1).expand(1, 2, -1)
    assert torch.equal(masked[:, 1:], expected)
    assert not torch.equal(note_encoder.mask_embedding(), note_encoder.pad_embedding())


def test_dinr_teacher_feedback_clamps_without_changing_loss_targets():
    config = _config()
    score_raw = torch.tensor([[[0.0, 100.0, 64.0], [100.0, 100.0, 64.0]]])
    predictions = torch.zeros(1, 2, 7)
    predictions[..., 0] = torch.tensor([[6.0, 3.0]])
    predictions[..., 1] = 3.0
    rows = _build_epr_decoder_rows(config, score_raw, predictions)
    performance_start = config.score_control_feature_dim
    assert torch.allclose(rows[0, :, performance_start], torch.tensor([5.0, torch.log(torch.tensor(100.0)) + 2.0]))
    assert torch.allclose(
        rows[0, :, performance_start + 1],
        torch.log(torch.tensor(100.0)) + 2.0,
    )


def test_dinr_floor_log_absolute_uses_full_positive_grid():
    config = _config(
        epr_timing_target="floor_log_absolute",
        dinr_timing_bins=512,
        dinr_zero_bin=0,
        dinr_timing_step=9.0 / 511.0,
        dinr_output_timing_bins=512,
        dinr_output_zero_bin=0,
        dinr_output_timing_step=9.0 / 511.0,
    )
    decoder = IntegratedContinuousDecoder(config)
    score_raw = torch.tensor([[[0.0, 100.0, 64.0]]])
    raw = decoder(torch.randn(1, 1, config.hidden_size), score_shared_raw=score_raw)
    logits = _split_dinr_logits(config, raw)
    assert torch.isfinite(_dinr_support_mask(config, logits["ioi"], "ioi", score_raw)).all()
    labels = torch.tensor([[[0.0, torch.log(torch.tensor(100.0)), 0.5, 0.0, 0.0, 0.0, 0.0]]])
    losses = _compute_integrated_loss_components(
        config, raw, labels, torch.ones(1, 1, dtype=torch.long), score_shared_raw=score_raw
    )
    assert all(torch.isfinite(value) for value in losses.values())


def test_dinr_separated_absolute_input_and_deviation_output_tables():
    config = _config(
        dinr_vocabulary_mode="separated",
        dinr_timing_bins=256,
        dinr_zero_bin=0,
        dinr_timing_step=9.0 / 255.0,
        dinr_output_timing_bins=256,
        dinr_output_zero_bin=128,
        dinr_output_timing_step=4.0 / 255.0,
    )
    note_encoder = IntegratedNoteEncoder(config, 12, "score")
    output_decoder = IntegratedContinuousDecoder(config)
    input_table = note_encoder.dinr_timing_table
    _share_dinr_value_tables(note_encoder, note_encoder, output_decoder)
    assert note_encoder.dinr_timing_table is input_table
    assert note_encoder.dinr_timing_table is not output_decoder.dinr_timing_table
    assert note_encoder.dinr_velocity_table is output_decoder.dinr_velocity_table
