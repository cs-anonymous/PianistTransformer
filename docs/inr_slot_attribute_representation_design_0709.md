# INR Slot-Attribute Representation Design 0709

## 1. Motivation

当前 INR-style EPR 模型已经尝试过多种表示和训练策略，包括 note raw-log timing、compact chord representation、random PAD dropout、DAgger-Lite / `tf_pred` feedback、stable dynamics loss 等。现有结果显示：

1. **raw-log timing 表示可以缓解一部分 timing scale 问题**，但不能消除 full-AR 随 rollout depth 增加而恶化的问题。
2. **compact chord representation 的收益有限**。它可能略微改善 IOI，但 duration、velocity、pedal 没有整体提升，甚至变差。因此 score-side IOI=0 continuation 不是当前 AR 闭环恶化的主因。
3. **随机将 decoder input 替换为 PAD 只能提供 missing-input robustness**，不能真正解决 wrong-but-plausible self-history 下的 distribution shift。
4. 当前一个核心疑点是：**将 pitch、score timing、performance timing、velocity、pedal、type/mask 等所有信息过早压入同一个 continuous embedding，可能导致不同属性通道在 AR 闭环中相互污染**。

因此，本设计提出一种 **slot-attribute representation**：保持 `1 note = 1 timestep`，但在 note 内部保留属性级结构。它不是回到 Pianist Transformer 那种 `8 tokens / note` 的长序列形式，而是将每个 note 的属性拆成若干 slot，每个 slot 独立编码，再聚合为一个 note-level embedding。

核心目标：

```text
保留 note-level AR 的效率
+ 保留属性边界
+ 区分 score condition 与 performance feedback
+ 支持 feature-level mask
+ 支持未来双向任务 / denoising / score-performance joint modeling
```

---

## 2. Design Principle

### 2.1 Structured input, unified hidden, structured output

本设计不要求 Transformer hidden state 显式分区。模型中间仍然使用统一的 `d_model` hidden。

```text
attribute slots -> slot embeddings -> note embedding -> Transformer -> factorized heads
```

即：

```text
structured in / unified middle / structured out
```

不建议第一版做 block-wise hidden 或 hard partitioned hidden，例如：

```text
h = [h_pitch, h_time, h_vel, h_ped]
```

因为这样会引入新的架构问题：不同分区如何通信、每个分区多少维、attention/FFN 是否分块等。当前阶段只在输入和输出端引入结构。

### 2.2 Score condition and performance feedback should be separated

AR 闭环恶化的主要风险来自 decoder input 中的 generated performance feedback。score-side attributes 是稳定条件，不会随 rollout 漂移；performance-side attributes 是 feedback，会随 rollout 漂移。

因此，不应将二者过早混入同一个 MLP。

```text
score-side: pitch, score IOI, score duration, score velocity, musical structure
perf-side: previous perf IOI, previous perf duration, previous perf velocity, previous perf pedal
```

### 2.3 IOI and duration should be different slots

IOI 和 duration 都属于 timing，但语义不同：

```text
IOI      = onset-to-onset time advance
Duration = note hold / articulation / overlap
```

在当前实验中，duration 往往比 IOI 更容易在 AR rollout 中恶化。因此，IOI 和 duration 不应一开始合成一个 timing feature，而应作为两个 slot 分别编码，再通过后续聚合层学习交互。

### 2.4 Musical information is score-side symbolic structure

musical onset、musical duration、musical length 等不是 raw seconds，也不是 logscale timing value，而是 categorical symbolic structure。它们应使用 embedding table 编码。

musical information 不是临时修补项，而是未来完整统一表示的一部分。它对于 EPR、CSR、score-performance bidirectional modeling、mask reconstruction 等任务都有价值。

---

## 3. Stage-A: 8-Slot Stability Representation

第一阶段先实现 8-slot，不加入 musical slots。目的不是构建最终表示，而是验证 attribute factorization 是否比当前 one-vector embedding 更稳定。

### 3.1 Slot definition

| Slot ID | Name | Type | Source | Role |
| ---: | --- | --- | --- | --- |
| 1 | `pitch` | categorical / multihot | score | pitch identity / alignment anchor |
| 2 | `score_ioi` | continuous raw-log | score | score-side time advance condition |
| 3 | `score_duration` | continuous raw-log | score | score-side duration condition |
| 4 | `score_velocity` | continuous / discrete | score | score-side dynamic condition |
| 5 | `perf_ioi` | continuous raw-log + mask | previous performance | AR feedback for IOI |
| 6 | `perf_duration` | continuous raw-log + mask | previous performance | AR feedback for duration |
| 7 | `perf_velocity` | continuous + mask | previous performance | AR feedback for dynamics |
| 8 | `perf_pedal` | continuous / binary pedal4 + mask | previous performance | AR feedback for pedal |

With `d_model = 768`, a simple equal-width design is:

```text
slot_dim = 768 / 8 = 96
```

Each slot is encoded to a 96-dimensional vector. The eight vectors are concatenated and projected to a final 768-dimensional note embedding.

```text
z_note = concat(e_pitch,
                e_score_ioi,
                e_score_duration,
                e_score_velocity,
                e_perf_ioi,
                e_perf_duration,
                e_perf_velocity,
                e_perf_pedal)        # 8 * 96 = 768

note_emb = MLP_slot_fusion(z_note)    # 768 -> 768
```

### 3.2 Encoder input under 8-slot design

Encoder input contains only score-side information. The four performance slots are set to feature-specific mask embeddings.

```text
encoder slots:
[pitch,
 score_ioi,
 score_duration,
 score_velocity,
 MASK_perf_ioi,
 MASK_perf_duration,
 MASK_perf_velocity,
 MASK_perf_pedal]
```

The encoder does not receive previous performance feedback.

### 3.3 Decoder input under 8-slot design

Decoder input contains score-side condition plus previous performance feedback.

```text
decoder slots:
[pitch,
 score_ioi,
 score_duration,
 score_velocity,
 prev_perf_ioi,
 prev_perf_duration,
 prev_perf_velocity,
 prev_perf_pedal]
```

During teacher forcing, previous performance feedback comes from GT. During AR inference, it comes from generated performance. During training, feature-level masking can be applied to the four performance slots.

---

## 4. Stage-B: 12-Slot Full Representation with Musical Structure

After 8-slot representation is verified to be stable, extend to 12-slot representation. This is the intended full representation for future bidirectional tasks.

### 4.1 Slot definition

| Slot ID | Name | Type | Source | Role |
| ---: | --- | --- | --- | --- |
| 1 | `pitch` | categorical / multihot | score | pitch identity |
| 2 | `score_ioi` | continuous raw-log | score | score time advance |
| 3 | `score_duration` | continuous raw-log | score | score duration |
| 4 | `score_velocity` | continuous / discrete | score | score dynamic condition |
| 5 | `perf_ioi` | continuous raw-log + mask | previous performance | AR feedback |
| 6 | `perf_duration` | continuous raw-log + mask | previous performance | AR feedback |
| 7 | `perf_velocity` | continuous + mask | previous performance | AR feedback |
| 8 | `perf_pedal` | pedal4 + mask | previous performance | AR feedback |
| 9 | `musical_onset` | categorical | score/music structure | symbolic onset / beat position |
| 10 | `musical_duration` | categorical | score/music structure | notated duration class |
| 11 | `musical_length_first` | categorical / mixed | score/music structure | phrase/group length, first/boundary role |
| 12 | `musical_binary` | binary vector | score/music structure | other symbolic flags |

With `d_model = 768`, use:

```text
slot_dim = 768 / 12 = 64
```

Each slot is encoded to 64 dimensions:

```text
z_note = concat(e_1, e_2, ..., e_12)  # 12 * 64 = 768
note_emb = MLP_slot_fusion(z_note)    # 768 -> 768
```

### 4.2 Musical slots should be categorical

Musical slots are not encoded using raw seconds or logscale.

Recommended encoding:

```text
musical_onset_id     -> Embedding(num_onset_classes, 64)
musical_duration_id  -> Embedding(num_duration_classes, 64)
musical_length_id    -> Embedding(num_length_classes, 64)
musical_binary_flags -> Linear/MLP(binary_flags) -> 64
```

Examples of possible binary annotations:

```text
is_downbeat
is_phrase_first
is_phrase_last
is_group_boundary
is_chord_continuation
is_grace
is_tie
is_slur
staff_id / hand indicator, if reliable
```

### 4.3 Why musical slots are needed

The 8-slot design is primarily a debugging/stability representation. The 12-slot version is the complete extensible representation.

Musical structure should affect performance rendering:

```text
beat / measure position -> timing and accent
notated duration -> articulation and duration control
phrase/group boundary -> rubato and velocity shaping
first/last position -> onset emphasis and release behavior
binary structure flags -> local performance conventions
```

Without explicit musical structure, the model must infer these roles from raw score timing and context alone. This may be insufficient under ASAP-only supervised training.

### 4.4 Why previous musical experiments may have failed

Earlier experiments showed that adding musical features to a one-vector input did not improve rollout stability and could worsen duration. This does not prove musical information is useless.

A more likely explanation is:

```text
musical features were mixed too early with continuous score/performance features,
causing them to interfere with timing feedback and other channels.
```

In the slot representation, musical features occupy independent symbolic slots. This makes their effect more interpretable and less likely to contaminate performance feedback.

---

## 5. Slot Encoding Details

### 5.1 Pitch slot

Options:

1. MIDI pitch ID embedding.
2. 88-dimensional piano multihot projected to slot dimension.
3. Chord-aware pitch multihot, if chord features are later reintroduced as auxiliary information.

For note-level representation:

```text
pitch_id -> Embedding(128, slot_dim)
```

or:

```text
pitch_multihot_88 -> Linear(88, slot_dim)
```

### 5.2 Score IOI slot

Use raw-log representation:

```text
score_ioi_raw_s = score_ioi_ms / 1000
score_ioi_log   = log1p(score_ioi_ms / 50)
```

Slot input:

```text
[score_ioi_raw_s, score_ioi_log]
```

Encoder:

```text
e_score_ioi = MLP_score_ioi([raw_s, log]) -> slot_dim
```

### 5.3 Score duration slot

Use raw-log representation:

```text
score_dur_raw_s = score_duration_ms / 1000
score_dur_log   = log1p(score_duration_ms / 50)
```

Slot input:

```text
[score_dur_raw_s, score_dur_log]
```

### 5.4 Score velocity slot

If score velocity is MIDI 0-127:

```text
score_velocity_norm = score_velocity / 127
```

Slot input:

```text
[score_velocity_norm]
```

If score velocity is categorical/dynamic class in future tasks, this slot can be changed to an embedding table.

### 5.5 Performance IOI slot

Performance feedback should use previous performance value. For EPR target, the head predicts deviations, but the decoder input can contain reconstructed absolute performance control.

Recommended decoder feedback value:

```text
perf_ioi_raw_s = perf_ioi_ms / 1000
perf_ioi_log   = log1p(perf_ioi_ms / 50)
```

Slot input:

```text
[perf_ioi_raw_s, perf_ioi_log, mask_flag/source_flag]
```

If masked:

```text
e_perf_ioi = MASK_perf_ioi_embedding
```

Do not set masked value to numeric zero alone, because 0 can be a valid value.

### 5.6 Performance duration slot

Same as performance IOI:

```text
perf_dur_raw_s = perf_duration_ms / 1000
perf_dur_log   = log1p(perf_duration_ms / 50)
```

### 5.7 Performance velocity slot

If performance velocity is MIDI 0-127:

```text
perf_velocity_norm = perf_velocity / 127
```

Masked performance velocity should use a feature-specific mask embedding.

### 5.8 Performance pedal slot

Use four pedal samples:

```text
pedal4 = [pedal_0, pedal_25, pedal_50, pedal_75]
```

If values are MIDI 0-127:

```text
pedal4_norm = pedal4 / 127
```

If already in 0-1, use directly.

Slot input:

```text
pedal4_norm -> MLP_pedal -> slot_dim
```

If masked:

```text
e_perf_pedal = MASK_perf_pedal_embedding
```

---

## 6. Mask and Source Design

### 6.1 Feature-specific mask embeddings

Do not use one shared `<PAD>` for all performance features.

Use separate mask embeddings:

```text
MASK_perf_ioi
MASK_perf_duration
MASK_perf_velocity
MASK_perf_pedal
```

For 12-slot representation, also allow musical masks for future bidirectional tasks:

```text
MASK_musical_onset
MASK_musical_duration
MASK_musical_length_first
MASK_musical_binary
```

### 6.2 Feedback source types

For decoder performance slots, the model should know the feedback source:

```text
GT feedback
sampled feedback
mean/greedy feedback
masked feedback
BOS feedback
```

This can be implemented in two ways:

1. Add a small source embedding to each performance slot.
2. Add a global decoder type/source slot.

First implementation recommendation:

```text
e_perf_slot = value_encoder(value) + source_embedding(source_type)
```

If masked:

```text
e_perf_slot = mask_embedding_for_this_slot + source_embedding(masked)
```

### 6.3 Feature-level feedback masking

Instead of replacing an entire decoder note embedding with PAD, mask individual performance slots.

Recommended first version:

```text
perf_ioi_mask_prob      = 0.10
perf_duration_mask_prob = 0.10
perf_velocity_mask_prob = 0.10
perf_pedal_mask_prob    = 0.10
full_perf_mask_prob     = 0.05
```

Avoid 50% masking in the first slot model. Excessive masking may teach the decoder to ignore performance feedback and collapse toward conditional mean prediction.

---

## 7. Slot Fusion Module

### 7.1 Basic fusion

For 8-slot:

```python
slot_vec = torch.cat([
    e_pitch,
    e_score_ioi,
    e_score_dur,
    e_score_vel,
    e_perf_ioi,
    e_perf_dur,
    e_perf_vel,
    e_perf_ped,
], dim=-1)  # [B, T, 768]

note_emb = slot_fusion_mlp(slot_vec)  # [B, T, 768]
```

For 12-slot:

```python
slot_vec = torch.cat([e1, e2, ..., e12], dim=-1)  # [B, T, 768]
note_emb = slot_fusion_mlp(slot_vec)              # [B, T, 768]
```

### 7.2 Recommended MLP

A light MLP is enough for first implementation:

```text
LayerNorm(768)
Linear(768 -> 4 * d_model)
SiLU/GELU
Linear(4 * d_model -> d_model)
LayerNorm(d_model)
```

If too heavy, use:

```text
LayerNorm(768)
Linear(768 -> d_model)
LayerNorm(d_model)
```

### 7.3 Optional slot gates

For interpretability and stability, add learnable slot gates:

```text
slot_vec_i = alpha_i * slot_vec_i
```

where `alpha_i` is a learned scalar or vector.

For musical slots in 12-slot version, initialize gates lower:

```text
alpha_musical_init = 0.1 or 0.3
```

This allows the model to use musical information gradually rather than letting it dominate early training.

---

## 8. Encoder, Decoder, and Head Design

### 8.1 Encoder

Encoder uses the same slot format but masks performance slots.

8-slot encoder:

```text
[pitch,
 score_ioi,
 score_duration,
 score_velocity,
 MASK_perf_ioi,
 MASK_perf_duration,
 MASK_perf_velocity,
 MASK_perf_pedal]
```

12-slot encoder:

```text
[pitch,
 score_ioi,
 score_duration,
 score_velocity,
 MASK_perf_ioi,
 MASK_perf_duration,
 MASK_perf_velocity,
 MASK_perf_pedal,
 musical_onset,
 musical_duration,
 musical_length_first,
 musical_binary]
```

### 8.2 Decoder input

8-slot decoder:

```text
[pitch,
 score_ioi,
 score_duration,
 score_velocity,
 prev_perf_ioi,
 prev_perf_duration,
 prev_perf_velocity,
 prev_perf_pedal]
```

12-slot decoder:

```text
[pitch,
 score_ioi,
 score_duration,
 score_velocity,
 prev_perf_ioi,
 prev_perf_duration,
 prev_perf_velocity,
 prev_perf_pedal,
 musical_onset,
 musical_duration,
 musical_length_first,
 musical_binary]
```

### 8.3 Decoder head

Decoder head should be factorized by output family.

```text
decoder_hidden h_t
├── IOI trunk
│   ├── log IOI dev distribution
│   └── raw IOI dev distribution
├── Duration trunk
│   ├── log duration dev distribution
│   └── raw duration dev distribution
├── Velocity trunk
│   └── velocity distribution
└── Pedal trunk
    └── pedal4 logits / distribution
```

Do not use a single giant linear layer for all output parameters in the first structured model.

---

## 9. Output Targets and Losses

### 9.1 Timing targets

Use raw-log deviation targets:

```text
log_ioi_dev = log1p(perf_ioi_ms / 50) - log1p(score_ioi_ms / 50)
log_dur_dev = log1p(perf_dur_ms / 50) - log1p(score_dur_ms / 50)

raw_ioi_dev_s = (perf_ioi_ms - score_ioi_ms) / 1000
raw_dur_dev_s = (perf_dur_ms - score_dur_ms) / 1000
```

### 9.2 Timing distribution

Use skew-normal head or mixture normal/skew-normal depending on implementation readiness.

Initial recommendation:

```text
epr_distribution = skew_normal
raw_timing_loss_lambda = 0.5
```

Loss:

```text
loss_ioi = SN_NLL(log_ioi_dev) + lambda_raw * SN_NLL(raw_ioi_dev_s)
loss_dur = SN_NLL(log_dur_dev) + lambda_raw * SN_NLL(raw_dur_dev_s)
```

### 9.3 Velocity target

If velocity is MIDI 0-127:

```text
velocity_norm = perf_velocity / 127
```

Loss options:

```text
SN_NLL(velocity_norm)
SmoothL1(velocity_norm)
CE if discretized
```

For consistency with current raw-log SN setup, use SN_NLL first.

### 9.4 Pedal target

Use pedal4:

```text
pedal4 = [p0, p25, p50, p75]
```

Loss:

```text
BCEWithLogits(pedal4)
```

or distributional regression if continuous half-pedal is meaningful.

### 9.5 Total loss

Initial version:

```text
loss =
  1.0 * loss_ioi
+ 1.0 * loss_duration
+ 1.0 * loss_velocity
+ 0.5 * loss_pedal
```

Do not add stable dynamics / rollout losses in the first slot embedding baseline. First isolate whether the representation itself improves rollout behavior.

---

## 10. Training Strategy

### 10.1 Stage-A experiments

Run the following clean ablations:

| Run | Representation | Musical | Feature mask | Stable | Purpose |
| --- | --- | --- | --- | --- | --- |
| S0 | current one-vector note raw-log SN | no | no | no | clean baseline |
| S1 | 8-slot note raw-log SN | no | no | no | isolate slot factorization |
| S2 | 8-slot note raw-log SN | no | perf-slot mask | no | channel robustness |
| S3 | 8-slot note raw-log SN | no | perf-slot mask | stable optional | only after S1/S2 |

Primary comparison:

```text
S0 vs S1
```

If S1 improves k-sweep slope or head regime shift, slot factorization is useful.

### 10.2 Stage-B experiments

After 8-slot is stable, add musical slots:

| Run | Representation | Musical | Gate | Purpose |
| --- | --- | --- | --- | --- |
| M0 | 8-slot | no | no | stable baseline |
| M1 | 12-slot | yes | small gate | test musical utility |
| M2 | 12-slot | yes | no gate | test full musical strength |

Priority:

```text
M0 vs M1
```

Do not interpret musical failure under one-vector representation as final evidence against musical information.

### 10.3 After representation stabilizes

Only after slot representation is evaluated, add distribution-level AR regularizers:

```text
rollout mean/std matching
head TF-vs-self regime matching
stable dynamics contraction
```

Recommended order:

```text
1. clean slot representation
2. feature-level feedback mask
3. rollout mean/std loss
4. head regime matching
5. stable dynamics if still useful
```

---

## 11. Evaluation Protocol

Use the same k-sweep diagnostics:

```text
k0
k1
k4
k16
full global W
AR-pp W
```

Metrics:

```text
IOI W
Duration W
Velocity W
Pedal W
```

Additional diagnostics:

```text
mean shift: raw IOI / raw duration
std ratio: raw IOI / raw duration
head mean under TF vs self-history
head std under TF vs self-history
single-channel feedback ablation
feature mask robustness
```

Most important comparisons:

```text
k0 -> k1 slope
k0 -> full slope
head mean/std shift under self-history
```

Do not judge only by final full AR W, because a representation may improve local robustness while still failing global stationary distribution.

---

## 12. Expected Outcomes and Interpretation

### 12.1 If 8-slot improves rollout slope

Interpretation:

```text
The previous one-vector embedding was causing harmful attribute mixing.
Separating score/performance and timing/velocity/pedal slots helps reduce feedback channel contamination.
```

Next:

```text
Add 12-slot musical representation.
Then add rollout mean/std matching.
```

### 12.2 If 8-slot improves k0 but not full AR

Interpretation:

```text
Slot representation improves local prediction quality but does not solve closed-loop dynamics.
```

Next:

```text
Keep 8-slot/12-slot as representation.
Add rollout-level distribution regularization.
```

### 12.3 If 8-slot does not improve anything

Interpretation:

```text
The main failure is not caused by input attribute mixing.
It is more likely dominated by closed-loop transition dynamics or distribution head behavior.
```

Next:

```text
Return to note raw-log SN baseline and prioritize rollout mean/std + head regime matching.
```

### 12.4 If 12-slot musical improves k0 but not full AR

Interpretation:

```text
Musical structure helps local expressive prediction but not long-horizon AR stability.
```

This is still useful for final system design.

### 12.5 If 12-slot musical worsens duration again

Interpretation:

```text
Musical features may be noisy or over-weighted.
Try smaller musical gates or only selected musical slots.
```

Do not immediately discard all musical information.

---

## 13. Implementation Checklist

### 13.1 Dataset / collator

- [ ] Build 8-slot input tensors.
- [ ] Build 12-slot input tensors after 8-slot is stable.
- [ ] Add feature-specific mask flags for performance slots.
- [ ] Add source type for decoder performance feedback.
- [ ] Ensure encoder performance slots are always masked.
- [ ] Ensure decoder performance slots are shifted previous-step feedback.
- [ ] Ensure BOS feedback uses feature-specific BOS/mask embeddings.

### 13.2 Model

- [ ] Implement `SlotAttributeEmbedding`.
- [ ] Implement per-slot encoders.
- [ ] Implement feature-specific mask embeddings.
- [ ] Implement optional slot gates.
- [ ] Implement slot fusion MLP.
- [ ] Keep Transformer hidden unchanged.
- [ ] Implement factorized decoder heads.

### 13.3 Training

- [ ] Run clean S0 one-vector baseline.
- [ ] Run S1 8-slot no mask.
- [ ] Run S2 8-slot with feature-level perf mask.
- [ ] Only then test stable / rollout regularizers.

### 13.4 Evaluation

- [ ] k-sweep W.
- [ ] full global W.
- [ ] AR-pp W.
- [ ] raw mean shift.
- [ ] raw std ratio.
- [ ] head TF-vs-self mean/std.
- [ ] single-channel feedback ablation.

---

## 14. Recommended Config Sketch

### 14.1 8-slot config

```json
{
  "representation": "slot_attribute",
  "slot_version": "8slot_rawlog_nomus_0709",
  "d_model": 768,
  "num_slots": 8,
  "slot_dim": 96,
  "slot_fusion": "mlp",
  "slot_gates": true,

  "slots": [
    "pitch",
    "score_ioi_rawlog",
    "score_duration_rawlog",
    "score_velocity",
    "perf_ioi_rawlog",
    "perf_duration_rawlog",
    "perf_velocity",
    "perf_pedal4"
  ],

  "timing_log_scale_ms": 50,
  "raw_time_scale_ms": 1000,
  "velocity_scale": 127,

  "decoder_perf_slot_mask": true,
  "perf_ioi_mask_prob": 0.10,
  "perf_duration_mask_prob": 0.10,
  "perf_velocity_mask_prob": 0.10,
  "perf_pedal_mask_prob": 0.10,
  "full_perf_mask_prob": 0.05,

  "epr_distribution": "skew_normal",
  "raw_timing_loss_lambda": 0.5,
  "factorized_decoder_heads": true,

  "musical_feature_mode": "none"
}
```

### 14.2 12-slot config

```json
{
  "representation": "slot_attribute",
  "slot_version": "12slot_rawlog_musical_0709",
  "d_model": 768,
  "num_slots": 12,
  "slot_dim": 64,
  "slot_fusion": "mlp",
  "slot_gates": true,
  "musical_gate_init": 0.1,

  "slots": [
    "pitch",
    "score_ioi_rawlog",
    "score_duration_rawlog",
    "score_velocity",
    "perf_ioi_rawlog",
    "perf_duration_rawlog",
    "perf_velocity",
    "perf_pedal4",
    "musical_onset_id",
    "musical_duration_id",
    "musical_length_first_id",
    "musical_binary_flags"
  ],

  "musical_feature_mode": "categorical_slots",
  "musical_onset_encoding": "embedding",
  "musical_duration_encoding": "embedding",
  "musical_length_first_encoding": "embedding",
  "musical_binary_encoding": "linear",

  "timing_log_scale_ms": 50,
  "raw_time_scale_ms": 1000,
  "velocity_scale": 127,

  "decoder_perf_slot_mask": true,
  "epr_distribution": "skew_normal",
  "raw_timing_loss_lambda": 0.5,
  "factorized_decoder_heads": true
}
```

---

## 15. Final Decision

The current recommended path is:

```text
1. Stop using compact chord representation as the main line.
2. Return to note raw-log SN representation.
3. Replace one-vector continuous embedding with 8-slot attribute embedding.
4. Validate whether slot factorization improves k-sweep stability.
5. After 8-slot is stable, extend to 12-slot with categorical musical slots.
6. After representation is clean, add rollout mean/std and head regime matching to address full-AR distribution shift.
```

Short version:

```text
8-slot = stability/debugging representation
12-slot = final extensible representation for EPR + future bidirectional tasks
```

This design keeps the efficiency of one note per timestep while restoring the attribute-level structure needed for stable performance feedback modeling.
