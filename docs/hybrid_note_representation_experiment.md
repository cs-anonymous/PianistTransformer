# Hybrid Note Representation 第一版实验规划

本文档定义一个可执行的第一版实验：保持 PianistTransformer 的 T5Gemma backbone 不变，仅将 MIDI 表示从 8-token note block 改为 note-level continuous node。实验目标是验证：在 EPR 任务中，使用结构化 note node 和连续回归头，是否优于当前离散 tokenizer + LM head 方案。

## 0. 当前实现状态

截至当前实现，第一版已经落地为 work-level PianoCoRe-A node SFT 流程：

- 数据生成脚本：[src/data_process/06_generate_sft_node_data_pianocore.py](/home/kaititech/EPR/PianistTransformer/src/data_process/06_generate_sft_node_data_pianocore.py)
- MIDI/node 工具：[src/utils/node_midi.py](/home/kaititech/EPR/PianistTransformer/src/utils/node_midi.py)
- Hybrid 模型：[src/model/hybrid_pianoformer.py](/home/kaititech/EPR/PianistTransformer/src/model/hybrid_pianoformer.py)
- SFT 训练入口：[src/train/sft_node.py](/home/kaititech/EPR/PianistTransformer/src/train/sft_node.py)
- 训练配置：[configs/sft_node_config_pianocore.json](/home/kaititech/EPR/PianistTransformer/configs/sft_node_config_pianocore.json)
- 启动脚本：[script/sft_node.sh](/home/kaititech/EPR/PianistTransformer/script/sft_node.sh)

数据格式已从原计划的 pair-level jsonl 改为 work-level JSON：每个作品一个 `*.node_a.json`，直接写在 refined score MIDI 旁边。`score.pitch` 和 `score.score_continuous` 只存一次，`performances[]` 中保存多个演奏的 `label_continuous` 与 `interpolated`。`interpolated` 使用短整数 `0/1`，连续值保留 5 位小数。

当前已完成全量 PianoCoRe-A 处理：

```text
output: data/pianocore/PianoCoRe/refined/**/*.node_a.json
summary: data/pianocore/PianoCoRe/refined/pianocore_a_node_summary.json
works_total: 1936
success_works: 1936
success_performances: 157198
failed_performances: 9
failed reason: pitch_mismatch
node JSON total size: ~16G
```

模型第一版为 `HybridPianoT5Gemma`：保留 T5Gemma encoder-decoder backbone，新增 `HybridNoteEncoder` 和 `HybridContinuousDecoder`。Pitch 只作为输入 embedding，不作为预测目标；decoder 输出 7 个连续字段并使用 sigmoid 限制在 `[0, 1]`。训练默认从旧 PianistTransformer checkpoint 迁移 encoder/decoder backbone，旧 token embedding 和 LM head 跳过，新 note encoder/decoder 随机初始化。

训练数据集已实现为 map-style Dataset，而不是 IterableDataset。样本索引映射到 `(work, performance, window)`，并使用每进程 LRU cache 缓存最近 work JSON；这样 DDP 的 DistributedSampler 可以稳定分片，避免 IterableDataset 在多卡下 batch dispatch 和尾部耗尽不一致的问题。

当前本地已启动 3 卡 node SFT：

```bash
tmux attach -t sft_node
tail -f logs/sft_node_*.log
```

默认配置为 1000 steps，`block_notes=512`，3 卡训练，`per_device_train_batch_size=2`，`gradient_accumulation_steps=16`，每 500 step 保存 checkpoint。

## 1. 核心假设

当前 PianistTransformer 已经在 encoder 端把每 8 个 token 合并为一个 note embedding，但 decoder 和 loss 仍然是 token-level classification。第一版 Hybrid Note Representation 将输入和输出都改为 note-level object：

```text
[pitch, ioi, duration, velocity, pedal_0, pedal_25, pedal_50, pedal_75]
    -> NoteEncoder
    -> PianistTransformer backbone
    -> NoteDecoder
    -> continuous performance fields
```

本实验不更换 backbone，不改变 attention、层数、hidden size 等主干结构。主要变化是：

- 数据从 token ids 变为 note feature tensor。
- 输入端从 `PianoEncoderEmbeddings` 改为 `HybridNoteEncoder`。
- 输出端从 `lm_head + cross entropy` 改为 `HybridNoteDecoder + regression loss`。
- Pedal 保持连续 CC value，使用 MSE。

## 2. 数据来源

第一版只使用 PianoCoRe refined pair，因为 refined data 已经提供 note-by-note aligned score/performance：

```text
data/pianocore/metadata.csv 或 data/pianocore/metadata_S.csv
data/pianocore/PianoCoRe/refined/<relative_path>
```

本地 refined 目录应在脚本中自动检测，优先级如下：

```python
refined_dir_candidates = [
    Path("data/pianocore/PianoCoRe/refined"),
    Path("data/pianocore/PianoCoRe-1.0/refined"),
]
```

当前仓库里的实际路径是：

```text
data/pianocore/PianoCoRe/refined
```

推荐第一版使用：

```text
metadata_S.csv
```

并过滤：

```python
df = df[df["tier_a_star"] == True]
df = df[df["refined_score_midi_path"].notna()]
df = df[df["refined_performance_midi_path"].notna()]
```

原因：

- `metadata_S.csv` 更小，适合首轮实验。
- `tier_a_star=True` 是最高置信 note-level alignment 子集。
- `refined_score_midi_path` 和 `refined_performance_midi_path` 的音符数一致。
- `refined_alignment_path` 里有 `interpolated` mask，可用于后续加权 loss 或分析。

暂不使用 `align_score_and_performance`。PianoCoRe refined pair 已经完成对齐和插值。

## 3. Note Feature Schema

每个 note 变成一个长度为 8 的 feature vector：

```text
[
  pitch,
  ioi,
  duration,
  velocity,
  pedal_0,
  pedal_25,
  pedal_50,
  pedal_75
]
```

字段定义：

- `pitch`: MIDI pitch，整数，范围 `[0, 127]`。
- `ioi`: 当前 note onset 与前一个 note onset 的差，单位 ms，保留 float。
- `duration`: 当前 note offset 与 onset 的差，单位 ms，保留 float。
- `velocity`: MIDI velocity，归一化到 `[0, 1]`。
- `pedal_0`: 当前 note onset 时刻的 CC64 value，归一化到 `[0, 1]`。
- `pedal_25`: 当前 note 到下一 note onset 的 25% 处 CC64 value，归一化到 `[0, 1]`。
- `pedal_50`: 当前 note 到下一 note onset 的 50% 处 CC64 value，归一化到 `[0, 1]`。
- `pedal_75`: 当前 note 到下一 note onset 的 75% 处 CC64 value，归一化到 `[0, 1]`。

注意：这里的 `ioi` 和 `duration` 使用真实播放时间，而不是原始 MIDI tick，也不再 round 到整数 ms。实现时应通过原 MIDI 的 tempo map 直接计算 float ms：

```python
start_ms = tick_to_time_mapping[note.start] * 1000.0
end_ms = tick_to_time_mapping[note.end] * 1000.0
```

如果为了兼容后续 MIDI 还原，仍然可以在写回 MIDI 时使用：

```text
ticks_per_beat = 500
tempo = 120 BPM
```

此时 1 tick 等价于 1 ms，但这是输出 MIDI 的表示方式，不应成为训练数据的量化约束。

## 4. 连续字段归一化

第一版使用稳定、可逆、抗长尾的归一化。

### 4.1 pitch

`pitch` 不归一化，保留整数，进入 embedding：

```python
pitch_id = note.pitch
```

### 4.2 ioi 和 duration

推荐使用 log normalization：

```python
MAX_TIME_MS = 10000

time_ms = min(max(time_ms, 0), MAX_TIME_MS)
time_norm = log1p(time_ms) / log1p(MAX_TIME_MS)
```

其中 `time_ms` 是 float，不做：

```python
round(time_ms)
```

反归一化：

```python
time_ms = expm1(time_norm * log1p(MAX_TIME_MS))
```

理由：

- 当前 tokenizer 对 timing 约 5 秒以上直接 clip，信息损失较大。
- EPR 的 timing/duration 分布长尾明显。
- 线性 `[0, 1]` 会让短音、快音型挤在很小区域。
- `MAX_TIME_MS=10000` 比当前 5000 更宽，首版更稳。

### 4.3 velocity 和 pedal

```python
velocity_norm = velocity / 127.0
pedal_norm = cc64_value / 127.0
```

模型输出建议经过 `sigmoid` 限制在 `[0, 1]`。

## 5. 数据文件格式

新增数据生成脚本：

```text
src/data_process/06_generate_sft_node_data_pianocore.py
```

输出：

```text
data/processed/sft/sft_pianocore_nodes.jsonl
```

每行格式：

```json
{
  "pitch": [60, 64, 67],
  "score_continuous": [[0.0, 0.12, 0.0, 0.0, 0.0, 0.0, 0.0], ...],
  "label_continuous": [[0.0, 0.10, 0.63, 0.0, 0.0, 0.12, 0.4], ...],
  "interpolated": [false, false, true],
  "score_source": ".../score_PDMX_refined.mid",
  "performance_source": ".../Aria_xxx_refined.mid",
  "alignment_source": ".../Aria_xxx_refined_align.npz",
  "split": "train"
}
```

其中：

- `pitch`: shared score/performance pitch sequence。
- `score_continuous`: score-side continuous features，shape `[num_notes, 7]`。
- `label_continuous`: performance-side target continuous features，shape `[num_notes, 7]`。
- `interpolated`: refined alignment 的 boolean mask。第一版默认不 mask loss，只记录；后续可用于加权。

生成数据时必须先检查：

```python
assert score_pitch == performance_pitch
```

通过后只保存一份 `pitch`。如果 pitch 不一致，说明该 pair 不满足第一版 EPR 的硬约束，应 skip 并记录 `pitch_mismatch_count`，而不是让模型预测 pitch。

`continuous` 的 7 个字段顺序固定为：

```text
[ioi_norm, duration_norm, velocity_norm, pedal_0, pedal_25, pedal_50, pedal_75]
```

Score-side `velocity_norm` 可以使用 score MIDI 原始 velocity，也可以统一设为 0。第一版推荐保留 score MIDI velocity，因为部分 score MIDI 的 velocity 可反映声部或动态信息，但后续需要做 ablation：

```text
score velocity kept vs score velocity zeroed
```

## 6. 序列切片

当前 token 方案的 `block_size=4096` 等价于：

```text
4096 tokens / 8 = 512 notes
```

Node 方案直接以 note 为单位切片。第一版配置：

```json
{
  "block_notes": 512,
  "overlap_ratio": 0.5,
  "min_notes": 64
}
```

切片逻辑：

```python
window_len = block_notes
stride = int(block_notes * (1 - overlap_ratio))
```

不足 `min_notes` 的片段丢弃。

`block_notes=512` 不是缩短 context，而是和旧方案的 `block_size=4096` 保持相同的实际音乐上下文。若设置 `block_notes=4096`，实际等价于旧 tokenizer 的 `32768 tokens`，会引入更长上下文和显存成本这两个额外变量，不适合作为第一版公平对照。

## 7. 模型结构

新增模型文件：

```text
src/model/hybrid_pianoformer.py
```

### 7.1 Config

新增：

```python
class HybridPianoT5GemmaConfig(PianoT5GemmaConfig):
    continuous_dim = 7
    max_time_ms = 10000
    pitch_vocab_size = 128
    pitch_pad_id = 128
```

注意：为了 padding，pitch embedding 可以设为 129：

```python
nn.Embedding(129, hidden_size, padding_idx=128)
```

### 7.2 HybridNoteEncoder

输入：

```python
pitch_ids: LongTensor[B, N]
continuous: FloatTensor[B, N, 7]
```

结构：

```python
pitch_emb = PitchEmbedding(pitch_ids)
cont_emb = ContinuousMLP(continuous)
note_emb = LayerNorm(pitch_emb + cont_emb)
```

推荐第一版：

```python
ContinuousMLP:
  Linear(7, hidden_size)
  GELU
  Linear(hidden_size, hidden_size)
```

### 7.3 Backbone

保持 PianistTransformer 的 encoder-decoder backbone：

```text
encoder_layers_num = 10
decoder_layers_num = 2
hidden_size = 768
intermediate_size = 3072
num_attention_heads = 8
num_key_value_heads = 4
head_dim = 128
```

Encoder 接受 note embeddings：

```python
encoder_outputs = self.model.encoder(
    inputs_embeds=score_note_embeds,
    attention_mask=attention_mask,
)
```

Decoder 第一版有两个可选方案。

推荐第一版采用 non-autoregressive decoder input：

```python
decoder_inputs_embeds = score_note_embeds
```

也就是：

```text
score node embeddings -> decoder cross-attends encoder -> performance node predictions
```

这避免了连续目标的 teacher forcing 复杂度，且更符合 EPR 的同长 structured prediction。

### 7.4 HybridNoteDecoder

输入：

```python
hidden_states: FloatTensor[B, N, hidden_size]
```

输出：

```python
continuous_pred: FloatTensor[B, N, 7]
```

结构：

```python
ContinuousHead:
  Linear(hidden_size, hidden_size)
  GELU
  Linear(hidden_size, 7)
  Sigmoid
```

第一版不预测 pitch，直接从 score copy pitch。原因：

- EPR 不应该改变音高。
- PianoCoRe refined score/performance 已经 note-by-note 对齐。
- pitch CE 会掩盖连续控制任务的主要信号。

后续可以增加 `pitch_head` 作为 auxiliary consistency loss，但不作为第一版目标。

## 8. Loss

目标连续字段：

```text
[ioi, duration, velocity, pedal_0, pedal_25, pedal_50, pedal_75]
```

第一版 loss：

```python
loss_ioi = masked_huber(pred[..., 0], target[..., 0], attention_mask)
loss_dur = masked_huber(pred[..., 1], target[..., 1], attention_mask)
loss_vel = masked_mse(pred[..., 2], target[..., 2], attention_mask)
loss_pedal = masked_mse(pred[..., 3:7], target[..., 3:7], attention_mask.unsqueeze(-1))

loss = (
    loss_ioi
  + loss_dur
  + loss_vel
  + loss_pedal
)
```

所有 loss 只在 `attention_mask == 1` 的 note 上计算。四个 group 第一版使用相同权重，避免过早引入人为调参。

`loss_pedal` 是 4 个 pedal snapshot 的 masked mean，而不是 4 项相加。因此它的量级仍然接近 `[0, 1]` 范围内的平均误差，不会天然变成其他 group 的 4 倍。

`interpolated` 第一版默认不降低权重。第二版可尝试：

```python
weight = torch.where(interpolated, 0.5, 1.0)
```

不建议第一版就加插值降权，因为它会让实验因素变多。

## 9. Trainer Batch 接口

新增训练脚本：

```text
src/train/sft_nodes.py
```

Data collator 输出：

```python
{
    "pitch_ids": LongTensor[B, N],
    "continuous": FloatTensor[B, N, 7],
    "labels_continuous": FloatTensor[B, N, 7],
    "attention_mask": LongTensor[B, N],
    "interpolated": BoolTensor[B, N],
}
```

Padding：

- `pitch_ids`: 使用 `pitch_pad_id=128`。
- `continuous`: 使用 0。
- `labels_continuous`: 使用 0。
- `attention_mask`: 有效 note 为 1，padding 为 0。
- `interpolated`: padding 为 `False`。

模型 `forward` 签名：

```python
def forward(
    self,
    pitch_ids=None,
    continuous=None,
    attention_mask=None,
    labels_continuous=None,
    interpolated=None,
    **kwargs,
):
```

返回 `Seq2SeqLMOutput` 或 `ModelOutput` 均可，但需要包含：

```python
loss
logits 或 continuous_pred
```

为了兼容 `Trainer`，可返回：

```python
return {"loss": loss, "continuous_pred": continuous_pred}
```

## 10. 配置文件

新增：

```text
configs/sft_node_config_pianocore.json
```

建议第一版：

```json
{
  "data_paths": ["data/processed/sft/sft_pianocore_nodes.jsonl"],
  "block_notes": 512,
  "min_notes": 64,
  "overlap_ratio": 0.5,
  "pretrained_model": null,
  "output_dir": "./models/sft_nodes/",
  "overwrite_output_dir": true,
  "num_train_epochs": 1,
  "save_steps": 500,
  "logging_steps": 50,
  "eval_steps": 1000,
  "per_device_train_batch_size": 4,
  "gradient_accumulation_steps": 8,
  "lr_scheduler_type": "cosine",
  "learning_rate": 0.0005,
  "warmup_ratio": 0,
  "save_total_limit": 10,
  "prediction_loss_only": true,
  "bf16": true,
  "report_to": "none",
  "logging_dir": "~/tf-logs/",
  "eval_strategy": "steps",
  "save_strategy": "epoch",
  "logging_strategy": "steps",
  "logging_first_step": true,
  "dataloader_num_workers": 16,
  "dataloader_prefetch_factor": 4
}
```

第一版 `pretrained_model` 设为 `null`，因为旧 checkpoint 的 embedding 和 LM head 与新模型不兼容。后续可以只迁移 backbone 权重。

## 11. 运行步骤

### 11.1 生成 node SFT 数据

```bash
python src/data_process/06_generate_sft_node_data_pianocore.py
```

预期输出：

```text
data/processed/sft/sft_pianocore_nodes.jsonl
```

脚本应打印：

```text
rows loaded
rows after tier_a_star filter
success pairs
failed pairs
total segments
```

脚本应在每个 pair 上执行以下检查：

```python
assert score_pitch == performance_pitch
pitch = score_pitch
assert len(pitch) == len(score_continuous)
assert len(score_continuous) == len(label_continuous)
assert len(pitch) == len(interpolated)
assert min(pitch) >= 0 and max(pitch) <= 127
```

### 11.2 训练

```bash
python src/train/sft_nodes.py --config configs/sft_node_config_pianocore.json
```

多 GPU 沿用现有 deepspeed/DDP 流程即可。

### 11.3 从预测还原 MIDI

新增工具函数：

```text
src/utils/node_midi.py
```

核心函数：

```python
def midi_to_note_features(midi_obj, normalize=True) -> dict
def note_features_to_midi(pitch, pred_continuous, target_ticks_per_beat=500, target_tempo=120) -> MidiFile
```

还原逻辑：

```python
ioi_ms = denormalize_time(pred[..., 0])
duration_ms = denormalize_time(pred[..., 1])
velocity = round(pred[..., 2] * 127)
pedal = round(pred[..., 3:7] * 127)
```

然后累积 IOI 得到 onset：

```python
start[i] = sum(ioi_ms[:i + 1])
end[i] = start[i] + duration_ms[i]
```

Pitch 直接使用 score pitch。

## 12. 评估指标

第一版评估分两类。

### 12.1 Feature-level metrics

在验证集上计算：

```text
MAE_IOI_ms
MAE_Duration_ms
MAE_Velocity
MSE_Pedal
```

注意：IOI/duration 的指标应在反归一化后的 ms 空间计算。

Velocity 指标用 0-127 空间：

```python
velocity_mae = mean(abs(pred_vel * 127 - target_vel * 127))
```

Pedal 可以同时报：

```text
Pedal_MSE_norm
Pedal_MAE_CC
```

### 12.2 MIDI-level metrics

复用或扩展 [src/evaluate/evaluate.py](/home/kaititech/EPR/PianistTransformer/src/evaluate/evaluate.py)：

```text
velocity distribution distance
duration distribution distance
IOI distribution distance
```

同时导出若干 MIDI 做主观听感检查：

```text
data/midis/node_sft_preview/
```

建议每轮固定 20 个验证样本，便于横向比较。

## 13. Baseline 对照

第一版至少比较：

```text
Baseline A: 当前 tokenizer + PianoT5Gemma SFT
Experiment B: Hybrid node + same backbone
```

需要保持：

- 同一 PianoCoRe subset。
- 同一 train/test split。
- 同一 note window 大小约 512 notes。
- 尽量接近的 effective batch size。

当前 tokenizer `block_size=4096` 对应 512 notes；node 方案使用 `block_notes=512`。

## 14. 最小实现清单

需要新增：

```text
src/utils/node_midi.py
src/data_process/06_generate_sft_node_data_pianocore.py
src/model/hybrid_pianoformer.py
src/train/sft_nodes.py
configs/sft_node_config_pianocore.json
```

建议暂不修改：

```text
src/model/pianoformer.py
src/train/sft.py
src/utils/midi.py
```

这样不会影响现有 tokenizer baseline。

## 15. 验收标准

数据阶段：

- `refined_score_note_count == refined_performance_note_count` 的 pair 全部可处理。
- 每个 segment 只保存一份 `pitch`，并且来自 `score_pitch == performance_pitch` 的 pair。
- 每个 segment 的 `score_continuous`、`label_continuous` shape 一致。
- `pitch` 在 `[0, 127]`，padding 仅在 collator 出现。
- continuous 字段均在 `[0, 1]`。

模型阶段：

- 单 batch forward 正常。
- `continuous_pred.shape == labels_continuous.shape`。
- loss 为 finite。
- attention mask 后 padding 不参与 loss。

训练阶段：

- 能跑通 100 step smoke test。
- eval loss 能正常计算。
- 能从预测输出还原 MIDI。

实验阶段：

- 和 tokenizer baseline 在同一验证集上比较 feature-level metrics。
- 至少保存 20 个固定验证样本的 MIDI 输出。

## 16. 后续实验方向

第一版稳定后，再做以下 ablation：

1. `MAX_TIME_MS=5000` vs `10000` vs `20000`。
2. `log normalization` vs linear normalization。
3. 预测 absolute IOI/duration vs 预测 score-relative ratio。
4. score velocity 保留 vs 置零。
5. interpolated notes loss weight `1.0` vs `0.5` vs masked。
6. 加 pitch auxiliary CE head。
7. 迁移旧 PianistTransformer checkpoint 的 backbone 权重，只随机初始化 note encoder/decoder。

## 17. 第一版结论预期

如果实验成立，应该看到：

- 序列长度减少为原来的 1/8。
- 不再有 timing token clip 到约 5 秒的问题。
- velocity、duration、pedal 的预测误差更自然，尤其是短时值和 pedal 连续变化。
- Decoder 不再需要学习 8-token 局部格式语法，训练目标更贴近 EPR。

这将支持论文中的核心表述：EPR 更适合作为 note-level conditional structured prediction，而不是 token-level language modeling。
