# PT / INR Slot-Attribute Representation Design

## 1. Motivation

当前 EPR 模型采用：

```text
1 note = 1 timestep
```

该设计能够显著缩短序列长度，避免将一个音符拆分成多个离散 token 所带来的长序列问题。但如果将 pitch、score timing、performance timing、velocity、pedal 以及 musical attributes 过早压缩为一个无结构的连续向量，可能产生以下问题：

1. 不同属性的统计尺度和语义角色不同，却在输入端过早混合；
2. score condition 与 performance feedback 缺少明确边界；
3. 某一 performance channel 的闭环误差可能通过共享 embedding 污染其他属性；
4. 不便于进行 feature-level masking、feedback source 标记和单通道消融；
5. 难以区分 absolute performance generation 与 relative score-performance deviation generation；
6. 当 decoder 使用自身预测作为下一步输入时，错误的 performance history 可能影响整个 note embedding，进而加剧 autoregressive distribution shift。

因此，本设计采用 **slot-attribute representation**：

```text
attribute values
    -> role-specific slot encoders
    -> 128-dim slot embeddings
    -> concatenate slots
    -> schema-shared SlotFusionMLP
    -> 768-dim note embedding
    -> Transformer
    -> factorized decoder heads
```

整体原则为：

```text
structured input
+ unified Transformer hidden
+ structured output
```

本设计不对 Transformer hidden state 做硬分区，也不改变 `1 note = 1 timestep` 的基本结构。

---

# 2. Core Design Principles

## 2.1 One note per timestep

每个时间步仍对应一个音符：

```text
note_t -> one 768-dim note embedding
```

slot 仅存在于 note embedding 的构造阶段，不会将一个音符展开成多个序列 token。

因此，该设计同时保留：

```text
note-level autoregressive modeling
+ attribute-level input structure
```

与 Pianist Transformer 中一个音符对应多个 token 的形式不同，本设计不会因为 slot 数增加而扩大 Transformer sequence length。

---

## 2.2 Unified slot dimension

所有属性 slot 统一使用：

```text
slot_dim = 128
d_model  = 768
```

无论 schema 包含 5、8、9 或 12 个 slot，每个 slot 的容量都保持为 128 维。

这样做具有以下优势：

1. 不同 schema 中同一属性的表示容量保持一致；
2. 不会因为 slot 数量增加而压缩每个 slot 的维度；
3. 便于复用属性编码器；
4. 便于添加或移除 musical slots；
5. 便于进行干净的 slot-level ablation；
6. slot representation 不需要强制拼接后恰好等于 `d_model`。

因此，不采用：

```text
5 slots  -> each slot approximately 153 dimensions
8 slots  -> each slot 96 dimensions
12 slots -> each slot 64 dimensions
```

而统一采用：

```text
each slot -> 128 dimensions
concat all slots
-> SlotFusionMLP
-> 768 dimensions
```

---

## 2.3 Role-specific attribute encoders

score attribute 与 performance attribute 即使表示同一种物理量，也不使用同一个 value encoder。

例如：

\[
e^{s}_{ioi}
=
f^{s}_{ioi}(x^{s}_{ioi})
\]

\[
e^{p}_{ioi}
=
f^{p}_{ioi}(x^{p}_{ioi})
\]

\[
e^{s}_{dur}
=
f^{s}_{dur}(x^{s}_{dur})
\]

\[
e^{p}_{dur}
=
f^{p}_{dur}(x^{p}_{dur})
\]

\[
e^{s}_{vel}
=
f^{s}_{vel}(x^{s}_{vel})
\]

\[
e^{p}_{vel}
=
f^{p}_{vel}(x^{p}_{vel})
\]

其中，每个 encoder 输出：

\[
e \in \mathbb{R}^{128}
\]

具体实现中使用：

```text
ScoreIOIEncoder      != PerfIOIEncoder
ScoreDurationEncoder != PerfDurationEncoder
ScoreVelocityEncoder != PerfVelocityEncoder
```

原因是 score-side 与 performance-side 属性具有不同角色。

### Score-side attributes

score IOI、duration 和 velocity 主要表示：

```text
symbolic condition
metrical structure
notated duration
score-side dynamic information
```

### Performance-side attributes

performance IOI、duration 和 velocity 主要表示：

```text
actual performance realization
autoregressive feedback
sampled model output
closed-loop state
```

此外，performance feedback 在不同阶段还可能来自：

```text
GT feedback
sampled feedback
mean / greedy feedback
masked feedback
BOS state
PAD state
```

因此，即使 score IOI 和 performance IOI 使用相同的 raw/log 数值格式，它们也不应通过完全相同的 MLP 映射到 slot embedding。

---

## 2.4 Shared FusionMLP within the same schema

虽然 score-side 和 performance-side 使用不同的 attribute encoder，但同一 schema 内部共享 note-level FusionMLP。

以 PT 5-slot schema 为例：

```text
Score note:
PitchEncoder
ScoreIOIEncoder
ScoreDurationEncoder
ScoreVelocityEncoder
NULLScorePedalEmbedding
        ↓
     shared Fusion5

Performance note:
PitchEncoder
PerfIOIEncoder
PerfDurationEncoder
PerfVelocityEncoder
PerfPedalEncoder
        ↓
     shared Fusion5
```

FusionMLP 的作用不是区分 score domain 与 performance domain，而是将已经完成 role-specific encoding 的 slot embeddings 聚合为统一的 note representation。

因此，整体原则为：

```text
domain-specific attribute encoding
+ schema-shared note fusion
```

即：

```text
different attribute encoders
shared slot-combination mechanism
```

---

## 2.5 PT and INR use different FusionMLPs

PT 与 INR 的 slot schema 不同，因此不能直接共享同一个 FusionMLP。

```text
PT without musical:
5 slots -> Fusion5

INR without musical:
8 slots -> Fusion8

PT with musical:
9 slots -> Fusion9

INR with musical:
12 slots -> Fusion12
```

对应输入维度为：

| Schema | Number of slots | Concat dimension | Output dimension |
|---|---:|---:|---:|
| A-5 | 5 | 640 | 768 |
| B-8 | 8 | 1024 | 768 |
| A-9 | 9 | 1152 | 768 |
| B-12 | 12 | 1536 | 768 |

---

## 2.6 Unified Transformer hidden representation

Transformer 内部仍然使用统一的 768 维 hidden state：

```text
attribute slots
-> note embedding
-> Transformer hidden
```

不采用显式 block-wise hidden：

```text
h = [h_pitch, h_timing, h_velocity, h_pedal]
```

因为 hidden partition 会额外引入以下问题：

1. 每个 hidden block 应分配多少维；
2. 不同 block 之间如何交换信息；
3. attention 是否需要分块；
4. FFN 是否需要分块；
5. cross-attention 中不同属性如何对齐；
6. 不同任务如何共享 hidden representation。

当前阶段只在输入和输出端保留属性结构：

```text
structured input
unified middle
structured output
```

---

# 3. Two Main Modeling Schemas

本设计比较两种完整的 EPR 建模范式：

```text
A. PT Slot:
   unified score/performance slot positions
   + absolute performance generation

B. INR Slot:
   separate score/performance slots
   + relative timing generation
```

两者都使用：

```text
encoder input = full score sequence
```

两者都要求 decoder input 始终是 **previous-note representation**。

禁止使用：

```text
[x_t, y_{t-1}]
```

因为这会把 current score note 与 previous performance note 混入同一个 note embedding，破坏 note index 的一致性。

---

# 4. Schema A: PT Slot Absolute Performance Generation

## 4.1 Formal definition

设 score sequence 为：

\[
x_1,x_2,\ldots,x_T
\]

performance sequence 为：

\[
y_1,y_2,\ldots,y_T
\]

PT Slot 使用：

\[
\text{Encoder input}
=
x_{1:T}
\]

\[
\text{Decoder input at step }t
=
y_{t-1}
\]

\[
\text{Target}
=
y_t
\]

即：

\[
p(y_t\mid y_{<t},x_{1:T})
\]

current score note 以及全局 score context 通过 encoder-decoder cross-attention 提供。

---

## 4.2 A-5: PT Slot without musical attributes

### 4.2.1 Slot schema

```text
[pitch, ioi, duration, velocity, pedal]
```

每个 slot 为 128 维：

```text
concat dimension = 5 × 128 = 640
```

然后：

```text
640
-> Fusion5
-> 768-dim note embedding
```

---

## 4.2.2 Encoder score note

encoder 输入 score note：

```text
[pitch_t,
 score_ioi_t,
 score_duration_t,
 score_velocity_t,
 NULL_score_pedal]
```

这里 pedal 并非被随机 mask，而是 score domain 中本来不存在 performance pedal。

因此，使用：

```text
NULL_score_pedal
```

而不是：

```text
MASK_pedal
```

二者语义不同：

```text
NULL:
attribute does not exist in this domain

MASK:
attribute exists but is intentionally hidden
```

---

## 4.2.3 Decoder performance note

训练时，decoder 输入 previous performance note：

```text
[pitch_{t-1},
 perf_ioi_{t-1},
 perf_duration_{t-1},
 perf_velocity_{t-1},
 perf_pedal_{t-1}]
```

其中 pitch 是 score-derived identity 和 alignment anchor，不需要由模型自由生成。

推理时，performance slots 来自上一步模型输出：

```text
[pitch_{t-1},
 predicted_perf_ioi_{t-1},
 predicted_perf_duration_{t-1},
 predicted_perf_velocity_{t-1},
 predicted_perf_pedal_{t-1}]
```

---

## 4.2.4 Target

模型预测 current performance note：

```text
perf_ioi_t
perf_duration_t
perf_velocity_t
perf_pedal_t
```

pitch 不作为自由生成目标，而由 score sequence 提供 hard constraint。

---

## 4.3 PT slot encoding

### Pitch

```text
pitch_id
-> PitchEmbedding
-> 128
```

或者对于 piano multihot：

```text
pitch_multihot
-> Linear / MLP
-> 128
```

### Score IOI

```text
score_ioi_raw_s = score_ioi_ms / 1000
score_ioi_log   = log1p(score_ioi_ms / 50)
```

```text
[score_ioi_raw_s, score_ioi_log]
-> ScoreIOIEncoder
-> 128
```

### Performance IOI

```text
perf_ioi_raw_s = perf_ioi_ms / 1000
perf_ioi_log   = log1p(perf_ioi_ms / 50)
```

```text
[perf_ioi_raw_s, perf_ioi_log]
-> PerfIOIEncoder
-> 128
```

### Score duration

```text
score_dur_raw_s = score_duration_ms / 1000
score_dur_log   = log1p(score_duration_ms / 50)
```

```text
[score_dur_raw_s, score_dur_log]
-> ScoreDurationEncoder
-> 128
```

### Performance duration

```text
perf_dur_raw_s = perf_duration_ms / 1000
perf_dur_log   = log1p(perf_duration_ms / 50)
```

```text
[perf_dur_raw_s, perf_dur_log]
-> PerfDurationEncoder
-> 128
```

### Score velocity

```text
score_velocity_norm = score_velocity / 127
```

```text
score_velocity_norm
-> ScoreVelocityEncoder
-> 128
```

### Performance velocity

```text
perf_velocity_norm = perf_velocity / 127
```

```text
perf_velocity_norm
-> PerfVelocityEncoder
-> 128
```

### Performance pedal

例如 pedal4：

```text
pedal4 = [
  pedal_0,
  pedal_25,
  pedal_50,
  pedal_75
]
```

```text
pedal4_norm
-> PerfPedalEncoder
-> 128
```

---

## 4.4 PT absolute timing target

PT timing 使用 absolute performance value。

定义 logscale function：

\[
g(v)
=
\log\left(1+\frac{v}{c}\right)
\]

建议：

```text
c = 50 ms
```

则 IOI target 为：

\[
y^{log}_{ioi,t}
=
g(\text{perf\_ioi}_{t})
\]

duration target 为：

\[
y^{log}_{dur,t}
=
g(\text{perf\_duration}_{t})
\]

模型预测：

\[
p(y^{log}_{ioi,t}\mid h_t)
\]

\[
p(y^{log}_{dur,t}\mid h_t)
\]

推理时，从预测分布获得：

\[
\hat y^{log}_t
\]

然后通过：

\[
g^{-1}(u)
=
c(e^u-1)
\]

还原为 absolute performance timing：

\[
\hat y_t
=
g^{-1}(\hat y^{log}_t)
\]

PT 虽然直接预测 absolute timing，但模型仍然可以通过 cross-attention 学习：

\[
y_t
=
x_t+\Delta_t
\]

区别在于，这一 score-performance decomposition 由模型隐式学习，而不是通过 target parameterization 显式规定。

---

# 5. Schema B: INR Slot Relative Timing Generation

## 5.1 Formal definition

INR Slot 的 decoder 输入为 previous paired note：

\[
r_{t-1}
=
[x_{t-1},y_{t-1}]
\]

模型使用：

\[
\text{Encoder input}
=
x_{1:T}
\]

\[
\text{Decoder input at step }t
=
r_{t-1}
\]

\[
\text{Timing target}
=
\operatorname{dev}(y_t,x_t)
\]

因此：

\[
p(\operatorname{dev}_t\mid r_{<t},x_{1:T})
\]

previous paired INR 使模型可以从同一个 note embedding 中提取 previous score-performance relationship：

\[
\operatorname{dev}_{t-1}
=
y_{t-1}-x_{t-1}
\]

---

## 5.2 B-8: INR Slot without musical attributes

### 5.2.1 Slot schema

```text
[pitch,
 score_ioi,
 score_duration,
 score_velocity,
 perf_ioi,
 perf_duration,
 perf_velocity,
 perf_pedal]
```

每个 slot 为 128 维：

```text
concat dimension = 8 × 128 = 1024
```

然后：

```text
1024
-> Fusion8
-> 768-dim note embedding
```

---

## 5.2.2 Encoder score note

encoder 中只存在 score condition，因此 performance slots 使用结构性 NULL embedding：

```text
[pitch_t,
 score_ioi_t,
 score_duration_t,
 score_velocity_t,
 NULL_perf_ioi,
 NULL_perf_duration,
 NULL_perf_velocity,
 NULL_perf_pedal]
```

这些 NULL state 与随机 mask 明确区分。

---

## 5.2.3 Decoder paired note

decoder 输入 previous paired INR：

```text
[pitch_{t-1},
 score_ioi_{t-1},
 score_duration_{t-1},
 score_velocity_{t-1},
 perf_ioi_{t-1},
 perf_duration_{t-1},
 perf_velocity_{t-1},
 perf_pedal_{t-1}]
```

所有属性必须来自同一个 previous note index。

禁止：

```text
[pitch_t,
 score_ioi_t,
 score_duration_t,
 score_velocity_t,
 perf_ioi_{t-1},
 perf_duration_{t-1},
 perf_velocity_{t-1},
 perf_pedal_{t-1}]
```

即禁止：

```text
[x_t, y_{t-1}]
```

因为这不是完整的 current note，也不是完整的 previous note，而是 mixed-index representation。

---

## 5.3 INR timing main target

INR timing 只使用一个主预测变量：

```text
logscale deviation
```

定义：

\[
g(v)
=
\log\left(1+\frac{v}{c}\right)
\]

建议：

```text
c = 50 ms
```

### IOI target

\[
d^{log}_{ioi,t}
=
g(\text{perf\_ioi}_{t})
-
g(\text{score\_ioi}_{t})
\]

### Duration target

\[
d^{log}_{dur,t}
=
g(\text{perf\_duration}_{t})
-
g(\text{score\_duration}_{t})
\]

模型预测：

\[
p(d^{log}_{ioi,t}\mid h_t)
\]

\[
p(d^{log}_{dur,t}\mid h_t)
\]

主损失为 distribution NLL：

\[
L^{ioi}_{log}
=
-\log p(d^{log,*}_{ioi,t}\mid h_t)
\]

\[
L^{dur}_{log}
=
-\log p(d^{log,*}_{dur,t}\mid h_t)
\]

第一版可以继续使用当前已实现的 skew-normal distribution。

后续可比较：

```text
normal
student-t
skew-normal
mixture normal
mixture skew-normal
student-t mixture
```

---

## 5.4 Reconstructing absolute timing

模型真正生成和反馈的仍然是 absolute performance timing。

首先，从预测分布获得确定性或采样的 log deviation：

\[
\hat d^{log}_t
\]

然后加入 current score baseline：

\[
\hat z_t
=
g(x_t)+\hat d^{log}_t
\]

再进行反变换：

\[
\hat y_t
=
g^{-1}(\hat z_t)
\]

其中：

\[
g^{-1}(u)
=
c(e^u-1)
\]

因此完整路径为：

```text
predicted logscale deviation
+ current score logscale timing
-> predicted performance logscale timing
-> inverse logscale
-> absolute performance timing
```

下一步 decoder feedback 使用：

```text
predicted absolute performance timing
```

而不是直接使用 deviation。

---

## 5.5 Raw-space regression loss

INR8-Dev 保留 timing 的双表示，并使用两个独立输出分支：

```text
logscale deviation:
  skew-normal distribution head

raw deviation:
  direct regression head
```

真实 raw deviation 为：

\[
d^{raw,*}_t
=
\frac{y_t-x_t}{1000}
\]

预测 raw deviation \(\hat d^{raw}_t\) 直接来自 raw regression head，
不由 logscale prediction 确定性重建。

raw regression loss 使用：

\[
L_{raw}
=
\operatorname{SmoothL1}
\left(
\hat d^{raw}_t,
d^{raw,*}_t
\right)
\]

因此 IOI timing loss 为：

\[
L_{ioi}
=
L^{ioi}_{log}
+
\lambda^{ioi}_{raw}
L^{ioi}_{raw}
\]

duration timing loss 为：

\[
L_{dur}
=
L^{dur}_{log}
+
\lambda^{dur}_{raw}
L^{dur}_{raw}
\]

第一版可设置：

```text
lambda_raw = 0.25
```

并进行：

```text
0.10
0.25
0.50
```

的消融。

两个损失的作用分别为：

```text
logscale distribution NLL:
  model the compressed long-tail distribution

raw-space SmoothL1:
  constrain raw timing deviation error
```

当前 INR8-Dev 不加入额外 consistency loss；两个 timing 表示由各自的
监督目标共同训练。

---

## 5.6 Raw regression estimate

raw branch 是确定性的 point regression，因此训练和推理都直接使用该
head 的输出，不从 skew-normal 分布随机采样。

INR8-Dev 的 9 维 materialized prediction 为：

```text
[log_ioi_dev, log_duration_dev,
 raw_ioi_dev, raw_duration_dev,
 velocity, pedal_0, pedal_25, pedal_50, pedal_75]
```

如果 raw branch 改为概率分布，raw loss 不应默认基于随机 sample 计算，否则会增加训练梯度方差。

推荐使用分布的确定性中心估计。

对于 normal distribution：

\[
\hat d^{log}
=
\mu
\]

对于 skew-normal distribution，可以选择：

### Option 1: location parameter

\[
\hat d^{log}
=
\xi
\]

### Option 2: distribution mean

若：

\[
\delta
=
\frac{\alpha}{\sqrt{1+\alpha^2}}
\]

则 skew-normal mean 为：

\[
\mathbb{E}[D]
=
\xi
+
\omega\delta\sqrt{\frac{2}{\pi}}
\]

这种可选设计中，推荐 raw loss 基于 distribution mean 计算：

\[
\hat d^{log}
=
\mathbb{E}[D]
\]

这样：

```text
NLL learns the full distribution
raw auxiliary loss constrains the predicted distribution center
sampling remains independent during inference
```

---

## 5.7 Optional consistency loss

INR8-Dev 已使用独立的非概率 raw regression head：

\[
\hat d^{raw,aux}_t
=
q_{raw}(h_t)
\]

则可加入：

\[
L^{aux}_{raw}
=
\operatorname{SmoothL1}
\left(
\hat d^{raw,aux}_t,
d^{raw,*}_t
\right)
\]

如果后续希望约束 raw head 与 log branch reconstruction 一致，可加入：

\[
L_{cons}
=
\left|
\hat d^{raw,aux}_t
-
\frac{
g^{-1}
\left(
g(x_t)+\hat d^{log}_t
\right)
-x_t
}{1000}
\right|
\]

总损失为：

\[
L
=
L_{log}
+
\lambda_{raw}L^{aux}_{raw}
+
\lambda_{cons}L_{cons}
\]

该 consistency loss 只作为后续可选实验，当前 INR8-Dev 不启用。

---

## 5.8 Velocity target

INR 第一版只对 timing 使用 relative target。

velocity 仍预测 absolute performance velocity：

\[
y^{vel}_t
=
\frac{\text{perf\_velocity}_t}{127}
\]

原因是：

1. score velocity 可能只是 MIDI score 默认值；
2. score velocity 未必等价于可靠的 notated dynamics；
3. velocity residual 不一定像 timing residual 一样具有稳定物理意义；
4. 直接预测 absolute performance velocity 更容易与 PT schema 对齐。

后续可以消融：

\[
d^{vel}_t
=
\frac{
\text{perf\_velocity}_t
-
\text{score\_velocity}_t
}{127}
\]

但不作为第一版默认设置。

---

## 5.9 Pedal target

pedal 没有稳定的 score baseline，因此始终预测 absolute performance pedal：

```text
target = perf_pedal4
```

可使用：

```text
BCEWithLogitsLoss
```

或者对连续 half-pedal value 使用 distributional regression。

---

# 6. Musical Slot Extension

Musical attributes 属于 score-derived symbolic information。

其角色与 pitch 相似：

```text
observed symbolic structure
not generated performance feedback
```

因此，它们可以同时出现在 encoder 和 decoder input 中。

---

## 6.1 Musical slot definitions

推荐四类 musical slots：

```text
musical_onset
musical_duration
musical_length
musical_binary
```

每个 slot 均输出 128 维。

---

## 6.2 Musical onset

表示 symbolic onset position，例如：

```text
beat position
measure position
subdivision class
metrical onset category
```

编码方式：

```text
musical_onset_id
-> Embedding(num_onset_classes, 128)
```

---

## 6.3 Musical duration

表示 notated duration class，例如：

```text
whole
half
quarter
eighth
sixteenth
dotted duration
triplet duration
other quantized class
```

编码方式：

```text
musical_duration_id
-> Embedding(num_duration_classes, 128)
```

---

## 6.4 Musical length

表示结构长度或位置，例如：

```text
phrase length bucket
group length bucket
position in phrase
position in group
first / middle / last
boundary category
```

编码方式：

```text
musical_length_id
-> Embedding(num_length_classes, 128)
```

---

## 6.5 Musical binary

表示多个 binary symbolic annotations，例如：

```text
is_downbeat
is_phrase_first
is_phrase_last
is_group_boundary
is_chord_continuation
is_grace
is_tie
is_slur
staff_id
hand indicator
```

编码方式：

```text
binary_flags
-> Linear / MLP
-> 128
```

---

# 7. A-9: PT Slot with Musical Attributes

## 7.1 Slot schema

```text
[pitch,
 ioi,
 duration,
 velocity,
 pedal,
 musical_onset,
 musical_duration,
 musical_length,
 musical_binary]
```

每个 slot 为 128 维：

```text
concat dimension = 9 × 128 = 1152
```

然后：

```text
1152
-> Fusion9
-> 768
```

---

## 7.2 Encoder score note

```text
[pitch_t,
 score_ioi_t,
 score_duration_t,
 score_velocity_t,
 NULL_score_pedal,
 musical_onset_t,
 musical_duration_t,
 musical_length_t,
 musical_binary_t]
```

---

## 7.3 Decoder performance note

```text
[pitch_{t-1},
 perf_ioi_{t-1},
 perf_duration_{t-1},
 perf_velocity_{t-1},
 perf_pedal_{t-1},
 musical_onset_{t-1},
 musical_duration_{t-1},
 musical_length_{t-1},
 musical_binary_{t-1}]
```

musical slots 使用 previous note 的 score-side structure，与 previous pitch index 保持一致。

---

# 8. B-12: INR Slot with Musical Attributes

## 8.1 Slot schema

```text
[pitch,
 score_ioi,
 score_duration,
 score_velocity,
 perf_ioi,
 perf_duration,
 perf_velocity,
 perf_pedal,
 musical_onset,
 musical_duration,
 musical_length,
 musical_binary]
```

每个 slot 为 128 维：

```text
concat dimension = 12 × 128 = 1536
```

然后：

```text
1536
-> Fusion12
-> 768
```

---

## 8.2 Encoder score note

```text
[pitch_t,
 score_ioi_t,
 score_duration_t,
 score_velocity_t,
 NULL_perf_ioi,
 NULL_perf_duration,
 NULL_perf_velocity,
 NULL_perf_pedal,
 musical_onset_t,
 musical_duration_t,
 musical_length_t,
 musical_binary_t]
```

---

## 8.3 Decoder paired note

```text
[pitch_{t-1},
 score_ioi_{t-1},
 score_duration_{t-1},
 score_velocity_{t-1},
 perf_ioi_{t-1},
 perf_duration_{t-1},
 perf_velocity_{t-1},
 perf_pedal_{t-1},
 musical_onset_{t-1},
 musical_duration_{t-1},
 musical_length_{t-1},
 musical_binary_{t-1}]
```

---

# 9. NULL, MASK, BOS, PAD and Source Types

## 9.1 Structural NULL

结构性不存在的属性使用 NULL state。

例如：

```text
NULL_score_pedal
NULL_perf_ioi
NULL_perf_duration
NULL_perf_velocity
NULL_perf_pedal
```

NULL 表示：

```text
this attribute is not defined in the current representation domain
```

---

## 9.2 Feature MASK

训练时人为遮盖一个原本存在的属性，使用 feature-specific MASK：

```text
MASK_perf_ioi
MASK_perf_duration
MASK_perf_velocity
MASK_perf_pedal
```

MASK 表示：

```text
the attribute exists but its value is intentionally hidden
```

不能仅用 numeric zero 表示 mask，因为 zero 可能是合法值。

---

## 9.3 BOS

sequence 起始处没有 previous performance note，因此使用：

```text
BOS_perf_ioi
BOS_perf_duration
BOS_perf_velocity
BOS_perf_pedal
```

或者使用统一 BOS source embedding 加各 slot 的 BOS value embedding。

---

## 9.4 PAD

padding timestep 使用：

```text
PAD_pitch
PAD_ioi
PAD_duration
PAD_velocity
PAD_pedal
```

并由 attention mask 阻止模型使用 padding position。

---

## 9.5 Feedback source embedding

performance slot 应包含 feedback source information。

source type 可定义为：

```text
GT
SAMPLED
MEAN
GREEDY
MASKED
BOS
PAD
```

推荐实现：

\[
e^{perf}_{slot}
=
f^{perf}_{slot}(value)
+
e_{source}
\]

例如：

```text
PerfIOIEncoder(value)
+ FeedbackSourceEmbedding(source_type)
```

如果属性被 mask：

```text
MASKPerfIOIEmbedding
+ FeedbackSourceEmbedding(MASKED)
```

这样模型能够区分：

```text
correct GT history
model-generated history
deterministic mean feedback
missing feedback
sequence boundary
```

---

# 10. SlotFusionMLP

## 10.1 Recommended architecture

各 schema 使用：

```text
LayerNorm(N × 128)
Linear(N × 128 -> 1536)
GELU or SiLU
Linear(1536 -> 768)
LayerNorm(768)
```

其中：

```text
N = 5, 8, 9 or 12
```

对应：

```text
Fusion5:
640 -> 1536 -> 768

Fusion8:
1024 -> 1536 -> 768

Fusion9:
1152 -> 1536 -> 768

Fusion12:
1536 -> 1536 -> 768
```

该参数量相对于完整 Transformer 和 factorized decoder heads 不大。

特别是 decoder head 本身可能采用：

```text
d
-> 2d
-> d
-> d/2
-> output
```

以：

```text
d = 768
```

计算，仅 trunk 部分就包含数百万参数。因此，FusionMLP 使用 1536 hidden dimension 是可接受的。

---

## 10.2 Why nonlinear fusion is needed

slot concat 后不建议仅做简单求和：

```text
sum(slot_embeddings)
```

因为不同属性之间存在交互，例如：

```text
score IOI × performance IOI
score duration × performance duration
velocity × pedal
metrical position × timing deviation
pitch × duration
```

FusionMLP 允许模型在压缩为 768 维之前学习 note-level cross-slot interactions。

---

## 10.3 Optional slot gates

可选地对每个 slot 加入 learnable gate：

\[
\tilde e_i
=
\alpha_i e_i
\]

其中：

\[
\alpha_i
=
\sigma(a_i)
\]

gate 可以是：

```text
scalar gate per slot
```

或者：

```text
128-dim vector gate per slot
```

第一版不强制使用 gate。

后续 gate 可用于分析模型是否实际使用某些属性，尤其是：

```text
score velocity
musical length
musical binary
pedal history
```

---

# 11. Encoder, Decoder and Cross-Attention

## 11.1 Encoder role

encoder 始终接收完整 score sequence：

```text
x_1, x_2, ..., x_T
```

encoder 提供：

```text
current score note
future score context
global score structure
metrical information
phrase-level information
alignment anchor
```

---

## 11.2 Decoder role in PT

PT decoder history 为：

\[
y_{<t}
\]

其主要作用是提供：

```text
previous performance timing
previous dynamics
previous articulation
previous pedal state
```

current score information由 cross-attention 提供。

---

## 11.3 Decoder role in INR

INR decoder history 为：

\[
[x_{<t},y_{<t}]
\]

其主要作用是提供：

```text
previous score-performance pair
previous local timing deviation
previous dynamics realization
previous expressive state
```

current score information仍由 cross-attention 提供。

---

## 11.4 No mixed-index representation

无论 PT 还是 INR，decoder input 都不能使用：

\[
[x_t,y_{t-1}]
\]

因为：

```text
x_t belongs to the current note
y_{t-1} belongs to the previous note
```

将二者压入同一个 note embedding 会导致 note identity 不明确。

正确形式为：

```text
PT:
y_{t-1}

INR:
[x_{t-1}, y_{t-1}]
```

---

# 12. Factorized Decoder Heads

decoder output 不使用一个单一 giant projection 同时预测所有参数。

推荐按输出 family 拆分：

```text
decoder hidden h_t
├── IOI trunk
│   └── logscale timing distribution
├── Duration trunk
│   └── logscale timing distribution
├── Velocity trunk
│   └── velocity distribution
└── Pedal trunk
    └── pedal4 logits or distribution
```

---

## 12.1 Timing trunk

推荐结构：

```text
768
-> 1536
-> 768
-> 384
-> distribution parameters
```

例如：

```text
Linear(768, 1536)
GELU / SiLU
LayerNorm
Linear(1536, 768)
GELU / SiLU
Linear(768, 384)
GELU / SiLU
Linear(384, output_dim)
```

IOI 与 duration 可使用独立 trunk。

---

## 12.2 Velocity trunk

```text
768
-> 1536
-> 768
-> 384
-> velocity distribution parameters
```

可使用：

```text
normal
beta
logistic-normal
mixture distribution
```

具体取决于 velocity normalization 与边界处理方式。

---

## 12.3 Pedal trunk

对于 pedal4：

```text
768
-> 1536
-> 768
-> 384
-> 4 logits
```

损失：

\[
L_{pedal}
=
\operatorname{BCEWithLogitsLoss}
\]

若 pedal4 不是四个独立 binary labels，而是单一 ordinal pedal state，则应改用 categorical or ordinal loss。

---

# 13. Total Loss

## 13.1 PT loss

PT 总损失可写为：

\[
L_{PT}
=
\lambda_{ioi}L^{abs}_{ioi}
+
\lambda_{dur}L^{abs}_{dur}
+
\lambda_{vel}L_{vel}
+
\lambda_{ped}L_{ped}
\]

其中：

```text
L_ioi_abs:
absolute logscale IOI distribution NLL

L_dur_abs:
absolute logscale duration distribution NLL
```

---

## 13.2 INR loss

INR 总损失可写为：

\[
L_{INR}
=
\lambda_{ioi}L_{ioi}
+
\lambda_{dur}L_{dur}
+
\lambda_{vel}L_{vel}
+
\lambda_{ped}L_{ped}
\]

其中：

\[
L_{ioi}
=
L^{ioi}_{log}
+
\lambda^{ioi}_{raw}L^{ioi}_{raw}
\]

\[
L_{dur}
=
L^{dur}_{log}
+
\lambda^{dur}_{raw}L^{dur}_{raw}
\]

因此：

\[
L_{INR}
=
\lambda_{ioi}
\left(
L^{ioi}_{log}
+
\lambda^{ioi}_{raw}L^{ioi}_{raw}
\right)
+
\lambda_{dur}
\left(
L^{dur}_{log}
+
\lambda^{dur}_{raw}L^{dur}_{raw}
\right)
+
\lambda_{vel}L_{vel}
+
\lambda_{ped}L_{ped}
\]

第一版可先设置：

```text
lambda_ioi = 1.0
lambda_dur = 1.0
lambda_vel = 1.0
lambda_ped = 1.0

lambda_raw_ioi = 0.25
lambda_raw_dur = 0.25
```

之后根据各损失尺度进行调整。

---

# 14. Main Experimental Plan

## 14.1 Stage 1: no musical attributes

首先运行两套主 schema：

| Run | Schema | Decoder input | Timing target | Musical |
|---|---|---|---|---|
| A5 | PT Slot | previous performance | absolute logscale | No |
| B8 | INR Slot | previous paired INR | logscale deviation + raw auxiliary | No |

这一比较回答：

```text
哪一套完整建模系统在当前 EPR 任务上更稳定？
```

但 A5 与 B8 同时改变了：

1. decoder history representation；
2. score/performance slot organization；
3. absolute vs relative timing target。

因此，不能仅通过 A5 与 B8 得出：

```text
absolute target is better than residual target
```

或：

```text
unified slots are better than separate slots
```

它们首先是两套完整系统的比较。

---

## 14.2 Stage 1.5: bridge ablations

为了分离 paired history 与 relative target 的作用，建议构造 \(2\times2\) 实验：

| Run | Decoder input | Timing target |
|---|---|---|
| U-Abs | previous performance | absolute timing |
| U-Dev | previous performance | relative timing |
| P-Abs | previous paired score/performance | absolute timing |
| P-Dev | previous paired score/performance | relative timing |

其中：

```text
U-Abs = A5
P-Dev = B8
```

桥接实验为：

### U-Dev

```text
decoder input:
previous performance

target:
current timing deviation
```

它主要隔离：

```text
relative target contribution
```

### P-Abs

```text
decoder input:
previous paired score-performance INR

target:
current absolute performance timing
```

它主要隔离：

```text
paired INR history contribution
```

如果完整训练成本较高，可以先使用：

```text
smaller dataset
shorter schedule
fewer epochs
fixed random seed
```

进行诊断。

---

## 14.3 Stage 2: musical attributes

在 no-musical schema 比较完成后，运行：

```text
A9
or
B12
```

同时建议进行 musical group ablation。

### Symbolic timing group

```text
musical_onset
musical_duration
```

### Structural group

```text
musical_length
musical_binary
```

实验组合：

```text
no musical
symbolic timing only
structural only
all musical
```

这样可以区分：

1. musical improvement 是否只是因为增加了 categorical timing encoding；
2. phrase/group structure 是否真正提供了额外信息。

---

## 14.4 Stage 3: feature-level feedback robustness

representation 确定后，再加入：

```text
feature-level slot masking
feedback source embedding
sampled feedback replacement
mean feedback replacement
```

推荐初始概率：

```text
perf_ioi_mask_prob      = 0.10
perf_duration_mask_prob = 0.10
perf_velocity_mask_prob = 0.10
perf_pedal_mask_prob    = 0.10
full_perf_mask_prob     = 0.05
```

不建议第一版直接使用：

```text
50% full performance masking
```

因为过强 masking 可能导致 decoder 学会忽略 performance history，退化为只依赖 score encoder 的 conditional mean predictor。

---

## 14.5 Stage 4: rollout-level training

只有在 representation 和 target 设计稳定后，再加入：

```text
TF vs self-history distribution matching
online feedback replacement
multi-step rollout training
DAgger-style data aggregation
stable dynamics regularization
rollout mean/std matching
```

不要在第一次 slot experiment 中同时修改：

```text
representation
target
feedback policy
rollout loss
sampling policy
```

否则难以判断提升来源。

---

# 15. Evaluation Protocol

## 15.1 k-sweep

保持统一的 rollout depth evaluation：

```text
k = 0
k = 1
k = 4
k = 8
k = 16
full AR
```

其中：

```text
k = 0:
teacher forcing

k = 1:
one-step self-feedback contamination

k > 1:
multi-step closed-loop feedback

full AR:
complete autoregressive rollout
```

---

## 15.2 Main metrics

主要指标：

```text
IOI Wasserstein distance
Duration Wasserstein distance
Velocity Wasserstein distance
Pedal Wasserstein distance
```

---

## 15.3 Additional diagnostics

额外记录：

```text
raw IOI mean shift
raw duration mean shift
raw IOI std ratio
raw duration std ratio
prediction quantiles
head mean under TF
head mean under self-history
head std under TF
head std under self-history
single-channel feedback ablation
feature-mask robustness
feedback-source robustness
```

---

## 15.4 Most important comparisons

重点观察：

```text
k0 -> k1 degradation
k1 -> k4 degradation
k4 -> k16 degradation
k0 -> full AR degradation
```

不能只看最终 full-AR Wasserstein。

如果：

```text
k0 and k1 are similar
but full AR suddenly degrades
```

则说明问题主要来自长期闭环 dynamics，而不是单步输入污染。

---

# 16. Interpretation Guide

## 16.1 A5 better than B8

可能说明：

```text
absolute generation
+ previous performance history
```

比显式 residual-state modeling 更容易优化。

但不能直接说明：

```text
relative target is useless
```

需要结合 U-Dev 和 P-Abs 判断。

---

## 16.2 B8 better than A5

可能说明以下一个或两个因素有效：

```text
previous paired score-performance state
explicit relative timing target
```

需要 bridge ablation 区分二者贡献。

---

## 16.3 k0 improves but full AR does not improve

说明 slot representation 改善了单步 conditional prediction，但没有解决：

```text
training-inference mismatch
closed-loop distribution shift
error accumulation
head calibration under self-history
```

下一步应转向 rollout training，而不是继续增加 slot complexity。

---

## 16.4 Musical attributes improve k0 only

说明 musical information有助于局部 expressive prediction，但不是长期 AR 稳定性的主要来源。

---

## 16.5 Musical duration improves timing but structural slots do not

说明提升可能主要来自：

```text
categorical symbolic timing encoding
```

而不是 phrase-level or group-level structure。

---

## 16.6 Both PT and INR degrade similarly

说明剩余问题更可能位于：

```text
closed-loop transition dynamics
distribution head sampling
feedback calibration
autoregressive exposure bias
```

而不是 note representation 本身。

---

# 17. Implementation Checklist

## 17.1 Dataset and collator

- [ ] Build A-5 tensors.
- [ ] Build B-8 tensors.
- [ ] Build A-9 tensors.
- [ ] Build B-12 tensors.
- [ ] Ensure decoder input is shifted by one note.
- [ ] Explicitly prohibit `[x_t, y_{t-1}]`.
- [ ] Add structural NULL states.
- [ ] Add feature-level MASK states.
- [ ] Add BOS states.
- [ ] Add PAD states.
- [ ] Add feedback source type.
- [ ] Verify score velocity source.
- [ ] Verify score/performance note alignment.
- [ ] Verify score baseline used in INR reconstruction is current \(x_t\).

---

## 17.2 Slot encoders

- [ ] Implement shared PitchEncoder.
- [ ] Implement ScoreIOIEncoder.
- [ ] Implement PerfIOIEncoder.
- [ ] Implement ScoreDurationEncoder.
- [ ] Implement PerfDurationEncoder.
- [ ] Implement ScoreVelocityEncoder.
- [ ] Implement PerfVelocityEncoder.
- [ ] Implement PerfPedalEncoder.
- [ ] Implement MusicalOnsetEncoder.
- [ ] Implement MusicalDurationEncoder.
- [ ] Implement MusicalLengthEncoder.
- [ ] Implement MusicalBinaryEncoder.
- [ ] Ensure every slot outputs 128 dimensions.

---

## 17.3 Fusion modules

- [ ] Implement Fusion5.
- [ ] Implement Fusion8.
- [ ] Implement Fusion9.
- [ ] Implement Fusion12.
- [ ] Share Fusion5 between PT score and performance notes.
- [ ] Share Fusion8 between INR encoder and decoder notes.
- [ ] Share Fusion9 between PT score and performance notes.
- [ ] Share Fusion12 between INR encoder and decoder notes.
- [ ] Keep PT and INR fusion parameters separate.

---

## 17.4 Decoder heads

- [ ] Implement factorized IOI trunk.
- [ ] Implement factorized duration trunk.
- [ ] Implement velocity trunk.
- [ ] Implement pedal trunk.
- [ ] Use skew-normal heads for logscale timing deviation.
- [ ] Use independent regression heads for raw timing deviation.
- [ ] Reconstruct absolute timing from INR logscale deviation.
- [ ] Compute raw-space SmoothL1 loss with weight 0.25.
- [ ] Implement optional consistency loss behind a config flag.

---

## 17.5 Feedback handling

- [ ] Add GT source embedding.
- [ ] Add sampled source embedding.
- [ ] Add mean source embedding.
- [ ] Add greedy source embedding.
- [ ] Add masked source embedding.
- [ ] Add BOS source embedding.
- [ ] Add PAD source embedding.
- [ ] Support feature-level feedback masking.
- [ ] Support full performance feedback masking.
- [ ] Ensure score-derived pitch and musical slots remain observed.

---

## 17.6 Evaluation

- [ ] Run k-sweep.
- [ ] Record per-channel Wasserstein.
- [ ] Record raw mean drift.
- [ ] Record raw standard-deviation drift.
- [ ] Compare TF and self-history head statistics.
- [ ] Run single-channel feedback ablation.
- [ ] Run bridge schema ablation.
- [ ] Run musical group ablation.
- [ ] Compare deterministic and sampled feedback.
- [ ] Record total model parameter count.

---

# 18. Recommended Configuration Sketches

## 18.1 A-5 PT Slot

```json
{
  "representation": "slot_attribute",
  "schema": "A5_pt_absolute_nomus",
  "d_model": 768,
  "slot_dim": 128,
  "num_slots": 5,
  "fusion_input_dim": 640,
  "fusion_hidden_dim": 1536,
  "fusion_output_dim": 768,
  "share_fusion_between_score_and_perf": true,
  "share_score_perf_attribute_encoders": false,
  "target_type": "absolute_logscale_performance",
  "timing_log_scale_ms": 50,
  "velocity_target": "absolute_performance",
  "pedal_target": "absolute_performance",
  "musical_feature_mode": "none",
  "factorized_decoder_heads": true,
  "slots": [
    "pitch",
    "ioi",
    "duration",
    "velocity",
    "pedal"
  ]
}
```

---

## 18.2 B-8 INR Slot

```json
{
  "representation": "slot_attribute",
  "schema": "B8_inr_logdev_nomus",
  "d_model": 768,
  "slot_dim": 128,
  "num_slots": 8,
  "fusion_input_dim": 1024,
  "fusion_hidden_dim": 1536,
  "fusion_output_dim": 768,
  "share_fusion_between_encoder_and_decoder": true,
  "share_score_perf_attribute_encoders": false,
  "target_type": "raw_log_deviation",
  "timing_log_scale_ms": 50,
  "raw_timing_head": "regression",
  "raw_timing_loss": "smooth_l1",
  "raw_timing_loss_lambda": 0.25,
  "consistency_loss": false,
  "velocity_target": "absolute_performance",
  "pedal_target": "absolute_performance",
  "musical_feature_mode": "none",
  "factorized_decoder_heads": true,
  "slots": [
    "pitch",
    "score_ioi",
    "score_duration",
    "score_velocity",
    "perf_ioi",
    "perf_duration",
    "perf_velocity",
    "perf_pedal4"
  ]
}
```

---

## 18.3 A-9 PT Slot with musical attributes

```json
{
  "representation": "slot_attribute",
  "schema": "A9_pt_absolute_musical",
  "d_model": 768,
  "slot_dim": 128,
  "num_slots": 9,
  "fusion_input_dim": 1152,
  "fusion_hidden_dim": 1536,
  "fusion_output_dim": 768,
  "share_fusion_between_score_and_perf": true,
  "share_score_perf_attribute_encoders": false,
  "target_type": "absolute_logscale_performance",
  "timing_log_scale_ms": 50,
  "velocity_target": "absolute_performance",
  "pedal_target": "absolute_performance",
  "musical_feature_mode": "categorical_slots",
  "factorized_decoder_heads": true,
  "slots": [
    "pitch",
    "ioi",
    "duration",
    "velocity",
    "pedal",
    "musical_onset",
    "musical_duration",
    "musical_length",
    "musical_binary"
  ]
}
```

---

## 18.4 B-12 INR Slot with musical attributes

```json
{
  "representation": "slot_attribute",
  "schema": "B12_inr_logdev_musical",
  "d_model": 768,
  "slot_dim": 128,
  "num_slots": 12,
  "fusion_input_dim": 1536,
  "fusion_hidden_dim": 1536,
  "fusion_output_dim": 768,
  "share_fusion_between_encoder_and_decoder": true,
  "share_score_perf_attribute_encoders": false,
  "target_type": "logscale_deviation",
  "timing_log_scale_ms": 50,
  "raw_timing_aux_loss": "smooth_l1",
  "raw_timing_aux_lambda": 0.25,
  "raw_auxiliary_head": false,
  "consistency_loss": false,
  "velocity_target": "absolute_performance",
  "pedal_target": "absolute_performance",
  "musical_feature_mode": "categorical_slots",
  "factorized_decoder_heads": true,
  "slots": [
    "pitch",
    "score_ioi",
    "score_duration",
    "score_velocity",
    "perf_ioi",
    "perf_duration",
    "perf_velocity",
    "perf_pedal4",
    "musical_onset",
    "musical_duration",
    "musical_length",
    "musical_binary"
  ]
}
```

---

# 19. Final Design Decision

最终设计如下：

```text
1. Keep one note per timestep.

2. Use 128 dimensions for every attribute slot.

3. Use separate score-side and performance-side encoders
   for IOI, duration and velocity.

4. Share the note-level FusionMLP within the same schema.

5. Keep PT and INR FusionMLPs separate.

6. PT uses a unified five-slot or nine-slot schema.

7. PT predicts absolute logscale performance timing.

8. INR uses separate score and performance slots.

9. INR8-Dev predicts logscale timing deviation with
   skew-normal distribution heads.

10. INR8-Dev also predicts raw timing deviation with
    independent regression heads.

11. Reconstruct absolute performance timing from:
    current score timing + predicted logscale deviation.

12. Use raw-space SmoothL1 with weight 0.25 for the raw heads.

13. Do not use a second raw timing distribution head.

14. Keep velocity and pedal as absolute performance targets
    in the first INR version.

15. Distinguish structural NULL, feature MASK, BOS and PAD.

16. Add feedback source embeddings for performance slots.

17. Add musical slots only after the no-musical PT/INR
    comparison is complete.

18. Use bridge experiments to separate paired-history effects
    from relative-target effects.

19. Separate representation experiments from rollout-level
    stability training.
```

核心结构可以概括为：

```text
PT Slot:
role-specific score/performance attribute encoders
-> shared Fusion5 or Fusion9
-> absolute performance prediction
```

```text
INR Slot:
separate score/performance slots and encoders
-> shared Fusion8 or Fusion12
-> logscale deviation prediction
-> absolute timing reconstruction
-> raw-space auxiliary loss
```

该设计在保留 note-level autoregressive efficiency 的同时，明确分离：

```text
score condition
performance feedback
symbolic musical structure
timing target parameterization
```

并为后续的：

```text
feature masking
feedback source modeling
bridge ablation
musical ablation
closed-loop training
DAgger-style data aggregation
rollout regularization
```

提供统一接口。
