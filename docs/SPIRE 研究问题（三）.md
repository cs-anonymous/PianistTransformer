
本文档已经形成了一条非常清晰的主线，而且相比我们前面的讨论有两个重要收敛：

1. **创新点不是 ED，而是 Integrated Note Representation。** Backbone 可以是 GPT、T5、BERT、Mamba，都可以直接替换，表示层与 Backbone 解耦。
2. **Note 成为一级对象（first-class object）**，控制符号（Bar、Dynamics、Slur 等）仍然保持语言化，而 Note 内部混合离散属性与连续属性，通过统一的 Note Encoder/Decoder 映射到共享表示空间。这一点我认为比单纯提出新的 MIDI tokenizer 或连续回归 head 更具有基础表示层面的意义。


# Toward a Integrated Note Representation for Symbolic Music Foundation Models

## 1 Motivation

近年来，大多数 Symbolic Music Foundation Model（MusicLM、MIDI-LM、PianistTransformer 等）均采用事件流（event stream）作为基本建模对象，将 MIDI 转换为一系列离散 token：

```
<N060><V080><T156><T043>
<N064><V076><T142><T039>
...
```

其中，Pitch、Velocity、Duration、Offset 等参数全部离散化为 token。

这种表示方式非常适合 Transformer 的语言建模目标（next-token prediction），但对于 Expressive Performance Rendering（EPR）任务存在天然缺陷。

EPR 本质上不是语言生成，而是：

```
Score
↓

Performance Parameters
```

即根据乐谱预测演奏参数，其目标是连续控制信号，而不是生成合理的 token 序列。

因此，将连续参数离散化再恢复，本身可能不是最优表示。

---

# 2 EPR 的本质

EPR 更接近：

```
Conditional Structured Prediction
```

而不是：

```
Language Modeling
```

模型需要预测：

- velocity
    
- onset shift
    
- duration ratio
    
- pedal value
    

这些变量本质上均为连续控制参数。

模型真正需要学习的是：

```
Structure
↓

Rendering
```

而不是：

```
Token
↓

Token
```

---

# 3 Integrated Note Representation

本文提出将 Music Representation 分为两类对象：

## (1) Symbol Tokens

用于表示真正的离散音乐符号。

例如：

```
<BAR>
<VOICE>
<SLUR_START>
<FERMATA>
<DYNAMIC_F>
<KEY_SIGNATURE>
<TIME_SIGNATURE>
<TEMPO_MARK>
```

这些信息天然属于语言符号，应继续采用 token 表示。

Transformer 对此具有天然优势。

---

## (2) Note Object

音符不再拆分为多个 token，而作为一个完整对象编码。

例如：

```
Note {

    pitch

    score_duration

    velocity

    onset_shift

    duration_ratio

    pedal_0

    pedal_25

    pedal_50

    pedal_75

}
```

其中：

Score 数据：

```
pitch
score_duration
```

存在；

Performance 数据：

```
velocity
onset_shift
duration_ratio
pedal_x
```

存在；

缺失字段统一使用 Mask/Null。

因此：

```
Score Note

↓

Unified Note Encoder

↓

Latent Note Embedding
```

Performance Note 同样进入同一 Encoder：

```
Performance Note

↓

Unified Note Encoder

↓

Latent Note Embedding
```

从而形成统一的 Note Latent Space。

---

# 4 Unified Note Encoder

Unified Note Encoder 接受离散属性与连续属性共同输入。

离散字段：

```
pitch
score_duration
articulation
dynamic
```

使用 Embedding：

```
PitchEmbedding

DurationEmbedding

...
```

连续字段：

```
velocity

onset_shift

duration_ratio

pedal
```

使用小型 MLP：

```
VelocityMLP

OnsetMLP

DurationMLP

PedalMLP
```

最终：

```
note_embedding

=

pitch_embedding

+

duration_embedding

+

velocity_embedding

+

onset_embedding

+

pedal_embedding

+ ...
```

Transformer Backbone 并不关心输入来源，只接受统一的 embedding sequence。

因此：

```
Token

↓

Embedding

```

与

```
Note

↓

Embedding

```

可以自然混合。

最终输入序列形如：

```
<BAR>

<NOTE>

<NOTE>

<DYNAMIC_F>

<NOTE>

<SLUR_END>

<NOTE>
```

其中：

控制符号采用普通 token embedding；

对应 Note Encoder 输出的连续 embedding。

---

# 5 Unified Note Decoder

Decoder 不再恢复 token，而恢复完整 Note Object。

对于离散字段：

```
pitch

score_duration

articulation
```

采用分类头：

```
CrossEntropy
```

对于连续字段：

```
velocity

duration_ratio

onset_shift

pedal
```

采用回归头：

```
Huber Loss

L1

MSE
```

总体损失：

```
L

=

L_discrete

+

λ L_continuous
```

因此 Decoder 实际恢复的是：

```
Note Object
```

而不是 token 序列。

---

# 6 Foundation Model 预训练

统一 Encoder/Decoder 的最大优势在于：

无需依赖 paired data。

对于大量无监督 MIDI：

```
Performance Note

↓

Encoder

↓

Decoder

↓

Performance Note
```

即可进行自监督学习。

对于大量乐谱：

```
Score Note

↓

Encoder

↓

Decoder

↓

Score Note
```

同样成立。

因此：

```
100M MIDI

+

ABCX

+

MusicXML

+

Score MIDI
```

均可参与 CPT。

相比 Score Encoder 与 Performance Encoder 分离设计，统一表示空间能够充分利用海量未配对数据。

---

# 7 EPR

EPR 不需要新的表示。

仅需：

```
Score Note

↓

Unified Encoder

↓

Transformer

↓

Unified Decoder

↓

Performance Fields
```

即可完成：

```
Σ

↓

Φ
```

反向：

```
Performance Note

↓

Unified Encoder

↓

Transformer

↓

Unified Decoder

↓

Score Fields
```

即可完成：

```
Φ

↓

Σ
```

从而天然支持双向建模。

---

# 8 关于 Pedal 的设计

Pedal 有两种建模方式。

## 方法一：独立事件流

```
Note

Note

Pedal

Note

Pedal
```

优点：

符合 MIDI 原始定义。

缺点：

长度变化；

需要模型学习 Pedal 插入位置；

增加 Decoder 复杂度。

---

## 方法二：Pedal Sampling（推荐）

参考 PianistTransformer，将两个 Note 之间均匀划分为四段：

```
Note_i

|

25%

|

50%

|

75%

|

Note_{i+1}
```

分别记录：

```
pedal_0

pedal_25

pedal_50

pedal_75
```

因此：

```
Note Object

=

pitch

+

performance parameters

+

pedal snapshots
```

Pedal 被视为局部控制轨采样，而不是事件流。

该设计具有：

- 固定长度；
    
- 易于对齐；
    
- 可自然作为 Note 连续属性编码；
    
- 无需模型学习 Pedal 插入位置。
    

相比独立 Pedal Event，更适合作为 Foundation Model 的统一表示。

---

# 9 核心思想

本文并不否定 MIDI Tokenizer。

真正需要修改的是：

```
所有音乐信息都必须 token 化
```

这一假设。

音乐表示应采用混合表示：

```
Control Symbol

↓

Discrete Token

```

```
Note

↓

Continuous Embedding
```

Transformer Backbone 完全可以同时接受：

```
Token Embedding

+

Note Embedding
```

形成统一序列。

因此，该方法与具体 Backbone 无关。

无论：

- Decoder-only
    
- Encoder-only
    
- Encoder-Decoder
    
- Mamba
    

均可直接采用该 Integrated Note Representation。

其创新点并非新的 Transformer，而是一种统一的 Symbolic-Continuous Music Representation，为 Symbolic Music Foundation Model 提供更加符合音乐本质的建模方式。