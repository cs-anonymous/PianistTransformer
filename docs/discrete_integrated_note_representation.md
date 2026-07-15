# Discrete Integrated Note Representation (DINR)

日期：2026-07-15

本文档提出一种面向 Expressive Piano Performance Rendering（EPR）的新离散表示：**Discrete Integrated Note Representation（DINR）**。该表示保留当前系统 `one note = one Transformer timestep` 的基本结构，但将CINR 中由 MLP 编码的 timing/value attributes 改为离散 sub-token，并将一个音符的多个属性组织为一个 compound note。

本文档同时说明 DINR 与**Continuous Integrated Note Representation（CINR）** 的关系、区别、量化方案、共享 embedding 设计、输出目标和建议实验。

---

## 1. 核心定义

DINR 的基本单位不是单个 MIDI event token，而是一个由多个 categorical sub-token 组成的 compound note：

```text
one aligned note
    -> multiple typed categorical sub-tokens
    -> field / role-aware embeddings
    -> slot-level representations
    -> SlotFusionMLP
    -> one 768-dim note embedding
    -> one Transformer timestep
```

形式上，第 $i$ 个音符表示为：

$$
C_i =
(P_i, I_i, D_i, V_i, M^o_i, M^d_i, A_i, \mathrm{Ped}_i),
$$

其中：

- $P_i$：pitch；
- $I_i$：IOI；
- $D_i$：duration；
- $V_i$：velocity；
- $M^o_i$：score musical onset；
- $M^d_i$：score musical duration；
- $A_i$：score annotations；
- $\mathrm{Ped}_i$：performance pedal states。

这些字段是同一个 compound note 内的 sub-token，不会展开为额外的 Transformer sequence positions。因此无论一个音符包含多少属性，外层序列长度始终为音符数：

$$
L_{\mathrm{DINR}}=N_{\mathrm{notes}}.
$$

---

## 2. 与 flat MIDI tokenizer 和 OctupleMIDI 的关系

### 2.1 Flat event tokens

普通 MIDI/event tokenizer 可能将一个音符展开为：

```text
POSITION_12
PITCH_60
DURATION_8
VELOCITY_72
```

一个音符占用多个 autoregressive steps，模型既要学习音乐关系，也要学习局部 token 顺序和合法 MIDI syntax。

### 2.2 Compound token / OctupleMIDI

Compound Word 和 OctupleMIDI 将多个离散属性放在同一个外层 timestep：

```text
(position, pitch, duration, velocity, ...)
```

DINR 属于同一大类思想，但专门面向 note-aligned EPR：

1. score note 与 performance note 已经对齐；
2. absolute timing 与 expressive timing deviation 使用同一个 additive log-coordinate vocabulary；
3. score、performance、IOI、duration 和 deviation 共享 value coordinates，由 field/role embeddings 区分语义；
4. score condition、performance feedback 和 deviation 具有明确 role；
5. pedal 和 annotations 使用结构化的 multi-field categorical representation。

因此，DINR 可以被描述为：

> A role-aware compound note representation that places absolute timing and expressive deviations on a shared additive logarithmic coordinate vocabulary.

---

## 3. 总体 schema

### 3.1 Score compound note

Score encoder 输入：

```text
ScoreCompoundNote_i
├── pitch
├── score IOI
├── score duration
├── score velocity
├── musical onset (mo)
├── musical duration (md)
└── annotations
```

### 3.2 Performance compound note

Autoregressive decoder feedback：

```text
PerformanceCompoundNote_i
├── pitch
├── performance IOI
├── performance duration
├── performance velocity
└── pedal states
```

Decoder 在 step $i$ 输入前一个 performance compound note，并预测当前音符的 expressive attributes。

### 3.3 Prediction target

推荐输出：

```text
Prediction_i
├── IOI deviation token
├── duration deviation token
├── performance velocity token
└── four pedal binary tokens
```

对于 score IOI 为零的音符，IOI log-ratio deviation 没有定义，因此使用单独的 absolute performance IOI head，详见第 8 节。

---

## 4. Unified additive log-timing vocabulary

### 4.1 统一坐标

Absolute timing 和 expressive deviation 都表示在自然对数坐标中：

$$
z_s=\log t_s,
\qquad
z_p=\log t_p,
$$

$$
d=\log t_p-\log t_s
=\log\frac{t_p}{t_s}.
$$

因此：

$$
z_p=z_s+d.
$$

DINR 不再为 absolute timing 和 deviation 建立两套 value vocabulary，而是让 score timing、performance timing 和 deviation 共享同一个 additive log-coordinate grid：

```text
UnifiedLogTimingVocabulary
├── score IOI absolute
├── score duration absolute
├── performance IOI absolute
├── performance duration absolute
├── IOI deviation
└── duration deviation
```

### 4.2 Zero-aligned 512-bin grid

推荐使用 512 个数值 bins，并令数值零精确落在 bin 93：

$$
K_{\mathrm{time}}=512,
\qquad q_0=93,
$$

$$
\Delta=\frac{2}{93}\approx0.021505.
$$

第 $q$ 个 bin 的 numerical coordinate 为：

$$
x_q=(q-q_0)\Delta,
\qquad q\in\{0,\ldots,511\}.
$$

由此得到：

$$
x_0=-2,
\qquad x_{93}=0,
\qquad x_{511}\approx8.989.
$$

该范围接近 $[-2,9]$，absolute timing 上限约为：

$$
e^{8.989}\approx8000\ \mathrm{ms},
$$

因此第一版 absolute timing support 设为 $[0,8000]$ ms，其中 0 ms 和 1 ms 共享 numerical coordinate 0。相邻 bins 的比例差约为：

$$
e^{\Delta}-1\approx2.17\%,
$$

最大舍入误差约为 $1.08\%$。

### 4.3 量化与反量化

任意 log coordinate $x\in[-2,8.989]$ 使用：

$$
Q(x)=
\operatorname{clip}
\left(
\operatorname{round}\left(\frac{x}{\Delta}\right)+q_0,
0,
511
\right).
$$

反量化：

$$
Q^{-1}(q)=(q-q_0)\Delta.
$$

Absolute timing 使用：

$$
q_{\mathrm{abs}}=Q(\log\max(t,1)),
$$

deviation 使用：

$$
q_{\mathrm{dev}}=Q(d).
$$

不需要为 numerical coordinate 的 $x=0$ 增加特殊 token。即使采用普通 $[-2,9]$ 等距 grid 导致零点存在约 $0.002$ 的偏差，其影响也可以忽略；zero-aligned grid 只是让加性坐标更整齐，而不是模型成立的必要条件。

### 4.4 Unified floor at 0/1 ms

DINR 不为 physical `IOI = 0 ms` 增加特殊 value token，而是统一定义：

$$
g(t)=\log(\max(t,1)).
$$

因此：

$$
g(0)=g(1)=0.
$$

0 ms 和 1 ms 都映射到 bin 93，并共享完整的 numerical coordinate。该 floor 吸收 chord 内部排序、MIDI tick rounding 和亚毫秒 alignment noise。Timing vocabulary 为：

```text
TIME_BIN_0
...
TIME_BIN_511
TIME_MASK
TIME_NULL
TIME_BOS
TIME_PAD
```

所有普通 timing tokens 都属于 numerical grid。`MASK`、`NULL`、`BOS` 和 `PAD` 仅表示结构状态，不作为输出 value token。

---

## 5. Shared timing embedding and semantic factorization

所有 timing values 共享：

$$
E_{\mathrm{time}}\in
\mathbb{R}^{512\times d_{\mathrm{slot}}},
\qquad d_{\mathrm{slot}}=128.
$$

共享关系为：

$$
E_{\mathrm{score\ IOI}}
=E_{\mathrm{score\ Dur}}
=E_{\mathrm{perf\ IOI}}
=E_{\mathrm{perf\ Dur}}
=E_{\mathrm{IOI\ dev}}
=E_{\mathrm{Duration\ dev}}
=E_{\mathrm{time}}.
$$

共享 value table 负责 log-coordinate identity；语义由 field 和 role embeddings 区分：

```text
field ∈ {IOI, Duration}
role  ∈ {ScoreAbsolute, PerformanceAbsolute, Deviation}
```

完整 timing representation 为：

$$
e_{\mathrm{time}}(q,f,r)=
E_{\mathrm{time}}[q]
+E_{\mathrm{field}}[f]
+E_{\mathrm{role}}[r]
+\phi_{\mathrm{time}}(x_q).
$$

例如：

$$
e_{\mathrm{score\ IOI}}
=E_{\mathrm{time}}[q]
+E_{\mathrm{IOI}}
+E_{\mathrm{score\ abs}}
+\phi_{\mathrm{time}}(x_q),
$$

$$
e_{\mathrm{duration\ dev}}
=E_{\mathrm{time}}[q]
+E_{\mathrm{Duration}}
+E_{\mathrm{deviation}}
+\phi_{\mathrm{time}}(x_q).
$$

这里的 $\phi_{\mathrm{time}}$ 是必选的 numerical coordinate encoder，而不是可选附加项。它可以采用 fixed sinusoidal/Fourier features 后投影到 128 维，或采用共享的小型 MLP。它使模型显式知道 bin 的顺序、距离和共同坐标；普通 lookup table 本身不保证这些关系。

---

## 6. Typed dynamic supports on the shared vocabulary

所有字段共享 512-bin value vocabulary，但 IOI head 根据 raw score IOI 使用动态 support mask：

```text
absolute timing:       [0, 8.989]  -> approximately [0, 8000] ms
IOI, score IOI > 0:    [-2, 2]
IOI, score IOI = 0:    [0, 5]
duration deviation:    [-2, 2]
```

Score IOI 是否为零不改变 head、vocabulary 或 token semantics。当 score IOI 为零时：

$$
d_I=g(I_p)-g(0)=g(I_p),
$$

所以 $[0,5]$ 覆盖约 1--148 ms 的 chord staggering。非零 score IOI 和 duration 使用较宽的 $[-2,2]$，让 categorical head从数据中学习概率集中区域。

训练标签超出当前样本对应 support 时，不把 clamp 后的边界 token 当作监督目标，而是将该属性从 loss 中排除。Loss-valid masks 为：

$$
m_I=
\begin{cases}
\mathbf 1[0\le d_I^*\le5],&I_s=0,\\
\mathbf 1[-2\le d_I^*\le2],&I_s>0,
\end{cases}
\qquad
m_D=\mathbf 1[-2\le d_D^*\le2].
$$

训练 loss：

$$
\mathcal L_{\mathrm{timing}}=
m_I\,\operatorname{CE}(Q(d_I^*),p_I)
+m_D\,\operatorname{CE}(Q(d_D^*),p_D).
$$

用于 teacher-forcing feedback 的安全值可以 clamp，但不能参与边界监督：

$$
d_{I,\mathrm{fb}}=
\begin{cases}
\operatorname{clip}(d_I^*,0,5),&I_s=0,\\
\operatorname{clip}(d_I^*,-2,2),&I_s>0,
\end{cases}
\qquad
d_{D,\mathrm{fb}}=\operatorname{clip}(d_D^*,-2,2).
$$

统一 floor-log deviations 为：

$$
d_I=g(I_p)-g(I_s),
\qquad
d_D=g(D_p)-g(D_s),
\qquad
g(t)=\log(\max(t,1)).
$$

模型预测共享 vocabulary 中的 deviation bin，再重建：

$$
\hat z_p=z_s+Q^{-1}(\hat q_d),
\qquad
\hat t_p=
\operatorname{clip}
\left(
\exp(\hat z_p),0,8000
\right).
$$

重建出的 performance absolute timing 再量化到同一个 vocabulary，作为下一步 decoder feedback：

```text
score absolute token
    + predicted deviation token
    -> reconstructed performance log timing
    -> performance absolute token
    -> next decoder step
```

### 6.1 Direct-absolute alternative

一个更接近 OctupleMIDI 的替代方案是不预测 deviation，而是直接预测 performance absolute timing。它使用统一 grid 中的非负部分：

$$
z=\log t\in[0,8.989],
$$

对应约 $[1,8000]$ ms，并继续使用统一步长：

$$
\Delta=\frac{2}{93}\approx0.021505,
$$

相邻 timing 比例约差 $2.17\%$。其流程为：

```text
score absolute timing token
    -> predict performance absolute timing token
    -> next decoder step
```

该方案结构最简单，不需要 zero-score-IOI 的 deviation 特例，但模型必须隐式学习 performance timing 通常围绕 score timing 变化。推荐把它作为 `DINR-Absolute` 消融；主设计仍使用共享 $[-2,8.989]$ grid，并在 $[-2,2]$ soft support 上预测 deviation，以保留当前 score-relative inductive bias。

---

## 7. Pitch and velocity tokens

### 7.1 Pitch

Pitch 保持 MIDI categorical vocabulary：

```text
PITCH_0 ... PITCH_127
PITCH_MASK
PITCH_NULL
PITCH_BOS
PITCH_PAD
```

对于钢琴数据可以只使用 21--108，但保留完整 MIDI 范围更便于数据兼容。

### 7.2 Velocity

Score velocity 和 performance velocity 都使用：

```text
VELOCITY_0 ... VELOCITY_127
```

推荐共享 velocity value embedding，并添加 role embedding：

$$
e_V(v,r)=E_V[v]+E_{\mathrm{role}}[r]+\phi_V(u_v),
\qquad r\in\{\mathrm{Score},\mathrm{Performance}\}.
$$

其中 numerical coordinate 为：

$$
u_v=\frac{v}{127}\in[0,1].
$$

$\phi_V$ 与 timing numerical encoder 采用同一原则：为 categorical identity 显式补充顺序和距离信息。它可以使用 fixed sinusoidal/Fourier features 或共享小型 MLP，再投影到 128 维。Velocity 0、64、127 不只是三个类别，还分别位于同一力度轴的低端、中部和高端。

Velocity numerical coordinate 是第一版 DINR 的必选组件。普通 `nn.Embedding(128, 128)` 不应作为最终 velocity representation，因为它不会自动保证 velocity 64 比 velocity 65 更接近、比 velocity 120 更远。

Performance velocity output 使用 128-class categorical head。

---

## 8. Unified IOI target

所有 score IOI 都使用同一个 floor-log difference target：

$$
d_I=g(I_p)-g(I_s),
\qquad
g(t)=\log(\max(t,1)).
$$

当 $I_s=0$ 时，$g(I_s)=0$，因此：

$$
d_I=g(I_p).
$$

模型不需要 zero-specific token 或 zero-specific head。唯一的条件分支是 support mask：

```text
UnifiedIOIHead -> UnifiedLogTimingVocabulary

score IOI > 0: support [-2, 2]
score IOI = 0: support [0, 5]
```

输入 token仍将 0 ms 与 1 ms 都映射到 coordinate 0；raw `score_shared_raw` 只用于选择 support mask，不改变 compound timing token。

---

## 9. Musical tokens

### 9.1 Musical onset $m_o$

当前 musical onset 采用 $1/24$ quarter-note grid，并支持 145 类：

$$
q_{m_o}=\operatorname{round}(24m_o),
\qquad q_{m_o}\in\{0,\ldots,144\}.
$$

使用：

```text
MO_0 ... MO_144
MO_NULL
MO_MASK
MO_PAD
```

直接保存 category ID，避免先构造 145 维 one-hot 再 `argmax`。

### 9.2 Musical duration $m_d$

Score musical duration 是符号化 quarter length，适合使用乐理 duration vocabulary，而不是毫秒 absolute timing vocabulary。例如：

```text
0
1/16
1/12
1/8
1/6
1/4
1/3
3/8
1/2
2/3
3/4
1
3/2
2
3
OTHER
```

使用：

```text
MD_0 ... MD_K
MD_OTHER
MD_NULL
MD_MASK
MD_PAD
```

Score musical duration 与 performance duration 不共享 table：前者表示 symbolic beat length，后者表示毫秒 physical timing。

---

## 10. Annotation tokens

Annotations 是 multi-label attributes，不应把所有组合视为一个普通类别。推荐将多个 categorical/binary sub-token 融合为一个 annotation slot：

```text
Annotation
├── hand:       left / right / unknown
├── trill:      no / yes
├── grace:      no / yes
├── staccato:   no / yes
└── stem:       none / up / down
```

每个字段分别 lookup：

$$
e_A=
E_{\mathrm{hand}}[h]
+E_{\mathrm{trill}}[t]
+E_{\mathrm{grace}}[g]
+E_{\mathrm{staccato}}[s]
+E_{\mathrm{stem}}[r].
$$

也可以 concatenate 后通过 `AnnotationFusionMLP` 投影到 128 维。第一版推荐求和，因为参数少且保留组合泛化能力。

这些 annotation sub-tokens 只存在于 compound note 内部，不展开为外层 sequence positions。

---

## 11. Pedal tokens

当前 pedal 表示为四个 binary snapshots：

```text
pedal at note onset
pedal at 25% duration
pedal at 50% duration
pedal at 75% duration
```

推荐使用四个位置相关的 binary embeddings：

$$
e_{\mathrm{Ped}}=
E_{P0}[p_0]+E_{P1}[p_1]+E_{P2}[p_2]+E_{P3}[p_3].
$$

输出保留四个 binary categorical heads：

```text
Pedal0Head -> {OFF, ON}
Pedal1Head -> {OFF, ON}
Pedal2Head -> {OFF, ON}
Pedal3Head -> {OFF, ON}
```

不建议把 pedal 展开成四个 Transformer steps。也不优先使用一个 16-class pattern token，因为独立 binary sub-token 能更好地共享局部 pedal state，并组合训练中较少出现的 pattern。

---

## 12. Compound note fusion

每个字段或字段组产生一个 128-dimensional slot：

```text
pitch slot
IOI slot
duration slot
velocity slot
musical-onset slot
musical-duration slot
annotation slot
pedal slot
```

CINR 和 DINR 在这里使用同一个配置与同一套骨架：encoder 和 autoregressive decoder 共用一个 note embedding module，只是传入的数据角色不同。encoder 传入 score slots，decoder 传入已右移的 performance feedback slots。没有输入值的 slot 使用 `MASK` embedding；`NULL` 只保留给语义上明确的“不适用”状态。所有 slots 拼接后，由同一个 fusion MLP 投影：

$$
h_i=
\operatorname{Fusion}
([e_{i,1};\ldots;e_{i,S}]),
\qquad h_i\in\mathbb{R}^{768}.
$$

因此这里不是“两套 encoder 再手动 tying 若干参数”，而是同一个 `IntegratedNoteEncoder` 实例：categorical/numerical value tables、field/role embeddings、slot MASK、LayerNorm 和 `SlotFusionMLP` 全部共享。DINR 改变的是 slot value encoder 和 output head，不改变 Transformer backbone。

### 12.1 Teacher forcing 的整音符 masking

CINR 和 DINR 共用同一组 teacher-forcing 配置：`tf_embedding_mask_decoder`、`tf_embedding_mask_keep_prob` 与 `slot_decoder_mask_mode`。当采用本文的 50% corruption 时，设置 `tf_embedding_mask_decoder=true`、`tf_embedding_mask_keep_prob=0.5`、`slot_decoder_mask_mode=whole_token`。masking 单位是融合后的完整 note embedding，而不是独立 slot：

$$
\tilde h_i^{\mathrm{dec}}=
\begin{cases}
e_{\mathrm{MASK}}^{\mathrm{note}}, & z_i=1,\\
h_i^{\mathrm{dec}}, & z_i=0,
\end{cases}
\qquad z_i\sim\operatorname{Bernoulli}(0.5).
$$

首个 BOS/起始位置不参与随机替换。slot-level `MASK` 只用于 score 侧不存在的 performance/pedal 字段、decoder 侧不存在的 musical 字段，或显式的 missing-field 实验；它不用于这项 50% teacher-forcing dropout。`PAD` 只表示 padding，不能作为 dropout replacement。

---

## 13. Output heads and weight tying

### 13.1 Typed categorical heads

与 CINR 的“MLP 到分布参数”不同，DINR head 输出合法 token 的 categorical logits：

```text
decoder hidden
    -> field-specific query projection
    -> dot product with shared token prototypes
    -> field/type support mask
    -> categorical distribution
    -> argmax or categorical sampling
```

设 decoder hidden 为：

$$
h_i\in\mathbb R^{768}.
$$

每个字段使用独立 query projection：

$$
g_i^I=H_I(h_i),
\qquad
g_i^D=H_D(h_i),
\qquad
g_i^V=H_V(h_i),
$$

其中 $g_i^I,g_i^D,g_i^V\in\mathbb R^{128}$。推荐每个 $H$ 使用：

```text
LayerNorm
-> Linear(768, 128)
-> GELU
-> Linear(128, 128)
```

### 13.2 Timing class prototypes

第 $k$ 个 timing class prototype 同时包含 learned identity 和 numerical coordinate：

$$
c_k^T=
E_{\mathrm{time}}[k]
+\phi_T(x_k),
$$

$$
x_k=(k-93)\frac{2}{93}.
$$

全部 512 个 prototypes 组成：

$$
C_T=
[c_0^T;\ldots;c_{511}^T]
\in\mathbb R^{512\times128}.
$$

不同 timing heads 共享 $C_T$，但保留独立 query projection 和 bias：

$$
\ell_i^{I}
=H_I(h_i)C_T^\top+b_I,
$$

$$
\ell_i^{D\text{-dev}}
=H_{D\text{-dev}}(h_i)C_T^\top+b_{D\text{-dev}}.
$$

IOI 无论 score IOI 是否为零都使用同一个 $H_I$ 和 $b_I$；只根据 raw score IOI 切换 support mask。

输出端不能只给所有 prototypes 加同一个 field/role embedding。若：

$$
c_k'=c_k+E_{\mathrm{IOI}},
$$

则 $g^\top E_{\mathrm{IOI}}$ 对所有类别相同，会在 softmax 中抵消。输出 field/role information 应通过独立 query heads、head conditioning 或 field-specific bias 注入。

### 13.3 Velocity class prototypes

Velocity prototype 为：

$$
c_v^V=E_V[v]+\phi_V(v/127),
\qquad v\in\{0,\ldots,127\}.
$$

组成：

$$
C_V\in\mathbb R^{128\times128}.
$$

Velocity logits：

$$
\ell_i^V=H_V(h_i)C_V^\top+b_V.
$$

预测 token ID 直接对应 MIDI velocity，不需要反量化。

### 13.4 Unified IOI head with dynamic support

Score IOI 是否为零在 forward 前已经由 raw condition 确定，但不路由到不同 heads：

```text
if score_ioi == 0:
    use IOIHead
    support = [0, 5]
else:
    use IOIHead
    support = [-2, 2]
```

两种情况的 target 都统一定义为 $d_I=g(I_p)-g(I_s)$。当 $I_s=0$ 时，$g(I_s)=0$，所以 target 自然退化为 performance IOI 的 absolute floor-log coordinate。Duration 使用独立 `DurationDeviationHead`，support 为 $[-2,2]$。

### 13.5 Typed vocabulary and support masking

Compound note 的每个字段在自己的 typed vocabulary 上预测：

```text
timing heads:   512 shared timing bins
velocity head:  128 velocity bins
pedal heads:    2 classes per snapshot
```

Timing head 先生成完整 logits，再应用 mask：

$$
m_k=
\begin{cases}
0,&x_k\text{ is valid for this head},\\
-\infty,&\text{otherwise},
\end{cases}
$$

$$
\tilde\ell_k=\ell_k+m_k.
$$

输出侧始终禁止 `MASK`、`NULL`、`BOS` 和 `PAD`。IOI head 按 score-zero condition 开放 $[0,5]$ 或 $[-2,2]$；duration head 开放 $[-2,2]$。

### 13.6 Numerical coordinates do not guarantee local prediction

Numerical coordinate 使相邻 token prototypes 共享数值结构，但不会硬性保证最大概率一定落在 target 附近。Categorical head 仍可以为远端 bin 分配高概率。

不同机制的作用应明确区分：

```text
numerical coordinate:
    exposes order and distance to the model

typed support mask:
    guarantees type and legal range

soft categorical target:
    encourages probability around neighboring coordinates

deviation prediction:
    anchors performance timing relative to score timing
```

第一版以 numerical prototypes、hard CE 和动态 support 为主；soft CE 和 coordinate auxiliary loss 作为消融。

### 13.7 Sampling

对任意 categorical head，先应用 support mask 和 temperature：

$$
p_k=
\operatorname{softmax}
\left(
\frac{\tilde\ell_k}{T}
\right).
$$

随机生成：

$$
\hat q\sim\operatorname{Categorical}(p).
$$

确定性生成：

$$
\hat q=\arg\max_k\tilde\ell_k.
$$

第一版使用 $T=1$ 且不使用 top-$k$/top-$p$，避免额外改变 PP/PN 分布；之后再单独调节 sampling policy。

对于 deviation token：

$$
\hat d=Q^{-1}(\hat q),
$$

$$
\hat t_p=
\operatorname{clip}
\left(
t_s e^{\hat d},0,8000
\right).
$$

再将 $\hat t_p$ 转为 performance absolute token，作为下一步 decoder feedback：

$$
\hat q_{p,\mathrm{abs}}
=Q(\log\max(\hat t_p,1)).
$$

Velocity 使用同样的 masked categorical sampling，但 token ID 直接得到 $\hat v\in\{0,\ldots,127\}$。Pedal 四个 snapshots 分别使用 2-class categorical sampling。

### 13.8 Recommended first implementation

```text
Shared prototypes:
├── TimingPrototype[512]
│   = TimingLookup + TimingNumericalCoordinate
└── VelocityPrototype[128]
    = VelocityLookup + VelocityNumericalCoordinate

Field-specific query heads:
├── UnifiedIOIQuery:         768 -> 128
├── DurationDeviationQuery:  768 -> 128
├── VelocityQuery:           768 -> 128
└── PedalQuery:              768 -> 4 x 2

Training:
  hard categorical CE
  IOI support: [-2, 2] when score IOI > 0, [0, 5] when score IOI = 0
  duration support: [-2, 2]
  exclude out-of-support targets from loss
  clamp excluded values only for feedback construction

Sampling:
  typed support mask
  temperature = 1.0
  categorical sample or argmax
  inverse coordinate
  reconstruct and clamp absolute timing to [0, 8000] ms
```

这种 head 保留 categorical distribution 的多峰能力；合法 token 类型和范围由 typed heads 与 support masks 保证，而不是依赖 numerical coordinate 自动产生局部预测。

---

## 14. Hard CE, soft CE and metric awareness

普通 `nn.Embedding` 和 hard categorical CE 不会自动知道相邻 bins 数值接近。即使 token ID 来自 floor-log grid，模型仍可能把 `TIME_100`、`TIME_101` 和 `TIME_200` 当作互不相关的类别。

因此建议比较以下训练方式。

### 14.1 Hard categorical CE

$$
\mathcal L_{\mathrm{hard}}
=-\log p(q^*).
$$

这是最纯粹的 discrete compound baseline。

### 14.2 Soft categorical CE

对真实 bin $q^*$ 构造邻域 target：

$$
y_k\propto
\exp\left(-\frac{|z_k-z_{q^*}|}{\tau}\right).
$$

然后：

$$
\mathcal L_{\mathrm{soft}}
=-\sum_k y_k\log p_k.
$$

距离应在实际 log-bin center $z_k$ 上定义，而不只是在 token ID 上定义。

### 14.3 Numerical coordinate encoding

Timing 和 velocity 必须加入固定或可学习 numerical coordinate encoding：

$$
e(q)=
E_{\mathrm{lookup}}[q]
+\phi(z_q),
$$

其中 $\phi$ 可以是 sinusoidal/Fourier features 或共享小型 MLP。这样 token 同时保留 categorical identity 和 metric position。

Timing 使用统一 coordinate：

$$
x_q=(q-93)\frac{2}{93}.
$$

Velocity 使用：

$$
u_v=\frac{v}{127}.
$$

推荐第一版使用：

```text
token lookup embedding
+ fixed Fourier/sinusoidal numerical features
+ learned projection
+ field embedding
+ role embedding
```

Numerical encoding 不需要额外 numerical-zero token。通过 $g(t)=\log(\max(t,1))$，0 ms 和 1 ms 都使用 coordinate 0 的普通 timing token。

第一版 DINR 即包含 numerical coordinate encoding。为了消融其贡献，应额外运行一个 `lookup-only` 版本，而不是把 lookup-only 当成最终主模型。Soft CE 仍可作为独立消融，从而区分收益来自 input metric encoding 还是 output neighborhood supervision。

---

## 15. DINR 与 CINR 的区别

### 15.1 CINR

当前 CINR：

```text
raw/continuous attribute values
    -> role-specific MLP encoders
    -> 128-dim slots
    -> SlotFusionMLP
    -> note embedding
    -> continuous probabilistic heads
```

Timing 输入保留连续数值，输出使用 DLM、logistic-normal 或其他连续/离散化概率分布参数。

### 15.2 DINR

DINR：

```text
quantized categorical attribute IDs
    -> shared value embeddings + numerical coordinates + field/role embeddings
    -> 128-dim slots
    -> SlotFusionMLP
    -> note embedding
    -> categorical token heads
```

### 15.3 对照表

| Dimension | CINR | DINR |
|---|---|---|
| Outer unit | one note per timestep | one note per timestep |
| Internal structure | attribute slots | categorical sub-token slots |
| Timing input | continuous floor-log value | quantized floor-log token |
| Timing target | continuous distribution parameters | categorical deviation token |
| Value encoder | MLP | embedding lookup + numerical coordinate encoder |
| IOI/duration sharing | usually separate role-specific encoders | shared unified log-value table + field embeddings |
| Score/performance sharing | usually separate role-specific encoders | shared unified log-value table + role embeddings |
| Output distribution | parameterized continuous family | flexible categorical distribution |
| Numerical distance | explicit in scalar input | explicit grid coordinate through mandatory numerical encoding |
| Quantization error | none before output distribution discretization | explicit but small |
| Multimodality | depends on distribution family/components | naturally represented by arbitrary bin probabilities |
| Pretraining compatibility | requires continuous-aware model interface | compatible with compound-token language modeling |

### 15.4 两者共享的核心

CINR 与 DINR 并不是完全不同的模型。二者共享：

1. one note per Transformer timestep；
2. attribute-level internal structure；
3. slot fusion；
4. score encoder / autoregressive performance decoder；
5. factorized expressive prediction；
6. musical and annotation conditioning slots。

真正的研究变量是：

```text
continuous value modeling
vs.
quantized categorical compound modeling
```

---

## 16. 优势与局限

### 16.1 DINR 的潜在优势

1. **灵活的输出分布**：categorical logits 可表达多峰、尖峰、端点质量和不规则分布；
2. **稳定的有界采样**：采样结果始终落在合法 token support；
3. **统一的 metric vocabulary**：absolute timing 和 deviation 在同一个 additive log-space 中表示；
4. **参数共享**：score/performance、IOI/duration 可以共享 value embeddings；
5. **短序列**：一个音符仍然只占一个 timestep；
6. **预训练兼容性**：更容易使用 categorical masking、CE 和 compound-token pretraining。

### 16.2 DINR 的局限

1. **量化误差**：虽然较小，但不再是完全连续表示；
2. **输出邻近关系不自动成立**：input numerical coordinates 显式提供距离，但 hard CE 本身仍不理解相邻输出 bins；
3. **稀疏 bins**：低端或极端 timing bins 可能样本不足；
4. **并行 head 假设**：factorized heads 不显式建模同一 note 内各输出采样值之间的依赖；
5. **zero IOI 特例**：需要 absolute/deviation 双路预测；
6. **通用 MIDI 生成能力仍有限**：该 schema 利用 note alignment，主要服务 EPR，而不是任意无对齐 composition。

### 16.3 CINR 的潜在优势

1. scalar input 显式保留数值顺序和距离；
2. 无固定量化 grid；
3. 可连续插值；
4. 更直接表达“EPR 是连续 structured prediction”的任务先验。

其主要风险是输出分布族可能过强约束真实的多峰和长尾分布。

---

## 17. 建议实验

### 17.1 主对比

保持以下因素一致：

```text
dataset and split
note alignment
Transformer backbone
hidden size
number of layers
training budget
autoregressive protocol
evaluation and MIDI export
```

比较：

1. **CINR**：当前 slot MLP + continuous/distribution heads；
2. **DINR-512 Lookup-only**：统一 log grid + lookup embedding + hard CE；
3. **DINR-512 Metric**：统一 log grid + timing/velocity numerical coordinates + hard CE；
4. **DINR-512 Metric-soft**：numerical coordinates + soft categorical CE；
5. **DINR-Absolute**：使用统一 grid 的 $[0,8.989]$ 非负部分，直接预测 performance absolute timing；
6. **DINR-Separate**：absolute/deviation 使用不同 value tables，作为共享方式消融。

### 17.2 共享方式消融

比较：

```text
separate absolute/deviation and score/perf embeddings
shared unified timing value embedding only
shared value + field embedding
shared value + field + role embedding
shared value + field + role + numerical coordinate
```

### 17.3 Target 消融

比较：

```text
absolute performance timing token
score-relative deviation token
```

### 17.4 评估指标

主 EPR 指标：

```text
IOI PP Wasserstein in raw ms
duration PP Wasserstein in raw ms
velocity Wasserstein
pedal Wasserstein
```

同时报告：

```text
PN Wasserstein
quantization-only reconstruction error
token/bin occupancy
prediction entropy
invalid/tail sampling rate
training throughput
peak GPU memory
inference latency
```

---

## 18. 推荐第一版实现配置

```text
representation_name: dinr512_unified_metric

unified_log_timing:
  bins: 512
  zero_bin: 93
  step: 0.021505376344086023  # 2 / 93
  coordinate_min: -2.0
  coordinate_max: 8.989247311827956
  absolute_transform: floor_log_ms
  absolute_positive_min_ms: 1
  absolute_max_ms: 8000
  merge_zero_and_one_ms: true
  shared_across_absolute_deviation: true
  shared_across_ioi_duration: true
  shared_across_score_perf: true
  absolute_valid_min: 0
  absolute_valid_max: 8.989247311827956
  ioi_deviation_valid_min: -2
  ioi_deviation_valid_max: 2
  zero_score_ioi_valid_min: 0
  zero_score_ioi_valid_max: 5
  duration_deviation_valid_min: -2
  duration_deviation_valid_max: 2
  out_of_support_target_loss: ignore
  out_of_support_feedback: clamp
  inference_absolute_timing_clamp_ms: [0, 8000]

velocity:
  bins: 128
  coordinate: value_div_127
  shared_across_score_perf: true

pedal:
  representation: binary_4

embedding:
  slot_dim: 128
  add_field_embedding: true
  add_role_embedding: true
  timing_numerical_encoding: fourier
  velocity_numerical_encoding: fourier
  numerical_encoding_required: true

output:
  loss: hard_categorical_ce
  tie_unified_timing_output_embedding: true
  separate_ioi_duration_projection: true
  unified_ioi_head: true
  dynamic_ioi_support_mask: true
```

---

## 19. 论文定位

DINR 不应被表述为“首次一个音符一个 token”，因为已有 compound token 和 OctupleMIDI 已采用 note-level tuple。更准确的贡献是：

> We introduce an EPR-specific categorical compound note representation that places score timing, performance timing, and expressive deviations on a shared additive log-coordinate vocabulary, augments timing and velocity tokens with explicit numerical coordinates, and preserves one aligned note per Transformer timestep.

与 CINR 联合起来，论文可以研究更一般的问题：

> For note-aligned expressive performance rendering, should timing be modeled as continuous structured values or as metric-aware categorical compound tokens?

因此，DINR 的价值不仅是一个替代实现，也是对 CINR 核心表示选择的直接实验检验：

```text
same note unit
same slots
same Transformer
same EPR task

continuous scalar representation
vs.
discrete categorical compound representation
```

这一对照能够区分已有收益究竟来自：

1. one-note-per-timestep；
2. attribute-level slot structure；
3. continuous numerical encoding；
4. categorical output flexibility；
5. score-relative deviation target。

---

## 20. 最终设计结论

推荐将 DINR 定义为一套统一的 additive log-timing vocabulary：

```text
512 bins
zero bin: 93
step: 2 / 93
coordinate range: approximately [-2, 8.989]
+ structural tokens

shared by:
  score IOI absolute
  score duration absolute
  performance IOI absolute
  performance duration absolute
  IOI deviation
  duration deviation
```

Absolute timing 和 deviation 共享 value embedding，因为它们满足 $z_p=z_s+d$，并位于同一个加性 log coordinate system。不同语义不通过拆分 vocabulary 解决，而通过以下部分分解：

```text
shared value embedding
+ explicit numerical coordinate encoding
+ IOI/Duration field embedding
+ ScoreAbsolute/PerformanceAbsolute/Deviation role embedding
+ field-specific output projection and support mask
```

Numerical coordinate 对 timing 和 velocity 都是主设计的一部分：

```text
timing coordinate:  x_q = (q - 93) * 2 / 93
velocity coordinate: u_v = v / 127
```

数值零点不需要额外 token。Zero-aligned grid 让 $x_{93}=0$，并通过 $g(t)=\log(\max(t,1))$ 将 physical 0 ms 与 1 ms 映射为同一个普通 timing token。

第一版 head 采用共享 numerical class prototypes 和 field-specific query projections：

```text
IOI, score IOI > 0:  unified IOI head, support [-2, 2]
IOI, score IOI = 0:  unified IOI head, support [0, 5]
duration deviation:  shared timing prototypes, soft support [-2, 2]
velocity:            shared velocity prototypes, support [0, 127]
```

超出当前动态 support 的 target 不计算 loss，只在 feedback 构造时 clamp。推理从 masked categorical distribution 采样或取 argmax，反量化后将 absolute timing clamp 到 $[0,8000]$ ms。

DINR 与CINR 共享 note-level slot architecture，但将：

```text
continuous value MLP
    -> categorical value lookup + numerical coordinate encoder

continuous distribution head
    -> categorical token head
```

从而形成一个结构清楚、可公平对照、兼容 compound-token 预训练的离散 EPR 表示。
