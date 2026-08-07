# Minimal Chord Representation Design

日期：2026-08-05

本文档规划一个最简 chord-aware EPR 表示。核心目标：保留当前 slot 结构，只把
`score IOI = 0` 的同 onset 音组合成 chord token，避免 chord 内伪 IOI 污染普通
`ioi_dev`。

现有 6 slot 保持不变：

```text
pitch, ioi, duration, velocity, pedal, musical
```

变化只发生在 slot 内部：

```text
pitch:
  single note id / one-hot -> chord pitch multihot

ioi:
  scalar value -> MLP(value, ioi_span_high_minus_low)

duration:
  scalar value -> MLP(value, duration_span_high_minus_low)

velocity:
  scalar value -> MLP(value, velocity_span_high_minus_low)

pedal:
  group-level shared pedal

musical:
  group-level or anchor-level musical attributes
```

其中 `span_high_minus_low` 是 chord 内部最高音 endpoint 减最低音 endpoint。
IOI 的 score 侧 span 为 0；duration 的 score 侧 span 按 score duration endpoint
difference 计算。Performance 侧 span 可正可负，表达 chord 内从低到高或从高到低的
展开。

---

## 1. Chord 认定

### 1.1 初始 onset group

先按 score onset 切分：

```text
new onset group starts when score_ioi > 0
score_ioi == 0 joins current onset group
```

不要求同 duration。同 onset、同手或同 staff 的音经常 duration 不同；要求 duration
相同会丢掉大量可合并 chord continuation。

### 1.2 Staff split

在每个 onset group 内，如果相邻 note 的 staff 均已知且发生变化，则切开：

```text
known staff 0 -> known staff 1: split
known staff 1 -> known staff 0: split
unknown staff: no staff split
```

Staff 作为强提示使用。Staff 缺失或投影不可靠时，交给 pitch span 规则处理。

### 1.3 Hand-span and size split

对每个 staff 子组，按 pitch span 和 chord size 两个约束递归切分：

```text
max_hand_span = 19 semitones  # about a twelfth
max_chord_size = 5

if max(pitch) - min(pitch) <= max_hand_span
   and chord_size <= max_chord_size:
    keep group
else:
    split at the largest adjacent pitch gap
    recursively apply the same rule
```

这样即使所有音被标成同一个 staff，过大的同 onset group 也会被切成更接近手型的
chord token。`max_chord_size=5` 用来处理音域不大但音数过多的 dense cluster。

切点选择：

```text
1. sort notes by pitch
2. find adjacent pitch gaps
3. split at the largest gap
4. if size is the only violation and the largest-gap split is extremely unbalanced,
   split pitch order into chunks with at most 5 notes
```

---

## 2. 6-Slot Schema

设第 `g` 个 chord group 包含 notes：

```text
C_g = {n_1, ..., n_K}
```

按 pitch 升序排列后：

```text
p_low  = min pitch in C_g
p_high = max pitch in C_g
```

### 2.1 Pitch Slot

Pitch slot 使用 88 维 multihot：

```text
pitch_multihot[k] = 1 if piano pitch (21 + k) appears in C_g else 0
```

单音 token 是只有一个 1 的 multihot。这样 pitch slot 不需要新增 chord_size、
pitch_min、pitch_max 等显式属性；这些信息可以由 multihot 隐式给出。

### 2.2 IOI Slot

IOI slot 输入两个属性：

```text
ioi_slot = MLP_ioi(ioi_value, ioi_span_high_minus_low)
```

Score 侧：

```text
score_onset_center_g = score_onset_g
score_ioi_value_g = score_onset_center_g - score_onset_center_{g-1}
score_ioi_span_high_minus_low_g = 0
```

Performance 侧：

```text
t_low_g  = onset time of p_low note
t_high_g = onset time of p_high note

perf_onset_center_g = (t_low_g + t_high_g) / 2
perf_ioi_value_g = perf_onset_center_g - perf_onset_center_{g-1}
perf_ioi_span_high_minus_low_raw_g = t_high_g - t_low_g
perf_ioi_span_high_minus_low_g =
    signed_log_raw(perf_ioi_span_high_minus_low_raw_g)
```

符号解释：

```text
perf_ioi_span_high_minus_low_g > 0: high pitch later than low pitch, low-to-high roll
perf_ioi_span_high_minus_low_g < 0: high pitch earlier than low pitch, high-to-low roll
perf_ioi_span_high_minus_low_g = 0: simultaneous endpoints
```

单音 token：

```text
perf_ioi_span_high_minus_low_g = 0
perf_onset_center_g = onset of the only note
```

### 2.3 Duration Slot

Duration slot 输入两个属性：

```text
duration_slot = MLP_duration(duration_value, duration_span_high_minus_low)
```

Score 侧第一版取 group 内 duration center，并保留 high-minus-low duration span：

```text
score_duration_value_g = mean(score_duration_j for n_j in C_g)

score_d_low_g  = score duration of p_low note
score_d_high_g = score duration of p_high note

score_duration_span_high_minus_low_raw_g = score_d_high_g - score_d_low_g
score_duration_span_high_minus_low_g =
    signed_log_raw(score_duration_span_high_minus_low_raw_g)
```

Performance 侧：

```text
d_low_g  = duration of p_low note
d_high_g = duration of p_high note

perf_duration_value_g = mean(perf_duration_j for n_j in C_g)
perf_duration_span_high_minus_low_raw_g = d_high_g - d_low_g
perf_duration_span_high_minus_low_g =
    signed_log_raw(perf_duration_span_high_minus_low_raw_g)
```

符号解释：

```text
perf_duration_span_high_minus_low_g > 0: high pitch longer than low pitch
perf_duration_span_high_minus_low_g < 0: high pitch shorter than low pitch
```

### 2.4 Velocity Slot

Velocity slot 输入两个属性：

```text
velocity_slot = MLP_velocity(velocity_value, velocity_span_high_minus_low)
```

Score 侧：

```text
score_velocity_value_g = mean(score_velocity_j for n_j in C_g)
score_velocity_span_high_minus_low_g = 0
```

Performance 侧：

```text
v_low_g  = velocity of p_low note
v_high_g = velocity of p_high note

perf_velocity_value_g = mean(perf_velocity_j for n_j in C_g)
perf_velocity_span_high_minus_low_raw_g = v_high_g - v_low_g
perf_velocity_span_high_minus_low_g = perf_velocity_span_high_minus_low_raw_g / 127
```

符号解释：

```text
perf_velocity_span_high_minus_low_g > 0: high pitch louder than low pitch
perf_velocity_span_high_minus_low_g < 0: high pitch softer than low pitch
```

### 2.5 Pedal Slot

Pedal slot 直接使用 group-level shared pedal：

```text
pedal4_g = pedal sampled around perf_onset_center_g
```

训练时 chord 内所有 note 不再各自提供 pedal target。展开 MIDI 时，group 内音共享同一
pedal curve。

### 2.6 Musical Slot

Musical slot 可沿用当前方案。第一版可采用 anchor 或 group aggregation：

```text
anchor: highest pitch note's musical attributes
aggregation: mean / first valid / staff-majority
```

为了保持实现简单，建议第一版先用 highest pitch anchor，与旧 chord 分支一致。

---

## 3. Targets

第一版输出目标：

```text
ioi_dev
duration_dev
velocity_value or velocity_dev
pedal4
ioi_span_high_minus_low
duration_span_high_minus_low
velocity_span_high_minus_low
```

其中 `ioi_dev`、`duration_dev`、`velocity_dev` 是 group center 的主要表达目标；
`ioi_span_high_minus_low`、`duration_span_high_minus_low`、
`velocity_span_high_minus_low` 是 chord 内 endpoint shape。

命名上避免把 span 称为 dev：

```text
ioi_dev: group center timing deviation
ioi_span_high_minus_low: chord endpoint onset difference
```

### 3.1 IOI Dev

使用当前 log-deviation 体系：

```text
score_ioi_g = score_onset_center_g - score_onset_center_{g-1}
perf_ioi_g  = perf_onset_center_g  - perf_onset_center_{g-1}

ioi_dev_g = log_time(perf_ioi_g) - log_time(score_ioi_g)
```

对第一个 token，可设为 0 或使用已有 BOS 规则。

### 3.2 Duration Dev

```text
score_dur_g = score_duration_value_g
perf_dur_g  = perf_duration_value_g

duration_dev_g = log_time(perf_dur_g) - log_time(score_dur_g)
```

### 3.3 Velocity

有两种选择：

```text
velocity_value_g = perf_velocity_value_g / 127
```

或：

```text
velocity_dev_g = (perf_velocity_value_g - score_velocity_value_g) / 127
```

如果当前主线已经使用 absolute velocity，第一版沿用 absolute velocity，减少变量。

---

## 4. Span Transform

Span 有正负号，使用与当前 raw-log timing 约定一致的 signed raw log：

```text
signed_log_raw(x)
  = sign(x) * log(max(1, abs(x)))
```

逆变换：

```text
inverse_signed_log_raw(y)
  = 0 if y == 0
  = sign(y) * exp(abs(y)) otherwise
```

因为 `log(max(1, abs(x)))` 会把 `|x| < 1` 映射到 0，1 ms 内 endpoint
difference 视为 simultaneous。

建议单位：

```text
ioi_span_high_minus_low:
  unit: ms
  transform: signed_log_raw(t_high - t_low)

duration_span_high_minus_low:
  unit: ms
  transform: signed_log_raw(d_high - d_low)

velocity_span_high_minus_low:
  unit: velocity
  transform: clip(x / 127, -1, 1)
```

`ioi_span_high_minus_low` 的抽样统计显示：

```text
<= 50 ms: 92.77%
<=100 ms: 97.96%
<=150 ms: 99.07%
```

因此绝大多数 chord onset endpoint difference 落在 150 ms 内。Raw-log transform
本身不设置 `xmax`；分布头或 loss 可单独设置 support / clipping。

---

## 5. Span 是否也要预测分布

建议：`ioi_span_high_minus_low` 使用分布头。

原因：

1. Inference 时需要从模型输出还原 chord 内展开。如果只有 `ioi_dev` 分布，chord
   内 roll 仍然缺少可采样变量。
2. `ioi_span_high_minus_low` 有明显多峰结构：接近 0、低到高、高到低。点估计容易向 0 收缩。
3. `ioi_span_high_minus_low` 的主质量集中在有限范围，适合 bounded distribution 或离散分布。

第一版可选两档：

```text
minimal:
  ioi_span_high_minus_low 使用 Huber / L1 点回归
  duration_span_high_minus_low 使用 Huber / L1 点回归
  velocity_span_high_minus_low 使用 Huber / L1 点回归

recommended:
  ioi_span_high_minus_low 使用 bounded/discrete distribution
  duration_span_high_minus_low 使用 point regression
  velocity_span_high_minus_low 使用 point regression
```

如果实现成本允许，更整齐的版本：

```text
ioi_dev: distribution
duration_dev: distribution
velocity_value/dev: point or distribution, follow current baseline

ioi_span_high_minus_low: distribution over signed raw-log values
duration_span_high_minus_low: regression or distribution over signed raw-log values
velocity_span_high_minus_low: bounded regression or bounded distribution
```

Span 分布头不必复用 IOI dev 的 unbounded head。更合适的形式：

```text
signed raw-log scalar:
  y = sign(delta) * log(max(1, abs(delta)))

candidate heads:
  DLM categorical bins over signed raw-log support
  bounded continuous head after clipping support
  skew-normal / logistic-normal on signed raw-log coordinate
```

考虑到 `ioi_span_high_minus_low` 有大量 near-zero mass，DLM / categorical bins 通常更直观：

```text
bins:
  zero bin around [-5 ms, 5 ms]
  dense bins in [-50 ms, 50 ms]
  tail bins up to +/-150 ms
```

---

## 6. Chord 展开公式

设 chord group 内 pitch 升序为：

```text
p_1 <= ... <= p_K
```

定义 pitch rank coordinate：

```text
if K == 1 or p_K == p_1:
    r_j = 0
else:
    r_j = (p_j - p_1) / (p_K - p_1) - 0.5
```

因此：

```text
r_low  = -0.5
r_high =  0.5
```

给定预测出的 center 和 span：

```text
ioi_span_raw = inverse_signed_log_raw(ioi_span_high_minus_low)
duration_span_raw = inverse_signed_log_raw(duration_span_high_minus_low)
velocity_span_raw = 127 * velocity_span_high_minus_low

onset_j = onset_center + r_j * ioi_span_raw
duration_j = duration_center + r_j * duration_span_raw
velocity_j = velocity_center + r_j * velocity_span_raw
```

其中：

```text
onset_center_g = onset_center_{g-1} + predicted_perf_ioi_value_g
duration_center_g = predicted_perf_duration_value_g
velocity_center_g = predicted_perf_velocity_value_g
```

边界处理：

```text
duration_j = max(duration_j, 1 ms)
velocity_j = round(clip(velocity_j, 1, 127))
```

该插值可表达：

```text
ioi_span_high_minus_low > 0: low-to-high
ioi_span_high_minus_low < 0: high-to-low
ioi_span_high_minus_low = 0: simultaneous
```

对于非单调 chord 内部 order，线性插值会产生残差。抽样诊断中，pitch-rank 线性插值
的 onset residual：

```text
<=10 ms: 88.74%
<=15 ms: 93.71%
<=20 ms: 96.27%
<=30 ms: 98.41%
```

这说明最简 span 表示能覆盖绝大部分真实 chord 展开。

---

## 7. 数据统计

统计对象：`data/ASAP_processed/**/*.json`，共 207 个 work JSON。

### 7.1 Staff 覆盖

```text
notes: 570,623
staff known notes: 521,061 = 91.31%
staff missing/unknown notes: 49,562 = 8.69%

score IOI = 0 transitions: 271,461
same previous staff: 242,618 = 89.37%
cross previous staff: 3,668 = 1.35%
staff missing/unknown: 25,175 = 9.27%
```

### 7.2 Duration 条件

在可合并的 same-staff zero-IOI transitions 中：

```text
same duration: 182,347 = 75.16%
different duration: 60,271 = 24.84%
```

Duration-equal 条件会丢掉约四分之一同 staff chord continuation。

### 7.3 新 heuristic 的 score-side 覆盖

规则：

```text
same onset group
-> split by known staff changes
-> recursively split when pitch span > 19 semitones or chord size > 5
-> split point: largest pitch gap, with balanced max-5 fallback
```

结果：

```text
original notes: 570,623
chord tokens: 363,596
token reduction: 36.28%

chord groups with size > 1: 137,121
notes inside non-singleton chord groups: 344,148 = 60.31%
```

Group size:

```text
1: 226,475
2: 87,720
3: 31,532
4: 15,233
5: 2,636
```

Final chord pitch span:

```text
q50: 12
q75: 12
q90: 16
q95: 18
q99: 19
max: 19
```

### 7.4 Performance 抽样诊断

抽样：随机 300 个 performance，直接从 aligned performance MIDI 和 alignment
`perf_idx` 还原真实 note start time；跳过包含 interpolated note 的 chord group。

Chord 内 onset order，10 ms 以内视为 simultaneous：

```text
groups: 196,164
near_simultaneous: 42.80%
low_to_high:       15.83%
high_to_low:       31.90%
non_monotonic:      9.47%

near_simultaneous + monotonic: 90.53%
```

Chord onset span:

```text
q50: 12.50 ms
q75: 25.00 ms
q90: 42.71 ms
q95: 60.42 ms
q99: 144.79 ms

<= 10 ms: 40.90%
<= 20 ms: 67.81%
<= 30 ms: 80.84%
<= 50 ms: 92.77%
<=100 ms: 97.96%
```

Pitch-rank linear interpolation residual:

```text
q50: 0.00 ms
q75: 3.54 ms
q90: 11.01 ms
q95: 17.19 ms
q99: 37.24 ms

<=10 ms: 88.74%
<=15 ms: 93.71%
<=20 ms: 96.27%
<=30 ms: 98.41%
```

---

## 8. 建议实验

### A. Minimal Span Baseline

```text
grouping: same onset + staff split + hand-span split
max_chord_size: 5
slots: pitch, ioi, duration, velocity, pedal, musical
pitch: 88-d multihot
ioi/duration/velocity encoder: dual-attribute MLP(value, span)
targets: ioi_dev, duration_dev, velocity, pedal4,
         ioi_span_high_minus_low, duration_span_high_minus_low,
         velocity_span_high_minus_low
span losses: point regression first, ioi_span_high_minus_low distribution next
reconstruction: center + pitch-rank span interpolation
```

### B. Span Head Ablation

比较：

```text
ioi_span_high_minus_low point Huber
ioi_span_high_minus_low DLM categorical
ioi_span_high_minus_low distribution over signed raw-log value
```

### C. Span Transform Ablation

比较：

```text
raw clipped span / xmax
signed_log_raw = sign(x) * log(max(1, abs(x)))
categorical ms bins
```

### D. Grouping Ablation

比较：

```text
same onset only
same onset + staff split
same onset + span split
same onset + size split
same onset + staff split + span split
same onset + staff split + span split + size split
same onset + staff split + duration-equal
```

---

## 9. 当前结论

推荐第一版采用最简 6-slot chord 表示：

```text
pitch slot:
  multihot

ioi slot:
  MLP(ioi_value, ioi_span_high_minus_low)

duration slot:
  MLP(duration_value, duration_span_high_minus_low)

velocity slot:
  MLP(velocity_value, velocity_span_high_minus_low)

pedal slot:
  group-level shared pedal

musical slot:
  anchor/group musical attributes
```

该方案只给 ioi/duration/velocity slot 增加一个 span 属性，整体复杂度低于显式添加
大量 chord attributes。数据上，它减少约 36.3% token，覆盖约 60.3% note；performance
抽样显示，signed span + pitch-rank interpolation 在 20 ms residual 内覆盖约 96.3%
chord onset pattern。
