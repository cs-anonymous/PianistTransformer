# INR Soft Categorical Head 设计记录

日期：2026-06-19

本文档记录当前 INR/FINE/PINE 实验中从 `Beta(mu/kappa)` probabilistic continuous head 转向 `soft categorical CE` head 的动机、诊断发现、分布建模问题、输入输出格式、loss 设计与 smoothing 参数估计。本文档是新增设计文档，不覆盖原有 `docs/integrated_note_representation_experiment.md`。

## 1. 背景

当前 INR 的 EPR 任务目标是从 score note sequence 生成 performance attributes：

```text
pitch:     copied / conditioned
ioi:       performance inter-onset interval
duration:  performance note duration
velocity:  MIDI velocity
pedal:     four sampled sustain pedal values
```

之前第一版 probabilistic EPR head 使用统一的 `Beta(mu/kappa)` 分布：

```text
hidden -> shared head -> [ioi_mu, dur_mu, vel_mu, ioi_kappa, dur_kappa, vel_kappa]
hidden -> pedal head  -> [pedal_mu_0..3, pedal_kappa_0..3]

alpha = mu * kappa
beta  = (1 - mu) * kappa
loss  = -log Beta(target | alpha, beta)
```

该设计的优点是形式统一、输出天然在 `[0, 1]` 内、可以采样。但是实验和分布诊断显示，单个 Beta 对 EPR 数据分布施加了过强的先验假设。

## 2. 实验发现

### 2.1 单 Beta 的 inductive bias 过强

单个 Beta 隐含假设：

```text
每个 feature 在给定上下文下可以由一个平滑、有界、低维参数化分布表示。
```

这对 velocity 较为合理，但对 ioi、duration、pedal 明显不足。

真实 EPR target 更接近：

```text
ioi:      多峰、包含 chord / near-zero mass、短 timing grid、正常 note spacing、长尾停顿
duration: 长尾、强 articulation / legato / phrase 依赖
velocity: 相对平滑，较接近单峰或弱多峰
pedal:    强端点膨胀，0/127 很多，中间半踏板值较少且结构复杂
```

因此 `Beta(mu/kappa)` 的问题不是“不能采样”，而是它的分布族太窄：

```text
deterministic mean/mu:
  容易变成保守的 mode-seeking / mean-seeking 输出

Beta sampling:
  可以产生随机性，但 timing 在 expm1 反变换后会放大尾部误差
  ioi/duration sampling 反而产生过大的长尾
```

### 2.2 K=3 sampling 没有解决问题

在 ASAP 子集上进行 INR K=3 sampling 后，结果显示 sampling 并没有修复 timing/pedal 分布 mismatch：

```text
FINE K=3 sampling:
  ioi MAE/Wass:      509.05 / 436.17
  duration MAE/Wass: 779.18 / 650.91
  velocity MAE/Wass: 19.98 / 8.13
  pedal MAE/Wass:    57.32 / 21.48

PINE K=3 sampling:
  ioi MAE/Wass:      411.90 / 335.56
  duration MAE/Wass: 996.50 / 890.32
  velocity MAE/Wass: 20.33 / 9.48
  pedal MAE/Wass:    55.67 / 22.07
```

这说明问题不是“没有随机性”，而是预测分布形状本身不适合。

### 2.3 Velocity 相对健康

Velocity 的拟合明显好于 ioi/duration/pedal。原因可能是：

```text
velocity 原始范围固定为 0..127
分布相对平滑
没有 expm1 反解码放大
端点膨胀不如 pedal 强
```

这进一步支持当前问题主要来自 output distribution assumption，而不是 INR backbone 或 FINE/PINE embedding 本身完全失效。

## 3. PT Categorical 为什么更稳

PianistTransformer 使用 token-level categorical modeling：

```text
pitch:     128 classes
ioi:       timing token, about 5000 classes
velocity:  128 classes
duration:  timing token, about 5000 classes
pedal:     128 classes each
```

训练使用 categorical CE，生成时通过 token slot 限制合法范围：

```text
ioi position      -> only timing token range
duration position -> only timing token range
pedal position    -> only pedal token range
```

这相当于一个非常自由的离散经验分布：

```text
p(bin | context)
```

它可以表达：

```text
多峰
尖峰
长尾
0/127 端点质量
量化 grid artifact
真实数据中的离散空洞
```

PT categorical 的优势不是“token 天然更高级”，而是它对输出分布几乎没有形状先验。相比之下，单 Beta 是强参数化分布，约束过强。

PT categorical 的缺点也很明确：

```text
类别之间没有天然距离感
100ms 和 101ms 被当成两个独立 class
量化 vocab 大
连续插值能力弱
```

因此新的 INR head 希望保留 categorical 的自由度，同时加入轻量的 ordinal / numerical distance 信息。

## 4. 新方案：Soft Categorical CE

### 4.1 核心形式

新的 EPR head 不再直接输出 normalized continuous feature，而是输出每个 feature 的 raw bin logits：

```text
ioi logits:       0..4999 timing bins
duration logits:  0..4999 timing bins
velocity logits:  0..127 bins
pedal logits:     4 x 0..127 bins
```

模型输出仍然是 categorical distribution：

```text
p(feature_bin | context) = softmax(logits)
```

训练 target 不是 one-hot，而是以真实 bin 为中心的 soft target：

```text
target bin = k
q(i | k) ∝ exp(-|i - k| / tau)
loss = CE(q, p)
```

这里使用的是 Laplace-shaped kernel。它不表示真实数据服从 Laplace 分布，只表示：

```text
离 target 越近，训练目标权重越高；
距离每增加 tau，权重大约乘以 e^-1。
```

例如 target 为 100，`tau=10`：

```text
q(100) ∝ 1
q(110) ∝ exp(-10/10) = 0.368
q(120) ∝ exp(-20/10) = 0.135
q(130) ∝ exp(-30/10) = 0.050
```

所以 soft categorical CE 不是 hard categorical，也不是点回归。它是：

```text
categorical output + ordinal-aware soft target
```

### 4.2 为什么不是 ordinal regression / EMD loss

Ordinal regression 或 EMD/CDF loss 也可以加入距离关系，但会引入额外复杂度：

```text
EMD/CDF loss:
  需要额外 loss term 和 lambda

cumulative ordinal regression:
  需要显式建模 CDF / threshold
  对 5000 timing bins 实现和计算都更复杂
  还可能引入新的单调 latent score 假设
```

当前阶段的目标是保持与 PT-style LM head 尽可能接近：

```text
hidden -> logits -> CE
```

因此第一版优先使用 soft categorical CE，而不是更复杂的 ordinal loss。

### 4.3 是否 probabilistic

Soft categorical CE 是 probabilistic。训练后模型输出完整离散分布：

```text
p_model(i | context)
```

推理时不是在预测点上加 Laplace noise，也不是使用训练时的 tau 采样。推理直接从模型分布采样：

```text
logits -> optional temperature/top-p/top-k -> softmax -> categorical sample
```

训练时的 `tau` 只用于构造 soft target，不直接参与 inference。

推理流程：

```text
1. decoder hidden -> feature logits
2. softmax 得到 p(bin | context)
3. sample 或 argmax 得到 raw bin
4. raw bin -> normalized input feature
5. normalized feature -> IntegratedNoteEmbedding
6. 作为下一步 AR decoder input
```

因此该方案保留真正的 probabilistic autoregressive generation。

## 5. 输出格式

采用 soft categorical CE 后，输出侧不再预测 `[0, 1]` continuous feature：

```text
ioi output:       raw timing bin, 0..4999
duration output:  raw timing bin, 0..4999
velocity output:  raw MIDI velocity bin, 0..127
pedal output:     raw MIDI CC64 bin, 0..127
```

训练 label 从当前 INR normalized label 反解得到 raw bin：

```text
ioi_ms      = denormalize_time(label_ioi)
duration_ms = denormalize_time(label_duration)
velocity    = round(label_velocity * 127)
pedal_i     = round(label_pedal_i * 127)
```

然后：

```text
ioi_bin      = clamp(round(ioi_ms), 0, 4999)
duration_bin = clamp(round(duration_ms), 0, 4999)
velocity_bin = clamp(round(velocity), 0, 127)
pedal_bin    = clamp(round(pedal), 0, 127)
```

这一点和 PT 的 timing / velocity / pedal tokenization 对齐，有利于公平比较。

## 6. 输入格式与归一化

输出改为 raw categorical bin，并不意味着输入也要使用 raw value。输入给 `IntegratedNoteEmbedding` 的连续 feature 仍建议使用 normalized scalar：

```text
raw bin -> normalized scalar -> embedding MLP
```

原因：

```text
raw timing 0..5000 与 velocity/pedal 0..127 尺度差异很大
MLP 输入更适合稳定的 bounded scalar
AR 推理时 sampled bin 仍需转换回 decoder input feature
```

### 6.1 timing input normalization

当前旧版 timing normalization 是：

```text
log1p(x) / log1p(10000)
```

该映射对小 timing 过度拉伸：

```text
0ms -> 0
1ms -> about 0.075
2ms -> about 0.119
5ms -> about 0.195
```

这导致 `[0, 0.2]` 区域出现明显离散断层，也会让 chord / near-zero timing 在 embedding 输入空间中被过度放大。

新方案建议改为与 PT timing bin 上限对齐的 scaled log：

```text
timing_input_norm(x) = log1p(min(x, 5000) / 10) / log1p(500)
```

即：

```text
x = 5000ms -> 1.0
```

该设计与 output timing bins 的有效域一致：

```text
output timing label: 0..4999
input timing norm:   x clipped to 5000ms
```

相比 `log1p(x / 10) / log1p(1000)`，使用 `/ log1p(500)` 的理由是：

```text
PT timing token 有效上限约为 5000ms
输入归一化和输出 bin 域保持一致
超过 5000ms 的长值统一 clip，和 PT token clamp 对齐
```

推荐：

```text
ioi_input      = log1p(clamp(ioi_ms, 0, 5000) / 10) / log1p(500)
duration_input = log1p(clamp(duration_ms, 0, 5000) / 10) / log1p(500)
```

### 6.2 velocity / pedal input normalization

Velocity 和 pedal 保持线性归一化：

```text
velocity_input = velocity / 127
pedal_input    = pedal / 127
```

## 7. Loss 设计

第一版保持简单，不引入 mixture，不引入额外 EMD lambda：

```text
L_ioi      = SoftCE(ioi_logits, ioi_bin, tau_ioi)
L_duration = SoftCE(duration_logits, duration_bin, tau_duration)
L_velocity = SoftCE(velocity_logits, velocity_bin, tau_velocity)
L_pedal    = mean_i SoftCE(pedal_i_logits, pedal_i_bin, tau_pedal)

L_epr = L_ioi + L_duration + L_velocity + w_pedal * L_pedal
```

当前为了避免 pedal loss 过强主导，可以继续沿用：

```text
w_pedal = 0.2
```

如果后续发现 categorical pedal CE 数值尺度不再极端，也可以重新评估是否恢复为 1.0。

## 8. Smoothing 参数估计

### 8.1 统计口径

为了避免拍脑袋选择 smoothing 半径，已在 train split 上统计同一 score、同一 note、不同 performance 的人类变异：

```text
对每个 score note:
  收集所有 train performance 的 raw feature value
  计算该 note 的 median value
  统计每个 performance 相对 median 的 abs deviation
```

优先使用 `non_interpolated` 统计，避免插值 note 使 label uncertainty 被估大。

结果文件：

```text
results/soft_ce_smoothing_estimate/train_within_score_note_absdev_summary.json
```

统计规模：

```text
works:              1624
train performances: 141554
note groups:        2899360
observations:       300877770
```

### 8.2 统计结果

`non_interpolated` 下的 abs deviation：

| feature | mean abs dev | p50 | p75 | p90 | p95 | p99 |
|---|---:|---:|---:|---:|---:|---:|
| ioi | 25.84 ms | 10 | 29 | 60 | 97 | 230 |
| duration | 221.05 ms | 80 | 220 | 530 | 890 | 2310 |
| velocity | 8.46 | 6 | 11 | 19 | 22 | 32 |
| pedal_0 | 22.98 | 0 | 0 | 127 | 127 | 127 |
| pedal_25 | 23.15 | 0 | 0 | 127 | 127 | 127 |
| pedal_50 | 22.67 | 0 | 0 | 127 | 127 | 127 |
| pedal_75 | 22.55 | 0 | 0 | 127 | 127 | 127 |

### 8.3 如何解释这些值

这些统计量表示同一 note 在不同 human performances 下的自然变化，不等于 soft CE 的 smoothing 参数必须直接取这些值。

关键区别：

```text
human variation:
  表示一对多 EPR 的真实风格差异

soft CE tau:
  表示单个 label 的局部数值容忍度 / 量化不确定性
```

完整的人类多峰分布应该由 categorical logits 学出来，而不是靠过大的 smoothing 人工抹开。

### 8.4 推荐第一版 tau

基于上述统计，推荐第一版使用保守的 feature-specific tau：

```text
tau_ioi      = 10
tau_duration = 20 to 40
tau_velocity = 4 to 8
tau_pedal    = 1 to 3
```

解释：

```text
ioi:
  p50 abs dev 为 10ms，适合作为局部 timing 容忍尺度。
  不建议直接取 p75=29 或 p90=60，否则容易抹掉节奏细节。

duration:
  p50=80ms, p75=220ms，但 duration 的人类差异包含 articulation / legato / style。
  tau 不应直接取 80 或 220，否则 target 太宽。
  建议从 20..40ms 起步，让模型自己学习 duration 的多峰和长尾。

velocity:
  p50=6, p75=11，分布健康。
  tau=4..8 可以表达 ordinal 关系，又不会过度平滑。

pedal:
  p50/p75 都是 0，但 p90 直接到 127。
  这说明 pedal 的差异常常是踩/不踩风格差异，而不是局部连续误差。
  因此 tau 只能很小，建议 1..3。
  0/127 多峰结构应由 categorical distribution 学出，不应靠 smoothing 展宽。
```

第一版可固定为：

```text
tau_ioi      = 10
tau_duration = 30
tau_velocity = 6
tau_pedal    = 2
```

后续可以在 validation 上对这些 tau 做小范围 grid search：

```text
ioi:      {5, 10, 20}
duration: {20, 30, 40, 80}
velocity: {4, 6, 8}
pedal:    {1, 2, 3}
```

## 9. Head 结构

第一版不引入复杂 mixture distribution。Head 仍保持简单的 MLP + logits：

```text
SharedHead:
  hidden -> hidden -> logits_ioi[5000]
                  -> logits_duration[5000]
                  -> logits_velocity[128]

PedalHead:
  hidden -> hidden -> logits_pedal[4, 128]
```

对于 FINE：

```text
head input = full hidden state
```

对于 PINE：

```text
shared head input = shared partition
pedal head input  = pedal / perf partition
```

这与原来的 FINE/PINE 表示比较保持一致，只改变 output distribution head。

## 10. AR 推理过程

训练使用 teacher forcing：

```text
previous GT raw bins -> normalized features -> IntegratedNoteEmbedding -> decoder input
current hidden -> feature logits -> soft categorical CE
```

推理使用 free-running AR：

```text
for note t:
  hidden_t -> logits_t
  sample raw bins:
    ioi_t, duration_t, velocity_t, pedal_t
  convert sampled raw bins to normalized features
  feed normalized feature embedding as previous performance condition for note t+1
```

采样策略：

```text
deterministic:
  argmax(logits)

sampling:
  logits / temperature
  optional top-p / top-k
  Categorical sample
```

训练时的 `tau` 不直接用于 sampling。推理随机性来自模型预测的 categorical distribution。

## 11. 与 Beta Head 的关系

新方案不是否定 continuous INR embedding，而是否定“输出必须是单 Beta continuous distribution”。

保留：

```text
INR note-level representation
FINE/PINE embedding
AR decoder
teacher forcing
same backbone scaffold
normalized feature as decoder input
```

改变：

```text
old output:
  hidden -> Beta(mu/kappa) -> normalized continuous feature

new output:
  hidden -> categorical logits -> raw feature bin
```

这使实验可以回答更干净的问题：

```text
INR 表示本身是否有效？
还是之前失败主要来自 Beta output distribution assumption？
```

如果 INR + soft categorical head 明显接近或超过 PT，则说明 Beta head 是主要问题。如果仍然明显落后，则需要继续检查 INR embedding、AR 流程、backbone 容量或数据协议。

## 12. 第一版实验定义

建议新的实验版本：

```text
FINE-SoftCat
PINE-SoftCat
```

共同设置：

```text
dataset: PianoCoRe-A full train split
backbone: same as current INR baseline
decoder mode: AR
training: teacher forcing
prior token keep/drop protocol: same as current controlled setting, unless explicitly ablated
window overlap: same as PT controlled protocol
output labels: raw bins aligned to PT token ranges
input timing norm: log1p(min(x, 5000) / 10) / log1p(500)
velocity/pedal input norm: value / 127
loss: soft categorical CE
```

Default tau:

```text
tau_ioi      = 10
tau_duration = 30
tau_velocity = 6
tau_pedal    = 2
```

Default loss:

```text
L_epr = L_ioi + L_duration + L_velocity + 0.2 * L_pedal
```

Evaluation should report:

```text
ASAP subset:
  MAE / Wasserstein for ioi, duration, velocity, pedal

Non-ASAP subset:
  MAE / Wasserstein for ioi, duration, velocity, pedal

Distribution diagnostics:
  raw feature histogram
  normalized input feature histogram
  predicted categorical entropy
  sampled vs deterministic output comparison
```

## 13. 后续可选方向

如果 soft categorical CE 仍不足，可再考虑：

```text
1. Hard categorical CE baseline
   用于确认 soft smoothing 是否有帮助。

2. CE + EMD ordinal penalty
   加强数值距离感，但会引入 lambda。

3. Discretized mixture logistic
   更连续、更参数化，但复杂度更高。

4. Pedal zero-one-inflated head
   专门处理 0/127 端点质量。

5. Feature-specific adaptive tau
   tau = data_estimated_tau * learned_or_validated_multiplier。
```

但第一版不建议直接引入这些复杂方法。当前首要目标是用最简单、最接近 PT LM head 的方式验证：

```text
INR 的问题是否主要来自过强的 Beta distribution inductive bias。
```

## 14. 2026-06-20 更新：Soft CE 与 mixture continuous head 后的重新判断

### 14.1 Soft categorical CE 的实验结论

后续实验显示，`soft categorical CE` 并没有成为更好的主线。虽然它在理论上比单 Beta 更自由，可以表达多峰、尖峰和端点质量，但实测存在明显问题：

```text
timing / pedal 输出分布很差
deterministic 输出容易集中在很窄的类别范围
sampling 虽然拉宽分布，但 free inference 下容易把序列带偏
pedal 仍然容易被推向边界模式
```

因此，原先“PT categorical 更稳，因此 INR 应该转向 soft categorical”的判断需要修正。问题不只是 output head 的分布族是否足够自由。更自由的 categorical head 没有自动解决 EPR free inference 的稳定性问题。

当前结论是：

```text
soft categorical CE 不是当前 INR 主线。
INR 需要回到 continuous / regression-oriented 表示，但不能只换一个连续分布 head。
```

这里的“回到回归”不是回到最初的 naive Beta，而是保留连续数值归纳偏置，同时修正 autoregressive 输入、teacher forcing mismatch、score/perf embedding 语义和 pedal 表示结构。

### 14.2 Mixture Logistic-Normal / Mixture Beta 的结论

为避免把失败简单归因于单 Beta 过窄，后续实现并训练了多个 continuous probabilistic head：

```text
Huber PINE
3-mixture logistic-normal FINE
3-mixture logistic-normal PINE
inflated 3-mixture logistic-normal PINE
pure logistic-normal PINE
3-mixture beta PINE
```

其中主线设想是：

```text
mixture logistic-normal:
  z = logit(y)
  z ~ mixture_k Normal(mu_k, sigma_k)
  y = sigmoid(z)

mixture beta:
  y ~ mixture_k Beta(alpha_k, beta_k)
```

这些 head 确实比单 Beta 有更强的表达能力，但实际结果并没有带来本质改善。`mln3`、`beta3` 与之前 beta-style head 的行为接近，没有显著解决 timing / pedal 的主要问题。

观察到的共同现象：

```text
deterministic free inference:
  输出分布明显变窄，呈现均值回归 / mode regression

sampling free inference:
  分布变宽，但序列容易被少数异常 sample 带偏
  timing 容易逐步变长，形成极值峰
  pedal 容易向 0 / 127 边界吸附
```

因此当前判断是：

```text
当前瓶颈大概率不只是 head distribution family。
即使换成更灵活的 mixture head，如果 AR 输入语义和训练-推理 mismatch 不修正，free inference 仍会不稳定。
```

### 14.3 Free inference 问题：timing drift 与 pedal collapse

后续诊断显示，INR free inference 存在严重的自回归滚动不稳定：

```text
timing drift:
  生成越往后 timing 越容易变长
  sampling 下更明显，偶发大 timing 会把后续历史状态带偏

pedal collapse:
  deterministic 下 pedal 可能整体坍缩为几乎全 127 或全 0
  sampling 下 pedal 边界质量被进一步放大
```

重要的是，teacher-forced 诊断并没有出现同等程度的 pedal collapse。这说明模型在给定真实历史时可以做相对正常的单步预测，但在 free-running rollout 中会进入坏吸引子。

因此当前核心问题被重新表述为：

```text
不是单步预测完全学不会，
而是 teacher forcing 训练分布与 free inference 自回归历史分布不一致。
```

已经检查过的方向：

```text
KV cache / cached rollout:
  cached autoregressive rollout 与 full recomputation 的差异约为 1e-6 量级
  因此 cache 不是主要原因

teacher-forced prediction:
  不呈现 free inference 那样的严重 pedal collapse
  因此 collapse 主要来自 self-conditioned rollout
```

### 14.4 Decoder input / embedding 语义问题

当前 AR decoder input 还存在一个结构性问题：decoder 侧历史 performance embedding 的语义不够干净。

需要修正的问题包括：

```text
1. decoder input pitch 使用方式不清楚，存在把当前 score pitch 与上一时刻 performance feature 混合的风险
2. score note embedding 与 performance note embedding 没有足够明确地区分
3. backbone 输出没有被强制解释为 clean performance note state
4. teacher forcing 时 decoder 看到的是真实右移 labels_continuous，而 free inference 时看到的是模型自身预测
```

这意味着模型训练时学到的是：

```text
score context + perfect previous performance history -> current performance
```

但推理时实际面对的是：

```text
score context + model-generated previous performance history -> current performance
```

如果 previous performance embedding 本身语义混乱或过度依赖真实历史，free inference 会很容易漂移。

### 14.5 为什么普通 dropout 不合适

之前尝试过对 continuous feature 做 dropout，但效果不好，原因是：

```text
0 对 timing / pedal / velocity 都是有语义的真实值
把某个 feature 置 0 不等于 unknown，而是在注入错误的 performance state
attribute-wise dropout 会制造不自然的半损坏音符
```

更合理的做法是 whole-note mask：

```text
用专门的 [MASK] performance-note embedding 替代上一 note 的完整 performance embedding
而不是把某几个属性置 0
```

这样 decoder 学到的是：

```text
有时上一 note performance history 是可用的
有时上一 note performance history 是 unknown / masked
```

这比 zero dropout 更接近 free inference 中“历史不可靠”的情况。

## 15. 当前 head / representation 的新要求

基于以上结果，新的 INR head 和 embedding 不应只追求更复杂的概率分布，而应满足以下要求。

### 15.1 保留数值归纳偏置

Soft categorical CE 的失败说明，完全离散分类不一定更好。当前更倾向于：

```text
输入和输出仍保留连续 / ordinal 数值结构
head 可以是 regression-oriented 或 structured ordinal
inference 后再 materialize / quantize 到 MIDI raw value
```

原因：

```text
timing / velocity / pedal 都有明确数值距离
100 和 101 比 100 和 127 更接近
continuous regression 能利用这种 inductive bias
```

但连续 head 不能再简单假设每个 feature 独立、平滑、单峰或无结构。

### 15.2 训练-推理一致性优先

新的设计必须显式考虑 free inference：

```text
不要只看 teacher-forced eval loss
必须评估 deterministic free inference
必须评估 sampling free inference
必须诊断 timing drift 和 pedal collapse
```

如果一个 head 在 teacher forcing 下 loss 更低，但 free inference 更容易进入坏吸引子，它就不是 EPR 可用方案。

### 15.3 Pedal 不应继续用 4 个自由 continuous value 建模

Pedal 的 4-slice 表示统计显示，数据有非常强的 morphology 结构。直接预测 4 个 continuous value 太自由，也容易在 free inference 中坍缩到边界。

统计结果位于：

```text
results/inr_eval/pedal_note_shape_summary_20260620/
```

关键结果：

```text
PianoCoRe train all:
  all0/all127 合计约 87.32%

PianoCoRe train ASAP:
  all0/all127 合计约 55.27%
  非边界连续形态占比明显更高

ASAP test:
  all0/all127 合计约 59.00%
  非边界连续形态约 41.00%
```

进一步按语义化 `mode + start + span` 统计，使用 tolerance=4：

```text
ASAP test:
  hold      81.02%
  ramp_up    7.93%
  ramp_down  7.35%
  valley     2.23%
  peak       0.58%
  complex    0.90%

PianoCoRe train ASAP:
  hold      80.54%
  ramp_up    8.20%
  ramp_down  7.55%
  valley     2.19%
  peak       0.36%
  complex    1.16%
```

这说明 pedal 主体不是任意 4D 连续向量，而是：

```text
hold + ramp_up + ramp_down + small valley/peak + rare complex
```

因此新的 pedal head 应该是 morphology-aware。

## 16. 新设计方向

### 16.1 Pedal head：主分类 + start，span 可选

新的 pedal 表示建议从 4 个 continuous values 改为结构化 morphology：

```text
pedal_mode:
  hold
  ramp_up
  ramp_down
  valley
  peak
  complex

pedal_start:
  ordinal / continuous value in 0..127

pedal_span:
  optional auxiliary value
```

当前倾向是：

```text
主任务:
  pedal_mode + pedal_start

辅助任务:
  pedal_span 可选
```

materialize 规则可以先使用强结构模板：

```text
hold:
  [start, start, start, start]

ramp_up:
  当前 note start 到下一 note start 之间插值

ramp_down:
  当前 note start 到下一 note start 之间插值

valley:
  在第 3 个位置放全曲统计上的 low anchor，通常接近 0
  第 2 / 第 4 个位置根据前后 start 插值

peak:
  在第 3 个位置放全曲统计上的 high anchor，通常接近 127
  第 2 / 第 4 个位置根据前后 start 插值

complex:
  第一版可退化为 hold 或小 span 模板
```

注意：

```text
使用下一 note start 做插值应当发生在整段 pedal mode/start 预测完成后的 materialize 阶段。
不要在当前 decoder step 中把未来 note 的 predicted start 作为输入泄露给模型。
```

这个设计引入了明确的 pedal morphology bias，但统计上是合理的。它的目标不是最大表达力，而是降低 free inference 下 pedal collapse 的自由度。

### 16.2 Decoder input：修正 pitch 语义

需要明确修复 decoder input pitch 的使用方式：

```text
performance decoder history 不应混入错误的当前 score pitch 语义
performance note embedding 应表达 previous generated performance state
score note embedding 应表达当前/全局 score condition
```

这是结构修正，不是额外的强先验。

### 16.3 添加 whole-note [MASK] embedding

引入专门的 performance-note `[MASK]` embedding，用于 teacher forcing dropout：

```text
训练时以一定概率将 previous performance note embedding 替换为 [MASK]
mask 作用于整音符 performance state
不对单个 timing / velocity / pedal 属性置 0
```

目标：

```text
减少模型对 perfect GT history 的依赖
让 decoder 学会在 previous performance history 不可靠时仍依赖 score context
缓解 free inference 的 exposure bias
```

建议第一版 ablation：

```text
prior_token_keep_prob = 1.0
prior_token_keep_prob = 0.8
prior_token_keep_prob = 0.6
prior_token_keep_prob = 0.4
```

最终选择不只看 eval loss，而要看 free inference drift diagnostics。

### 16.4 明确区分 score_note_embedding 与 perf_note_embedding

新的 embedding 结构建议明确分离 score/performance 两类 note embedding。

建议结构：

```text
pitch:
  固定 88 维 one-hot

remaining hidden:
  680 维 MLP embedding
```

Score note embedding：

```text
score_features -> MLP(in_dim, 680)
concat pitch_onehot[88] + score_mlp[680] -> 768
```

Performance note embedding：

```text
performance_features -> MLP(in_dim, 680)
concat pitch_onehot[88] + perf_mlp[680] -> 768
```

这里的关键不是 88/680 本身，而是：

```text
pitch 明确占据独立 one-hot 子空间
score feature 与 performance feature 不共享模糊语义
head / decoder 可以明确知道自己处理的是 score condition 还是 performance history
```

如果后续发现 performance history 不应含 pitch，也可以进一步 ablate：

```text
perf embedding with pitch
perf embedding without pitch
```

但第一版先明确分离 score/perf embedding，避免当前混合状态继续污染判断。

### 16.5 Decoder head 从结构化表示解码

Decoder head 也应与新的 680 维 non-pitch 表示对齐：

```text
decoder hidden -> shared/perf head input
head 解码 timing / duration / velocity / pedal_mode / pedal_start / optional pedal_span
```

Pedal 不再直接输出 4 个 independent continuous pedal values，而是输出可 materialize 的 structured pedal state。

## 17. 建议 ablation 顺序

这些修改会同时改变 head、embedding、teacher forcing 和 materialization。如果一次全部修改，很难判断收益来源。因此建议分阶段：

```text
A. 当前最好 INR baseline
   例如 mln3_pine / beta3_pine / huber_pine 中保留当前代表性结果

B. 只修 decoder input 与 score/perf embedding 分离
   不改 pedal head
   不加 [MASK]

C. 在 B 基础上加入 whole-note [MASK] teacher-forcing dropout
   比较 keep_prob = 1.0 / 0.8 / 0.6 / 0.4

D. 在 C 基础上加入 pedal morphology head
   pedal_mode + pedal_start
   pedal_span 作为 optional auxiliary

E. 可选：pedal_span ablation
   D without span
   D with span auxiliary loss
```

每一步都必须同时报告：

```text
teacher-forced eval loss
deterministic free inference MAE/Wass
sampling free inference MAE/Wass
timing drift diagnostics
pedal collapse diagnostics
feature distribution overlays
```

当前最重要的评估不是“loss 是否更低”，而是：

```text
free inference 是否不再 timing drift
pedal 是否不再 collapse 到全 0 / 全 127
生成分布是否接近 ASAP label
```

## 18. 当前总判断

截至 2026-06-20，INR continuous head 的失败不能再简单解释为：

```text
单 Beta 太弱，所以换成更自由的 head 就能解决。
```

更准确的判断是：

```text
1. soft categorical CE 没有解决问题，分类自由度本身不是充分条件。
2. mln3 / beta3 等更灵活 continuous probabilistic head 与 beta head 结果近似，说明 head family 不是唯一瓶颈。
3. free inference 的 exposure bias、decoder input 语义、score/perf embedding 混合、pedal 4D 自由表示，可能共同导致 timing drift 和 pedal collapse。
4. 下一阶段应从结构修正入手：clean embedding、whole-note MASK、pedal morphology。
```

因此新的 INR 主线是：

```text
continuous / ordinal-aware regression-oriented INR
+ clean score/perf embedding separation
+ whole-note [MASK] teacher-forcing dropout
+ morphology-aware pedal head
+ free-inference-first diagnostics
```
