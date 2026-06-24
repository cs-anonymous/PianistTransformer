# Representation Metrics

本文档说明当前 EPR 实验中不同 note representation 的评估口径。这里的核心目标不是比较不同模型规模或不同训练策略，而是比较 score-to-performance 建模时，`score note` 被表示成什么形式，以及 decoder/head 从什么 latent space 读出 performance attributes。

当前建议把 representation 分成四类：

```text
PT, fine, pine, sp
```

其中 `fine / pine / sp` 都属于 INR-style note-level representation：每个音符对应一个 embedding，输出目标一致，训练策略一致。`PT` 是 PianistTransformer 原始 token-level baseline。

## 1. 四种 Representation

### 1.1 PT

`PT` 指 PianistTransformer 原始表示。

主要特点：

- 每个音符被展开为 8 个 token。
- encoder 端有专门的 note encoder，会把 score note token block 编码成 note-level 向量。
- decoder 端仍然生成 token 序列。
- 输出层是 LM head，训练目标是 token-level categorical cross entropy。
- EPR 输出通过离散 token 反序列化成 MIDI performance。

因此，PT 和 INR 的差异不只在 input embedding，也包括：

- 输出空间：token sequence vs note-level structured attributes。
- head 类型：LM head vs continuous / distributional heads。
- loss 类型：categorical CE vs attribute NLL。
- 推理误差形态：token 级错误会通过反序列化影响 note-level timing / duration / velocity / pedal。

所以 PT 适合作为原始系统 baseline，但不应被解释为单纯的 `note_embedding_mode` ablation。

### 1.2 fine

`fine` 是 INR 的 full-space integrated note embedding。

每个音符被编码为一个完整 hidden-size 向量：

```text
z_note = z_pitch + z_shared + z_score + z_pedal + z_type
```

其中不同 feature group 都投影到完整 hidden space 后相加。decoder/head 默认读取完整 hidden state：

```text
head_input_mode = full
```

主要特点：

- 每个 note 一个 embedding。
- pitch / shared / score / pedal 在同一个 hidden space 内融合。
- head 可以从完整 hidden state 读取信息。
- 表示最自由，结构约束最弱。

### 1.3 pine

`pine` 是 INR 的 partitioned integrated note embedding。

每个音符仍然是一个 hidden-size 向量，但 hidden dimension 被切成语义分区：

```text
z_note = [z_pitch ; z_shared ; z_score ; z_perf]
```

例如当前常用配置：

```text
pitch: 128
shared: 256
score: 256
perf: 128
```

decoder/head 默认只读取对应分区：

```text
head_input_mode = partitioned
```

主要特点：

- 每个 note 一个 embedding。
- latent space 有显式语义分区。
- 输出 head 只能从对应语义 block 解码。
- 结构约束更强，有助于解释，但可能限制跨属性信息流。

### 1.4 sp

`sp` 指当前代码中的 `score_perf` / `score_perf_split` 表示。

它也是每个音符一个 embedding，但 score-side 和 performance-side 使用不同的 feature layout：

```text
score input: pitch one-hot + score feature projection
perf input:  pitch one-hot + performance feature projection
```

当前实现中，score encoder 使用 `shared + score` 特征，performance/decoder side 使用 `shared + pedal` 特征。head 在 `feature` / `partitioned` 模式下主要从 feature projection 部分读出。

主要特点：

- 每个 note 一个 embedding。
- score 和 performance 不是完全统一的 note object layout。
- 输入侧更接近 score/perf 分离建模。
- 如果性能变差，可能说明“score/perf 使用不同表示空间”削弱了统一 INR 接口。

因此，`sp` 可以作为一个重要 ablation：它测试的不是 FINE/PINE 的 full vs partition 差异，而是统一 note representation 是否必要。

## 2. 可比性原则

为了让 representation 指标可解释，`fine / pine / sp` 的实验应保持以下变量一致：

- 同一训练数据，例如 `processed_raw + pedal4`。
- 同一 train/test split。
- 同一 backbone scaffold。
- 同一 head family，例如 `mln3 + inflated`。
- 同一 `prior_token_keep_prob`。
- 同一 batch size、训练步数、学习率 schedule。
- 同一 deterministic 和 sampling 推理脚本。
- 同一 PN/PP Wasserstein 评估脚本。

`PT` 可以放在同一评估表中，但需要标注为 original token baseline，因为它的 decoder/head/loss 与 INR 三种 representation 不同。

## 3. 主指标

当前建议主报告两个 Wasserstein 指标：

```text
PN-Wass
PP-Wass
```

并且分别报告：

```text
deterministic
sampling
```

### 3.1 PN-Wass

`PN-Wass` 表示 per-note Wasserstein。

计算口径：

- 对同一个 score note，收集多个 reference performances 的 label 分布。
- 对同一个 score note，收集模型生成的 pred 分布。
- 在每个 note 上计算 pred distribution 和 label distribution 的 Wasserstein distance。
- 再对所有 valid notes 聚合平均。

它衡量的是：

```text
模型是否在每个具体 score note 上生成了合理的演奏分布
```

因此 PN-Wass 更关注 note-conditioned expressiveness。它比 piece-level 指标更严格，因为不能只匹配全曲边缘分布，还要在对应 note 上匹配。

### 3.2 PP-Wass

`PP-Wass` 表示 per-piece Wasserstein。

计算口径：

- 对每首曲子，把所有 notes 的某个属性合并成 piece-level distribution。
- 分别得到 reference performances 的 piece-level label distribution 和模型 pred distribution。
- 在曲目级计算 Wasserstein distance。
- 再对所有 pieces 聚合平均。

它衡量的是：

```text
模型生成的整首曲子的 timing / duration / velocity / pedal 边缘分布是否像真实演奏
```

PP-Wass 比 PN-Wass 更宽松。模型即使没有在每个 note 上预测准确，只要整体分布像真实演奏，PP-Wass 也可能较好。

## 4. 为什么不把 MAE 作为主指标

MAE 衡量的是 point-wise absolute error：

```text
|pred_i - label_i|
```

但 EPR 是 one-to-many 任务。同一个 score note 可以有多个合理演奏，因此 deterministic prediction 和某一个 reference 的点对点误差不一定能反映生成质量。

MAE 仍然可以作为 diagnostic：

- 检查 deterministic 输出是否偏离过大。
- 定位 timing / duration 是否出现系统性 shift。
- 和旧实验结果对齐。

但主指标建议用 PN-Wass / PP-Wass，因为它们更符合多演奏分布匹配的任务定义。

## 5. 推荐结果表

### 5.1 Deterministic

| Representation | Data | Head | PN-IOI | PN-Dur | PN-Vel | PN-Pedal | PP-IOI | PP-Dur | PP-Vel | PP-Pedal |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PT | PT data protocol | LM head | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| fine | processed_raw + pedal4 | mln3 + inflated | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| pine | processed_raw + pedal4 | mln3 + inflated | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| sp | processed_raw + pedal4 | mln3 + inflated | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

### 5.2 Sampling

| Representation | Samples per score | Data | Head | PN-IOI | PN-Dur | PN-Vel | PN-Pedal | PP-IOI | PP-Dur | PP-Vel | PP-Pedal |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PT | TBD | PT data protocol | LM head | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| fine | TBD | processed_raw + pedal4 | mln3 + inflated | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| pine | TBD | processed_raw + pedal4 | mln3 + inflated | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| sp | TBD | processed_raw + pedal4 | mln3 + inflated | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

## 6. 解释指标时的优先级

推荐解读顺序：

1. 先看 deterministic PN-Wass。
   这能判断模型在最稳定输出下是否学到了 note-conditioned mapping。

2. 再看 deterministic PP-Wass。
   如果 PN 差但 PP 好，说明模型可能学到了整体风格分布，但没有对准具体 note。

3. 再看 sampling PN-Wass。
   如果 sampling PN 明显好于 deterministic PN，说明采样确实恢复了每个 note 的多样性。

4. 最后看 sampling PP-Wass。
   如果 sampling PP 好但 PN 不好，说明采样只是在全曲层面扩大了分布，不一定是正确的 note-level expressiveness。

## 7. 当前实验口径

当前要比较的 INR 三组 representation 应使用：

```text
fine
pine
sp / score_perf
```

共同配置：

```text
data: processed_raw + pedal4
head: mln3 + inflated
prior_token_keep_prob: 0.5
evaluation: PN-Wass + PP-Wass
inference: deterministic + sampling
```

这组实验最适合回答：

```text
在输出、head、训练策略都统一时，note representation 本身如何影响 EPR 分布匹配？
```

如果 `fine` 明显优于 `pine`，说明完整 hidden-space 融合更适合当前任务。  
如果 `pine` 明显优于 `fine`，说明语义分区提供了有效 inductive bias。  
如果 `sp` 明显差于 `fine/pine`，说明 score/perf 分离表示可能是核心问题，统一 note representation 更重要。  
如果 `sp` 与 `fine/pine` 接近，则说明此前退化更可能来自 head、pedal 表示、数据版本或训练策略，而不是 score/perf split 本身。

