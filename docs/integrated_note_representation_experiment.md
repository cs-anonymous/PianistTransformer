# INSPIRE / INR 表示对比实验规划

本文档定义一个可执行的 `INSPIRE` 实验：`INSPIRE = Integrated Note-based Score Performance Interpretation, Reconstruction and Expression`。其核心表示层命名为 `INR (Integrated Note Representation)`：将 MIDI 表示从 PianistTransformer 的 8-token note block 改为 note-level continuous node，并在同一表示、同一 PianoCoRe-A 数据源、同一固定 backbone scaffold 下比较两种表示参数化：`FINE` 与 `PINE`。实验目标是先验证表示层本身是否带来收益，再把 `backbone` 与更深的 `embedding/head` 结构作为后续 inductive bias ablation；同一表示也会扩展到 `CSR (Canonical Score Reconstruction)`，把它作为 `EPR` 的反向任务来研究。

当前仓库中的文档标题、脚本文件名与主要实现命名统一使用 `Integrated*` / `INR-*` 口径。

## 0. 当前实现状态

截至当前实现，数据处理已整理为两步 work-level PianoCoRe-A node 流程：

- paired MIDI 生成 JSON：[generate_json_with_paired_midi.py](/home/kaititech/EPR/PianistTransformer/src/data_process/generate_json_with_paired_midi.py)
- XML 更新 score feature：[update_json_score_feature_with_xml.py](/home/kaititech/EPR/PianistTransformer/src/data_process/update_json_score_feature_with_xml.py)
- XML/MIDI 对齐 helper：[score_xml_alignment.py](/home/kaititech/EPR/PianistTransformer/src/data_process/score_xml_alignment.py)
- MIDI/node 工具：[src/utils/inr_midi.py](/home/kaititech/EPR/PianistTransformer/src/utils/inr_midi.py)
- Integrated 模型：[src/model/integrated_pianoformer.py](/home/kaititech/EPR/PianistTransformer/src/model/integrated_pianoformer.py)
- SFT 训练入口：[src/train/train_inr.py](/home/kaititech/EPR/PianistTransformer/src/train/train_inr.py)
- 训练配置：[configs/inr_config_pianocore.json](/home/kaititech/EPR/PianistTransformer/configs/inr_config_pianocore.json)
- 启动命令：`python src/train/train_inr.py --config configs/inr_config_pianocore.json`

数据格式已从原计划的 pair-level jsonl 改为 work-level INR JSON：每个作品一个 `*.json`，直接写在 refined score MIDI 旁边。第一步生成 paired MIDI JSON；第二步从 XML/MXL 投影 `score.score_feature` 与 `score.has_score_feature`，并在 `meta.xml_to_refined_score_alignment` 中记录 coverage。

当前已完成全量 PianoCoRe-A 处理：

```text
output: data/pianocore/PianoCoRe/refined/**/*.json
summary: data/pianocore/PianoCoRe/refined/pianocore_a_node_summary.json
works_total: 1936
success_works: 1936
success_performances: 157198
failed_performances: 9
failed reason: pitch_mismatch
INR JSON total size: ~16G
score feature note-level coverage: 91.14%
```

当前已落地的第一版模型为 `IntegratedPianoT5Gemma`：它只作为固定 scaffold，不进入第一阶段比较；第一阶段只比较 `FINE` 与 `PINE`。Pitch 只作为输入 embedding，不作为预测目标；decoder 使用最简 3-head 方案分别处理 `Shared / Pedal / Score`。

需要特别说明：当前落盘数据已经是 `pianocore_integrated_node_work_v2` schema，但当前训练代码仍主要服务第一阶段 EPR baseline；CSR 和完整对称 encoder/decoder 训练会在下一步实现。

第一阶段 FINE/PINE 正式对比不默认迁移旧 PianistTransformer checkpoint，而是从随机初始化开始直接在 PianoCoRe-A 上训练。原因是 PianoCoRe-A 已有约 15.7 万个 aligned performance pairs，监督数据规模足够支撑目标任务训练；如果使用大规模 unpaired MIDI 预训练，会额外引入预训练语料、预训练目标和初始化差异，不利于公平比较表示参数化本身。迁移旧 checkpoint 与 backbone 对比都放到后续 inductive bias ablation。

训练数据集已实现为 map-style Dataset，而不是 IterableDataset。样本索引映射到 `(work, performance, window)`，并使用每进程 LRU cache 缓存最近 work JSON；这样 DDP 的 DistributedSampler 可以稳定分片，避免 IterableDataset 在多卡下 batch dispatch 和尾部耗尽不一致的问题。

当前本地已启动 3 卡 INR SFT：

```bash
tmux attach -t inr
tail -f logs/inr_*.log
```

默认配置为 1000 steps，`block_notes=512`，3 卡训练，`per_device_train_batch_size=2`，`gradient_accumulation_steps=16`，每 500 step 保存 checkpoint。

## 1. 核心假设

当前 PianistTransformer 已经在 encoder 端把每 8 个 token 合并为一个 note embedding，但 decoder 和 loss 仍然是 token-level classification。`INR (Integrated Note Representation)` 将输入和输出都改为 note-level object：

```text
unified note object
    -> IntegratedNoteEmbedding
    -> Transformer backbone
    -> decoder heads
    -> target note attributes
```

本实验将表示层和 backbone 解耦。所有模型共享同一 `INR` interface：

- 数据从 token ids 变为 note feature tensor。
- 输入端从 `PianoEncoderEmbeddings` 改为 `IntegratedNoteEmbedding`。
- 输出端从 `lm_head + cross entropy` 改为 `decoder heads + probabilistic / structured loss`。
- EPR performance-side continuous variables 统一使用 `Beta(mu/kappa)` NLL，包括 timing、velocity 与 pedal。

当前第一阶段只比较 `FINE` 与 `PINE`。`backbone`、更深的 `embedding/head` 以及不同 backbone family 作为后续 inductive bias ablation，不进入第一阶段主实验。

### 1.1 任务族：EPR 与 CSR

在 Integrated Note Representation 下，可以把当前工作统一看成一类 note-level conditional mapping，而不仅仅是单一的 `score -> performance`。

#### EPR

```text
EPR = Expressive Performance Rendering
input:  score note sequence
output: performance note sequence
```

EPR 的目标是在不改变音高拓扑的前提下，为每个 note 预测演奏层参数，例如：

- onset / IOI
- duration
- velocity
- pedal trajectory

这本质上是：

```text
canonical symbolic structure -> expressive realization
```

#### CSR

```text
CSR = Canonical Score Reconstruction
input:  performance note sequence
output: canonical score note sequence
```

CSR 可以看作 EPR 的反向任务。给定一条演奏序列，模型需要恢复更规范、更接近乐谱语义的 note-level 表示，例如：

- 将 rubato 后的 IOI 恢复到 canonical timing
- 将 performance duration 恢复到更接近记谱时值的 duration
- 去除 pedal / sustain 对表面时值的干扰
- 保留音高与 note 对齐关系，恢复更稳定的 score-side note attributes

它本质上是：

```text
expressive realization -> canonical symbolic structure
```

#### 为什么 CSR 很重要

如果 EPR 成立，说明 Integrated Note Representation 可以把“从乐谱到演奏”的表达映射建模为 note-level conditional prediction。  
而如果 CSR 也成立，则说明同一表示还能支持“从演奏回到规范乐谱”的逆向映射。两者合起来会让 INR 不只是一个 EPR trick，而是一套更通用的 note-level symbolic-performance interface。

#### 为什么 CSR 可能更适合 BERT

当前 `INR-BERT` 在 EPR 上的结果不理想，但这并不意味着 encoder-only 思路没有价值。相反，CSR 很可能正是更适合 BERT 的任务类型。原因是：

1. `EPR` 更接近 one-to-many conditional generation。  
   同一个 score 往往对应多种合理演奏，模型容易学成 conditional mean。

2. `CSR` 更接近 many-to-one canonicalization。  
   多种具体演奏细节会被投影回同一个更稳定的 canonical score 表示，因此目标熵通常更低。

3. 如果 CSR 仍保持 note 对齐、输入输出长度一致，那么它天然适合：

```text
performance note sequence
  -> bidirectional contextual encoder
  -> per-note canonical score prediction
```

也就是 encoder-only 的并行结构回归。

因此，当前可以提出一个明确假设：

```text
INR-BERT 可能不适合 high-entropy 的 EPR，
但可能适合 lower-entropy 的 CSR。
```

这也是后续非常值得单独验证的方向。

## 2. 数据来源

本实验只使用 PianoCoRe-A refined pair，因为 refined data 已经提供 note-by-note aligned score/performance：

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

推荐正式 FINE/PINE 对比使用全量 PianoCoRe-A：

```text
metadata.csv
tier_a=True
```

原因：

- PianoCoRe-A 数据量已经足够大，当前已处理成功 `157198` 个 performance pairs。
- 本实验第一阶段目标是比较 FINE/PINE 表示参数化，而不是验证 unpaired pretraining 或不同 backbone 的收益。
- 不使用大规模 unpaired MIDI pretrain 可以避免额外变量，使比较更干净。

如果需要快速 smoke test 或首轮调参，可以临时使用：

```text
metadata_S.csv
```

并过滤：

```python
df = df[df["tier_a_star"] == True]
df = df[df["refined_score_midi_path"].notna()]
df = df[df["refined_performance_midi_path"].notna()]
```

临时使用 `metadata_S.csv` 的原因：

- `metadata_S.csv` 更小，适合 smoke test 或首轮 debug。
- `tier_a_star=True` 是最高置信 note-level alignment 子集。
- `refined_score_midi_path` 和 `refined_performance_midi_path` 的音符数一致。
- `refined_alignment_path` 里有 `interpolated` mask，可用于后续加权 loss 或分析。

暂不使用 `align_score_and_performance`。PianoCoRe refined pair 已经完成对齐和插值。

## 3. Note Feature Schema

这一版 Integrated Note Representation 不再把 note 仅仅视为 “8 维 EPR 连续向量”，而是把它定义为一个统一的 note object。  
这个 object 同时支持：

- `EPR`: `score -> performance`
- `CSR`: `performance -> canonical score`

也就是说，score note 和 performance note 共用同一套外层结构，但拥有不同的 feature group。

需要区分两层 schema：

- **Storage schema**: 磁盘上的 `*.json` 字段统一视为 INR payload，优先保持简单、兼容旧数据。
- **Model schema**: encoder / decoder 内部按统计性质拆分的 feature group。

因此，文档中的 `timing`, `velocity`, `pedal`, `score_position`, `measure_structure`, `score_annotation` 是模型概念分组；落盘时统一合并为更紧凑的 `score_feature` 数组字段，并用 `has_score_feature` 标记该 note 是否成功投影到 XML-derived score feature。

### 3.1 统一 note object

每个 note object 由下列字段组成：

| Group | Fields | Type | Scope | 说明 |
|------|------|------|------|------|
| `pitch` | `pitch` | categorical id | shared | MIDI pitch，范围 `[0, 127]` |
| `note_type` | `has_score_feature`, `has_pedal_feature` | 2-d binary | shared | 分别表示该 note 是否拥有 score-side feature / pedal-side feature |
| `timing` | `dur`, `ioi` | continuous | shared | note 时值与相邻 onset 间隔 |
| `velocity` | `vel` | continuous | shared | note dynamics |
| `pedal` | `pedal_1..4` | continuous | perf-only | 4 个 pedal snapshot |
| `score_position` | `mo`, `md` | normalized ordinal | score-only | 小节内位置相关的 canonical score timing |
| `measure_structure` | `ml`, `first` | normalized ordinal + binary | score-only | 小节长度与是否为小节首音 |
| `score_annotation` | `staff`, `trill`, `grace`, `staccato` | binary | score-only | 基本 note-level 记谱属性 |

除了 `pitch` 之外，note object 中所有数值字段都应该归一化到 `[0, 1]` 后再进入模型。  
因此这里的 `ordinal` 不是 raw integer / raw quarter-length，而是归一化后的 ordinal scalar；它仍然保留自然顺序，但数值尺度和其它连续 / binary 字段一致。

这里最重要的设计点是：

1. `score note` 并不是“没有 performance 属性”。  
   score note 仍然有 `vel`, `dur`, `ioi`，因为它来自 score-to-MIDI 渲染。
2. `performance note` 才是没有 score-side 属性。  
   它没有 `mo`, `md`, `ml`, `staff`, `first`, `trill`, `grace`, `staccato`。
3. `pedal` 不属于 shared group，而是明确属于 performance-only。
4. presence 不再用单个 `score_note` 表示，而是使用二维 `note_type = [has_score_feature, has_pedal_feature]`。  
   这样可以表达三种常见 note object：只有 shared feature、shared + score feature、shared + pedal feature。
5. 对落盘 score sequence 来说，`has_score_feature = 1` 表示该 refined score note 成功从 XML/MXL 投影得到 `score_feature`；如果投影失败，则该 note 仍保留 `pitch + score_continuous`，但 `has_score_feature = 0`。

### 3.2 常见 note object 的实例化

#### Score Note with XML-derived score feature

| Field | Value Type | Used |
|------|------|------|
| `pitch` | categorical id | always |
| `note_type` | `[1, 0]` | always |
| `dur`, `ioi` | continuous | always |
| `vel` | continuous | always |
| `pedal_1..4` | not stored | perf-only |
| `mo`, `md` | normalized ordinal scalar | if `has_score_feature = 1` |
| `ml` | normalized ordinal scalar | if `has_score_feature = 1`; mainly meaningful at measure start |
| `first` | binary | if `has_score_feature = 1` |
| `staff`, `trill`, `grace`, `staccato` | binary | if `has_score_feature = 1` |

#### Score Note without XML-derived score feature

| Field | Value Type | Used |
|------|------|------|
| `pitch` | categorical id | always |
| `note_type` | `[0, 0]` | always |
| `dur`, `ioi` | continuous | always |
| `vel` | continuous | always |
| `pedal_1..4` | not stored | perf-only |
| `mo`, `md`, `ml`, `first` | masked / ignored | not supervised |
| `staff`, `trill`, `grace`, `staccato` | masked / ignored | not supervised |

这种 note 是 XML/MXL 到 refined score MIDI 映射未覆盖的位置。它仍然是 score-side input 的一部分，因为 pitch 与 score-rendered timing/velocity 仍可用；只是不能声称拥有可靠的 score-side notation feature。

#### Performance Note with pedal feature

| Field | Value Type | Used |
|------|------|------|
| `pitch` | categorical id | always |
| `note_type` | `[0, 1]` | always |
| `dur`, `ioi` | continuous | always |
| `vel` | continuous | always |
| `pedal_1..4` | continuous | always |
| `mo`, `md` | not stored | score-only |
| `ml`, `first` | not stored | score-only |
| `staff`, `trill`, `grace`, `staccato` | not stored | score-only |

因此，最常见的模式是：

```text
score note with XML feature: [has_score_feature=1, has_pedal_feature=0]
score note without XML feature: [has_score_feature=0, has_pedal_feature=0]
performance note: [has_score_feature=0, has_pedal_feature=1]
```

### 3.3 为什么这样分组

这样划分不是为了“把字段尽量堆在一起”，而是为了让统计性质接近的字段进入同一组：

- `timing = [dur, ioi]`  
  都是连续时间量，但它们描述的是 physical time。
- `velocity = [vel]`  
  单独成组，因为它的分布和 timing 明显不同。
- `pedal = [pedal_1..4]`  
  是 performance-only 的连续控制，和 note body 本身不同。
- `score_position = [mo, md]`  
  是 canonical score-time 的 ordinal 变量，不应和 `log1p` 的 physical timing 混用。
- `measure_structure = [ml, first]`  
  `ml` 和 `first` 语义绑定最紧，而且 `ml` 主要在小节起始音上最有意义。
- `score_feature[..., 4:8] = [staff, trill, grace, staccato]`  
  都是轻量记谱标签，适合合并成一个 binary group。

### 3.4 Ordinal 不等于 categorical one-hot

在这个表示里：

- `mo`, `md`, `ml` 属于 `ordinal`
- `staff`, `first`, `trill`, `grace`, `staccato` 属于 `binary`

我们不把 `mo/md/ml` 当成普通 one-hot categorical class。  
更合理的做法是：

1. 输入侧使用归一化到 `[0, 1]` 的 ordinal scalar；
2. 经轻量 projection 后进入模型，而不是 one-hot class embedding；
3. 输出侧预测 `[0, 1]` 范围内的连续标量；
4. decode 时再反归一化并 quantize 到合法的 score grid。

这样更符合这些变量本身具有自然顺序的事实。

## 4. 连续字段归一化

第一阶段需要明确区分两类“时间”：

1. `performance-like physical time`: `dur`, `ioi`
2. `canonical score time`: `mo`, `md`, `ml`

这两类量的语义不同，不应强行共用同一种归一化。

### 4.1 全局数值约束

除了 `pitch` 之外，所有输入模型的 note features 都应该位于 `[0, 1]`：

| Field Type | Normalized Range |
|------|------|
| `pitch` | integer id, not normalized |
| binary fields | `{0, 1}` |
| `vel`, `pedal_1..4` | `[0, 1]` |
| `dur`, `ioi` | `[0, 1]` after `log1p` normalization |
| `mo`, `md`, `ml` | `[0, 1]` after ordinal range normalization |

这个约束非常重要，因为它让 group-wise projection、decoder heads 和 regression loss 都工作在相近的数值尺度上。

### 4.2 Pitch 与二值字段

- `pitch` 保留整数 id，直接进入 embedding
- `has_score_feature`, `has_pedal_feature`, `first`, `staff`, `trill`, `grace`, `staccato` 保留为 `0/1`

### 4.3 Shared continuous: `dur`, `ioi`, `vel`

对于 `dur` 和 `ioi`，继续使用对长尾更稳定的 `log1p` 归一化：

```python
MAX_TIME_MS = 10000

time_ms = min(max(time_ms, 0.0), MAX_TIME_MS)
time_norm = log1p(time_ms) / log1p(MAX_TIME_MS)
```

注意这里的 `ioi` 仍然定义为：

```text
当前 note onset 相对前一个 note onset 的差
```

而不是相对该小节首音或该乐句首音的绝对位置。  
这说明它描述的是 performance-like local flow，而不是 canonical score grid。

`vel` 使用简单线性归一化：

```python
vel_norm = velocity / 127.0
```

### 4.4 Perf-only continuous: `pedal_1..4`

```python
pedal_norm = cc64_value / 127.0
```

它们依然是连续目标，不建议在第一阶段强行二值化。  
因为真实数据里可能存在半踏板与非理想转录，直接二值化会让目标分布和训练设定产生偏差。

### 4.5 Score-only ordinal: `mo`, `md`, `ml`

`mo`, `md`, `ml` 不应使用 `log1p` continuous time normalization。  
它们更接近 canonical score lattice 上的 ordinal 坐标，但进入模型前仍然必须归一化到 `[0, 1]`。

归一化常量应直接来自 `MIDI2ScoreTransformer` 的 tokenizer 参数定义，而不是从当前训练集统计 observed max。当前仓库实际引用的是：

- `offset`: `[0, 6]`，步长 `1/24`
- `duration`: `[0, 4]`，步长 `1/24`
- `downbeat`: `[-1/24, 6]`，步长 `1/24`

在当前 Integrated Note schema 中，这三者对应关系是：

- `mo = offset`，表示 note 在小节内的 onset offset
- `md = duration`，表示 note 的记谱时值
- `ml = measure length`，来自 `downbeat` 的小节长度语义；同时单独保留 `first` 作为是否为小节首音的 binary 标记

因此推荐在实现中显式定义：

```python
MO_MIN, MO_MAX = 0.0, 6.0
MD_MIN, MD_MAX = 0.0, 4.0
ML_MIN, ML_MAX = 0.0, 6.0
DOWNBEAT_MIN = -1.0 / 24.0
SCORE_GRID = 1 / 24
```

当前表示里，`mo/md/ml` 都已经转成 `[0, 1]` 中的 scalar，正向映射公式应写死为：

```python
mo_norm = clamp((mo - MO_MIN) / (MO_MAX - MO_MIN), 0.0, 1.0)
md_norm = clamp((md - MD_MIN) / (MD_MAX - MD_MIN), 0.0, 1.0)
ml_norm = clamp((ml - ML_MIN) / (ML_MAX - ML_MIN), 0.0, 1.0)
```

也就是在当前范围下可化简为：

```python
mo_norm = clamp(mo / 6.0, 0.0, 1.0)
md_norm = clamp(md / 4.0, 0.0, 1.0)
ml_norm = clamp(ml / 6.0, 0.0, 1.0)
```

反向还原时，先从 `[0, 1]` 恢复到原始 quarter-length 标度，再量化到 `1/24` 拍格点：

```python
x = x_norm * (X_MAX - X_MIN) + X_MIN
x = round(x / SCORE_GRID) * SCORE_GRID
```

对 `mo/md/ml` 分别展开，就是：

```python
mo = round((mo_norm * 6.0) / (1.0 / 24.0)) * (1.0 / 24.0)
md = round((md_norm * 4.0) / (1.0 / 24.0)) * (1.0 / 24.0)
ml = round((ml_norm * 6.0) / (1.0 / 24.0)) * (1.0 / 24.0)
```

等价整数写法：

```python
mo = round(mo_norm * 144.0) / 24.0
md = round(md_norm * 96.0) / 24.0
ml = round(ml_norm * 144.0) / 24.0
```

如果需要还原成 `MIDI2ScoreTransformer` 的原始 `downbeat` 语义，则应结合 `first`：

```python
if first == 1:
    downbeat = ml
else:
    downbeat = DOWNBEAT_MIN  # 即 -1/24，表示当前 note 不是小节首音
```

这也解释了为什么当前 schema 里 `ml` 和 `first` 要拆开：

- `first` 决定是否是 measure start
- `ml` 只在 `first == 1` 时真正有意义
- 非小节首音处的 `ml` 可以视为 masked / ignored，或者保留为 0

关于落盘精度，`[0, 1]` 保留 5 位小数已经足够覆盖这个量化体系。因为：

- `mo` 的归一化最小步长是 `(1/24) / 6 = 1/144 ≈ 0.00694444`
- `md` 的归一化最小步长是 `(1/24) / 4 = 1/96 ≈ 0.01041667`
- `ml` 的归一化最小步长同样是 `1/144 ≈ 0.00694444`

而 5 位小数的舍入误差上界只有 `0.000005`，比最小量化步长小约 `694x` 到 `2083x`。  
因此只要 decode 时仍然执行“反归一化后再 round 到 `1/24` grid”，5 位小数不会破坏 `mo/md/ml` 的离散档位。

因此完整策略是：

1. raw score parameter 按论文理论范围归一化到 `[0, 1]`；
2. encoder 看到的是 normalized ordinal scalar；
3. decoder 输出 normalized ordinal scalar；
4. 最终 decode 时反归一化并量化到合法 grid，例如 `1/24` 拍分辨率。

也就是说：

```text
dur/ioi 负责描述“演奏时间”
mo/md/ml 负责描述“乐谱时间”
```

这两套时间轴在表示层应该并存，而不是合并。

## 5. 数据文件格式

正式数据处理入口：

```text
src/data_process/generate_json_with_paired_midi.py
src/data_process/update_json_score_feature_with_xml.py
```

当前第一阶段已落地的数据写成 work-level `*.json`。第一步只依赖 paired refined MIDI 与 alignment；第二步在同一个 JSON 上补充 XML-derived score-side feature。独立的 `07_audit_score_xml_to_refined_alignment.py` 不再作为主流程入口保留，coverage 由第二步的 summary/details 直接输出。

推荐输出：

```text
data/pianocore/PianoCoRe/refined/**/*.json
```

### 5.1 推荐的简化存储 schema

```json
{
  "schema": "pianocore_integrated_node_work_v2",
  "meta": {
    "score_source": ".../score_PDMX_refined.mid",
    "score_xml_source": ".../score.mxl",
    "score_midi_source": ".../score_PDMX.mid",
    "xml_to_refined_score_alignment": {
      "method": "midi2scoretransformer_parse_mxl + pitch_aware_monotonic_alignment",
      "matched": 403,
      "unmatched": 0
    }
  },
  "score": {
    "pitch": [60, 64, 67],
    "score_continuous": [[0.0, 0.12, 0.63], "..."],
    "score_feature": [[0.0, 0.25, 0.75, 1.0, 0.0, 0.0, 0.0, 1.0], "..."],
    "has_score_feature": [1, 1, 0],
    "note_count": 403
  },
  "performances": [
    {
      "id": "PianoCoRe_xxxxxx",
      "performance_source": ".../Aria_xxx_refined.mid",
      "alignment_source": ".../Aria_xxx_refined_align.npz",
      "split": "train",
      "tier_a_star": true,
      "label_continuous": [[0.0, 0.10, 0.63, 0.0, 0.0, 0.12, 0.4], "..."],
      "interpolated": [0, 0, 1]
    }
  ]
}
```

字段顺序固定为：

```text
score.pitch:
  [pitch]

score.score_continuous:
  [ioi_norm, duration_norm, velocity_norm]

score.score_feature:
  [mo_norm, md_norm, ml_norm, first, staff, trill, grace, staccato]

score.has_score_feature:
  [1/0]

performance.label_continuous:
  [ioi_norm, duration_norm, velocity_norm, pedal_0, pedal_25, pedal_50, pedal_75]
```

这个 schema 的好处是：

- `score.pitch` 和 `performances[].label_continuous` 基本沿用当前实现。
- 旧版 `score.score_continuous` 从 7 维收缩为 3 维，只保留 score note 自身存在的 shared continuous fields。
- 新增 score-side 信息集中在一个 8 维 `score_feature` 数组里，便于存储和切片。
- `has_score_feature` 明确区分该 refined score note 是否成功从 XML/MXL 对齐并获得 score-side feature；没有成功投影时仍保留 shared feature，但 score feature 不参与对应 loss。
- 每个 work JSON 的 performance 列表不需要因为 score-side feature 增加而重复写入冗余字段。
- EPR 训练仍然可以直接读取 `score` 作为输入、`performances[].label_continuous` 作为目标。

### 5.2 与模型内部 feature group 的对应关系

存储 schema 和模型内部 group 的对应关系如下：

| Storage Field | Shape | Model Group |
|------|------|------|
| `score.pitch` | `[N]` | `pitch` |
| `score.score_continuous[..., 0:2]` | `[N, 2]` | `timing = [ioi, duration]` |
| `score.score_continuous[..., 2:3]` | `[N, 1]` | `velocity = [velocity]` |
| `score.score_feature[..., 0:2]` | `[N, 2]` | `score_position = [mo, md]` |
| `score.score_feature[..., 2:4]` | `[N, 2]` | `measure_structure = [ml, first]` |
| `score.score_feature[..., 4:8]` | `[N, 4]` | `score_annotation = [staff, trill, grace, staccato]` |
| `score.has_score_feature` | `[N]` | score-side feature mask / `note_type[..., 0]` |
| `performances[].label_continuous[..., 0:3]` | `[N, 3]` | target `timing + velocity` |
| `performances[].label_continuous[..., 3:7]` | `[N, 4]` | target `pedal` |

## 6. 序列切片

当前 token 方案的 `block_size=4096` 等价于：

```text
4096 tokens / 8 = 512 notes
```

Node 方案直接以 note 为单位切片。第一版配置：

```json
{
  "block_notes": 512,
  "overlap_ratio": 0.125,
  "min_notes": 64
}
```

切片逻辑：

```python
window_len = block_notes
stride = int(block_notes * (1 - overlap_ratio))
# With block_notes = 512 and overlap_ratio = 0.125, this yields stride = 448
```

不足 `min_notes` 的片段丢弃。

`block_notes=512` 不是缩短 context，而是和旧方案的 `block_size=4096` 保持相同的实际音乐上下文。若设置 `block_notes=4096`，实际等价于旧 tokenizer 的 `32768 tokens`，会引入更长上下文和显存成本这两个额外变量，不适合作为第一版公平对照。

## 7. 模型结构

新增模型文件：

```text
src/model/integrated_pianoformer.py
```

### 7.1 Config

新增：

```python
class IntegratedPianoT5GemmaConfig(PianoT5GemmaConfig):
    continuous_dim = 7
    max_time_ms = 10000
    pitch_vocab_size = 128
    pitch_pad_id = 128
```

这里的 `continuous_dim = 7` 是当前第一阶段 EPR baseline 的实现参数。  
如果切换到完整对称 schema，配置层会从“单个 flat continuous_dim”转向“按 feature group 显式建模”。

注意：`pitch` 本身只有 128 个合法 MIDI pitch：

```text
0..127
```

为了 padding，pitch embedding 可以设为 129：

```python
nn.Embedding(129, hidden_size, padding_idx=128)
```

这里 `128` 不是真实 pitch，而是 `pitch_pad_id`。它只用于 batch padding，配合 `attention_mask=0` 让 padding note 不参与 attention / loss。

但 `PAD` 与 `BOS/EOS/MASK/PLAY` 语义不同。PT 的 token vocabulary 中保留了：

```text
0 = PAD
1 = MASK
2 = BOS
3 = EOS
4 = PLAY
```

INR/FINE/PINE 也应保留 note-level special embeddings。推荐定义：

```text
special_note_id:
  0 = PAD_NOTE
  1 = MASK_NOTE
  2 = BOS_NOTE
  3 = EOS_NOTE
  4 = PLAY_NOTE
```

真实 note 不使用 `special_note_id`，而是使用：

```text
pitch in [0, 127]
feature groups
```

特殊 note 不代表真实 pitch，也不应伪装成某个 pitch。实现上可以使用一个单独的 learned table：

```python
special_note_embeddings = nn.Embedding(5, hidden_size)
```

对于 PINE，也可以使用同一个 full hidden-size special embedding，并在特殊 note 位置直接替换完整 `768d` note embedding。这样最简单，也避免为每个 partition 单独设计 BOS/PAD。

第一版中：

- `PAD_NOTE` 用于 batch padding，仍然由 `attention_mask=0` 屏蔽。
- `BOS_NOTE` 用于 AR decoder 的第一个输入 embedding。
- `EOS_NOTE` 只有在未来支持变长生成时才需要。
- `MASK_NOTE` 可用于后续 denoising / masked note pretraining。
- `PLAY_NOTE` 可作为未来 task/control token，第一版可先不用。

### 7.2 IntegratedNoteEmbedding: FINE 与 PINE

数据 loader 从简化 storage schema 读入：

```python
pitch_ids = batch["score"]["pitch"]
score_continuous = batch["score"]["score_continuous"]      # [ioi, duration, velocity]
score_feature = batch["score"]["score_feature"]            # [mo, md, ml, first, staff, trill, grace, staccato]
has_score_feature = batch["score"]["has_score_feature"]    # [1/0]
```

进入模型前再拆成内部 feature group：

```python
pitch_ids: LongTensor[B, N]
note_type: FloatTensor[B, N, 2]            # [has_score_feature, has_pedal_feature]
timing: FloatTensor[B, N, 2]              # [dur, ioi]
velocity: FloatTensor[B, N, 1]            # [vel]
score_position: FloatTensor[B, N, 2]      # [mo, md]
measure_structure: FloatTensor[B, N, 2]   # [ml, first]
score_annotation: FloatTensor[B, N, 4]    # [staff, trill, grace, staccato]
```

注意：EPR 的 score 输入不包含 `pedal`。  
`pedal` 只存在于 performance-side target，也就是 `performances[].label_continuous[..., 3:7]`。如果后续做 CSR 并把 performance note 作为输入，才会在 input encoder 侧启用 pedal projection。

EPR 输入通常是：

```text
note_type = [has_score_feature, 0]
```

CSR 输入通常是：

```text
note_type = [0, 1]
```

如果某个 score note 没有成功投影到 XML-derived score feature，则：

```text
note_type = [0, 0]
```

它仍然保留 `pitch + timing + velocity` 这类 shared feature，但不携带 score-side feature。

这一层更准确地说不是传统意义上的 sequence encoder，而是 `IntegratedNoteEmbedding`：它只负责把一个 note object 的结构化属性映射到 backbone 所需的 `hidden_size` 向量。Transformer encoder / decoder / BERT / GPT 才负责上下文建模。

因此，后续统一使用：

```text
IntegratedNoteEmbedding(note object) -> note embedding
```

而不是把 score-side 和 target-side 分别命名为不同的 encoder。  
无论是 score note、performance note，还是 AR 解码时已经生成的 target note，都应通过同一个 Integrated Note interface 构造 embedding；差别只在于哪些 feature slot 被占用。

#### 7.2.1 FINE: Full Integrated Note Embedding

`FINE = Full Integrated Note Embedding`。  
它对应“所有 feature group 都编码到同一个 768 维语义空间，然后相加融合”的方案。

结构：

```python
pitch_emb = PitchEmbedding(pitch_ids)
type_emb = TypeProjection(note_type)
shared_emb = SharedProjection(shared_features)
score_emb = ScoreProjection(score_features)
pedal_emb = PedalProjection(pedal_features)

note_emb = LayerNorm(
    pitch_emb
  + type_emb
  + shared_emb
  + score_emb
  + pedal_emb
)
```

对于 EPR，如果实现上希望复用同一个 general-purpose encoder，也可以保留 `pedal_emb = PedalProjection(zeros[B, N, 4])`，但更推荐在第一版 EPR encoder 中不加入 score-side pedal 输入，避免把不存在的 score pedal 当成真实特征。

推荐第一版采用 group-wise projection + summation，而不是把所有字段直接拼成一个大向量后喂给单个 MLP。continuous feature 到 hidden space 的映射默认使用 2-layer MLP，因为连续变量与 hidden representation 之间通常不是简单线性关系：

```python
SharedProjection:
  Linear(3, hidden_size)
  GELU
  Linear(hidden_size, hidden_size)

ScoreProjection:
  Linear(8, hidden_size)
  GELU
  Linear(hidden_size, hidden_size)

PedalProjection:
  Linear(4, hidden_size)
  GELU
  Linear(hidden_size, hidden_size)
```

也就是说，第一版 embedding/head 结构保持对称的 2-layer MLP，但仍然只在 feature group 内部做局部投影；复杂的上下文建模交给 backbone。

这样做有几个好处：

1. 不同统计性质的字段不会在输入第一层就完全混在一起；
2. 各 group 的 inductive bias 更清晰；
3. encoder 的组织方式可以与 decoder 的 group heads 完整对称。

这里说的“对称”是 feature ontology 对称，不是参数共享。  
也就是说，encoder 和 decoder 围绕同一套 group 组织，但并不需要 weight tying。

FINE 中 `note_type` 或 presence mask 更重要。原因是所有 group embedding 最终都加到同一个 `hidden_size` 空间中，如果没有显式类型或 mask，模型需要仅凭零向量判断 feature group 是否存在：

```text
score_emb = 0 是缺失，还是 score feature 全为 0？
pedal_emb = 0 是缺失，还是真实 pedal 全为 0？
```

因此 FINE 推荐保留：

```text
note_type = [has_score_feature, has_pedal_feature]
has_score_feature
has_pedal_feature
```

它们可以实现为 768 维 learned type embedding，也可以作为 group projection / gate 的输入。  
这里的 type embedding 类似 BERT segment embedding：它不是把两个 bit 本身“扩展”为 768 维信息，而是给 hidden state 加一个可学习的偏置方向，帮助模型区分 score-side 和 performance-side manifold。

#### 7.2.2 PINE: Partitioned Integrated Note Embedding

`PINE = Partitioned Integrated Note Embedding`。  
它对应“按照 feature category 在 hidden vector 中划分固定语义槽位”的方案。一个推荐分区是：

```text
hidden_size = 768

z_note = [
  z_pitch,   # 128 dims
  z_shared,  # 256 dims: ioi / duration / velocity
  z_score,   # 256 dims: mo / md / ml / first / staff / trill / grace / staccato
  z_perf     # 128 dims: pedal_1..4 and future perf-only controls
]
```

Score note 与 performance note 使用同一个 `IntegratedNoteEmbedding`，但占用不同 slot：

```text
score note:
  [pitch_block, shared_block, score_block, zero_perf_block]

performance note:
  [pitch_block, shared_block, zero_score_block, perf_block]

score note without XML-derived score feature:
  [pitch_block, shared_block, zero_score_block, zero_perf_block]
```

对应实现可以是：

```python
pitch_block = PitchEmbedding128(pitch_ids)
shared_block = Linear(3, 256)(shared_features)
score_block = Linear(8, 256)(score_features) * has_score_feature
perf_block = Linear(4, 128)(pedal_features) * has_pedal_feature

note_emb = concat([pitch_block, shared_block, score_block, perf_block])
note_emb = LayerNorm(note_emb)
```

特殊 note 位置（`PAD_NOTE / BOS_NOTE / EOS_NOTE / MASK_NOTE / PLAY_NOTE`）不走上述 group projection，而是直接使用：

```python
note_emb = special_note_embeddings[special_note_id]
```

并覆盖完整 `768d` note embedding。这样 AR decoder 的第一个输入可以直接使用 `BOS_NOTE`，而不是伪造一个 normal note。

PINE 的核心好处是：

1. `note type` 被隐含编码在 occupied semantic slots 中；
2. score-side 和 performance-side feature 不会在 embedding 入口处直接相加冲突；
3. decoder head 可以只读取对应分区，形成更集中的输出路径；
4. 表示本身更容易解释，适合作为 `INR` 的结构化版本。

需要注意：PINE 只保证输入层的槽位结构清晰。进入 Transformer backbone 之后，Q/K/V projection 和 FFN 都是 dense linear，信息仍然可以跨分区混合。这不是问题，因为我们希望 backbone 能利用全局上下文；PINE 约束的是“输入输出接口的语义槽位”，不是把整个网络做成 block-diagonal。

#### 7.2.3 PINE 中 type feature 是否需要保留

PINE 可以删除大规模 additive type embedding。  
因为：

```text
score note = score slot occupied, perf slot empty
perf note  = score slot empty, perf slot occupied
```

slot occupancy 本身已经提供了 type signal。  
但不建议完全删除所有 presence / mask 信息，因为零向量仍然存在歧义：

```text
pedal = 0          # 真实没有踩踏板
score feature = 0  # 真实 false / zero feature
group absent = 0   # 该 feature group 不存在
```

因此推荐：

```text
FINE:
  保留 note_type / presence mask，必要性较高

PINE:
  删除 768d additive type embedding
  保留轻量 group presence mask / has_feature bit
```

更具体地说，PINE 推荐默认配置为：

```text
PINE-mask:
  fixed block concat
  absent group 置零
  has_score_feature / has_pedal_feature 用于 group gate 或 loss mask
  不额外加入 full hidden-size type embedding
```

可以作为 ablation 比较：

```text
PINE-no-type:
  只靠分区 + zero block

PINE-mask:
  分区 + has_score_feature / has_pedal_feature

PINE-type:
  分区 + learned type embedding
```

预期上，`PINE-mask` 更稳；`PINE-type` 不一定更好，因为它可能削弱分区表示的纯净性。

#### 7.2.4 FINE/PINE 与两个独立 encoder 的关系

不推荐设计两个完全独立的：

```text
ScoreNoteEncoder(pitch, shared, score_feature)
PerfNoteEncoder(pitch, shared, pedal)
```

原因是它们会让 score note 和 performance note 漂到两套 embedding space 中，削弱 INR 的统一接口叙事，也会让 EPR / CSR / AR decoder 在不同输入分布之间切换。

更推荐：

```text
one IntegratedNoteEmbedding
  score note occupies score slots
  performance note occupies perf slots
  missing groups use zero + mask
```

也就是说，AR decoder 中已经生成的 target note 不需要单独的 `TargetNoteEncoder`。它只是一个 performance note object：

```text
Generated performance note:
  pitch = score_pitch_i
  shared = sampled [ioi, duration, velocity]
  score_feature = zero / absent
  perf_feature = sampled pedal

IntegratedNoteEmbedding(generated performance note) -> decoder input embedding
```

这能保证 teacher forcing 和推理回灌使用完全相同的 embedding interface。

#### 7.2.5 第一版 embedding 结构

第一版 embedding 端使用 2-layer MLP。所有 continuous feature group 都采用 `Linear -> GELU -> Linear`：

```text
FINE embedding:
  pitch:   Embedding(129, 768, padding_idx=128)
  special: Embedding(5, 768)
  shared:  Linear(3, 768) -> GELU -> Linear(768, 768)
  score:   Linear(8, 768) -> GELU -> Linear(768, 768)
  pedal:   Linear(4, 768) -> GELU -> Linear(768, 768)
  fusion:  sum + LayerNorm(768)

PINE embedding:
  pitch:   Embedding(129, 128, padding_idx=128)
  special: Embedding(5, 768)
  shared:  Linear(3, 256) -> GELU -> Linear(256, 256)
  score:   Linear(8, 256) -> GELU -> Linear(256, 256)
  pedal:   Linear(4, 128) -> GELU -> Linear(128, 128)
  fusion:  concat + LayerNorm(768)
```

其中：

- `Embedding(129, *)` 中的 `129 = 128 real pitches + 1 pitch_pad_id`。
- `special: Embedding(5, 768)` 对应 `PAD/MASK/BOS/EOS/PLAY` note-level special embeddings。
- normal note 使用 pitch + feature group projection。
- special note 直接使用 special embedding，覆盖完整 note embedding。
- 第一版 embedding 使用 `embedding_depth = 2`。`embedding_depth = 1` 可以作为后续 ablation，用来验证连续变量是否真的需要非线性投影。

#### 7.2.6 为什么 embedding 和 head 不要求完全对称

`IntegratedNoteEmbedding` 和 decoder heads 在语义上是相反方向：

```text
embedding:
  note attributes -> hidden vector

head:
  hidden vector -> note attributes / distribution params
```

但它们不是数学上的可逆反函数，也不要求结构镜像、层数相同或参数共享。原因是：

1. embedding 的输入是已知、干净、低维的结构化变量，例如 `pitch / shared / score / pedal`，主要任务是把这些变量投到 backbone hidden space。
2. head 的输入是上下文化后的 hidden state，任务是从其中解析目标变量，EPR 中还要输出条件分布参数 `mu/kappa`，比单纯编码已知属性更难。
3. AR 中 head 预测的是下一步 note attributes，而不是把当前 embedding 反解回原 note；生成出的 attributes 会重新组成 note object，再通过同一个 `IntegratedNoteEmbedding` 回灌。

因此第一版采用语义对称、任务不强制可逆的接口：

```text
embedding:
  2-layer MLP = Linear -> GELU -> Linear

head:
  2-layer MLP = Linear -> GELU -> Linear
```

这样做的目的不是把 embedding/head 设计成数学反函数，而是让 continuous attributes 与 hidden space 之间的双向映射都具有基本非线性表达能力。  
`2-layer MLP` 可以提供必要的非线性读出；但输出分布族仍由 head 定义，例如单个 `Beta(mu/kappa)` 仍然只是单个 Beta，而不是任意复杂 mixture distribution。

### 7.3 Backbone 作为后续 inductive bias

Backbone 指位于 `IntegratedNoteEmbedding` 和 decoder heads 之间的 Transformer 主干，包括 self-attention、cross-attention、FFN、norm、position encoding/rotary embedding 等上下文建模模块。它不包括 note feature schema、pitch/continuous embedding、continuous regression head 和 loss。

当前第一阶段不比较 backbone。为了把实验变量集中在 `FINE` 与 `PINE`，backbone 只作为固定 scaffold 使用；T5/GPT/BERT、encoder-decoder vs encoder-only、decoder depth、attention variant 等都作为后续 inductive bias ablation。

为了后续公平比较，所有 backbone ablation 应使用相同的输入输出接口：

```text
score pitch + score continuous + score feature mask
  -> IntegratedNoteEmbedding
  -> fixed or ablated Backbone
  -> decoder heads
  -> performance continuous
```

除结构本身外，应尽量保持以下设置一致：

```text
hidden_size = 768
intermediate_size = 3072
num_attention_heads = 8
num_key_value_heads = 4
head_dim = 128
block_notes = 512
loss = same masked regression loss
data = PianoCoRe-A
pretrained_model = null
```

后续 backbone ablation 不使用大规模 unpaired MIDI pretrain。所有 backbone 直接按 EPR 目标在 PianoCoRe-A 上训练，从而把实验变量集中在 backbone 结构本身。

### 7.3.1 后续 backbone 公平比较协议

后续目标是比较 backbone inductive bias，而不是比较预训练收益、模型规模或 attention variant。因此采用以下约束：

1. **全部从随机初始化训练。**

   不使用 text-pretrained T5/GPT/BERT，不使用 MIDI-pretrained PT checkpoint，也不使用大规模 unpaired MIDI object pretraining。所有模型只在 PianoCoRe-A 上按 EPR 目标训练。

2. **Note interface 架构一致，但参数不共享。**

   后续 backbone ablation 应固定同一套 `IntegratedNoteEmbedding` 和 decoder heads。具体使用 FINE 还是 PINE，应由第一阶段结果决定：

   ```text
   IntegratedNoteEmbedding:
     FINE or PINE -> hidden_size

   Decoder heads:
     SharedHead -> Beta(mu/kappa) for [ioi, duration, velocity]
     PedalHead  -> Beta(mu/kappa) for [pedal_0..3]
     ScoreHead  -> CSR score features
   ```

   但每个模型训练自己独立的一套参数：

   ```text
   INR-T5-10+2.note_embedding / decoder_heads
   INR-T5-6+6.note_embedding / decoder_heads
   INR-GPT.note_embedding / decoder_heads
   INR-BERT.note_embedding / decoder_heads
   ```

   原因是不同 backbone 的 hidden state 分布不同，输入 embedding space 和输出 head 都需要与各自 backbone 共适应。共享已训练的 node encoder/head 会引入额外依赖，不利于公平解释。

3. **统一基础宽度和 attention 设置。**

   后续 backbone ablation 固定：

   ```text
   hidden_size = 768
   intermediate_size = 3072
   num_attention_heads = 8
   num_key_value_heads = 4
   head_dim = 128
   attention = GQA
   ```

   这里沿用 PT/T5Gemma 风格的 grouped-query attention。虽然 `hidden_size=1024, intermediate_size=4096` 更符合传统 `hidden_size = num_heads * head_dim` 的整齐配置，但它会显著增加模型参数和计算量，因此不放入后续 backbone ablation 的基础设置。`hidden_size=1024` 作为后续 scale ablation。

4. **近似参数量匹配并报告效率。**

   后续 backbone ablation 优先匹配参数量，而不是强行匹配 block 数。T5 decoder block 包含 cross-attention，单层参数量高于 encoder-only / decoder-only block；因此 GPT/BERT 需要更多层才能和 T5 接近：

   ```text
   INR-T5-10+2: 10 encoder blocks + 2 decoder blocks  ~= 124.6M params
   INR-T5-6+6:  6 encoder blocks + 6 decoder blocks   ~= 134.1M params
   INR-GPT:     17 decoder-only blocks                 ~= 124.5M params
   INR-BERT:    17 encoder-only blocks                 ~= 126.1M params
   ```

   GPT/BERT 的 17 层配置让参数量落在 T5 两个设置之间，满足后续 fair comparison 的同量级约束。由于 GPT/BERT/T5 的单层计算量仍不会完全相同，因此实验记录必须报告：

   ```text
   total_params
   trainable_params
   notes/sec or samples/sec
   GPU memory
   wall-clock time per step
   ```

   参数量目标是同一量级、尽量接近，而不是强行做到完全相等。若某个 backbone 参数量偏离超过约 `10%`，应调整层数或明确标注为不同规模。

5. **统一训练预算。**

   保持相同：

   ```text
   PianoCoRe-A split
   block_notes = 512
   effective batch size
   max_steps / epochs
   optimizer and scheduler
   loss weights
   evaluation set
   fixed MIDI preview samples
   ```

### 7.3.2 INR-T5-10+2

结构：

```text
score nodes
  -> bidirectional encoder, 10 layers
  -> cross-attention decoder, 2 layers
  -> continuous prediction
```

这是与 PianistTransformer 最接近的 Integrated Note 版本。PT 原本使用深 encoder + 浅 decoder，是因为 encoder 端已经压缩到 note-level，而 decoder 端仍然是 token-level 自回归生成，decoder 计算更贵。INR-T5-10+2 保留这个非对称设计，适合作为结构基线。

优点：

- 与当前 `IntegratedPianoT5Gemma` 实现最接近，工程改动最小。
- encoder 容量强，适合建模长程 score context。
- 可以直接回答：只替换 PT 表示层和输出头后，原 10+2 非对称结构是否仍有效。

缺点：

- INR decoder 已经是 note-level，不再有 PT token-level decoder 的 4096-token 瓶颈，2 层 decoder 可能容量不足。
- 非对称结构可能继承 PT 的效率取向，但未必是 INR 的最优结构。

### 7.3.3 INR-T5-6+6

结构：

```text
score nodes
  -> bidirectional encoder, 6 layers
  -> cross-attention decoder, 6 layers
  -> continuous prediction
```

这是对称 encoder-decoder ablation。由于 INR 的 decoder input 也是 note-level，decoder 序列长度约为 `block_notes=512`，不再是 PT 的 `4096` token 序列，因此可以合理增加 decoder 深度。

优点：

- encoder 和 decoder 容量更均衡。
- 更适合检验 INR 里 decoder 是否仍是性能瓶颈。
- 仍保留明确的 seq2seq 结构和 cross-attention，对 score-to-performance 映射解释性较好。

缺点：

- 推理和训练成本高于 10+2。
- 如果 EPR 在 note-aligned 条件下主要是 per-note regression，深 decoder 可能收益有限。

### 7.3.4 INR-GPT

结构：

```text
<score> score nodes ... <performance> performance nodes ...
  -> causal decoder-only Transformer
  -> loss only on performance nodes
```

INR-GPT 将 score 和 performance 放在同一个 causal object sequence 中。score 段作为 prefix condition，performance 段作为需要预测的目标。对于连续 node，可以使用 teacher-forced performance node embedding、masked performance placeholder，或 shifted performance node embedding；loss 只在 performance 段计算。

优点：

- 最接近现代 LLM 的 decoder-only scaling recipe。
- 容易扩展到 prompt、style token、performer token、多任务控制等统一序列形式。
- 如果未来单独研究 object-level pretraining，GPT 形式可以自然做 causal modeling；但该因素不进入本阶段公平比较。

缺点：

- 对 note-aligned EPR 来说，causal prefix conditioning 可能不如 encoder-decoder 的 cross-attention 高效。
- score-performance 对齐关系需要通过 causal self-attention 学习，没有显式 cross-attention。
- 保留连续 node 时不能直接使用标准 LM head，需要 mixed discrete-continuous head。

### 7.3.5 INR-BERT

结构：

```text
score nodes
  -> bidirectional encoder-only Transformer
  -> per-note continuous prediction
```

INR-BERT 把 EPR 视为 aligned note-level structured regression，而不是生成任务。由于 PianoCoRe-A refined pair 已经满足 score/performance note-to-note alignment，输出长度与输入长度一致，pitch 也直接 copy，因此 decoder 在这个设定下并非必要。

优点：

- 训练和推理最简单、最快。
- 每个输出 note 可以看到完整 score context，没有自回归误差积累。
- 最贴合 PianoCoRe-A 的 aligned EPR 设定，是检验 “decoder 是否必要” 的强 baseline。

缺点：

- 不适合变长生成、插入/删除 note、performance continuation 等更开放任务。
- 生成建模能力弱于 T5/GPT，更像 performance parameter predictor。
- 如果未来单独研究大规模 object-sequence generative pretraining，需要额外设计 masked denoising 目标；但该因素不进入本阶段公平比较。

### 7.3.6 三类 backbone 的 TikZ 结构图

三类 backbone 的独立 TikZ 图已写成 standalone `tex` 文件：

- [docs/figures/integrated_backbone_structures.tex](/home/kaititech/EPR/PianistTransformer/docs/figures/integrated_backbone_structures.tex)

该文件包含：

- `INR-T5` encoder-decoder 图
- `INR-GPT` decoder-only 图
- `INR-BERT` encoder-only 图

设计上统一展示：

- 输入 note nodes
- `IntegratedNoteEmbedding`
- backbone 主体
- separate decoder heads
- 最终 note attributes 输出

### 7.3.7 Backbone 比较重点

后续 backbone ablation 重点比较以下问题：

```text
1. INR-T5-10+2 vs INR-T5-6+6:
   PT 的浅 decoder 设计在 note-level INR 中是否仍然合理？

2. INR-T5 vs INR-BERT:
   在 aligned EPR 中，encoder-decoder 是否优于 encoder-only regression？

3. INR-GPT vs INR-T5:
   decoder-only LLM-style causal conditioning 是否适合 note-level EPR？

4. INR family vs PT tokenizer baseline:
   提升来自 Integrated Note Representation，还是来自 backbone 变化？
```

### 7.3.8 为什么 INR-BERT 看起来适合这个任务，但当前结果反而更差

先澄清一点：这里的 `INR-BERT` 并不是“只能输出一个向量”的分类器。当前实现中的 encoder-only backbone 会输出整条序列的 note-level hidden states，再由每个位置的 output head 回归对应的 performance 参数。对应代码位置：

- note encoder: [src/model/integrated_pianoformer.py](/home/kaititech/EPR/PianistTransformer/src/model/integrated_pianoformer.py:63)
- BERT backbone: [src/model/integrated_pianoformer.py](/home/kaititech/EPR/PianistTransformer/src/model/integrated_pianoformer.py:426)
- per-note continuous heads: [src/model/integrated_pianoformer.py](/home/kaititech/EPR/PianistTransformer/src/model/integrated_pianoformer.py:86)
- encoder-only wrapper: [src/model/integrated_pianoformer.py](/home/kaititech/EPR/PianistTransformer/src/model/integrated_pianoformer.py:476)

因此，`INR-BERT` 在能力上并非“不能做 seq2seq”，而是在做一种更窄的任务形式：

```text
score note sequence -> same-length performance parameter sequence
```

也就是 aligned, parallel, per-note structured regression。

但“适合任务定义”不等于“当前配置下最容易学好”。截至 `2026-06-08` 的实测结果（见 [results/BACKBONE_EVALUATION_STATUS.md](/home/kaititech/EPR/PianistTransformer/results/BACKBONE_EVALUATION_STATUS.md)）里，`INR-BERT` 明显落后于两种 T5：

- ASAP overall JS:
  - `T5-6+6 = 0.1806`
  - `T5-10+2 = 0.1992`
  - `BERT-17 = 0.3648`
- PianoCoRe-only overall JS:
  - `T5-6+6 = 0.1550`
  - `T5-10+2 = 0.1625`
  - `BERT-17 = 0.3645`

当前更合理的解释不是 “BERT 天生不能做 EPR”，而是下面几个因素叠加使它在第一阶段 from-scratch 公平比较里处于劣势。

#### 1. EPR 虽然 note-aligned，但不只是局部回归

理论上，PianoCoRe-A refined pair 已经 note 对齐，输入输出长度一致，所以 encoder-only regression 看起来很自然。  
但实际 EPR 并不只是：

```text
第 i 个 score note -> 第 i 个 performance 参数
```

它往往还包含：

- phrase-level shaping
- harmonic tension release
- pedal span over multiple notes
- rubato trajectory
- 同一 score pattern 在不同全局上下文下的不同演奏决策

也就是说，任务虽然“对齐”，但目标空间仍然很像一个条件生成问题，而不是纯粹逐点回归。  
T5 的 decoder hidden states 为每个 note 提供了一个显式的 target-side latent space；BERT 则只能直接把 source-side hidden state 映射成输出，这个归纳偏置更容易收缩成 conditional mean。

#### 2. T5 有显式 target query；BERT 没有

这是当前实现里最关键的结构差异。

T5 版本的流程是：

```text
score note embeds
  -> encoder states
  -> decoder note queries
  -> cross-attention
  -> output heads
```

也就是说，T5 的 decoder 并不是简单重复 encoder，而是在构造“我要预测什么”的一组 target-side states。  
这对 EPR 很重要，因为 performance parameter 不是 score embedding 的线性重标定，而更像“在 score 条件下生成一个演奏版本”。

而当前 BERT 版本是：

```text
score note embeds
  -> bidirectional hidden states
  -> output heads
```

它缺少：

- target query token
- source / target state 分离
- cross-attention 这条显式条件路径

所以它更像一个强大的 regressor，而不是 conditional renderer。

#### 3. encoder-only 结构更容易平均化，尤其在 pedal 上

当前结果里最明显的问题是 pedal。  
`BERT-17` 的 pedal 指标非常差，说明模型很容易把难学、模态多、局部不稳定的目标压到平均值附近。

这和 encoder-only + point-regression 风格回归是相符的：

- 当同类 score context 对应多个可能演奏细节时，回归模型倾向输出均值。
- pedal 又比 velocity / IOI 更容易受乐句和和声层面的长程策略影响。
- 没有 target-side latent space 时，这种平均化更明显。

所以当前结果更像：

```text
INR-BERT 学成了一个“保守的条件平均器”
```

而不是一个真正有表现力的 renderer。

#### 4. “BERT” 在这里其实是 encoder-only Transformer baseline，不是完整 BERT recipe

这一点很重要。当前实现虽然叫 `INR-BERT`，但本质上是：

- bidirectional self-attention
- absolute position embedding
- 17 层 encoder-only block
- 从随机初始化开始
- 只用 EPR regression loss 训练

它并没有使用经典 BERT 最依赖的那整套 recipe：

- MLM / denoising pretraining
- 大规模预训练语料
- 预训练后再微调

所以当前实验能证明的是：

```text
在我们这个 from-scratch, note-level, regression-only 设定下，
encoder-only baseline 不如 T5 family
```

但它还不能证明：

```text
所有 BERT-style 方法都不适合 EPR
```

#### 5. 当前实现对 BERT 其实偏“朴素”，还没有给它最强版本

如果后续真要把 encoder-only 路线做强，至少还可以尝试：

1. 给 BERT 增加 target query slots，而不是直接在 source hidden state 上回归。  
   本质上会变成 “Perceiver-style latent query regression” 或轻量 decoder。
2. 把 pedal head 做得更强，甚至改成 mixture / discretized + regression integrated head。
3. 使用 masked denoising pretraining，再做 EPR fine-tuning。
4. 在 loss 上降低 conditional mean 倾向，例如对 pedal 引入分段建模或更稳定的目标变换。

所以当前结论应该写成：

- `INR-BERT` 作为“最简 encoder-only aligned regression baseline”是成立的。
- 但它不是 encoder-only 方案的上限。
- 当前负结果说明：在 EPR 里，显式的 target-side 建模很可能是有价值的。

### 7.4 Decoder Heads

需要先明确一个和 PianistTransformer 的关键差异：decoder 序列长度已经从 token-level 变为 note-level。

原 PT SFT 中：

```text
block_size = 4096 tokens
每个 note = 8 tokens

encoder input length  = 4096 token ids
encoder hidden length = 512 note states   # 8-token compressed to 1-note
decoder input length  = 4096 token ids
decoder output length = 4096 token logits
```

也就是说，原 PT 的 encoder 端已经压缩到 note-level，但 decoder / LM head / loss 仍然是 token-level classification。

INR/FINE/PINE 中：

```text
block_notes = 512

encoder input length  = 512 note embeddings
decoder input length  = 512 note embeddings
decoder output length = 512 note predictions
```

每个 decoder position 对应一个 note object，而不是一个 token。head 输出的是该 note 的属性组：

```text
shared attributes: [ioi, duration, velocity]
perf attributes:   [pedal_1..4]
score attributes:  [mo, md, ml, first, staff, trill, grace, staccato]
```

输入：

```python
hidden_states: FloatTensor[B, N, hidden_size]
```

输出：

```python
shared_pred:
  timing_pred: FloatTensor[B, N, 2]             # [dur, ioi]
  velocity_pred: FloatTensor[B, N, 1]           # [vel]

perf_pred:
  pedal_pred: FloatTensor[B, N, 4]              # [pedal_1..4]

score_pred:
  score_position_pred: FloatTensor[B, N, 2]     # [mo, md]
  measure_structure_pred: FloatTensor[B, N, 2]  # [ml, first]
  score_annotation_pred: FloatTensor[B, N, 4]   # [staff, trill, grace, staccato]
```

#### 7.4.1 FINE decoder heads

FINE 中所有 group 的 head 都读取完整 hidden state：

```python
SharedHead:
  Linear(768, 768)
  GELU
  Linear(768, shared_output_dim)

PedalHead:
  Linear(768, 768)
  GELU
  Linear(768, pedal_output_dim)

ScoreHead:
  Linear(768, 768)
  GELU
  Linear(768, score_output_dim)
```

在 FINE 中：

```text
head input = full 768d hidden state
head hidden width = 768
```

优点是每个 head 可以直接使用所有上下文信息；缺点是 embedding 和 head 都较大，而且不同输出组之间的责任边界不如 PINE 清晰。

#### 7.4.2 PINE decoder heads

PINE 中 decoder head 只读取对应语义分区：

```text
z_note = [z_pitch, z_shared, z_score, z_perf]

shared_head(z_shared) -> [ioi, duration, velocity]
score_head(z_score)   -> [mo, md, ml, first, staff, trill, grace, staccato]
perf_head(z_perf)     -> [pedal_1..4]
```

推荐默认分区：

```text
z_pitch:  128 dims
z_shared: 256 dims
z_score:  256 dims
z_perf:   128 dims
```

对应 head：

```python
SharedHead:
  Linear(256, 256)
  GELU
  Linear(256, shared_output_dim)

ScoreHead:
  Linear(256, 256)
  GELU
  Linear(256, score_output_dim)

PerfHead:
  Linear(128, 128)
  GELU
  Linear(128, perf_output_dim)
```

PINE 的 head hidden width 默认等于对应 partition dim，而不是统一扩到 `768`。这样 head 仍然只能从自己的语义分区读取并解码信息，避免削弱 PINE 的分区归纳偏置。

PINE 的输出侧约束很重要：即使 backbone 内部已经通过 dense attention / FFN 融合了所有信息，最终模型仍然必须把待解码信息写回对应 semantic block，head 才能读到它。这让 representation layout 和 decoder head 的归纳偏置绑定起来。

因此 PINE 不是简单的“小 head 省参数”，而是结构化 latent note representation：

```text
z_i = [z_i^pitch ; z_i^shared ; z_i^score ; z_i^perf]

y_i^shared = H_shared(z_i^shared)
y_i^score  = H_score(z_i^score)
y_i^perf   = H_perf(z_i^perf)
```

建议保留一个 ablation 开关：

```text
head_input_mode = "partitioned"  # head 只读对应 block
head_input_mode = "full"         # head 读完整 768d hidden
```

如果 `partitioned` 明显不弱于 `full`，说明分区表示确实能承载对应属性；如果 `full` 明显更好，则说明某些输出属性需要更直接地访问跨组信息，PINE 的分区约束可能过强。

#### 7.4.3 EPR 与 CSR 的对称 supervision

```text
EPR: score note sequence -> predict performance-side groups
CSR: performance note sequence -> predict score-side groups
```

具体来说：

| Task | Input | Predict Groups |
|------|------|------|
| `EPR` | score note object | `timing`, `velocity`, `pedal` |
| `CSR` | performance note object | `score_position`, `measure_structure`, `score_annotation` |

如果后续需要，也可以让 shared groups 在两个任务里都被监督：

- `EPR` 中监督 `dur/ioi/vel`
- `CSR` 中也可恢复 canonicalized `dur/ioi/vel`

但第一阶段最清晰的定义仍然是：

- `EPR` 负责生成 performance-side expressive attributes
- `CSR` 负责恢复 score-side canonical attributes

相比一个大一统 joint head，separate heads 的额外参数量很小，但能显著减少不同目标族之间的梯度干扰。

第一版不预测 pitch，直接从 score copy pitch。原因：

- EPR 不应该改变音高。
- PianoCoRe refined score/performance 已经 note-by-note 对齐。
- CSR 第一阶段也同样不把 pitch 设为主要目标，先把问题限定在 aligned note attribute reconstruction。

后续可以增加 `pitch_head` 作为 auxiliary consistency loss，但不作为第一版目标。

#### 7.4.4 AR-mode, teacher forcing 与 score-mode

当前第一版 INR 训练默认使用真正的 note-level AR：

```text
decoder_input_mode = "ar"
```

训练时使用 note-level teacher forcing：

```text
step 0:
  BOS_NOTE -> special_note_embeddings[BOS_NOTE]

step t > 0:
  gold previous performance note object
  -> IntegratedNoteEmbedding
  -> decoder input embedding
  -> decoder hidden state
  -> continuous / probabilistic heads
  -> predict current note attributes
```

这样生成过程更接近 PT 的逻辑：模型必须把已经生成的 target-side note object 回灌到 decoder，而不是每个位置都直接拿 score note embedding 做 parallel query。

为了减轻 exposure bias，第一版在训练时还引入 `prior_token_keep_prob = 0.5` 的 decoder-side dropout：

```text
score / pitch 相关输入保留
previous target-side continuous condition 按 0.5 概率保留
其余 previous target-side condition 随机置空
```

也就是说，teacher forcing 仍然存在，但不会永远把“完整正确的历史”喂给 decoder。

原 PT SFT 的 teacher forcing 是：

```text
labels:        y1 y2 y3 ... y4096
decoder input: BOS y1 y2 ... y4095
target:        y1 y2 y3 ... y4096
```

推理时则是：

```text
previous token id
  -> token embedding
  -> decoder
  -> vocab logits
  -> sample token id with temperature / top-p
  -> next step
```

因此 PT 的 decoder AR 是 token-level categorical sampling。

推理时使用模型自己生成的 note object 回灌：

```text
step 0:
  decoder input = BOS_NOTE embedding

step t:
  generated performance notes y_<t
    -> IntegratedNoteEmbedding
    -> decoder input embeddings
    -> decoder hidden_t
    -> probabilistic continuous head
    -> distribution params
    -> sample concrete note attributes y_t
    -> assemble performance note object
    -> feed y_t back through IntegratedNoteEmbedding at next step
```

`PAD_NOTE` 不能作为第一个 decoder input，因为它表示空位且会被 mask；`BOS_NOTE` 才表示生成开始。  
在当前 PianoCoRe refined aligned EPR 中，输出长度等于输入 note 数，因此第一版推理不需要依赖 `EOS_NOTE` 停止生成；`EOS_NOTE` 主要留给未来变长生成。

这和 PT 的 AR 过程是同构的，只是离散 token 被替换成了 structured note object：

```text
PT:
  prev token id -> token embedding -> categorical logits -> sample token id

INR:
  prev note attrs -> IntegratedNoteEmbedding -> continuous distribution params -> sample note attrs
```

关键点是：head 输出的不是下一步 embedding。  
head 输出的是属性值或分布参数。采样得到具体属性值后，必须重新组成 note object，再通过同一个 `IntegratedNoteEmbedding` 编码成下一步 decoder input embedding。

`score-mode` 仍然可以作为后续 ablation：

```text
decoder_input_mode = "score"
```

它不是传统 teacher forcing，而是 parallel conditional prediction：

```text
score note embeddings
  -> decoder
  -> performance note attributes
```

也就是说，decoder 的每个 position 直接以 score note embedding 作为 query，输出同位置的 performance 参数。它不使用上一个 performance note 作为输入，因此没有自回归误差累积，也没有 token sampling。  
该模式训练和推理更简单，但生成逻辑弱于 AR，因此不作为第一版主实验。

#### 7.4.5 Probabilistic continuous heads

EPR 是 one-to-many conditional generation。同一个 score 可以对应多种合理演奏，因此纯 `MSE/Huber` point regression 容易学成 conditional mean。第一版 EPR 不再混用 probabilistic 与 non-probabilistic loss，而是对所有 performance-side continuous variables 统一使用 Beta distribution。

第一版只保留 3 个 decoder heads，避免过细分组：

```text
SharedHead -> [ioi, duration, velocity]
PedalHead  -> [pedal_0, pedal_25, pedal_50, pedal_75]
ScoreHead  -> [mo, md, ml, first, staff, trill, grace, staccato]
```

EPR 使用：

```text
SharedHead + PedalHead
```

CSR 使用：

```text
ScoreHead
```

EPR 的 Beta head 使用 `mu / kappa` 参数化，而不是直接预测 `alpha / beta`。两者表达能力等价：

```text
mu = alpha / (alpha + beta)
kappa = alpha + beta

alpha = mu * kappa
beta  = (1 - mu) * kappa
```

选择 `mu / kappa` 的原因是训练更稳、语义更清楚：

```text
mu    = predicted center / mean
kappa = concentration / confidence
```

实现：

```python
mu = sigmoid(raw_mu)
kappa = softplus(raw_kappa) + kappa_min

alpha = mu * kappa + eps
beta = (1.0 - mu) * kappa + eps

target = target.clamp(eps, 1.0 - eps)
loss = -Beta(alpha, beta).log_prob(target)
```

SharedHead 输出：

```text
SharedHead:
  input -> [
    ioi_mu, duration_mu, velocity_mu,
    ioi_kappa, duration_kappa, velocity_kappa
  ]
```

PedalHead 输出：

```text
PedalHead:
  input -> [
    pedal_0_mu, pedal_25_mu, pedal_50_mu, pedal_75_mu,
    pedal_0_kappa, pedal_25_kappa, pedal_50_kappa, pedal_75_kappa
  ]
```

推理：

```python
deterministic_output = mu
sampled_output = Beta(alpha, beta).sample()
```

Beta distribution 天然约束输出在 `[0, 1]`，适合当前所有 EPR target：

```text
ioi_norm, duration_norm, velocity_norm, pedal_0..3
```

如果后续发现单个 Beta 仍不足以表达复杂多峰分布，可以再升级为 mixture of Betas 或 discretized-continuous hybrid head。但第一版不引入 mixture，先使用统一的 `mu/kappa Beta`。

CSR 则不一定需要概率连续头。CSR 更接近 many-to-one canonicalization，目标熵较低，score grid / notation fields 更适合 ordinal classification + BCE。

## 8. Loss

loss 也按 group 组织，这与 encoder / decoder 的分组保持一致。

### 8.1 EPR Loss

```python
loss_ioi = masked_beta_nll(ioi_mu, ioi_kappa, ioi_target, attention_mask)
loss_dur = masked_beta_nll(duration_mu, duration_kappa, duration_target, attention_mask)
loss_vel = masked_beta_nll(velocity_mu, velocity_kappa, velocity_target, attention_mask)

loss_pedal_0 = masked_beta_nll(pedal_0_mu, pedal_0_kappa, pedal_0_target, attention_mask)
loss_pedal_25 = masked_beta_nll(pedal_25_mu, pedal_25_kappa, pedal_25_target, attention_mask)
loss_pedal_50 = masked_beta_nll(pedal_50_mu, pedal_50_kappa, pedal_50_target, attention_mask)
loss_pedal_75 = masked_beta_nll(pedal_75_mu, pedal_75_kappa, pedal_75_target, attention_mask)

loss_shared = loss_ioi + loss_dur + loss_vel
loss_pedal = mean([loss_pedal_0, loss_pedal_25, loss_pedal_50, loss_pedal_75])
loss_epr = loss_shared + loss_pedal
```

其中：

- `loss_ioi` / `loss_dur` / `loss_vel` 来自 `SharedHead`
- `loss_pedal_0..75` 来自 `PedalHead`
- `loss_pedal` 对 4 个 pedal snapshots 取均值，避免 pedal 因为维度数量天然放大
- 第一版不引入手动 loss weights，先观察无权重基线

推荐理由：

- `ioi` / `duration` / `velocity` / `pedal` 都是 `[0, 1]` 连续变量，因此统一使用 Beta NLL。
- `mu/kappa` 参数化能同时表达中心趋势与不确定度，比点估计式 regression 更适合 EPR 的 one-to-many 条件分布。
- 第一版不引入变量级权重或先验，避免把调参变量混入表示层与 backbone 对比。

Beta NLL 的实现建议：

```python
def beta_params(raw_mu, raw_kappa, eps=1e-5, kappa_min=1e-3):
    mu = raw_mu.sigmoid()
    kappa = softplus(raw_kappa) + kappa_min
    alpha = mu * kappa + eps
    beta = (1.0 - mu) * kappa + eps
    return mu, kappa, alpha, beta

target = target.clamp(eps, 1.0 - eps)
loss = -Beta(alpha, beta).log_prob(target)
```

如果后续发现某一类变量主导训练，再单独做权重 ablation；第一版先保持：

```text
loss_epr = L_shared + L_pedal
```

### 8.2 CSR Loss

```python
score_mask = attention_mask * has_score_feature
ml_mask = score_mask * first_target

loss_mo = 1.0 * masked_ordinal_ce(mo_logits, mo_bin_target, score_mask)
loss_md = 1.0 * masked_ordinal_ce(md_logits, md_bin_target, score_mask)
loss_first = 1.0 * masked_bce_with_logits(first_logit, first_target, score_mask)
loss_ml = 1.0 * masked_ordinal_ce(ml_logits, ml_bin_target, ml_mask)

loss_staff = 1.0 * masked_bce_with_logits(staff_logit, staff_target, score_mask)

loss_trill = 1.0 * masked_class_balanced_bce(
    trill_logit,
    trill_target,
    score_mask,
    pos_weight=trill_pos_weight,
)
loss_grace = 1.0 * masked_class_balanced_bce(
    grace_logit,
    grace_target,
    score_mask,
    pos_weight=grace_pos_weight,
)
loss_staccato = 1.0 * masked_class_balanced_bce(
    staccato_logit,
    staccato_target,
    score_mask,
    pos_weight=staccato_pos_weight,
)

loss_csr = (
    loss_mo
  + loss_md
  + loss_first
  + loss_ml
  + loss_staff
  + loss_trill
  + loss_grace
  + loss_staccato
)
```

这里：

- `mo`, `md`, `ml` 的最终推荐形式不是纯连续回归，而是固定 score grid 上的 ordinal classification
- `first`, `staff` 走普通 binary prediction
- `trill`, `grace`, `staccato` 是极不均衡的稀疏 notation labels，不建议直接使用 plain BCE
- CSR 的 score-side loss 应额外乘以 `has_score_feature` mask；没有成功投影 XML/MXL score feature 的 note 不参与 score-side feature loss。
- `ml` 只在小节首音真正有意义，因此应再额外乘 `first` mask；否则会往大量非首音位置注入无意义监督。

推荐理由：

- `mo`, `md`, `first` 共同定义 canonical score structure，因此是 CSR 的主目标。
- `ml` 只在小节首音位置有意义，因此需要 `ml_mask = score_mask * first_target`；同时由于 `ml_mask == 1` 的位置远少于其它目标，它的系数不应再压低，默认设为 `1.0` 更合理。
- `staff` 是重要的结构标签，默认权重设为 `1.0`。
- `trill/grace/staccato` 是稀疏但音乐意义明确的 notation labels；它们的 loss term 系数仍设为 `1.0`，但 criterion 内部需要处理类别不均衡。

稀疏 notation labels 的第一版推荐是 class-balanced BCE，而不是 plain BCE：

```python
pos_weight_c = num_negative_c / max(num_positive_c, 1)
pos_weight_c = clamp(pos_weight_c, min=1.0, max=pos_weight_cap)

loss_c = masked_bce_with_logits(
    logit_c,
    target_c,
    mask=score_mask,
    pos_weight=pos_weight_c,
)
```

其中 `pos_weight_c` 使用训练集级别统计，而不是 batch 内动态统计，避免 batch 间波动。`pos_weight_cap` 可先设为 `20` 或 `50`，防止极端稀疏标签产生过大的梯度。  
如果 class-balanced BCE 仍然产生过多 false positives 或训练不稳，再将 `trill/grace/staccato` 的 criterion 做 ablation：

```text
plain BCE
class-balanced BCE
focal BCE
asymmetric focal loss
```

关于 `mo/md/ml` 的 loss family，需要特别说明：

- 它们虽然进入 encoder 前以 `[0, 1]` scalar 存储
- 但在 CSR decoder 端，最终推荐目标是离散 score lattice 上的 ordinal bins
- `Huber` 可以作为快速 baseline，但不是正式版的首选

当前 score grid 为：

- `mo`: `[0, 6]`，步长 `1/24`，共 `145` bins
- `md`: `[0, 4]`，步长 `1/24`，共 `97` bins
- `ml`: `[0, 6]`，步长 `1/24`，共 `145` bins

这里建议把 `ml` mask 升级为明确实现规则，而不是“可选优化”：

- `ml` 默认就应该只在 `first == 1` 上计算 loss。

### 8.3 对称但不完全同构

这套设计是“任务对称”的：

```text
EPR: score  -> performance
CSR: performance -> score
```

但不是说两边必须使用一模一样的 loss family。

- performance-side 更偏连续控制，第一版适合 probabilistic continuous modeling
- score-side 包含 ordinal 与 binary，适合 ordinal CE、BCE 与稀疏标签 class-balanced BCE 混合

因此，更准确的说法是：

```text
encoder / decoder 在 feature ontology 上对称
EPR / CSR 在 supervision direction 上对称
loss family 按目标类型分别设计
```

所有 loss 只在 `attention_mask == 1` 的 note 上计算。  
`interpolated` 第一阶段默认不降低权重，避免过早引入新变量。

## 9. Trainer Batch 接口

新增或沿用训练脚本：

```text
src/train/train_inr.py
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
configs/inr_config_pianocore.json
```

建议第一版：

```json
{
  "refined_dir": "data/pianocore/PianoCoRe/refined",
  "metadata_path": "data/pianocore/metadata.csv",
  "block_notes": 512,
  "min_notes": 64,
  "overlap_ratio": 0.125,
  "pretrained_model": null,
  "load_pianoformer_backbone": false,
  "backbone_type": "t5",
  "encoder_layers_num": 10,
  "decoder_layers_num": 2,
  "hidden_size": 768,
  "intermediate_size": 3072,
  "num_attention_heads": 8,
  "num_key_value_heads": 4,
  "head_dim": 128,
  "continuous_dim": 7,
  "note_embedding_mode": "fine",
  "special_note_vocab_size": 5,
  "pine_partition_dims": {
    "pitch": 128,
    "shared": 256,
    "score": 256,
    "perf": 128
  },
  "special_note_ids": {
    "pad": 0,
    "mask": 1,
    "bos": 2,
    "eos": 3,
    "play": 4
  },
  "use_full_type_embedding": true,
  "use_group_presence_mask": true,
  "head_input_mode": "full",
  "decoder_input_mode": "ar",
  "embedding_depth": 2,
  "head_depth": 2,
  "head_activation": "gelu",
  "epr_distribution": "beta_mu_kappa",
  "shared_head": "beta_mu_kappa",
  "pedal_head": "beta_mu_kappa",
  "score_sparse_label_loss": "class_balanced_bce",
  "attention_variant": "gqa",
  "output_dir": "./models/inr_models/",
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

第一阶段 FINE/PINE 对比中 `pretrained_model` 设为 `null`。旧 checkpoint 的 embedding 和 LM head 与新模型不兼容；即使只迁移 backbone 权重，也会引入额外初始化变量，因此不放入第一阶段公平比较。迁移 PT backbone 可以作为后续单独实验。

FINE / PINE 使用独立配置文件更清楚，例如：

```text
configs/fine_pianocore.json
configs/pine_pianocore.json
```

除 `note_embedding_mode`、`pine_partition_dims`、`head_input_mode` 与 type/mask 开关外，第一阶段配置应保持一致。`embedding_depth=2` 与 `head_depth=2` 在 FINE/PINE 之间保持相同。

FINE / PINE 对比建议使用同一固定 backbone scaffold：

```json
{
  "note_embedding_mode": "fine",
  "use_full_type_embedding": true,
  "use_group_presence_mask": true,
  "head_input_mode": "full"
}
```

```json
{
  "note_embedding_mode": "pine",
  "use_full_type_embedding": false,
  "use_group_presence_mask": true,
  "head_input_mode": "partitioned",
  "pine_partition_dims": {
    "pitch": 128,
    "shared": 256,
    "score": 256,
    "perf": 128
  }
}
```

其中：

- `note_embedding_mode = "fine"` 表示 group-wise projection 后在完整 `hidden_size` 空间相加。
- `note_embedding_mode = "pine"` 表示 group-wise projection 到固定分区后 concat。
- `use_full_type_embedding` 只建议 FINE 默认开启；PINE 默认关闭。
- `use_group_presence_mask` 建议 FINE/PINE 都开启，用于解决 zero ambiguity 和 loss masking。
- `head_input_mode = "full"` 表示每个 head 读完整 hidden state。
- `head_input_mode = "partitioned"` 表示每个 head 只读对应 PINE block。
- `embedding_depth = 2` 表示 feature group 使用 `Linear -> GELU -> Linear`。
- `head_depth = 2` 表示 decoder heads 使用 `Linear -> GELU -> Linear`。

AR 推理时可以额外配置采样温度：

```json
{
  "sampling_temperature": 1.0
}
```

其中 AR 推理必须执行：

```text
sample attributes -> assemble note object -> IntegratedNoteEmbedding -> next decoder input
```

不能直接把 head 输出当作下一步 embedding。

## 11. 运行步骤

### 11.1 生成 INR SFT 数据

```bash
python src/data_process/generate_json_with_paired_midi.py --overwrite
python src/data_process/update_json_score_feature_with_xml.py
```

预期输出：

```text
data/pianocore/PianoCoRe/refined/**/*.json
data/pianocore/PianoCoRe/refined/pianocore_a_node_summary.json
data/pianocore/PianoCoRe/refined/pianocore_a_integrated_score_feature_update_summary.json
data/pianocore/PianoCoRe/refined/pianocore_a_integrated_score_feature_update_details.jsonl
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
assert len(pitch) == len(score_feature)
assert len(pitch) == len(has_score_feature)
assert len(pitch) == len(label_continuous)
assert len(pitch) == len(interpolated)
assert len(score_continuous[0]) == 3
assert len(score_feature[0]) == 8
assert len(label_continuous[0]) == 7
assert all(value in (0, 1) for value in has_score_feature)
assert min(pitch) >= 0 and max(pitch) <= 127
```

### 11.2 训练

```bash
python src/train/train_inr.py --config configs/inr_config_pianocore.json
```

多 GPU 沿用现有 deepspeed/DDP 流程即可。

### 11.3 从预测还原 MIDI

新增工具函数：

```text
src/utils/inr_midi.py
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
MAE_Pedal_CC
RMSE_Pedal_CC
BetaNLL_EPR
```

注意：IOI/duration 的指标应在反归一化后的 ms 空间计算。

Velocity 指标用 0-127 空间：

```python
velocity_mae = mean(abs(pred_vel * 127 - target_vel * 127))
```

Pedal 可以同时报：

```text
Pedal_MAE_CC
Pedal_RMSE_CC
```

`BetaNLL_EPR` 用于评估 probabilistic head 的 likelihood / calibration；`MAE/RMSE` 用于和旧 point-regression baseline 做直观数值比较。

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

### 12.3 Testset protocol

第一阶段 testset evaluation 分为两套协议：

```text
Protocol-Det: deterministic generation
Protocol-Sample: stochastic generation / multi-sample evaluation
```

`Protocol-Det` 用于主表：

- 每个 score 只生成 1 个 prediction。
- prediction 与该 score 的多个 GT performance 分别计算指标。
- score-level 指标最后再平均。

`Protocol-Sample` 用于补充分析：

- 每个 score 生成 `K` 个 samples。
- prediction set 与 GT set 在 score level 上做多对多比较。
- 重点衡量 one-to-many 能力，而不是单一最优样本。

多对多聚合定义如下：

```text
E[MAE](x) = mean_{k,m} d(p_k, g_m)
```

其中 `p_k` 是第 `k` 个 prediction，`g_m` 是第 `m` 个 GT。

Wasserstein 采用 pooled empirical distribution：

```text
W1(pool(predictions), pool(GT))
```

Sampling 版同时报告：

- `expected_pairwise MAE/Wass`
- `pooled distribution Wass`
- `model-model diversity`
- `human-human diversity`

当前 INR 第一版已切换为 `Beta(mu/kappa)` probabilistic head；`Protocol-Det` 取 `mu` 作为确定性输出，`Protocol-Sample` 从 Beta 分布采样。

## 13. 第一阶段对照

第一阶段至少比较：

```text
Reference A: PT tokenizer + PianoT5Gemma SFT
Experiment B: FINE under fixed backbone scaffold
Experiment C: PINE under fixed backbone scaffold
```

需要保持：

- 同一 PianoCoRe subset。
- 同一 train/test split。
- 同一 note window 大小约 512 notes。
- 尽量接近的 effective batch size。
- 不使用大规模 unpaired MIDI pretrain。
- Integrated Note 模型均从随机初始化开始直接训练。
- 相同固定 backbone scaffold。
- 相同 embedding/head 深度：embedding 和 head 都使用 `Linear -> GELU -> Linear`。
- 只改变 `FINE` 与 `PINE` 两种 Integrated Note embedding/head 参数化。
- 相同 `hidden_size=768`、`intermediate_size=3072`、GQA attention 设置。
- 报告 `total_params`、`trainable_params`、吞吐、显存和单 step 时间。

当前 tokenizer `block_size=4096` 对应 512 notes；node 方案使用 `block_notes=512`。

如果需要把 PT tokenizer baseline 也做成完全 from-scratch，应明确记录；如果 PT baseline 使用已有 pretrain checkpoint，则只能作为 “PT full recipe” 参考，不应和 from-scratch FINE/PINE 直接解释为纯表示差异。

## 14. 最小实现清单

需要新增：

```text
src/utils/inr_midi.py
src/data_process/generate_json_with_paired_midi.py
src/data_process/update_json_score_feature_with_xml.py
src/data_process/score_xml_alignment.py
src/model/integrated_pianoformer.py
src/train/train_inr.py
configs/inr_config_pianocore.json
```

新增 testset 工具：

```text
src/inference/infer_inr_testset.py
src/evaluate/evaluate_inr_saved_midis.py
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
- 每个 segment 只保存一份 `score.pitch`，并且来自 `score_pitch == performance_pitch` 的 pair。
- 每个 segment 的 `score.score_continuous`、`score.score_feature`、`score.has_score_feature`、`performances[].label_continuous`、`performances[].interpolated` 第一维一致。
- `score.score_continuous.shape[-1] == 3`，`score.score_feature.shape[-1] == 8`，`performances[].label_continuous.shape[-1] == 7`。
- `score.has_score_feature` 只包含 `0/1`；当值为 `0` 时，该 note 的 `score_feature` 不参与 score-side feature loss。
- `pitch` 在 `[0, 127]`，padding 仅在 collator 出现。
- 除 `pitch` 之外，所有 note object 数值字段均在 `[0, 1]`。

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

- 和 tokenizer reference、FINE、PINE 在同一验证集上比较 feature-level metrics。
- FINE / PINE 使用同一固定 backbone scaffold 与同一训练预算。
- 至少保存 20 个固定验证样本的 MIDI 输出。

## 16. 后续实验方向

第一阶段 FINE/PINE 对比稳定后，再做以下 inductive-bias ablation：

1. Backbone family: fixed T5 scaffold vs T5-6+6 vs GPT vs BERT。
2. Attention variant: GQA vs MHA vs MQA。
3. Model scale: `hidden_size=768, intermediate_size=3072` vs `hidden_size=1024, intermediate_size=4096`。
4. Embedding/head depth ablation：embedding `1 vs 2`，head `1 vs 2 vs 3`，验证当前非对称接口是否足够。
5. `head_input_mode = full` vs `partitioned`，验证 PINE 的输出侧分区约束是否真正有益。
6. `decoder_input_mode = ar` vs `score`，比较 note-level autoregressive generation 与 parallel conditional prediction。
7. `EPR Beta(mu/kappa) NLL` vs point regression，验证 probabilistic continuous head 对 EPR 一对多目标的收益。
8. PINE 中 `PINE-no-type` / `PINE-mask` / `PINE-type`，验证 type feature 是否可以由 slot occupancy 与轻量 mask 替代。
9. `MAX_TIME_MS=5000` vs `10000` vs `20000`。
10. `log normalization` vs linear normalization。
11. 预测 absolute IOI/duration vs 预测 score-relative ratio。
12. score velocity 保留 vs 置零。
13. interpolated notes loss weight `1.0` vs `0.5` vs masked。
14. 加 pitch auxiliary CE head。
15. 迁移旧 PianistTransformer checkpoint 的 backbone 权重，只随机初始化 note embedding/head；该实验单独报告，不与 from-scratch fair comparison 混在一起。
16. 大规模 unpaired MIDI object pretraining；该实验用于研究预训练收益，不进入第一阶段 FINE/PINE fair comparison。
17. 将同一 Integrated Note interface 迁移到 `CSR (Canonical Score Reconstruction)`，并在 CSR 中重新比较 FINE/PINE 以及后续 backbone inductive bias。

## 17. 第一版结论预期

如果实验成立，应该看到：

- 序列长度减少为原来的 1/8。
- 不再有 timing token clip 到约 5 秒的问题。
- velocity、duration、pedal 的预测误差更自然，尤其是短时值和 pedal 连续变化。
- Decoder 不再需要学习 8-token 局部格式语法，训练目标更贴近 EPR。
- FINE 和 PINE 会呈现不同 trade-off：FINE 更自由，PINE 更结构化且更可解释。
- 如果 note-level AR 成立，采样得到的属性值必须重新组成 note object，再通过 `IntegratedNoteEmbedding` 回灌，而不是直接把 head 输出当作下一步 embedding。
- Probabilistic continuous head 应能缓解 EPR 中 point regression 的 conditional mean 问题，尤其是 timing 相关指标与听感。
- 同一 INR 表示不只适用于 `score -> performance`，还应能自然扩展到 `performance -> canonical score (CSR)`。
- 如果 CSR 的目标熵确实低于 EPR，则 `INR-BERT` 有可能在 CSR 上显著优于其在 EPR 上的表现。

这将支持论文中的核心表述：Integrated Note Representation 将 symbolic-performance 映射的表示层与 Transformer backbone 解耦，使 EPR 与 CSR 都可以作为 note-level conditional structured prediction 来研究，而不是被固定在 token-level language modeling 范式中。
