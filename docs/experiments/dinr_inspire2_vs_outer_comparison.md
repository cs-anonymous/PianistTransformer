# INSPIRE2 DINR 与外层 DINR 对照

这两个模型都被称为 DINR，但它们不是同一个训练问题。两者在 held-out-12-work 对照中使用了相同的主要窗口划分：valid 为 12 个完整作品、507 个 perf-window instances，train 实际为约 10,280 个 instances；缺少 median sidecar 的 Scriabin work 两边都没有进入训练。因此当前过拟合差异主要来自模型、目标和优化协议。

## 关键差异

| 项目 | INSPIRE2 DINR-Absolute | 外层 INR DINR |
|---|---|---|
| timing target | 绝对 performance IOI/duration，10 ms 量化，0--4990 ms | score-relative log deviation；nonzero IOI/duration 主要为 `[-2,1]`，score-IOI=0 的 IOI 为 `[0,5]` |
| IOI heads | `grouped_ioi_head` 与 `chord_ioi_head` 两个独立 500-way head | `dinr_vocabulary_mode=unified`，一个 IOI head，用 support mask 区分 zero/nonzero 范围 |
| timing output | 500 个绝对 timing 类别 | 256 个 deviation 类别，IOI 与 duration 共用 DINR timing value table |
| pedal | 每个 pedal 两个 logits，CE；四个 CE 的平均值 | 输出 4 个 logits，当前配置实际走 BCEWithLogitsLoss |
| score input | tokenized pitch/score IOI/score duration/score velocity | continuous score controls + musical51 score features |
| decoder feedback | 完整 shifted performance token 作为 teacher forcing 输入 | score-relative continuous feedback；训练时 decoder token embedding 随机 mask 50% |
| output parameterization | 每个属性基本是独立 linear head | query MLP 与 categorical value-table prototype 点积；IOI/duration 共享 timing table，并与 DINR 输入 value table 绑定 |
| PT initialization | 使用官方 PT checkpoint 初始化 backbone 和兼容 token embedding | 本次外层配置 `pretrained_model=null`、`load_pianoformer_backbone=false`，随机初始化 |
| backbone depth | encoder 10 层、decoder 2 层 | encoder 8 层、decoder 4 层 |
| effective batch | 32，gradient accumulation 1 | 64，per-device 32、gradient accumulation 2 |
| optimizer schedule | AdamW 默认 `weight_decay=0`，linear，warmup 0 | cosine，warmup 50 steps |

## 为什么 INSPIRE2 更容易出现过拟合

### 1. 绝对目标更容易记住作品和速度状态

INSPIRE2 直接预测绝对毫秒值。即使 performance 已做 median scaling，绝对 IOI 仍然强依赖 score 的音符时值、作品的局部节奏结构和作品级时间分布。模型可以逐渐记住 train works 的绝对 timing pattern。

外层 DINR 预测的是相对 score 的 log deviation。score 时值被移到条件输入中，输出主要描述 expressive deviation，跨作品的目标分布更集中。这本身就降低了“记住每个作品的绝对时间表”的机会。

### 2. INSPIRE2 使用完整 teacher forcing，外层有 50% decoder feedback mask

INSPIRE2 的 decoder 输入是完整 shifted performance token。训练时每个位置都能看到前一个真实 performance note 的全部属性，模型很容易拟合训练作品中的局部 performance 序列。

外层配置 `tf_embedding_mask_decoder=true`、`tf_embedding_mask_keep_prob=0.5`、`slot_decoder_mask_mode=whole_token`。训练时约一半 decoder note embedding 被替换为 mask embedding，模型必须更多依赖 score 和上下文。这是明确的正则化来源，也是两边最可能直接影响过拟合曲线的差异之一。

### 3. INSPIRE2 的 IOI loss 更重、head 更自由

INSPIRE2 的 IOI loss 是

`L_ioi = CE(grouped_ioi) + CE(chord_ioi)`。

两个 CE 都是各自有效样本的平均值后再相加，因此 grouped 与 chord 两个子任务各得到完整权重。外层 unified DINR 只有一个 IOI CE，zero/nonzero 样本在同一个平均中按出现频率加权。

同时，INSPIRE2 的 grouped/chord/duration head 是独立的 `hidden -> 500` linear head。外层 DINR 使用 value-table prototype 点积，IOI 与 duration 还共享 timing value table，并将 value table 同时用于输入 embedding。这种参数绑定和共享结构限制了输出头自由度，具有更强的正则化效果。

### 4. 外层还截掉了少量异常尾部目标

外层配置启用 `tail_mask_enabled=true` 和 `tail_mask_tf_clamp=true`。训练中超出范围的 log deviation 不参与对应 timing loss，decoder feedback 也会被 clamp。全体 ASAP median 数据上，nonzero IOI 约 2.0% 超出 `[-2,1]`，duration 约 5.0% 超出 `[-2,1]`；这些样本在外层不是完整监督。INSPIRE2 则把绝对值量化并 clamp 到 500 个 timing bins，仍然对每个 note 计算 CE。

### 5. 优化步数和学习率调度不同

相同 epoch 下，INSPIRE2 的 global batch 为 32，每个 epoch 约 322 个 optimizer updates；外层 global batch 为 64，每个 epoch 约 161 个 updates。也就是说，INSPIRE2 在 6 个 epoch 内执行约 1,932 次更新，而外层在 8 个 epoch 内执行 1,288 次更新。

INSPIRE2 没有 warmup，使用 linear scheduler；外层有 50-step warmup 和 cosine scheduler。INSPIRE2 更早以完整 `3e-4` 学习率拟合，train loss 更快下降，随后更早出现 train/valid gap。

### 6. 初始化不同，不能把两条曲线解释成纯架构对照

INSPIRE2 使用官方 PT 权重初始化。外层本次实验是随机初始化。PT initialization 会让模型更快学到 score 的结构和 decoder 的基本表示，因此也可能更快拟合训练作品的绝对 timing；随机初始化的外层模型在 8 epoch 内仍处于较慢的拟合阶段。这个差异与目标变换、decoder mask 和 batch/schedule 混在一起，不能单独归因于 DINR head。

## 对现有曲线的解释

INSPIRE2 CE no-coordinate 的 held-out valid loss 为 `10.948 (ep3) -> 11.324 (ep6)`，而 train loss 继续下降，属于明显的早期过拟合。

外层 DINR no-coordinate 的 valid loss 为 `12.828 (ep1) -> 12.231 (ep8)`，train loss 为 `13.366 -> 12.147`；coordinate 结果也类似。外层曲线不能证明外层 DINR 一定不会过拟合，只能说明在 8 epoch、随机初始化、relative target、50% decoder mask、global batch 64 和当前 tail mask 协议下，模型还没有进入同样的过拟合阶段。

## 同一 INSPIRE2 模型内的控制实验

为避免跨实现混杂，在 INSPIRE2 内使用相同的 PT 初始化、held-out-12-work token cache、seed 42、6 epochs、CE pedal head 和 DINR 结构，依次拆分 decoder mask、有效 batch、学习率日程，并把 timing target 从 absolute 改成 score-relative log。所有组都在每个 epoch 结束后对同一个 507-instance valid set 评估。

验证曲线图：[valid_loss_delta.png](../../INSPIRE2/models/overfit_factorial_20261030/valid_loss_delta.png)；机器可读汇总：[loss_summary.json](../../INSPIRE2/models/overfit_factorial_20261030/loss_summary.json)。图中每条曲线都减去了自身 ep1 valid loss，只展示组内变化，不比较不同 target 的 loss 绝对值。
train/valid 曲线：[train_valid_curves.png](../../INSPIRE2/models/overfit_factorial_20261030/train_valid_curves.png)。train 是 Trainer 每 10 step 的 loss 日志按 epoch 分组求均值，属于近似值；valid 是完整 valid 集的 epoch-end loss。

### Absolute target 优化与 mask 对照

| 配置 | 关键变化 | Valid loss ep1→6 | 最低值 | ep6 相对最低值 |
|---|---|---|---:|---:|
| 基线 | global batch 32、linear、无 warmup、无 mask | 11.320, 10.975, 10.948, 10.999, 11.160, 11.324 | 10.948 (ep3) | +0.376 |
| decoder mask | 仅加 50% shifted-note mask | 11.507, 11.326, 11.244, 11.173, 11.105, 11.084 | 11.084 (ep6) | 0 |
| batch 64 | 仅将 effective batch 加倍 | 11.468, 11.140, 11.001, 10.924, 10.850, 10.896 | 10.850 (ep5) | +0.046 |
| cosine + warmup | 仅换 cosine，warmup 50 steps | 11.386, 10.972, 10.489, 10.520, 10.725, 10.884 | 10.489 (ep3) | +0.395 |
| batch 64 + cosine + warmup | 无 mask；之前的合并 optimizer 对照 | 11.519, 11.189, 10.915, 10.555, 10.575, 10.640 | 10.555 (ep4) | +0.085 |
| batch 64 + cosine + warmup + mask | 组合检验 | 11.640, 11.430, 11.324, 11.169, 11.075, 11.067 | 11.067 (ep6) | 0 |

batch64-only 的 epoch-6 evaluation 已完成，但 epoch-6 checkpoint 的 optimizer 写入因文件系统空间耗尽报错；valid loss 已在日志中记录，不能从该次运行恢复训练。其它本节训练正常结束。为恢复磁盘空间，实验目录中的 `optimizer.pt` 已清理；所有 epoch 的模型权重、trainer state、日志和评估结果仍保留，因此仍可加载/评估 checkpoint，但不能从这些 checkpoint 续训。

### Timing target 对照

在同一 INSPIRE2 模型中，对 target 与 shifted decoder teacher-forcing token 同时做如下编码：score-IOI 大于零时，IOI 与 duration 使用 clipped log ratio `[-2,1]`；score-IOI 为零的 chord IOI 使用 clipped `log(performance IOI in ms)` `[0,5]`。score encoder 输入、网络结构、类别数、loss、初始化和数据 split 均保持不变。relative target 的 CE 绝对值与 absolute target 的 CE 绝对值不可横向比较，只看每个配置内部的 epoch 趋势。

| 配置 | Valid loss ep1→6 | 最低值 | ep6 相对最低值 |
|---|---|---:|---:|
| score-relative log，无 mask | 12.117, 11.715, 11.092, 10.817, 10.873, 11.025 | 10.817 (ep4) | +0.208 |
| score-relative log + decoder mask | 12.370, 12.073, 11.858, 11.623, 11.299, 11.226 | 11.226 (ep6) | 0 |
| score-relative log + batch64/cosine/warmup/mask | 12.557, 12.117, 11.903, 11.585, 11.382, 11.360 | 11.360 (ep6) | 0 |

relative-target 诊断代码位于 `INSPIRE2/src/data_process/dinr_dataset.py` 和 `INSPIRE2/src/train/dinr_absolute.py`，训练时通过 `--timing-target-mode score_relative_log` 启用。本轮使用 `--no-checkpointing`，保留完整训练/验证日志但不写模型 checkpoint；target 变换和 shifted decoder 输入已做 smoke check。

### 因果判断

1. **Decoder feedback mask 是这里最清楚的过拟合抑制因素。** 无论 absolute 还是 score-relative log target，加 50% mask 后，六轮 valid loss 都持续下降。它比 target 变换更稳定地改变了“valid 何时回升”。
2. **Score-relative log target 有帮助，但不足以单独消除过拟合。** 无 mask 时，absolute baseline 在 ep4 开始回升；relative-target baseline 到 ep5 才回升，说明目标分布有贡献，但仍会在继续训练时过拟合。
3. **Cosine + warmup 单独改善前几轮拟合，但不能单独防止后期回升。** ep3 达到最低 valid 后，ep6 回升 0.395。batch64 单独把最低点推迟到 ep5，之后仅回升 0.046，作用较小。
4. **PT 初始化不是过拟合主因。** 去掉初始化导致 valid loss 整体更高、收敛更慢，ep4 后仍略回升；该结果不支持“预训练权重导致过拟合”。
5. **外层的训练曲线更平稳，与 decoder mask 及更温和的优化协议吻合。** 但外层还用了共享 value-table head、tail masking 和不同 backbone 深度；当前实验没有逐一拆分这些结构因素，因此不能把两套实现的全部差异归给 mask 或 target。

这些结果能回答“INSPIRE2 为什么早期回升”的主要机制，但不是最终性能对照：本轮 relative-target 版本沿用 500 类 linear heads，chord log-ms 也按 INSPIRE2 的单独 head 处理，不等同于外层 256-bin、共享 DINR value table 的精确实现。若要完全归因外层结构，还需在同一个代码实现中进一步拆分 tied value table、tail masking 与 encoder/decoder 深度。

## 最重要的结论

当前比较不是“同一 DINR 在两份代码中的不同实现”，而是至少同时改变了：

1. absolute target vs score-relative log target；
2. grouped/chord 双 head vs unified IOI head；
3. 独立 linear output vs tied value-table query head；
4. 完整 teacher forcing vs 50% decoder feedback masking；
5. PT initialization vs random initialization；
6. global batch 32/linear/no-warmup vs global batch 64/cosine/50-step warmup；
7. CE pedal vs BCE pedal；
8. 全部 timing rows vs tail-masked timing rows。

因此，“内层过拟合、外层暂未出现”是多个因素叠加。控制实验支持 **decoder mask 是最强的直接正则化因素，score-relative target 会延后但不会消除过拟合，cosine/warmup 改善早期拟合，batch64 的独立效果有限，PT 初始化不是原因**。两套代码的共享 DINR head、tail masking 和 backbone 深度仍是未拆分因素；当前对训练协议与 timing target 的主要因果判断已有直接对照支持。
