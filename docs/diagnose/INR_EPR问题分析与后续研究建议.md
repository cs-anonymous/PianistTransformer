# INR / INSPIRE 当前问题分析与后续研究建议

日期：2026-10-09

本文档基于仓库中已经完成的实验结果、训练日志、配置、评估脚本和数据审计整理。重点分析四个已被实验暴露的问题：

1. INR 在 ASAP 子集上的收益能否扩展到 PianoCoRe；
2. IOI、duration、velocity、pedal 之间是否存在系统性的目标冲突；
3. PN/PP Wasserstein 是否能代表真实听感；
4. 训练、checkpoint 选择和采样结果为什么不稳定。

文中优先使用已经落盘的汇总结果。没有直接重新运行的内容会明确标为“尚未验证”，不会把假设写成实验结论。

## 1. 现有证据范围

仓库中同时存在论文主线、较早的 INR 探索、全量 PianoCoRe 处理实验和 2026 年 8 月的验证代理实验。原始文件数量非常大，其中包含数十万条 MIDI/JSON 原始输出；本文主要读取了可复核的汇总文件，而不是把每一个 raw output 当成独立实验。核心证据包括：

| 证据 | 位置 | 用途 |
|---|---|---|
| 26 配置 INR 队列 | [`queue_experiment_report.md`](../backup/repo_cleanup_20260804/results/inr_epr_pipeline/asap_processed_musical51_nolossnorm16_slot_effective_20260723/queue_experiment_report.md) | CINR、DINR、slot、musical feature 的统一比较 |
| 21 配置 loss/表示/架构队列 | [`RESULTS_SUMMARY.md`](../backup/results0722/inr_epr_pipeline/lossnorm_ep20_ablation_20260718_094039/RESULTS_SUMMARY.md) | lossnorm、GradNorm、容量、task-token、分布和 backbone 变体 |
| 负结果总报告 | [`asap_musical51_negative_exploration_report_20260725.md`](../backup/repo_cleanup_20260804/results/inr_epr_pipeline/asap_musical51_negative_exploration_report_20260725.md) | 多目标冲突与 loss/metric mismatch 的集中分析 |
| 5 次相同核心配置复跑 | [`BASELINE_REPEAT_LOSS_ANALYSIS.md`](../backup/repo_cleanup_20260804/results/inr_epr_pipeline/musical_slot_effective_20260722_gated_baseline_component_v2_tmux/BASELINE_REPEAT_LOSS_ANALYSIS.md) | 训练路径和下游指标方差 |
| checkpoint 验证代理 | [`validation_proxy_analysis.md`](../backup/results0807/cinr_baseline_ckpt_signal_probe_0805_20260805_142256/validation_proxy_analysis.md) | validation 指标与 test rollout 的相关性 |
| 完整窗口 rollout | [`validation_window_full_ar_20w_analysis.md`](../backup/results0807/cinr_baseline_ckpt_signal_probe_0805_20260805_142256/validation_window_full_ar_20w_analysis.md) | teacher forcing、短 rollout、full-window AR 的差异 |
| k-rollout 代理 | [`k_rollout_hr_cost_correlation_analysis.md`](../backup/results0807/cinr_baseline_ckpt_signal_probe_0805_20260805_142256/k_rollout_hr_cost_correlation_analysis.md) | 评估成本、窗口数和 checkpoint 排名稳定性 |
| 和弦顺序审计 | [`asap_subset_perf_chord_order.json`](../backup/results0714/analysis/asap_subset_perf_chord_order.json) | alignment 对 IOI/PN 的影响 |
| 和弦规模与窗口审计 | [`asap_subset_chord_window_drop_estimate_20260709.json`](../backup/results0714/analysis/asap_subset_chord_window_drop_estimate_20260709.json) | chord-level 表示对序列长度的影响 |

论文主实验本身只使用 ASAP：19 个测试 score、82 个 ground-truth performance，训练数据为 188 个 work、896 个 performance；这组数字见 [`technical_supplement.tex`](../INSPIRE-AAAI2027/technical_supplement.tex:34)。早期完整 ASAP 处理审计使用 207 个 score work、969 个 performance task、887/82 train/test，570,623 个 score notes、368,472 个 chord units；见 [`chord_asap_build_summary.json`](../backup/results0714/analysis/chord_asap_build_summary.json)。

## 2. 论文当前能被数据支持到什么程度

论文中 CINR 在 ASAP 上的主结果是有效的，但它支持的是一个较窄的结论：在固定的、高质量、已对齐的 ASAP 子集上，score-relative timing 加连续分布头能够改善部分 PP/PN 指标，并显著减少 note-level 解码步骤。它还不能支持以下更强的结论：

- 数据规模增大后性能必然提升；
- INR 的收益与数据来源无关；
- 训练 loss 更低就会生成更好的 MIDI；
- PP/PN Wasserstein 排名就是听感排名；
- 相同配置可以稳定复现；
- 一个共享的 integrated note state 可以同时协调所有演奏属性。

仓库中的后续实验已经反复说明这些强结论不成立或尚未得到证实。因此下一版论文或后续工作应把贡献表述为“受控条件下的高效 note-level rendering”，并把 scaling、属性协调、评价和稳定性作为开放问题。

## 3. 数据规模：为什么 ASAP 有效不等于 PianoCoRe 可扩展

### 3.1 观察到的现象

PianoCoRe 的处理规模远大于 ASAP。仓库文档记录的 PianoCoRe-A 处理结果为 1,936 个 works、157,198 个 performance pairs；但当前论文实验只在 ASAP 子集上训练和评估。全量 PianoCoRe 训练没有带来预期的改善，这个现象和仓库中的早期 full PianoCoRe 日志、`fine_pianocore_full` / `pine_pianocore_full` 配置及相关 INR 日志一致。

这里不能把“没有提升”解释成单纯的模型容量不足，因为 ASAP 队列已经显示：增大容量、换表示、换分布和继续训练，都不会稳定改善所有指标。更合理的解释是，全量数据中增加了大量与目标相关但不一致的信息。

### 3.2 数据异质性和标签污染有直接证据

ASAP 的和弦审计显示，性能真实起音顺序与 score 顺序并不一致：

- 1,544,090 个 chord 内相邻对中，true onset inversion 为 780,016，占 **50.52%**；
- 542,887 个 chords 存在至少一次 inversion，占 **66.71%**；
- 只有 79,989 个 chords 的真实起音完全相等，占 **9.83%**；
- inversion delta 的中位数约 **10.42 ms**，95 分位约 **47.92 ms**，最大值 500 ms。

当前 JSON pipeline 为了保持 score 对齐顺序，对这些 note 强制使用单调起音，导致 chord 内部的真实顺序被变成非负 IOI。这样得到的 timing target 混合了音乐表达、MIDI tick 量化和 alignment/order artifact。数据审计还显示，score-order 与 true-onset-then-pitch order 只有 **33.29%** 的 chord observation 一致。上述数字来自 [`asap_subset_perf_chord_order.json`](../backup/results0714/analysis/asap_subset_perf_chord_order.json)。

MusicXML 投影也不是完整观测。当前文档报告的 note-level score feature coverage 约为 **91.14%**，缺失值通过 mask 进入模型；见 [`integrated_note_representation_experiment.md`](../docs/integrated_note_representation_experiment.md)。因此 PianoCoRe 中不同来源的 alignment quality、XML coverage、pedal 记录和 MIDI 清洗质量很可能不同。

这说明全量训练面对的不是“同一分布的数据更多”，而是：

```text
有效音乐信号 + 来源差异 + alignment 噪声 + MusicXML 缺失 + 作品/演奏数量不平衡
```

如果同一 score 的 timing、duration、velocity 或 pedal 标签在不同来源中遵循不同规范，共享模型会学到平均化的 compromise。数据越多，冲突梯度越多，性能未必上升。

### 3.3 需要避免的错误结论

“PianoCoRe 更大但不更好”不能直接证明 INR 不具备 scaling 能力，也不能证明数据越多越有害。当前缺少的是控制变量实验：

- 同一作品集合上逐步加入数据，而不是直接比较 ASAP 和全量 PianoCoRe；
- 按来源、alignment quality、XML coverage、作曲家和演奏者分层；
- work 等权、note 等权、performance 等权三种聚合；
- 去除低质量/高 inversion 样本后的重训；
- 训练数据来源标签或 domain adapter。

在这些实验完成前，最稳妥的判断是：**当前 INR 的 scaling 受到数据异质性和标签可靠性的限制，ASAP 的结果不能外推到 PianoCoRe 全量。**

## 4. 属性干扰：这是重复出现的结构性 trade-off

### 4.1 统一队列中已经看到“一项变好、另一项变差”

26 配置统一队列每个配置都使用同一 rollout 口径：19 个 score、82 个 human MIDI、每个 score 三个 prediction。结果非常清楚：

| 配置 | PN IOI | PN Dur. | PN Vel. | PN Pedal | PP IOI | PP Dur. | PP Vel. | PP Pedal |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CINR baseline | 32.17 | 117.88 | 18.54 | 0.425 | 11.48 | 49.38 | 8.88 | 0.225 |
| CINR 10% bound | 35.86 | 136.45 | **16.31** | 0.426 | 11.88 | 55.51 | **6.66** | **0.208** |
| DINR no-coordinate | 42.85 | 150.79 | 25.80 | 0.397 | 24.71 | 71.14 | 23.62 | 0.209 |
| extra/slot-sum | 32.93 | 119.11 | **16.99** | 0.427 | **11.87** | **47.67** | **6.18** | 0.218 |
| CINR LN-k3 | 45.73 | 144.33 | 16.42 | 0.427 | 12.82 | **40.40** | 6.33 | 0.217 |

`extra/slot-sum` 是队列中最好的整体折中，综合相对 canonical baseline 约改善 4.7%，但它也没有在每个指标上都胜出。`CINR 10% bound` 的 velocity 和 pedal 较好，却牺牲了 timing；`LN-k3` 的 PP duration 最好，却明显损害 PN timing。这些数字见 [`queue_experiment_report.md`](../backup/repo_cleanup_20260804/results/inr_epr_pipeline/asap_processed_musical51_nolossnorm16_slot_effective_20260723/queue_experiment_report.md)。

同一现象在 21 配置队列中再次出现：`t5_8_4` 综合排名最好，但单项最佳分别属于 DINR、CINR、slot-sum 等不同配置；没有一个配置全面统治八个 PN/PP 属性。完整表见 [`RESULTS_SUMMARY.md`](../backup/results0722/inr_epr_pipeline/lossnorm_ep20_ablation_20260718_094039/RESULTS_SUMMARY.md)。

### 4.2 针对属性干扰的实验没有解决根因

仓库已经尝试了多种看似合理的方案：

- loss normalization 和 GradNorm；
- timing/expression 拆分 embedding；
- four-embedding、hidden size 1024；
- slot-direct、slot-sum、slot256-MLP、sine representation；
- 2N/task-token decoder 和 10+2 backbone；
- velocity PN/PP W1 辅助损失；
- 全部 PN/PP W1 辅助损失。

容量实验中，`hidden1024` 和 `two_embedding` 的 velocity 有改善，但不是稳定的全面改善；`four_embedding` 反而变差。`velocity W1` 实验将 PN velocity 从 18.54 降到 17.13、PP velocity 从 8.88 降到 7.08，但 pedal 从 PN/PP 0.425/0.225 变为 0.437/0.250。`allW1` 更明确地失败：它在训练内部 loss 下降，但 rollout 的 PN/PP HR 比 baseline 更差。详细数字见 [`asap_musical51_negative_exploration_report_20260725.md`](../backup/repo_cleanup_20260804/results/inr_epr_pipeline/asap_musical51_negative_exploration_report_20260725.md)。

因此目前不能把问题归因于“loss 权重还没调好”。现有证据更支持以下判断：

1. timing、duration、velocity、pedal 在演奏中有真实的条件依赖，不是四个独立回归目标；
2. 当前把所有属性融合进一个 hidden state，再由多个 head 读取，会让共享参数承受相互竞争的梯度；
3. 简单的静态 loss weight、增加 embedding 宽度或增加 task token 不能消除这种竞争；
4. pedal 还具有跨音符连续性，不适合完全作为四个独立 note snapshot 处理。

需要注意：仓库目前还没有直接记录每个属性梯度夹角，因此“梯度冲突”的定量程度尚未测量。它是由大量 trade-off 结果支持的机制假设，下一步应通过 gradient cosine、参数更新方向和属性条件互信息验证。

## 5. PN/PP Wasserstein 的有效范围与局限

### 5.1 它们测量了什么

PP 将一首曲子的 note values 汇总后比较边缘分布，适合测量整首曲子的 timing、duration、velocity、pedal 数值范围和总体离散程度。PN 在相同 score note 上比较 prediction/reference 分布，理论上更关注 note-conditioned accuracy。

两者都没有直接测量：

- 句法和拍点组织；
- phrase-level rubato；
- 旋律与伴奏的层次；
- duration、velocity、pedal 的联合关系；
- 踏板跨音符连续性；
- 和弦组内的相对起音；
- 音频播放后的整体听感。

因此“Wass 更低但听感更差”与指标定义并不矛盾：它说明边缘分布距离不是音乐结构和感知质量的充分统计量。

### 5.2 仓库中的实验已经显示指标排名不稳定

不同配置的单项最佳分散在不同模型：例如 `dinr_no_coord` 获得最佳 PP IOI/PP pedal，`t5_8_4` 获得最佳 PP velocity，`slot-sum` 获得更好的 PP duration/velocity，而 CINR baseline 在 PN timing 上更强。这种 Pareto trade-off 说明 HR 的单一平均值会隐藏维度间冲突。

checkpoint 代理实验也显示验证指标并不等价：20 个 validation windows、每个窗口 2 次 rollout 时，validation PP velocity 与 test PP velocity 的 Spearman 相关只有 **0.067**；validation PN velocity 与 test PN velocity 的 Spearman 只有 **0.033**。相同实验中 duration 的相关约为 0.783/0.800，IOI PN 相关约为 0.850，而 pedal 相关在 PP 上为 0.933、PN 上只有 0.317。见 [`validation_window_sampling_wass_20w_s2_analysis.md`](../backup/results0807/cinr_baseline_ckpt_signal_probe_0805_20260805_142256/validation_window_sampling_wass_20w_s2_analysis.md)。

这不是说 PN/PP 完全无效，而是说它们对不同属性、不同样本规模和不同 validation population 的可靠性不同。尤其 velocity 不能仅靠当前 validation Wass 选择 checkpoint。

### 5.3 评估样本量和聚合方式限制结论

主线 sampling 结果一般只有每个 score 三个生成样本；部分 PianoCoRe 非 ASAP 评估甚至是 49 个 score、每个 score 一个 sample。用这么少的样本估计 per-note distribution，Monte Carlo 方差可能与配置间差异同量级。

此外，HR 将四个属性先除以 human denominator 再平均。pedal 的 human denominator 很小，因此 pedal 的微小变化可能显著改变 HR；这会让“平均 HR 排名”与整体听感更不一致。仓库的 baseline 验证分析已经指出 raw Wasserstein ranking 比 HR ranking 更稳定，HR 对小 human denominator 尤其敏感；见 [`baseline0807_validation_hr_analysis.md`](../docs/baseline0807_validation_hr_analysis.md)。

### 5.4 应保留 Wasserstein，但不能继续单独使用

最合理的做法不是删除 PP/PN，而是把它们降为 distribution-level metrics，并增加：

- note-level：alignment、missing/insertion、robust MAE、极端 timing 比例；
- phrase-level：beat-relative timing、句尾 ritardando、dynamic contour、左右手同步；
- joint-level：IOI-duration、timing-velocity、velocity-pedal 的 covariance 或 rank correlation；
- continuity：pedal change、跨窗口 jump、相邻 note 的 timing/duration smoothness；
- perceptual-level：双盲听感实验和维度评分。

最有价值的验证不是再发明一个距离，而是取现有模型和样本，计算各种指标与人类听感评分的 Spearman/Kendall 相关，直接测量哪些指标能预测“听起来更好”。

## 6. 稳定性：训练、checkpoint 和采样分别存在问题

### 6.1 相同核心配置的复跑显示下游指标方差很大

`BASELINE_REPEAT_LOSS_ANALYSIS.md` 汇总了五个核心配置几乎相同的 run。训练 loss 差异很小：best eval loss 的均值 3.1177、标准差 0.0184、CV 0.59%；final train loss 的 CV 只有 0.38%。但下游 sampling 指标差异明显：

| 指标 | 均值 | 标准差 | CV |
|---|---:|---:|---:|
| PP IOI | 12.687 | 0.609 | 4.80% |
| PP duration | 60.966 | 4.542 | 7.45% |
| PP velocity | 7.912 | 3.414 | **43.15%** |
| PN IOI | 33.473 | 0.280 | 0.84% |
| PN velocity | 18.195 | 1.645 | 9.04% |
| PP pedal | 0.1915 | 0.0088 | 4.61% |

这直接证明“相同参数结果不稳定”主要来自训练路径和模型落点，而不只是 inference sampling。最好的 eval loss run 的 PP/PN velocity 反而最差，说明 eval loss 不能作为 velocity Wasserstein 的可靠 checkpoint selector。

### 6.2 训练后期仍在变化，且 loss 与 rollout 不同步

复跑报告中五个 run 的 best checkpoint 都在最后一个评估点，last-5 eval slope 仍为负或接近 0，说明 20 epoch 可能还未稳定收敛。但更长训练也不是简单答案：allW1 实验中内部 eval loss 持续下降，最终 rollout 指标仍然变差；`asap_musical51_negative_exploration_report_20260725.md` 给出了 allW1 epoch 14.76/18.43/24 的具体结果。

因此目前至少有两个独立问题：

1. 优化还没有完全收敛；
2. 即使 loss 继续下降，teacher-forced loss 也不保证 rollout MIDI 变好。

### 6.3 验证代理的排序只在较高 rollout 成本下较可靠

`k_rollout_hr_cost_correlation_analysis.md` 显示，40 个窗口、k=4 的平均 HR Spearman 只有 0.483，k=32 才达到 0.800；80 个窗口时 k=32 的平均 HR Spearman 为 0.667。full token-wise AR 20 windows/2 samples 的平均 HR Spearman 为 0.683，但成本约 420 秒；40 windows、k=32 的约 43.5 秒，成本约为十分之一。

这说明短 validation probe 不能随意替代 full rollout。增加窗口数也不保证更好：80 windows 不如 40 windows，原因可能是 validation population 改变、human baseline 改变或加入了区分度较低的窗口。该分析明确指出这些相关性只来自一个训练 run 的九个 checkpoint，尚未跨 seed 验证。

### 6.4 采样温度和 top-p 是质量—多样性旋钮，不是固定“正确参数”

仓库已经做过 DINR 和 matched DLM 的 temperature/top-p/top-k matrix。主论文选择 temperature 1.0、top-p 0.9，是在 PP/PN HR 之间取折中，而不是证明它对所有模型和数据都最优。sampling sweep 显示更强截断通常改善 PN，却损害 PP 多样性；更高温度或更宽候选集可能改善 PP，却增加 note-level 偏差。

因此相同参数产生不同听感是预期现象的一部分；但当前没有报告每首曲子跨 seed 的方差，也没有把 quality-diversity Pareto frontier 作为正式结果。后续必须把“平均质量”和“样本稳定性”同时报告。

## 7. 其他会削弱论文结论的问题

### 7.1 对齐依赖和 chord order

INR 的输入输出都依赖 note-level alignment，并把 pitch 继承为 oracle。它没有解决插入、遗漏、装饰音和 chord order。上面的 inversion 审计说明 PN timing 在部分样本上可能测到强制 score order 后的 artifact，而不是纯粹的 expressive timing。

优先级最高的数据改进是 chord-level set alignment：同一 score onset 的 notes 先作为无序集合匹配，再从 chord group 中定义相对 onset spread，而不是把 chord 内部每个相邻 note 当作普通 IOI。

### 7.2 pedal representation 不匹配控制信号本质

四个 pedal snapshots 保持了固定长度，但它们不是连续 pedal curve，也没有显式建模跨 note 的 sustain、release 和和声变化。当前 pedal 可能在边缘分布上较好，却在听感上出现不自然的断续或错误换踏板。

### 7.3 efficiency comparison 需要更谨慎

论文中的 INR 从零训练于 ASAP 子集，而 Pianist Transformer 使用大规模预训练模型。INR 的推理步骤确实因每 note 一个 timestep 而显著减少，但训练 GPU-hour 的比较同时混合了不同数据量、初始化和训练目标。后续应报告同数据、同初始化条件下的 wall-clock、FLOPs、每 note latency，以及达到相同感知质量所需的成本。

### 7.4 score-span normalization 改变了任务定义

所有 human performance 被按 score onset span 做全局缩放，这有助于比较局部 rubato，却移除了全局 tempo 风格。当前结论应限定为“score-span-normalized expressive rendering”。需要补充 raw-tempo 结果，或单独建模 global tempo，再评价 local expression。

## 8. 综合诊断

把已有实验放在一起后，当前最可信的因果链是：

```text
PianoCoRe 数据异质性 / alignment 噪声
                 ↓
不同作品和来源的属性条件分布不一致
                 ↓
共享 note state 的多属性目标发生竞争
                 ↓
teacher-forced loss 与 rollout MIDI 分布脱钩
                 ↓
PN/PP 边缘 Wass 对结构和听感覆盖不足
                 ↓
checkpoint 排名、采样结果和最终听感不稳定
```

这条链条比“CINR 分布头不够好”更符合全部结果：

- CINR、DINR、slot 和 distribution sweep 只改变 trade-off；
- lossnorm/GradNorm 没有解决 rollout mismatch；
- allW1 surrogate 也没有解决 mismatch；
- identical-core rerun 的 train loss 很稳定但 PP velocity 方差达到 43.15%；
- validation velocity 与 test velocity 的相关很低；
- chord inversion 和 XML coverage 说明 target 本身并不完全干净。

因此当前主要瓶颈不是单个 head 或单个超参数，而是 **数据—目标—表示—评估四者没有形成闭环**。

## 9. 是否值得继续研究

值得，但下一阶段不应继续做无约束的 configuration search。已有实验已经证明，单纯换 distribution family、slot fusion、hidden size、loss normalization 或 task-token 不足以解决问题。继续研究应围绕三个可检验的新问题。

### 9.1 先建立可靠诊断基线

第一轮不改模型，完成：

1. 至少 5 个 training seeds，统一保存 checkpoint 和完整 rollout；
2. 每个 checkpoint 至少 10 个 sampling seeds，报告均值、标准差和分位数；
3. 按 data source、alignment quality、XML coverage、composer、piece length 分组；
4. 计算属性梯度 cosine，验证多任务冲突是否真实存在；
5. 对 PP/PN、结构指标和双盲听感做相关性分析；
6. 报告 raw-tempo 与 score-span 两种评估。

这一步的产出是稳定的诊断协议，而不是新的模型分数。

### 9.2 解决属性冲突，而不是继续调静态权重

最有根据的模型方向是 shared trunk 加属性专家或 adapter：

```text
shared score representation
        ├── timing expert
        ├── duration/articulation expert
        ├── dynamics expert
        └── pedal curve expert
```

专家之间只在低层或显式 cross-attribute module 交换信息。训练时比较 GradNorm、PCGrad、CAGrad 和 uncertainty weighting，但必须以多 seed、rollout 和听感为判据。

属性之间还应显式条件化：duration 可以依赖 timing，velocity 可以依赖 phrase/timing latent，pedal 使用跨 note curve decoder。这样比把所有变量独立地从同一个 hidden state 读出更符合演奏生成的因果结构。

### 9.3 把评价升级为结构和感知一致性

保留 PN/PP 作为分布描述，但新增：

- chord-level onset spread；
- beat-relative timing curve；
- phrase-final slowing；
- melody/accompaniment velocity contrast；
- pedal continuity and harmonic-boundary alignment；
- joint distribution distance；
- multi-sample diversity and calibration；
- 人类双盲听感评价。

研究重点应从“哪个模型 W1 最低”改为“哪些指标能预测人类认为更好的演奏”。如果指标与听感排序不一致，新的结构感知指标本身可以形成独立贡献。

## 10. 推荐执行顺序

| 优先级 | 工作 | 原因 | 成功标准 |
|---:|---|---|---|
| 1 | 复跑 baseline、slot-sum、10% bound，5 seeds | 确认已有正信号是否可复现 | 主要指标置信区间和 seed 方差可报告 |
| 2 | 完成 source/quality 分层的 PianoCoRe scaling | 区分数据量问题和数据质量问题 | 找到收益随数据量/质量的曲线 |
| 3 | 计算 gradient conflict 和 rollout drift | 验证机制假设 | 属性梯度夹角、窗口边界误差与指标相关 |
| 4 | 做指标—听感相关性实验 | 判断 PN/PP 是否能用于选模 | 指标与听感的 rank correlation |
| 5 | chord-set alignment + pedal curve decoder | 修正最明确的表示/标签问题 | chord timing 和 pedal continuity 改善 |
| 6 | shared + attribute experts | 解决多属性共享冲突 | 在多个 seed 和多个数据域上同时改善 |

不建议优先做的工作是继续扩展 distribution family、继续微调 temperature，或只增加 hidden size。已有队列已经表明这些方法大多只移动 Pareto trade-off，不能稳定扩大整体收益。

## 11. 当前结论

现有数据并不支持“INR 已经解决了大规模 expressive piano rendering”。它支持的是一个更具体、也更可靠的结论：

> 在小规模、高质量、已对齐的 ASAP 子集上，CINR/INR 的 note-level 解码可以降低序列长度，并在部分 timing 和 piece-level distribution 指标上优于 token-level baseline；但当数据扩展到异质的 PianoCoRe、属性目标共同优化、模型进入长序列 rollout 后，收益受到标签噪声、属性冲突、训练目标错位、评价盲点和随机性共同限制。

这个结论仍然很有研究价值。下一阶段真正值得研究的主题不是“再找一个更低的 Wasserstein 配置”，而是：

**如何在异质对齐数据上学习相互协调的 timing、dynamics、articulation 和 pedal，并用能够预测听感的结构化指标稳定地选择模型。**

