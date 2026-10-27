# INSPIRE 审稿意见分析与行动计划

本文档基于 [review.md](./review.md)、仓库中的 INR 实验结果和后续诊断文档整理。目标不是把所有意见都塞进一次 rebuttal，而是区分论文当前的可信度问题、需要补实验的研究问题，以及可以留到后续工作的长期问题。

## 1. 总体判断

当前评分结构是 `5/5/7/3`。评审并没有否定 INSPIRE 的基本想法：多个审稿人认可 note-level rendering 的实用性、score-relative timing target 的有效性，以及明显的推理效率优势。真正的问题是，论文对这些结果的解释超出了实验能够支持的范围。

目前最严重的风险不是模型完全无效，而是以下证据链没有闭合：

```text
主表数字与 canonical 配置不一致
        ↓
headline gain 的可信度受到影响
        ↓
评价指标又不能测量音乐结构和听感
        ↓
无法确认收益来自 INR、timing target 还是其他训练差异
        ↓
论文的 novelty、causal claim 和 human-like claim 都显得过强
```

因此，短期目标应该是把论文改成一个审慎、可核查的结果报告；长期目标才是证明 INR 对质量的独立贡献，并建立更可靠的音乐结构和听感评价。

## 2. 评审共识

### 2.1 已经被认可的部分

- 问题重要：score-to-performance rendering 具有明确的音乐生成和实际应用价值。
- 设计合理：把已对齐的一个音符组织成一个 note-level representation，适合减少自回归长度。
- score-relative log timing residual 有实证价值，绝对 timing 的结果明显更差。
- INR/CINR 带来显著的推理成本下降，效率方向是可信的。
- 实验包含较多 ablation，论文的基本技术路线并非没有依据。

这些优点意味着不需要推翻整篇论文，但必须收窄论文声称的范围，并修复可信度问题。

### 2.2 反复出现的主要问题

四份有效评审和 AI review 反复指出以下问题：

1. **配置和数字不一致**：Table 2、补充材料、正文 canonical config 似乎不是同一个模型；CINR 的 13.8% headline 可能对应 `extra/rep_slot_sum`，而非正文描述的 slot+MLP。DINR 的 coordinate augmentation 也存在类似问题。
2. **评价指标过窄**：PP Wasserstein 忽略 note order，PN Wasserstein 逐 note 比较边缘属性；它们不能证明 phrase structure、beat-level tempo shaping、cross-attribute coherence、pedal continuity 或听感。
3. **贡献归因不充分**：INR、relative timing、连续/离散输出头、backbone、预训练和数据设置同时变化，无法把质量提升单独归因于 INR。
4. **新颖性表述过强**：one timestep per note、typed fields 和 compound representation 都有先例。贡献应表述为面向 aligned EPR 的组合设计，而不是首次提出 note-level/compound decoding。
5. **相关工作和基线不完整**：Compound Word Transformer、VirtuosoNet、ScorePerformer、DExter、SyMuPe/PianoFlow 的关系或结果说明不足。
6. **表达和符号不够友好**：DINR 公式、slot 的输入/反馈/输出角色、zero-score-IOI、pitch copy、pedal reconstruction 等没有清楚定义。
7. **数据和效率范围没有说清楚**：论文只在 ASAP 子集上实验；训练和推理成本的 GPU、precision、batch、cache、checkpoint 和 baseline 复测条件需要明确。

## 3. 按严重程度分级

### P0：必须立即处理的可信度问题

这些问题不修，任何新增实验都会被“结果不可核查”抵消。

| 问题 | 需要做的事 | 适合 rebuttal 还是长期修改 |
| --- | --- | --- |
| Table 2 与 canonical config 不一致 | 建立“表格行 -> config -> checkpoint -> sampling protocol -> output metric”映射；重新核对 CINR/DINR 主结果；必要时更正摘要中的 13.8% | 一周内必须完成 |
| 训练/采样结果没有不确定性 | 至少报告已有多次 run、checkpoint 或 sampling seed 的均值、标准差和样本数；解释 2.38、2.54、2.08 等差异来自哪里 | 一周内完成已有结果；正式扩展留后续 |
| 质量收益的因果表述过强 | 明确声明当前比较证明的是“整体系统差异”，不能证明 INR 单独造成全部质量提升；把 relative timing 的贡献单独描述为 target ablation 结果 | 一周内必须完成 |
| “no quantization / human-like / rhythmic organization”措辞不准确 | 将 CINR 改称 continuous parameterization with discretized likelihood；把 human-like 结论改为 distributional similarity；删除或限定无法由 PP/PN 支持的表述 | 一周内必须完成 |
| canonical model 定义不统一 | 正文、摘要、表格、补充材料、配置 manifest、代码默认值使用同一命名和同一模型 | 一周内必须完成 |

### P1：直接影响两个 5 分审稿人的研究证据

这些问题决定能否从“边缘拒稿”提升到“边缘接收”。

| 问题 | 建议动作 | 优先级判断 |
| --- | --- | --- |
| Wasserstein 无法衡量音乐结构 | 基于已有生成 MIDI 增加 beat-level tempo correlation、velocity contour correlation、相邻 onset/duration smoothness、chord timing consistency、pedal continuity 等至少一组顺序敏感指标 | 高；优先使用已有输出 |
| 缺乏听感验证 | 若时间允许，做小规模随机双盲听测；至少报告已有媒体样本的协议和限制。旧项目中的听测不能直接当作 INSPIRE 结果，除非对当前输出重新执行 | 高；听测比新增 KL/JS 更有价值 |
| INR 与 timing target 未隔离 | 最理想是同 backbone、同数据、同训练预算的 serial-token + relative timing baseline；如果做不到，明确将其列为未解决的 attribution limitation | 高；成本可能超过 rebuttal 周期 |
| DINR/CINR 不是纯 head ablation | 写一张 ablation matrix，标出 input representation、output head、value table、coordinate augmentation 分别是否改变；禁止把 DINR/CINR 差异解释成单一因素 | 一周内可完成文字；实验留后续 |
| 与 prior work 的定位不清楚 | 逐项比较 Compound Word、VirtuosoNet、ScorePerformer、DExter：外层时间步、输入信息、数值编码、输出参数化、是否 aligned EPR、是否支持连续分布 | 一周内必须补齐 |
| 只在 ASAP 子集上实验 | 明确 ASAP 的 works、performances、test scores 和 full PianoCoRe-A 的规模差异；把论文定位为 controlled ASAP study，不声称已验证 scaling | 一周内可完成；规模实验留后续 |

### P2：应补充的技术细节和可复现性信息

这些问题通常不会单独决定接收与否，但会影响审稿人对论文严谨度的判断。

- 明确 score pitch 是 copy-through，不由 performance head 预测。
- 解释 `score IOI = 0` 时的 target、mask、support 和 reconstruction 规则。
- 说明 bounded discretized logistic 的 support、端点 tail handling，以及 “5% bound” 的确切定义。
- 说明 note-level pedal snapshots 如何合成为全局 MIDI controller trajectory。
- 说明 validation windows 的切分顺序、overlap 处理，以及 Figure 4 使用 validation 还是 test。
- 完整报告硬件型号、GPU 数量、dtype、batch、worker、cache、warm-up、计时范围和 baseline 实现。
- 在可能的情况下补充 KL/JS，但必须固定 binning、平滑和 support；不能用它替代结构指标或听测。

### P3：不应在本次 rebuttal 中承诺完成的长期问题

这些问题是真正的研究缺口，通常需要重做数据、模型或评价协议：

- 在 full PianoCoRe-A、第二个 aligned corpus 或跨来源数据上验证 scaling。
- 对 alignment quality、chord density、ornamentation、XML coverage 分层训练和评估。
- 用 chord-set alignment 处理 chord 内部 onset inversion，而不是把所有 note 当作普通序列。
- 将 pedal 从四个 note snapshots 扩展为连续的跨音符控制轨迹。
- 建模 timing、duration、velocity、pedal 的联合分布或 attribute-conditioned experts。
- 进行多 seed、多 checkpoint、多 sampling seed 的完整稳定性分析。
- 设计严格的同 backbone serial-vs-note-level efficiency/quality 对照实验。

## 4. 一周 rebuttal 的建议顺序

### 第 1 天：结果审计

1. 锁定唯一 canonical config。
2. 重新生成主表、摘要数字和补充表的 config mapping。
3. 对每个 headline metric 记录 checkpoint、sampling seed、temperature、top-p、sample 数和 test score 数。
4. 决定是否必须把 13.8% 改为 canonical 配置的实际提升。

### 第 2 天：重写贡献和 claim

将贡献拆成三个可检验假设：

1. note integration 减少 autoregressive decoding cost；
2. numerical parameterization 提供 ordinal/continuous inductive bias；
3. score-relative timing 简化 timing prediction。

明确哪些结果支持哪个假设，避免把所有收益都归因于 INR。

### 第 3 天：补可读性和符号

增加一张完整的 role-by-slot 表，重写 DINR 公式，补充 pitch copy、zero IOI、pedal reconstruction 和 support 定义。面向一般深度学习读者重写问题定义和方法概览。

### 第 4--5 天：利用已有输出补评价

优先计算不需要重新训练的指标：

- beat-level tempo curve correlation；
- velocity/dynamics contour correlation；
- adjacent-note timing smoothness；
- chord-level onset consistency；
- pedal transition/continuity statistics。

如果有可靠的参与者和音频流程，再做小规模双盲听测。听测样本应固定 score、随机化顺序、隐藏模型名称，并报告样本数和限制。

### 第 6 天：写 rebuttal 的限制和基线说明

对无法完成的 matched baseline、SyMuPe/PianoFlow、full PianoCoRe-A，给出具体原因和明确边界，不要用泛泛的“时间有限”作为唯一解释。把“当前工作证明了什么”和“尚未证明什么”并列写清楚。

### 第 7 天：一致性检查

- 摘要、主文、补充材料和代码配置是否使用同一个 canonical 名称；
- 所有百分比是否能从表格重新计算；
- 所有 “first / human-like / coherent / SOTA” 是否有足够证据；
- 新增指标是否使用 test 数据，是否存在调参泄漏；
- rebuttal 是否承认 metric blind spot 和 attribution limitation。

## 5. 中长期论文修改路线

### 阶段一：可信度修复

目标是让读者可以复现每个数字，并准确理解论文的贡献边界。交付物包括统一配置、修正后的主表、完整符号表、效率协议和 claim audit。

### 阶段二：因果和基线实验

在同一数据、同一 backbone、同一训练预算下比较：

```text
serial-token + absolute timing
serial-token + relative timing
note-level INR + absolute timing
note-level INR + relative timing
```

再分别固定输入表示或输出 head，拆分 CINR/DINR 的影响。只有完成这一阶段，才能比较有力地讨论 INR 本身的质量贡献。

### 阶段三：结构化评价和泛化

在现有 PN/PP Wasserstein 之外，加入顺序、联合关系、连续性和听感评价，并在 PianoCoRe-A 或第二个 corpus 上做分层实验。该阶段才适合重新讨论 scaling、human relevance 和跨数据集泛化。

## 6. 应避免的策略

- 不要继续把 “one timestep per note” 写成单独的首次创新。
- 不要用 PP/PN Wasserstein 直接证明 rhythmic coherence 或 human-like quality。
- 不要把 extra/slot-sum 的最好数字包装成 canonical baseline。
- 不要把 CINR 优于 DINR 解释成 CINR 全面解决 expressive rendering；DINR 仍然是表示和输出参数化的对照。
- 不要在没有 matched baseline 的情况下声称质量提升来自 INR。
- 不要为了回应审稿人而堆积大量新的 distribution family；当前瓶颈首先是配置一致性、评价有效性和归因设计。

## 7. 最终目标

本轮 rebuttal 的现实目标不是证明论文已经完成，而是让审稿人相信：

1. 数字和配置可以被准确核查；
2. 作者理解并承认当前指标和归因的边界；
3. INSPIRE 的真正贡献被重新、准确地表述；
4. 已有结果支持一个窄但可信的结论：在高质量、已对齐的 ASAP 子集上，INR 可以显著减少 note-level 解码成本，并在部分 timing 和 piece-level distribution 指标上优于 token-level baseline。

长期论文修改则应围绕“独立贡献、结构化评价、数据质量和泛化”展开，而不是继续只寻找一个更低的 Wasserstein 数字。
