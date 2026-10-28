# 第一条链路：60 epoch 的 PN/PP teacher-forcing 诊断

## 结论

这轮实验支持“训练早期的 loss 与 PN/PP teacher-forcing HR 总体同向变化”，暂时没有证据表明 PN 和 PP 的 HR 与 validation loss 存在根本性脱钩。原始 12 epoch 结果中的 epoch 5 仍然是一个异常 checkpoint，但在 6 条独立的 60 epoch 训练轨迹中，从 epoch 4、8、12、24、36、48 到 60，loss、PN HR 和 PP HR 都整体下降。

这不能写成“模型已经收敛”：6 条 run 的预注册 plateau 检查均为 `inconclusive_not_at_plateau`。因此本轮结果是延长训练后的诊断证据，不是收敛模型上的正式 loss 优劣结论。

## 指标口径

- 评估对象是固定 validation split 的 68 个 eligible ASAP works，每个 work 使用一个固定窗口，`rollout_k=0`。
- 每个 checkpoint 都使用同一套 68 个 work/window；评估文件位于 `results/pn_convergence_constant60/{baseline,cached}/seed_*/tf_eval68/`。
- 论文 HR 使用四个属性（IOI、duration、velocity、pedal）；数值越低越好。
- `pn_hr`/`pp_hr` 均使用四个属性；当前汇总中的 composite HR 是 PN 与 PP 的算术平均。
- 后续相关性分析统一沿用论文已有 HR 定义，不引入 relevance、倒数变换或新的 HR 命名。
- 本文先以 PN-only 为主指标，PP 单独报告；不能把 PP 当成 PN 的替代品。

## 60 epoch 结果

下表给出每条轨迹的 epoch 4 到 epoch 60。论文 HR 与 loss 的方向相同，均为下降更好。

| 条件 | seed | loss 4 -> 60 | PN HR 4 -> 60 | PP HR 4 -> 60 | loss-PN HR Spearman | loss-PP HR Spearman |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 11 | 18.269 -> 15.045 | 2.071 -> 1.193 | 5.735 -> 2.168 | 1.000 | 0.964 |
| baseline | 22 | 17.823 -> 15.116 | 1.773 -> 1.239 | 4.560 -> 2.424 | 1.000 | 1.000 |
| baseline | 33 | 18.675 -> 14.997 | 1.647 -> 1.174 | 4.405 -> 2.103 | 1.000 | 1.000 |
| cached | 11 | 19.147 -> 15.026 | 1.675 -> 1.179 | 4.542 -> 2.089 | 0.964 | 0.857 |
| cached | 22 | 18.597 -> 15.126 | 1.769 -> 1.218 | 4.817 -> 2.255 | 1.000 | 0.964 |
| cached | 33 | 18.683 -> 15.027 | 1.655 -> 1.167 | 4.441 -> 2.129 | 1.000 | 1.000 |

把 42 个 checkpoint pooled 后，Trainer validation loss 与论文 HR 及 grouped TF loss 的相关性为：

| 指标 | Pearson | Spearman | Kendall |
|---|---:|---:|---:|
| grouped TF loss | 0.997 | 0.996 | 0.961 |
| PN HR | 0.905 | 0.966 | 0.849 |
| PP HR | 0.923 | 0.953 | 0.830 |
| PN HR | 0.933 | 0.978 | 0.884 |
| PP HR | 0.946 | 0.972 | 0.870 |

其中 Pearson 反映数值近似线性关系，Spearman/Kendall 更适合回答“loss 能否正确选择 checkpoint”。在每条单独 seed 轨迹内，PN HR 的秩相关几乎全部为 1；跨 seed pooled 后仍为 0.966。PP 的秩相关略低，但仍保持同向且较强。

## epoch 5 异常

原始 baseline 12 epoch sweep 中，epoch 5 的 grouped TF loss 为 `18.495`，低于 epoch 4 的 `19.840`，但 PN HR 从 `1.622` 恶化到 `2.050`，PP HR 从 `4.397` 恶化到 `5.713`；epoch 6 又分别回到 `1.579` 和 `4.154`。因此 epoch 5 不是简单的“loss 和 HR 都变差”，而是 loss 下降、HR 同时变差的局部异常。

去掉 epoch 5 后，原始 1--11 个精确 validation event 上，Trainer loss 与 PN HR 的 Pearson/Spearman/Kendall 为 `0.848/0.939/0.822`，与 PP HR 为 `0.874/0.952/0.867`。这说明异常点会显著拉低 Pearson，但不是整个链路的常态。

60 epoch 多 seed 结果表明：这个异常没有在更长训练中持续出现，不能据此认定现有 loss 与 PN/PP 指标根本不匹配；但由于没有保存每个早期 epoch 的 60-run checkpoint，仍不能证明 epoch 5 的原因已经定位。后续应对原始 epoch 4/5/6 做逐 work、逐属性对照，检查是否由少数窗口、IOI tail、mask 或 checkpoint/evaluation 对齐造成。

## 当前决策

1. 先保留现有 loss，不立即加入新的 PN/PP auxiliary loss；延长训练和多 seed 已经显示出较稳定的 checkpoint 排序关系。
2. 报告中将 PN 与 PP 分开，并继续使用论文已有 HR 定义；composite HR 明确写成 `mean(PN HR, PP HR)`。
3. 本诊断只研究 HR 与 loss/rollout 的相关性，不重新定义 HR，也不引入 relevance 或倒数变换。
4. 60 epoch 仍未通过 plateau gate。若要形成正式结论，应预先决定继续训练到 plateau，或在不改变阈值的情况下把本轮明确标为 diagnostic。
5. 只有当更长、已通过 plateau 的多 seed 结果仍出现 loss 下降而 PN HR 系统性上升时，才进入 loss 聚合、CDF-W1 surrogate 或 auxiliary loss 的改动。

