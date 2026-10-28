# 五类方差分离实验运行结果

## 运行状态

截至 2026-10-13，已完成并通过文件检查的任务如下：

| 阶段 | 实际完成 | 结果 |
|---|---:|---|
| M0 evaluator implementation | 8 条件 | 完成，所有条件结果逐项完全一致 |
| M1 fixed-checkpoint sampling | 24 seeds | 完成 |
| M2 checkpoint selection | 5 checkpoints | 完成 |
| M3 training paths | 5 training seeds，12 epoch | 完成 |
| M3 checkpoint evaluation | 45 = 5 seeds × 3 epochs × 3 sampling seeds | 完成 |
| M4 window rollout | 3 个 seed，m=40，k=4/32 | 完成诊断子集；m=80 因只有 68 个 eligible works 未执行 |

所有 GPU 进程均已结束，当前没有残留 inference 或训练进程。

## 主要数值

### M0：评估实现方差

单个 prediction manifest 在 worker 数 1/2/4/8、原始/排序文件顺序下，PN/PP 全部字段的最大差异为 **0**。因此当前实现层面的 worker、文件顺序和聚合顺序没有制造可见方差。

### M1：固定 checkpoint 的 sampling 方差

24 个 sampling seed 使用同一 checkpoint、同一 full-score sampling 协议。关键 CV 如下：

| 指标 | CV |
|---|---:|
| PN IOI | 0.58% |
| PN duration | 0.23% |
| PN velocity | 1.02% |
| PP IOI | 1.71% |
| PP duration | 0.83% |
| **PP velocity** | **5.11%** |

固定 checkpoint 后，sampling 对 PP velocity 有可见影响，但规模明显小于训练路径和 checkpoint 演化造成的变化。

### M2：同一训练轨迹的 checkpoint 方差

5 个 checkpoint 使用同一个 sampling seed 42。PP velocity 从 **7.93**（checkpoint-140）变化到 **13.81**（checkpoint-700），随后在 checkpoint-816 为 **9.37**。PP IOI 从 **11.61** 变化到 **88.00**，CV 达 **108.92%**；PN IOI CV 为 **55.51%**。

这说明 checkpoint 选择会改变下游分布指标，且变化不是单调的。训练 loss 或最后 checkpoint 不能自动代表最佳 PN/PP。

### M3：训练路径方差

5 个 training seed 均完成 12 epoch。对 epoch-12 的每个 seed 取 3 个 sampling seed平均：

| training seed | PN velocity | PP velocity | PP IOI |
|---:|---:|---:|---:|
| 11 | 18.14 | 7.76 | 14.36 |
| 22 | 17.81 | 6.22 | 12.84 |
| 33 | 19.99 | 11.69 | 14.81 |
| 44 | 20.14 | 12.62 | 12.58 |
| 55 | 19.31 | 10.46 | 11.49 |

epoch-12 跨 training seed 的结果：

- PN velocity CV：**5.64%**；
- PP velocity CV：**26.23%**；
- PN IOI CV：**2.83%**；
- PP IOI CV：**9.55%**；
- PP duration CV：**16.82%**。

把 epoch 4/8/12 和 sampling seed 合并后，PP velocity CV 为 **35.45%**，PN duration CV 为 **10.50%**。因此当前最大问题是训练路径与 checkpoint 落点，而不是 evaluator 实现；sampling 是第二级来源。

### M4：window rollout 诊断

对 seed 11/22/33 的 epoch-12 checkpoint，使用 40 个 distinct works，比较 k=4 和 k=32。m=80 没有运行，因为当前 valid manifest 只有 68 个 eligible works。

| 协议 | PN HR CV | PP HR CV | PP velocity raw W1 CV |
|---|---:|---:|---:|
| m=40, k=4 | 1.00% | 1.90% | 14.43% |
| m=40, k=32 | 3.30% | 7.74% | 29.61% |

k 从 4 增加到 32 后，velocity 相关指标方差反而变大，且 rollout 预算会改变模型之间的相对结果。该结果支持把 rollout protocol 作为独立实验因素固定和报告。

## 结论

1. **评估实现不是主要方差来源**：M0 八种条件完全一致。
2. **sampling 有影响但不是主因**：固定 checkpoint 时 PP velocity CV 为 5.11%。
3. **checkpoint 选择有显著影响**：同一条训练轨迹中 PP IOI CV 达 108.92%，PP velocity 跨 checkpoint 从 7.93 到 13.81。
4. **training seed 是主要来源之一**：epoch-12 的 PP velocity CV 为 26.23%；加入 checkpoint 演化后为 35.45%。
5. **loss 不能直接作为 PN/PP 代理**：不同 checkpoint 和 training seed 的下游指标变化不呈单调关系。
6. **window rollout 需要单独固定**：k=4 与 k=32 的排名和方差不同，不能把低成本 rollout 直接当作 full-score 结果。

## 当前 baseline 建议

正式比较应至少使用：固定 checkpoint 规则、固定 full-score rollout、多个 training seed、每个 checkpoint 多 sampling seed，并报告均值和置信区间。M4 的 window rollout 只能作为诊断表；PN/PP Wasserstein 定义保持不变。

当前 M4 仍是诊断子集，不是完整五 seed 矩阵：只有 seed 11/22/33，且 m=80 因数据量不足未运行。因此不能把 M4 的三次结果当成最终泛化结论。
