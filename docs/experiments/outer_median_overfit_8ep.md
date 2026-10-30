# 外层 median 数据的 8 epoch 过拟合对照

## 实验目的

检验 INSPIRE2 中观察到的早期过拟合是否是 DINR-absolute 特有，还是由外层模型、数据划分或验证集构造造成。三组实验使用同一份 median-scaled performance 标签、同一个 work-level train/valid/test protocol，以及 50% window overlap：

- DINR slot-sum，无 numerical coordinate；
- DINR slot-sum，加入 timing/velocity numerical coordinate；
- CINR logistic-normal，slot-MLP fusion。

三组最大训练轮数均为 8，global batch size 为 64，学习率为 $3\times10^{-4}$。每个完整 epoch 结束后在 valid 集评估一次；test split 保持原来的 17 个作品，本实验不进行 test inference。

## 数据协议

median+musical51 侧车使用新 tag `ASAP_MUSICAL51_MEDIAN`，保留原外层 musical51 score feature，并将 grouped-median sidecar 的 `grouped_ioi_raw` 写入外层 IOI 标签槽。原始 sidecar 没有被覆盖。

固定 split 使用 12 个完整 held-out works 作为 valid，包含 141 个 score windows、47 条 performances 和 507 个 valid perf-window instances。原始 token cache 中 train 有 175 个有效 works、837 条 train performances 和 10,280 个 train perf-window instances；外层 Trainer 每个 epoch 为 161 个 optimizer steps，与 10,280/64 的 batch 计数吻合。manifest 打印的 10,642 是基于 metadata 的预估数，不是最终实际加载的 dataset 长度。valid work 的所有 canonical windows 都从 train 排除，因此不存在 train/valid window 重叠。

## 结果

结果文件：[`loss_summary.json`](../../results/outer_median_overfit_8ep/loss_summary.json)；曲线：[`loss_curves.png`](../../results/outer_median_overfit_8ep/loss_curves.png)。train 曲线是根据 Trainer 每 20 step 的 loss 日志按 epoch 加权汇总的近似均值，valid 是完整 valid 集的 epoch-end loss。

| 设置 | train loss epoch 1 → 8 | valid loss epoch 1 → 8 | valid 最低值 |
|---|---:|---:|---:|
| DINR no-coordinate | 13.366 → 12.147 | 12.828 → 12.231 | 12.231 (ep 8) |
| DINR coordinate | 13.396 → 12.156 | 12.895 → 12.235 | 12.235 (ep 8) |
| CINR | 0.580 → -0.557 | 0.806 → -0.570 | -0.570 (ep 8) |

CINR 的 NLL 与 DINR 的分类/回归损失不在同一量纲，不能比较表中的绝对数值；只能在同一模型内部比较 train-valid 趋势。

## 判断

外层的两种 DINR 都表现为 train 和 valid 同步下降，epoch 8 仍未出现 valid 回升。DINR no-coordinate 的 valid 从 12.828 降到 12.231，coordinate 从 12.895 降到 12.235，说明在新的整曲 held-out valid 上没有观察到持续过拟合。

CINR 的 valid 在 epoch 2 和 epoch 5 有短暂波动，但之后继续下降，epoch 8 达到整个 8 epoch 的最低值；这更像分布似然训练中的随机波动，而不是 train 持续下降、valid 持续上升的过拟合。

因此，INSPIRE2 中的过拟合现象不能简单归因于 DINR 结构。更可能的因素是旧 valid 构造（曲尾 edge windows 的 duration 分布偏移）、较小或不稳定的验证样本，以及模型/学习率带来的短期波动。当前外层对照支持先采用 work-disjoint valid，再用完整 epoch-end valid 曲线判断过拟合。
