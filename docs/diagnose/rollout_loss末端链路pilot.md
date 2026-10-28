# Rollout loss 末端链路 pilot

## 目的

当前主要断点是 `test TF HR -> test AR HR`。合理假设是 teacher forcing 训练时模型看到真实历史，而 AR 评估时使用自己的预测；误差会通过 decoder feedback、采样和窗口续接逐步放大。为验证这个假设，加入一个可反向传播的一步 model-feedback loss：

```text
L = L_TF + lambda * L_rollout(k=1)
```

第一版保留原有 TF loss，`k=1`，由第一次前向生成 detached feedback，再做第二次前向并对第二次 loss 反向传播。默认 `rollout_train_weight=0`，因此旧配置行为不变。

## 实现与验证

- 训练逻辑：`src/train/train_inr.py` 的 `NodeSFTTrainer.compute_loss`。
- 模型配置字段：`rollout_train_weight`、`rollout_train_k`、`rollout_train_feedback_strategy`。
- 实验入口：`script/run_rollout_loss_experiment.sh`。
- smoke：baseline、`lambda=0.10`、`lambda=0.25` 各 2 step，均完成；rollout 日志出现 `train_loss_rollout_k1`，没有 shape 或反传错误。
- 额外修复：短 development manifest 没有 valid rows 时，训练脚本关闭 Trainer 内置 eval，只保存 epoch checkpoint；正式评估仍走独立 protocol。

## Development 结果

设置为 seed 11、10 个 work、每 work 最多 3 个 performances、每 work 2 个 windows、4 epochs、每卡 batch 8、gradient accumulation 12。三条件使用完全相同的数据和优化设置：

| 条件 | `lambda` | epoch-4 TF loss | k=0 PN HR3 | k=1 PN HR3 | k=0 PP HR3 | k=1 PP HR3 |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 0 | 29.8835 | 1.7482 | 1.7526 | 5.4873 | 5.5072 |
| rollout10 | 0.10 | 29.8731 | 1.7464 | 1.7509 | 5.4799 | 5.4995 |
| rollout25 | 0.25 | 29.8728 | 1.7464 | 1.7509 | 5.4795 | 5.5001 |

这里 HR 越低越好。相对 baseline，rollout10 的 k=1 PN HR3 下降约 0.0017，PP HR3 下降约 0.0077；rollout25 没有继续改善。差异很小，且只有一个 training seed、10 个 work，不能据此声称末端 test AR 已被修复。

## 当前结论

这条训练路径是可行的，且没有破坏 TF 指标。在小规模 validation 上，它对 model-feedback 后的 HR 有轻微改善，方向与 exposure bias 假设一致，但证据还不足以证明能改善 `test TF HR -> test AR HR`。

本次 test MIDI sampling 没有完成：19-score inference 的单个 score 约需 5.5 分钟，首个 score 尚未完成就停止了这批不完整任务。因此本 pilot 没有产生可用于论文结论的 test AR HR，也没有计算新的 endpoint correlation。

## 下一步正式实验

1. 固定同一 development protocol，跑 baseline、`lambda=0.10`、`lambda=0.25` 至少 3 个 training seeds。
2. 先用独立 grouped TF/one-step evaluator 选择预注册的 lambda；不能用 test AR 选择权重。
3. 对选定 lambda 和 baseline 运行相同 12-score test 子集、相同 sampling seed、相同生成窗口和 GT subset，分别得到 test TF HR、test AR HR、generated length/failure rate。
4. 只有完成 paired test TF/AR 后，才重新计算 `test TF HR -> test AR HR` 和 `train loss -> test AR HR`；当前 pilot 不更新这两个相关性。
5. 如果 one-step validation 改善而 test AR 不改善，应继续拆分 sampling、window continuation 和 MIDI materialization 三个因素，不能直接把失败归因于 rollout loss。
