# PN/PP 训练损失与最终指标对齐方案

## 结论

当前最主要的问题不是 PN/PP 定义需要立刻修改，而是训练时优化的对象与最终评估对象不同：

- 训练：单个 GT、teacher-forcing、逐 note 条件似然；
- 评估：模型 prefix 下的长程 rollout，多次采样后形成 distribution，再计算 PN/PP Wasserstein。

已有实验直接支持这个判断：

- M0 中改变 evaluator worker 数和文件顺序，PN/PP 最大差异为 0；
- M1 固定 checkpoint 的 sampling 只造成 PP velocity CV **5.11%**；
- M3 epoch-12 的 training-seed PP velocity CV **26.23%**，但最终 train loss CV 只有 **0.68%**；
- M2 同一训练轨迹中，PP IOI checkpoint CV 达 **108.92%**，且指标随 checkpoint 非单调变化。

所以第一目标是让训练目标更能预测 rollout PN/PP，同时保留正式 PN/PP 作为最终指标。

## 1. CRPS 和 CDF-W1 不是一回事

一维 CDF-W1 比较两个完整分布：

$$
W_1(P,Q)=\int |F_P(t)-F_Q(t)|dt.
$$

如果模型 PMF 是 `p`，人类经验 PMF 是 `q`，固定 bins 后就是累计概率差的加权绝对值：

$$
L_{CDF-W1}=\sum_b \Delta z_b
\left|CDF(p)_b-CDF(q)_b\right|.
$$

CRPS 比较的是一个预测分布 `P` 和一个观测值 `y`：

$$
CRPS(P,y)=\int(F_P(t)-1[y\le t])^2dt.
$$

它使用平方 CDF 差，目标是概率预测的校准；它不是 Wasserstein，也不能被报告为 PN/PP-W1。其 sample form 中的

$$
E|X-y|-\frac12E|X-X'|
$$

第二项会惩罚预测分布无意义地塌缩到单点。

本项目当前不需要同时引入 CRPS。只要存在足够多同 score 的 GT，直接使用 train-only cached CDF-W1 更贴近 PN 的参考分布；只有 score 确实只有一个 GT 时，CDF-W1 target 不可识别，才把 NLL 作为主损失，CRPS 作为后续单 GT 概率校准实验，而不是第一版必需项。

## 2. 单 GT 为什么不能直接做 PN-W1

把一个 GT `y` 当成分布 `delta_y` 会得到：

$$
W_1(P,\delta_y)=E_{X\sim P}|X-y|,
$$

这只是 MAE 型的绝对误差期望，不是人类 performance distribution loss。它不能学习同一 score 的人类差异、多峰或尾部。

当前仓库还有一个实现事实：`src/train/train_inr.py` 的 `_pn_pp_metric_w1_loss` 对少于 2 条 row 的 group 直接跳过。因此默认 `multi_perf_group_size=1` 时，PN/PP W1 项实际上没有梯度；不能通过单纯增大 W1 lambda 修复问题。

如果某个 score 只有一个 train GT，正确处理是：

- 保留 DLM NLL，学习单个观测来自的条件分布；
- 不把 `W1(P, delta_y)` 称为 distribution loss；
- 用跨 score 的共享模型先验表达未观测 variability；
- 只有在后续有多个同 score GT 或正式 rollout risk 时，才监督分布形状。

## 3. 是否必须把同曲目的不同 GT 放在同一 batch

要分成两个情况。

### 3.1 直接用 batch 内样本估计 human target

如果 W1 的 target 是“当前 batch 中同一 score 的 GT 样本”，那么必须在同一计算单元里放入至少两条同 score GT。不同 score 的 GT 不能混合，否则得到的不是 PN 的条件分布。这正是当前 `multi_perf_group_size` 路径的要求。

这种做法有三个问题：batch 划分会改变 target、有效 group 数受限、不同 score 的 performance 数量会改变采样权重。因此它适合小规模验证，不适合作为主方案。

### 3.2 离线构建 train-only target cache

推荐先采用这种方式。对每个 `(score_source, note_index, feature)`，使用 train split 中该 score 的所有有效 GT 构造：

$$
Q_{g,i,a}=\frac1{M_g}\sum_m\delta(y_{g,m,i,a}).
$$

训练时每条单 GT row 都查同一个 `Q`，所以不要求同曲目不同 GT 出现在同一 batch，也不会因为 batch 恰好如何组成而改变 human target。

这只解决参考侧。训练时可将每个 aligned note 的条件预测 PMF 与 `Q_{g,i,a}` 比较，但这仍是 proxy：teacher-forcing 的条件分布与正式 rollout 后的 note 边缘分布并不相同，不能宣称二者已经严格对齐。PP 还存在预测侧 pooling：要严格模拟正式 PP，仍然需要在同一个 score 的多个 notes 上 pooling prediction，或者用固定 window/score accumulator；不需要同曲目的多个 GT 同 batch，但需要明确 note pooling 的近似和权重。

因此第一轮应先做 **PN velocity cached CDF-W1**。PP cached CDF-W1 只有在 score/window pooling 的实现和正式 evaluator 对齐后再加入。

## 4. target cache 是否会慢、是否有顺序偏差

不会因为“查字典”本身成为主要瓶颈，但不建议每个样本用字符串三元组做 Python 深层字典查找。建议：

1. 构建阶段为每个 score 分配稳定整数 `score_id`，note 使用原始对齐 index；
2. 每个 score 保存一个连续 tensor/压缩数组，形状类似 `[num_notes, num_features, num_bins]`；
3. manifest 保存 `score_id -> cache file, offset, length`；
4. Dataset worker 首次访问某 score 时加载整个 score 的 cache，使用现有 prepared-work LRU；
5. 后续 note 查找只做整数索引，不做字符串解析；
6. cache 只保存 `M_g >= 2` 的 score；单 GT score 直接走 NLL，不保存无意义的 delta target。

如果 notes 很多，PMF 不必保存为 float32：固定 bins 下可保存计数或 float16 PMF，并保存有效样本数。更省空间的方案是保存每个 note 的 sorted human values；但第一版为了 GPU 向量化，优先使用连续 PMF/CDF tensor。

正确实现后，cache 读取量约为“每个 worker 每个 score 一次”，远小于每个 note 重复读取原始 GT 文件。应在 smoke test 中记录 cache hit rate、加载秒数和 dataloader throughput，不能只凭理论判断。

### 顺序与数据偏好

聚合 PMF/CDF 是交换的，GT 文件顺序不应影响结果。为避免隐性偏差：

- performance source 先按稳定 `performance_id/source` 排序；
- note 使用 score 的 canonical aligned index，不能使用窗口内局部 index；
- 同一 score 的多个窗口共享同一 target cache；
- cache 只能由 train split 构建，不能读 validation/test GT；
- loss 按 score 或 note 做明确权重，不能让 GT 更多的 score 因为重复 row 自动获得更大权重；
- 对每个 score 保存 `M_g`、有效样本率和 manifest hash，报告 cache coverage。

如果训练 dataset 仍然按“每个 performance 一条 row”采样，那么一个有 20 个 GT 的 score 会比只有 2 个 GT 的 score 出现更多次。使用 cached target 并不会自动消除这个 sampling bias；应在 sampler 或 loss 中采用 score-balanced 权重，例如每条该 score 的 row 乘 `1/M_g`，再按 score 内 note 数归一化。

## 5. 第一版针对性 loss

第一阶段不要同时加入 CRPS、Sinkhorn、Energy Score 和 REINFORCE。使用：

$$
L=L_{DLM-NLL}+\lambda_{PN}L_{PN\text{-}CDF-W1}^{velocity}.
$$

其中：

- 所有 score 都保留 NLL；
- `M_g >= 2` 的 score 使用 cached PN velocity CDF-W1；
- `M_g < 2` 的 score 的 PN-CDF 项 mask 掉；
- 先只做 velocity，避免 IOI/duration/velocity 的属性冲突掩盖结果；
- hard bucket assignment 先固定 bins，后续再单独比较 soft assignment，避免同时改变目标和实现。

然后在同一个 loss 上比较：

1. 100% teacher-forcing；
2. 25% model prefix；
3. 50% model prefix。

prefix 采样 stop-gradient，当前 note 仍用 NLL/CDF target 监督。仓库已有 `tf_pred`、`k1_twopass` 和 DAgger cache，可复用，不需要先引入全序列 REINFORCE。

推荐实验矩阵：NLL baseline、NLL+cached PN-CDF-W1、以及这两者各自的 25%/50% model-prefix；至少 5 training seeds，固定 epoch 4/8/12，多 sampling seeds。记录 NLL、PN-CDF-W1、teacher-forcing validation、rollout PN/PP 和 cache coverage。

这些是后续候选实验，而不是当前立即执行的大矩阵。应先完成第 7 节的充分训练对照，再用相同评估协议、多 training seeds 判断 proxy 与 rollout 的关系。历史 CV 只有在采样协议、聚合方式和标准差定义一致时才能比较。

## 最终决定

当前选择是：

- 正式 PN/PP 定义不变；
- CRPS 暂不加入第一版，它与 CDF-W1 不同；
- 不要求同曲目不同 GT 同 batch，改用 train-only per-score target cache；
- 先实现 PN velocity cached CDF-W1，PP 延后到 pooling 对齐后；
- 单 GT score 不伪造 PN distribution target，继续使用 NLL；
- 用 DAgger/mixed-prefix 单独处理 rollout gap。

这条路线只引入一个新的主变量：**是否使用 cached PN velocity CDF-W1**，因此能直接回答 loss 是否真的缩小了与最终 PN/PP 的 gap。

## 6. Pilot 实施结果（epoch-4，单 training seed）

已按上述方案完成一个可复现 pilot。由于 `cinr__default_dlm_k1_bounded5.json` 在当前工作区缺少 fixed-window summary 中的 work，pilot 使用 `configs/inr_epr/baseline0807.json`，三组都使用完全相同的训练数据、模型和 seed=11：

| 条件 | cached CDF-W1 | prefix | train loss | train runtime(s) |
|---|---:|---:|---:|---:|
| baseline | 0 | 0% | 19.1698 | 548.8 |
| cached | 0.10 | 0% | 19.3488 | 533.4 |
| cached_p25 | 0.10 | 25% | 19.4218 | 669.8 |

训练日志中 cached 项在后期约为 `1.20--1.32`，这是乘入 $\lambda=0.10$ 之前的统计量；结合此前 smoke 的梯度检查，确认该 loss 路径已接入。训练日志估计 188 个 score、6678 个 examples。另一次使用普通 manifest、未套用 fixed-window split 的覆盖审计得到 6683 个 examples，其中 146/188（77.7%）score 有至少两条 train performance，42 个 score 为单 GT。这是该审计口径下的覆盖率，不能当作实际 fixed-window 训练 manifest 已被逐项核验的结果。

epoch-4 使用正式 `evaluate_inr_saved_midis.py`，测试集为 19 个 ASAP score。先做单次 sampling 得到：

| 条件 | PN velocity Wass | PP velocity Wass | PN duration Wass | PN IOI Wass |
|---|---:|---:|---:|---:|
| baseline | 15.5074 | 7.6992 | 134.0941 | 48.7931 |
| cached | 15.6619 | 7.8702 | 132.5834 | 43.9649 |
| cached_p25 | 15.8183 | 7.9830 | 116.6929 | 43.1411 |

这组单次结果不能支持“cached loss 已降低 PN velocity gap”：cached 相对 baseline 的 PN velocity 变差约 1.0%，cached_p25 变差约 2.0%；同时观测到 duration/IOI 数值较低，但单 seed 结果不足以确认这种属性差异能稳定复现。由于此前固定 checkpoint 的 sampling CV 已达到数个百分点，1--2% 的差异不能与训练效应区分，必须使用多 sampling seeds 后再判断。

随后对同一批 epoch-4 checkpoint 使用 `num_samples=3`（19 个 score，每个 score 3 个独立 sampling task）并重新评估。三次预测在 evaluator 中作为 prediction distribution 与相同 GT 集合比较：

| 条件 | PN velocity Wass | PP velocity Wass | PN duration Wass | PN IOI Wass | PP IOI Wass |
|---|---:|---:|---:|---:|---:|
| baseline | **14.3783** | **7.8115** | 128.9659 | 47.0372 | 29.6739 |
| cached | 14.5376 (+1.1%) | 7.9864 (+2.2%) | 127.2942 | 41.9431 | 23.2002 |
| cached_p25 | 14.7200 (+2.4%) | 8.0982 (+3.7%) | **111.3361** | **40.9321** | **22.0124** |

按 score 做 paired comparison，cached 相对 baseline 的 PN velocity 在 19 个 score 中只有 7 个下降，均值差为 `+0.1593`；cached_p25 只有 5 个下降，均值差为 `+0.3417`。这组 pooled 评估同样没有显示 velocity 改善。但 `num_samples=3` 是将三条预测汇成一个分布后计算一次指标，并不是三次独立指标复跑；不能排除采样波动，更不能估计 training-seed 方差。它与单 sample 指标的样本量不同，也不宜直接比较绝对数值。

这个 pilot 已经给出两个工程结论：

1. target cache 的加载和 loss 计算没有造成训练失败；cached 的 4 epoch runtime 与 baseline 同量级，只有 p25 因 DAgger cache refresh 增加约 121 s。
2. 当前实现仍然是 teacher-forcing 的 per-note PN proxy，且存在单 GT score 不参与辅助项的覆盖限制；它没有直接监督 rollout prefix，也没有 PP pooling。因此即使单次 PN velocity 不下降，也不能把结果解释为 Wasserstein 目标理论错误。

目前不能据此优先调 $\lambda$：4 epoch 尚未验证训练充分性，应先固定 $\lambda=0.10$，完成与 baseline 匹配的长训练。即使充分训练后辅助项没有改善 rollout，也不能唯一归因于 prefix；条件分布与边缘分布、采样截断以及 MIDI 后处理仍是未分离因素。

### lambda pilot 的恢复训练混杂：撤回强弱排序

`results/pn_cached_velocity_lambda_pilot/` 中 $\lambda=0.30$、$0.50$ 的训练曾中断，随后从 step-210 checkpoint 恢复。核查发现 `src/train/train_inr.py` 的 `_save_checkpoint` 调用 `_drop_optimizer_state`，删除了 `optimizer.pt`。这些恢复不构成原优化轨迹的连续延长：原日志 step260 的学习率约为 $2.53\times10^{-6}$，恢复日志 step220/240/260 则约为 $5.4\times10^{-5}$、$1.74\times10^{-4}$、$2.94\times10^{-4}$，重新经历了学习率爬升。

| 条件 | PN velocity Wass（1 sample） | PP velocity Wass（1 sample） | 解释限制 |
|---|---:|---:|---|
| baseline | 15.5074 | 7.6992 | 完整 4-epoch 原始运行 |
| cached $\lambda=0.30$ | 15.2958 | 6.2316 | 中断后恢复，优化器与学习率轨迹不匹配 |
| cached $\lambda=0.50$ | 19.3011 | 13.5511 | 同上 |

保留这些产物供审计，但**撤回“0.50 过强、0.30 更合适”的判断**。表中差异不能归因于 $\lambda$。恢复后的累计 `train_loss` 也受到已恢复 step 计数的分母影响，不与完整运行的累计 loss 比较。今后需要从头运行匹配的长训练，或完整保存并恢复模型、optimizer、scheduler 和 RNG 状态。

复现实验目录：

- 训练：`results/pn_cached_velocity_pilot_baseline2/`
- 单次 rollout 与正式评估：`*/seed_11/inference_epoch4/`
- 三次 sampling rollout 与正式评估：`*/seed_11/inference_epoch4_ns3/`
- 运行入口：`script/run_pn_cached_velocity_experiment.sh`
- rollout/evaluator 并行入口：`script/run_pn_cached_velocity_eval_3gpu.sh`

## 7. 当前优先级：先排除训练不足，再判断 loss–metric gap

此前 cached 实验只训练了 4 epoch，**没有做过“持续训练直到 loss 不再下降”的验证**。优先做 $\lambda$ 小范围搜索过早。用户指出的训练不足应先排除；严格地说，loss 仍下降支持“优化尚未完成”，尚不能单独诊断模型容量不足意义上的欠拟合。

### 7.1 历史 M3 能说明什么

历史目录 `results/variance_matrix_3gpu/M3_training/` 有 5 个 training seeds、12-epoch baseline。它与当前 pilot 都基于 `baseline0807`，但旧 M3 的 gradient accumulation 为 1、有效 batch 为 32；当前为 3、有效 batch 为 96。旧 M3 约 203 个 optimizer steps/epoch，当前约 68。它不能作为新 cached 模型的匹配对照。

下表 loss 是对应 epoch alias 保存时**最近一次已记录的验证 NLL**；epoch-12 的最近验证实际在约 11.325 epoch，并非终点重新验证。指标来自 `M3_checkpoint_eval/`：先对每个 training seed 的 3 次独立单 sample 指标取均值，再对 5 个 training seeds 汇总。

| checkpoint alias | 最近验证 NLL，5 seeds 均值 | PN velocity Wass | PP velocity Wass | PP velocity training-seed CV |
|---:|---:|---:|---:|---:|
| 4 | 16.8495 | 18.7964 | 10.9964 | 40.96% |
| 8 | 15.9490 | 19.4727 | 11.8491 | 40.71% |
| 12 | 15.7153 | 19.0779 | 9.7493 | 27.59% |

这里 CV 使用 5 个 seed 均值的样本标准差（`ddof=1`）除以均值；开头历史报告的 26.23% 来自此前汇总口径，不能不经核对就与此表直接比较。这些独立单 sample 指标的均值也不能当作第 6 节 pooled 3-sample 指标。

历史数据支持：更长训练仍能降低 NLL，但 PN velocity 不随之单调降低。它不证明当前 4-epoch cached 已充分训练，也不证明 epoch 12 已收敛。

### 7.2 新的匹配长训练

目录：`results/pn_cached_velocity_longpilot_seed11/`。从头训练两个条件：NLL baseline 与 NLL + cached velocity CDF-W1（$\lambda=0.10$）。固定 seed=11、batch32、gradient accumulation=3、有效 batch96、12-epoch cosine schedule、相同训练数据与其余配置。不使用发生 optimizer-state 丢失的恢复训练，也不增加 prefix 变量。

cached 已完成 816 optimizer steps（12 epoch）。其日志 `logs/cached_seed_11.log` 中同一次长训练的验证记录为：

| 实际 epoch | 验证 NLL（不含 cached 辅助项） | velocity NLL |
|---:|---:|---:|
| 4.1182 | 18.6615 | 4.4278 |
| 6.1773 | 17.0338 | 4.0026 |
| 8.2365 | 16.1548 | 3.8805 |
| 10.2956 | 15.8810 | 3.8469 |
| 11.3251 | 15.8100 | 3.8432 |

从 4.118 到 11.325 epoch，验证 NLL 下降约 15.28%，velocity NLL 下降约 13.20%。这是训练超过 4 epoch 仍有实质收益的直接证据。baseline 匹配运行正在 GPU1 执行；在其完成且使用同一 rollout 协议评估前，不判断辅助 loss 优劣。单 training seed 也不能估计训练路径 CV。

注意：新长训练从开始就采用 12-epoch cosine schedule，因此它的第 4 epoch 与旧的完整 4-epoch cosine 运行学习率路径不同。只在这次长训练内部比较 checkpoint，或在两个匹配长训练之间比较条件。epoch-4/8 alias 对应实际约 4.118/8.236，报告时同时保留实际 step/epoch。

评估固定 epoch-4/8/12 的正式 PN/PP，保持 score 列表、采样参数、采样数一致。先用 `num_samples=3, seed=11` 的 pooled 协议做诊断；如需估计采样方差，要多次独立重复整个 pooled 评估。不能将三条 pooled samples 当作三次指标重复。测试结果用于本次诊断，后续 $\lambda$ 选择应在 validation 上完成，不以测试最优 checkpoint 作为最终结果。

### 7.3 怎样判断是否还需要更长训练

**12 epoch 是观察点，不是收敛判据。** 当前到 11.325 epoch 的验证 NLL 仍下降；cosine 后期学习率已很低，曲线减缓可能由学习率造成。因此现在只能说比 4 epoch 充分，不能说 loss 已不再下降。

下一步依次执行：

1. 完成当前匹配的 12-epoch 两组训练及固定 checkpoint 的 PN/PP 对照，暂缓 $\lambda$ 搜索。
2. 将训练 NLL、辅助 CDF-W1、各属性验证 NLL 分开看；不能用不同目标的总 loss 直接排序。训练 minibatch loss 有波动，判断趋势应使用固定训练 probe（eval 模式）与固定 validation 的平均 loss。
3. 如果到末期仍持续下降，追加更长时域实验。两组使用相同计划、从头运行，或先修复完整 optimizer/scheduler/RNG 保存再严格续训。不能仅从模型权重重新 warmup 后称为连续训练。
4. 为排除“学习率降到零造成假平台”，更长实验应预先设定保留非零学习率的平台诊断段；例如 warmup 后保持固定学习率，连续 5 次完整 epoch 的验证中，最佳验证 NLL 改善不足 0.1% 作为操作性平台标准，并用固定 train probe 区分训练仍在改善与验证停滞。该阈值只是预先规定的停止规则，不是数学收敛保证。
5. 若 train probe 继续下降、validation 停滞或恶化，检查泛化，不能再把问题一概归为训练不足；若两者均在有效学习率下稳定，再判断 cached 项是否降低以及 rollout PN/PP 是否跟随。只有完成这些检查后才决定调 $\lambda$ 或处理 proxy 的结构性偏差。

当前建议是：**先保证优化充分，再诊断 loss–metric 对齐；不把低 epoch pilot 的负结果解释成新 loss 无效。**

## 8. 已实现的充分训练实验协议

为了避免在欠拟合阶段比较 PN/PP，新增 `script/run_pn_convergence_experiment.sh`。默认运行 baseline 与 cached $\lambda=0.10$ 两个条件、training seeds `11,22,33`，每个条件使用相同 seed 配对。每个训练进程独占一张 GPU；`script/launch_pn_convergence_when_idle.sh` 会等待当前长训练和 inference 释放三张 GPU 后再启动矩阵，避免抢占正在运行的实验。

这不是把 epoch 数随意调大，而是预先固定一个可审计的优化协议：

- 最多 60 epoch，前 24 epoch 使用学习率 $3\times10^{-4}$，之后保持非零的 $3\times10^{-5}$，避免 cosine 学习率降到零后把“没有更新”误认为平台；
- 每个 epoch 在相同的 score-round-robin probe 上记录 teacher-forcing NLL、velocity NLL、cached CDF-W1 和 mixed objective；
- 同一 probe 再使用固定随机掩码测量训练态 loss，用来区分真实优化变化与偶然的 dropout 波动；
- 在最后 3 个 epoch 扫描完整的实际训练 dataset，不能只用 512 条 probe 宣称收敛；
- 操作性平台规则为至少 36 epoch 后，连续 8 次 probe 记录的相对跨度不超过 0.2%，且最后 3 次完整训练集扫描也满足同一条件；
- 即使平台规则通过，也只写为 `optimization_plateau`，不声称模型达到容量意义上的全局最优；60 epoch 仍未通过则写为 `inconclusive_not_at_plateau`，禁止用其 PN/PP 结果作 loss 优劣结论。

每个 run 另外保存 `convergence_probe_manifest.json`，其中包含实际 sidecar-filtered training examples、score 数、multi-GT score 数、probe hash 和 dataset hash。这样不同 seed 的 probe 和训练数据能被逐项核对，而不是依赖 manifest 的 estimated examples。每个 epoch 的诊断写入 `convergence_history.jsonl`，最终判定写入 `convergence_result.json`。

为避免恢复训练改变优化轨迹，协议保留 `optimizer.pt`、`scheduler.pt`、`rng_state.pth` 和 `trainer_state.json`；严格续训时若缺少任一文件会直接失败。旧的删除 optimizer state 的 checkpoint 不能用于该协议。PN/PP 仍只在充分训练判定之后，用相同 checkpoint、相同 sampling seeds 和相同 pooled/independent 评估协议计算。

正式评估入口为 `script/run_pn_convergence_eval_if_ready.sh`。它首先要求所有 seed 和两个条件的 `convergence_result.json` 都是 `optimization_plateau`；任何一个 run 为 `inconclusive_not_at_plateau`、缺文件或训练失败，脚本都会阻止 PN/PP 评估。这条门禁保证实验不会用一个仍在下降的模型去支持 loss–metric 结论。默认只评估 epoch 36/48/60；这些 checkpoint 的结果仍需在 paired training seeds 和多次 sampling 重复下汇总，不能由单个 checkpoint 的最优值挑选结论。

### 当前运行状态

第一次启动在第 1 epoch 失败，未产生可用的收敛结果：两个 run 的 `metric_for_best_model` 仍继承了不存在的 rollout 指标，另一个 run 与 inference worker 发生显存竞争。该批次被判为实验失败，不能解释为模型未收敛。已修正为 `eval_loss`，并将空闲判断改为同时检查 GPU 利用率和显存占用后重新启动。当前三张 GPU 正在执行 baseline 的三个 training seeds；只有这批完整运行结束后才会启动 cached 批次。目前没有任何 run 通过 `optimization_plateau`，因此尚未开始新的正式 PN/PP 结论评估。

最新检查时 baseline 三个 seed 约在第 51--55 epoch。相对第 1 epoch，固定 probe 的 NLL 已从约 19.5--19.95 降到 14.88--15.01，velocity NLL 从约 4.47--4.53 降到 3.76--3.77，验证 NLL 约为 15.03--15.14。最近 8 次 probe 的相对跨度仍高于 0.2%，所以这些数值证明长训练仍有收益，但不构成收敛判定。cached 条件尚未开始；当前没有新的 60-epoch PN/PP 结果。

### 60 epoch 矩阵最终结果

截至 2026-10-17，baseline/cached 两个条件的 3 个 training seeds 均完成 60 epoch、4080 optimizer steps。第一次启动失败的 3 个 run 已保留在 manifest 和日志中；第二次启动的 6 个 run 均以 `done` 结束，且 checkpoint 保留了 optimizer、scheduler 和 RNG 状态。

但 6 个 run **没有一个通过 `optimization_plateau`**。原因是最后 8 个 probe epoch 的相对跨度仍超过预先规定的 0.2%，并不是训练异常：

| 条件 | seed | 最佳 validation NLL | epoch-60 validation NLL | epoch-60 probe NLL | epoch-60 probe velocity NLL | 判定 |
|---|---:|---:|---:|---:|---:|---|
| baseline | 11 | 15.0374 | 15.0446 | 14.9232 | 3.7636 | inconclusive |
| baseline | 22 | 15.1158 | 15.1158 | 14.9801 | 3.7602 | inconclusive |
| baseline | 33 | 14.9970 | 14.9970 | 14.8627 | 3.7570 | inconclusive |
| cached $\\lambda=0.10$ | 11 | 15.0263 | 15.0263 | 14.8740 | 3.7506 | inconclusive |
| cached $\\lambda=0.10$ | 22 | 15.1261 | 15.1261 | 14.9804 | 3.7743 | inconclusive |
| cached $\\lambda=0.10$ | 33 | 15.0066 | 15.0266 | 14.8879 | 3.7627 | inconclusive |

3 个 seed 的均值为：baseline 的最佳 validation NLL `15.0501`，cached `15.0530`；epoch-60 validation NLL 分别为 `15.0524` 和 `15.0597`。cached 的 probe NLL 均值为 `14.9141`，baseline 为 `14.9220`，但 probe velocity NLL 均值 cached 为 `3.7625`、baseline 为 `3.7603`。这些差异都小于约 0.1%，且方向在不同 seed 间不完全一致，不能据此声称 cached 更好。

平台门禁的具体失败证据也很明确：baseline 三个 seed 的最后 8 epoch probe NLL 相对跨度为 `0.45%--0.65%`，probe velocity 为 `0.21%--0.43%`；cached 分别为 `0.37%--0.55%` 和 `0.16%--0.24%`，cached 的 CDF-W1 probe 跨度还达到 `0.65%--1.67%`。最后 3 个完整训练集扫描中，baseline velocity 相对跨度为 `0.30%--0.38%`，cached 为 `0.03%--0.86%`。因此当前协议只能给出“60 epoch 后仍有可测优化变化”，不能给出“已充分收敛后的 PN/PP 对照”。

由于 6 个 `convergence_result.json` 全部为 `inconclusive_not_at_plateau`，`script/run_pn_convergence_eval_if_ready.sh` 正确阻止了正式 PN/PP 评估；`results/pn_convergence_constant60/` 中没有新的 `eval_pn_pp_metrics.json`。因此本轮回答是：训练实验完成，收敛门禁未通过，loss--metric 的正式比较尚未完成。下一步需要先决定是延长相同非零学习率协议，还是把 0.2% 作为过严的诊断阈值进行预注册调整；在该决定前，不应使用 epoch-60 PN/PP 结果作论文结论。
