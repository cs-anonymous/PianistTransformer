# INR/EPR 五类方差分离实验设计

## 1. 目标与约束

当前最关键的问题是：相同配置复跑时，训练 loss 很稳定，但最终 PN/PP Wasserstein 指标方差很大。因此本阶段的目标不是修改 PN/PP 定义，也不是继续搜索模型结构，而是建立一个可复现、可诊断的 baseline，回答以下问题：

1. 方差来自训练轨迹本身吗？
2. 方差来自 checkpoint 选择吗？
3. 固定模型后，采样随机性有多大？
4. 不同 rollout 协议是否改变 PN/PP 结果或 checkpoint 排名？
5. 评估实现、数据顺序和聚合方式是否引入了额外方差？

正式报告仍使用当前 PN/PP Wasserstein 定义。window AR、MAE、teacher-forcing loss 等只作为诊断变量，不替换正式指标。

## 2. 已有数据给出的基线事实

已有 `BASELINE_REPEAT_LOSS_ANALYSIS.md` 的五次相同核心配置复跑显示：

| 变量 | 五次复跑 CV |
|---|---:|
| best eval loss | 0.59% |
| PP IOI | 4.80% |
| PP duration | 7.45% |
| PP velocity | **43.15%** |
| PN velocity | 9.04% |

五个 run 的 best checkpoint 都位于最后评估点，因此当前方差不能简单解释为“某些 run 选到了较差的早期 checkpoint”。同时，best eval loss 最低的 run 并不对应最好的 PP/PN velocity。

已有 checkpoint probe 还显示：

- validation PP velocity 与 test PP velocity 的 Spearman 相关仅为 **0.067**；
- validation PN velocity 与 test PN velocity 的 Spearman 相关仅为 **0.033**；
- validation PN IOI 相关约 **0.850**；
- validation PP duration 相关约 **0.783**。

这说明 loss 或 validation 指标对 velocity 下游指标的排序能力很弱。已有 rollout 代理分析中，40 windows 时 k=4 的 HR 排名相关约 **0.483**，k=32 时约 **0.800**；80 windows、k=32 又降至约 **0.667**。因此 rollout 和评估预算本身可能改变 checkpoint 排名。

## 3. 五类方差的操作性定义

### 3.1 训练路径方差

固定数据、超参数和评估流程，仅改变训练随机因素：初始化 seed、batch/data-loader 顺序、worker seed、CUDA/D​​DP 非确定性和并行归约顺序。它回答“同一配置是否会走到不同模型解”。

### 3.2 Checkpoint 选择方差

固定一条训练轨迹，比较固定 epoch、best eval loss、best PN/PP validation、rollout probe 等选择规则。它回答“模型本身的变化”和“我们选择哪个 checkpoint”各贡献多少方差。

### 3.3 采样方差

固定 checkpoint、固定评估数据和 rollout 协议，仅改变 sampling seed、temperature/top-p 和生成样本数。它回答“同一个模型一次评估得到的 PN/PP 是否已经不稳定”。

### 3.4 Rollout 方差

固定 checkpoint、sampling seed 和评估集，仅改变 teacher forcing、1-step AR、固定长度 window AR、continuation window AR、full-score AR 等推理路径。它回答“训练时看到的条件分布”和“最终长程生成路径”之间是否存在不一致。

### 3.5 评估实现与聚合方差

固定模型输出和随机种子，仅改变 evaluator 的 worker 数、文件枚举顺序、batch 划分、缓存读取、浮点聚合顺序、GT manifest 和 score-level 聚合实现。它回答“代码实现是否制造了看似模型方差的数值差异”。

## 4. 总体原则

所有实验必须保存原始 score-level 结果，而不是只保存最终均值。对每个样本、每个属性、每个 sampling seed 保存生成序列、GT 对齐结果、PN/PP Wasserstein 值和 evaluator 版本。每次运行同时记录 git commit、配置文件哈希、checkpoint 路径哈希、数据 manifest 哈希、CUDA/PyTorch 版本、GPU 型号和环境变量。

所有阶段使用同一份冻结的 test manifest。除非实验因素明确要求改变，否则固定 temperature、top-p、长度、batch size、数据切分和 PN/PP 计算代码。

每个实验报告均需同时给出：均值、标准差、CV、95% bootstrap CI、最大最小差、score-level 分布，以及 PN 与 PP 各属性的结果。不能只报告一个总分。

## 5. 实验阶段与矩阵

### Phase A：评估实现复现性

目的：先证明同一份模型输出经过评估代码会得到同一结果。

选一个冻结 checkpoint，导出一份不可变的 prediction manifest。对以下条件各运行 3 次：

| 条件 | 设置 |
|---|---|
| A1 | 单进程、单 worker、固定文件顺序 |
| A2 | 多 worker，worker 数 2/4/8 |
| A3 | 原始文件顺序 vs 排序后顺序 |
| A4 | 逐样本聚合 vs batch 聚合 |
| A5 | float32 聚合 vs float64 聚合 |
| A6 | 清空缓存后重算 vs 读取缓存 |

判定：同一 prediction manifest 的 PN/PP 差异必须小于 `1e-5`（或小于现有实现的机器精度范围）。若超过该阈值，先修复评估实现，再进行后续方差实验。

### Phase B：固定模型的采样方差

目的：估计“单个 checkpoint 本身”的随机生成方差。

固定 Phase A 已通过的 evaluator、full-score AR 协议和一个代表性 checkpoint。使用 20 个 sampling seeds，并比较样本数 `N={3,10,20}`。每个 seed 保存完整 score-level 结果。

输出：对每个属性分别计算 sampling CV；绘制样本数随 CV 下降曲线；报告 N=3、10、20 对 checkpoint 排名的 Spearman 相关。

判定：若 sampling CV 已接近或超过训练复跑 CV，当前单次 PN/PP 报告就不足以作为可靠比较，应将多 seed 平均和置信区间纳入 baseline。

### Phase C：固定训练轨迹的 checkpoint 方差

目的：把 checkpoint 演化与训练随机性分开。

从一条固定训练 run 保存 epoch `4, 8, 12, 16, 20`（或所有实际评估点）checkpoint。每个 checkpoint 用相同的 10 个 sampling seeds、相同 test manifest 和 rollout 协议评估。

同时记录：training loss、teacher-forcing validation loss、PN/PP validation、PN/PP test、生成长度和失败率。

分析：

- loss 与 PN/PP 的 Pearson/Spearman 相关；
- 固定 epoch 与 best-loss 选择的差异；
- checkpoint 间 PN/PP 曲线是否单调；
- 不同属性是否出现此消彼长。

判定：若 PN/PP 在训练后期没有稳定改善，而 loss 继续下降，则说明训练目标与最终评估目标存在明显错位；此时应把“固定 epoch + 多 seed 平均”作为临时 baseline，而不是使用单一 best-loss checkpoint。

### Phase D：训练路径方差

目的：直接估计同一配置的训练随机性。

至少运行 5 个显式 training seeds。每个 seed 保存 5 个固定 epoch checkpoint，并在初筛阶段使用 3 个 sampling seeds；对候选 checkpoint 再使用 10 个 sampling seeds。建议先使用单卡、固定 worker、确定性算子，随后做单卡/多卡对照。

推荐矩阵：

| 因素 | 水平 |
|---|---|
| training seed | 5 个预注册 seed |
| data-loader worker | 0、4 |
| device | 单卡（主 baseline）、多卡（对照） |
| checkpoint | epoch 4/8/12/16/20 |
| sampling seed | 初筛 3；最终 10 |

分解方法：以 score-level PN/PP 为响应变量，使用随机效应模型或分层 ANOVA：

`Y = 总均值 + training_seed + checkpoint + sampling_seed + residual`。

先在固定 checkpoint 上估计 training seed 方差，再在固定 training seed 上估计 checkpoint 方差，避免把两者混在一起。报告每一项的方差占比和 bootstrap CI。

### Phase E：Rollout 协议方差

目的：确定最终生成路径是否是主要不稳定来源，同时保持 PN/PP 的定义不变。

选择 3 个代表性 checkpoint：一个 loss 最低、一个 PN/PP 综合较好、一个 velocity 异常的 checkpoint。固定 10 个 sampling seeds，比较：

| 协议 | 说明 |
|---|---|
| R0 | teacher-forcing evaluation |
| R1 | 1-step AR |
| R2 | 固定 64-note window AR，每个 window 独立采样 |
| R3 | 固定 512-note window AR |
| R4 | continuation window AR，窗口之间传递状态 |
| R5 | full-score AR |

window AR 的结果只用于诊断 rollout 长度和误差累积；正式 PN/PP 仍按现有评估定义计算。对每个协议报告属性级均值、CV、与 full-score AR 的 rank correlation，以及不同 checkpoint 的排名一致性。

判定：如果不同协议对 checkpoint 的 Spearman 相关显著低于 0.8，说明“模型选择”和“推理路径”不可分离；此时论文必须固定一个预注册 rollout 协议，并报告协议敏感性。

## 6. 建议的最小可执行矩阵

资源有限时，优先完成以下最小版本：

- **M0（实现）**：1 个 checkpoint × 1 个 prediction manifest × 6 个 evaluator 条件 × 3 次重复；
- **M1（采样）**：1 个 checkpoint × 20 sampling seeds × 样本数 3/10/20；
- **M2（checkpoint）**：1 条训练轨迹 × 5 个 epoch checkpoint × 10 sampling seeds；
- **M3（训练）**：5 training seeds × 5 个 checkpoint × 初筛 3 sampling seeds；
- **M4（rollout）**：3 个 checkpoint × 6 个 rollout 协议 × 10 sampling seeds。

执行顺序必须是 M0 → M1 → M2 → M3 → M4。M0 未通过时，后续方差数字无法解释；M1 未完成时，训练 seed 方差会混入 sampling noise。

## 7. 数据记录与统计输出

每次运行建立一行 run-level manifest，并关联 score-level parquet/JSON 文件。最少字段包括：

- `train_seed`, `sampling_seed`, `checkpoint_epoch`, `checkpoint_hash`；
- `rollout_protocol`, `temperature`, `top_p`, `num_samples`；
- `evaluator_commit`, `gt_manifest_hash`, `prediction_manifest_hash`；
- `device`, `num_workers`, `torch_version`, `cuda_version`；
- 每个属性的 PN/PP Wasserstein、样本数、有效样本率和失败原因。

统计上优先使用 score-level bootstrap，而不是把同一 run 内的多个样本当成独立 training replicate。训练 seed 是最高层 replicate，sampling seed 是其内层 replicate。最终表格应同时给出：

1. 单次运行结果；
2. sampling 平均及 CI；
3. training-seed 平均及 CI；
4. 五类方差的估计占比。

## 8. 如何解释结果与后续决策

- 若 Phase A 不通过：优先修复 evaluator、manifest、排序或缓存问题。
- 若 Phase B 占主导：正式 baseline 采用多 sampling seed 平均，并报告 CI。
- 若 Phase C 占主导：固定 epoch 或预注册 checkpoint 规则，暂停使用单一 best-loss 选择。
- 若 Phase D 占主导：增加 training seeds，使用 seed-level CI，并检查初始化、数据顺序和非确定性算子。
- 若 Phase E 占主导：固定唯一 rollout 协议，单独报告协议敏感性；暂不据此修改 PN/PP 定义。
- 若 loss 与 PN/PP 在 Phase C 中持续低相关：把 loss 作为训练诊断量，不再把它当作下游质量的代理指标。

只有在五类方差均被测量、主导来源有证据支持、并且同一配置的置信区间足够窄时，才能声称 baseline 稳定。此后再讨论指标改造、window AR 作为正式评估、或训练目标重设计，才不会把评估噪声误判为模型改进。

## 9. 当前推荐的 baseline 版本

在实验完成前，建议使用以下临时 baseline 规则：

1. 固定 test manifest、evaluator commit 和 full-score AR 协议；
2. 不使用随机的 best-loss checkpoint 作为唯一结果，预注册固定 epoch 或固定选择规则；
3. 每个 checkpoint 使用至少 10 个 sampling seeds，报告均值、CV 和 95% CI；
4. 至少 5 个 training seeds，training seed 作为统计 replicate；
5. PN/PP 定义、Wasserstein 距离和属性聚合方式全部保持当前版本；
6. 所有 window AR、MAE、teacher-forcing loss 仅作为诊断表，不进入正式主结果。

该 baseline 的目的，是先回答“结果是否可重复、方差来自哪里、loss 是否能预测下游指标”。在这些问题没有被数据回答之前，不应把单次复跑的 PN/PP 改善解释为模型能力提升。

## 10. 三张 GPU 的执行安排

仓库新增了 [run_variance_matrix_3gpu.sh](../script/run_variance_matrix_3gpu.sh)。它把每个独立实验绑定到一张 GPU，采用“三个一批、完成后补位”的调度方式，并在 `OUT_ROOT/manifests/runs.jsonl` 中记录 GPU、seed、checkpoint、输出目录和 git commit。已有完整 `eval_pn_pp_metrics.json` 的任务会自动跳过，因此中断后可以直接重跑。

推荐按以下顺序执行：

```bash
# M0 不需要 GPU；先确认评估实现的重复性
PREDICTION_MANIFEST=/path/to/prediction_manifest.json \
SCORE_SOURCE_LIST=data/asap_test_score_sources.txt \
OUT_ROOT=results/variance_matrix_3gpu \
bash script/run_variance_matrix_3gpu.sh M0

# M1：固定 checkpoint，20 个 sampling seed，三卡并行
GPU_LIST=0,1,2 CHECKPOINT=/path/to/checkpoint-best \
CONFIG=configs/inr_epr/cinr__default_dlm_k1_bounded5.json \
OUT_ROOT=results/variance_matrix_3gpu \
bash script/run_variance_matrix_3gpu.sh M1

# M2：同一条训练轨迹的 5 个 checkpoint，三卡并行
GPU_LIST=0,1,2 CHECKPOINT=/path/to/training \
CHECKPOINTS=/path/to/training/epoch-4,/path/to/training/epoch-8,/path/to/training/epoch-12,/path/to/training/epoch-16,/path/to/training/epoch-20 \
bash script/run_variance_matrix_3gpu.sh M2

# M3：5 个 training seed，第一轮 3 张卡，第二轮补 2 张卡
GPU_LIST=0,1,2 CONFIG=configs/inr_epr/cinr__default_dlm_k1_bounded5.json \
SEEDS=11,22,33,44,55 EPOCHS=12 \
OUT_ROOT=results/variance_matrix_3gpu \
bash script/run_variance_matrix_3gpu.sh M3
```

因此 M1 约为 `ceil(20/3)=7` 个 GPU 批次，M2 为 2 个批次，M3 为 2 个训练批次。M4 应在 M1–M3 确定主导方差来源后再运行；脚本中的 M4 会用六个固定 seed 产生可比的 full-score 结果并记录 rollout 标签。若仓库中已有窗口/continuation 专用入口，应将其作为 M4 的协议命令分别运行，不能把脚本中的标签误当成已经改变了 rollout 实现。

所有任务的 CPU worker 数由 `NUM_WORKERS` 控制。三卡同时生成时建议先用 `NUM_WORKERS=2`，避免三份评估进程争抢 CPU；单任务调试可提高到 4 或 8。正式报告只读取 `eval_pn_pp_metrics.json` 和 `runs.jsonl`，不读取日志中的临时均值。
