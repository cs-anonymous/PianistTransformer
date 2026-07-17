# Script Directory Guide

`script/` 现在只保留真正的 shell pipeline 和少量实验 launcher。Python 工具脚本已经整理进 `src/train` 和 `src/evaluate`。

## 当前在用的脚本

### 1. EPR 主流水线

主入口：`script/run_inr_epr_pipeline.sh`

最小启动方式：

```bash
CUDA_VISIBLE_DEVICES=0,1 \
CONFIG=configs/inr0624_epr_mln3_cine_mslog.json \
bash script/run_inr_epr_pipeline.sh
```

常用可选环境变量：

- `RUN_DIR_OVERRIDE`: 指定结果目录
- `PIPELINE_STAGE_START=train|adapt|infer`: 从某个阶段继续
- `RESUME_CHECKPOINT_OVERRIDE=/path/to/checkpoint-*`

### 2. 当前这次双双卡任务

当前实际使用的是 `tmux + script/run_inr_epr_pipeline.sh`，不是 `setsid`。

`cine`：

```bash
tmux new-session -d -s inr0624_cine_d4w1 \
  "cd /home/sy/EPR/PianistTransformer && \
   CUDA_VISIBLE_DEVICES='0,1' \
   CONFIG='results/inr0624_musical51_mln3_d4w1_dual/configs/inr0624_epr_mln3_cine_musical51_s50_d4w1_splitzero_seed42.json' \
   RUN_DIR_OVERRIDE='results/inr0624_musical51_mln3_d4w1_dual/inr0624_epr_mln3_cine_musical51_s50_d4w1_splitzero_seed42_tmux' \
   bash script/run_inr_epr_pipeline.sh 2>&1 | tee 'results/inr0624_musical51_mln3_d4w1_dual/launcher_logs/cine_tmux.log'"
```

`sine`：

```bash
tmux new-session -d -s inr0624_sine_d4w1 \
  "cd /home/sy/EPR/PianistTransformer && \
   CUDA_VISIBLE_DEVICES='2,3' \
   CONFIG='results/inr0624_musical51_mln3_d4w1_dual/configs/inr0624_epr_mln3_sine_musical51_s50_d4w1_splitzero_seed42.json' \
   RUN_DIR_OVERRIDE='results/inr0624_musical51_mln3_d4w1_dual/inr0624_epr_mln3_sine_musical51_s50_d4w1_splitzero_seed42_tmux' \
   bash script/run_inr_epr_pipeline.sh 2>&1 | tee 'results/inr0624_musical51_mln3_d4w1_dual/launcher_logs/sine_tmux.log'"
```

查看：

```bash
tmux ls
tmux attach -t inr0624_cine_d4w1
tmux attach -t inr0624_sine_d4w1
```

### 3. 其他仍保留的 shell 脚本

- `script/build_pianocore_inr_sidecars.sh`: 当前标准 INR 数据处理链
- `script/run_inr_removed_task_pipeline.sh`: removed_task 流水线
- `script/run_head_capacity_pipeline.sh`: head capacity 对比实验
- `script/run_pt_pipeline.sh`: Pianist Transformer 旧主线流水线
- `script/launch_inr0624_epr_logscale_4gpu.sh`: 特定 INR0624 EPR 批量启动器
- `script/launch_inr0624_removed_task_4gpu.sh`: 特定 INR0624 removed_task 批量启动器

## Python 工具的新位置

- `src/data_process/prebuild_inr_work_pt.py`
- `src/evaluate/plot_target_distribution_diagnostic.py`
- `src/evaluate/plot_zero_nz_ioi_dev_scales.py`
- `src/evaluate/plot_duration_musical_diagnostics.py`
- `src/evaluate/diagnose_timing_per_note.py`
- `src/evaluate/analyze_asap_train_test_timing.py`
- `src/evaluate/analyze_timing_distributions.py`
- `src/evaluate/analyze_timing_delta_ratio.py`
- `src/evaluate/report_inr0624_wass.py`

## INR 数据预处理

当前标准入口：

```bash
cd /home/sy/EPR/PianistTransformer

bash script/build_pianocore_inr_sidecars.sh
```

该脚本会顺序完成：

- 生成 paired INR JSON
- 补 XML score feature
- 写固定 `train_valid_asap3_nonasap05_v1` valid split
- 预构建 base `.pt`
- 预构建 ASAP `.ASAP.pt`

如果要直接启动 INR 训练，不再通过旧短壳，改用：

```bash
python src/train/train_inr.py --config <config.json>
```
