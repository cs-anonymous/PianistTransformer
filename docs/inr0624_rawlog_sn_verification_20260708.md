# INR0624 Raw-Log + Skew-Normal 修改核对记录

日期：2026-07-08

本文档用于核对以下 5 项修改是否已经在当前仓库中落地，并记录当前实现状态与代码证据。

## 结论摘要

| 项目 | 状态 | 结论 |
| --- | --- | --- |
| 1. timing 改为双输入 `[raw_s, logscale]` | 已完成 | 当前 `raw_log` 模式已经把 timing control 编码为 `[s, log1p(20 * s)]`，不再除以 `g(5000)` |
| 2. timing head 改为双 head，raw dev + logscale dev，loss 相加 | 部分完成 | 双 head 和双 target 已完成，但当前 loss 是 `log_loss + 0.5 * raw_loss`，还不是“直接相加” |
| 3. musical 中 `mo/md/ml` 不除 6，只有 `vel` 除 127 | 已完成 | `build_score_musical_rows` 已改为直接保留原始 `mo/md/ml` 标量 |
| 4. head 不再用 MLN / logistic，改用 SN 或 MSN | 部分完成 | 新配置已切到 `skew_normal`，但仓库默认启动脚本/README 仍主要指向旧的 `mln3` 配置 |
| 5. infer 时只用 logscale 采样，raw timing 只用于训练 | 已完成 | 当前 SN 推理只读取 `timing_log_*` 参数，修改 `timing_raw_*` 不影响 infer 输出 |

## 逐项核对

### 1. Timing 双输入 `[s, log1p(20 * s)]`

已完成。

- `src/train/train_inr.py` 中 `timing_control_feature_dim(..., mode='raw_log')` 返回 5，表示 score control 两个 timing 字段各占 2 维，再加 velocity 1 维。
- `src/train/train_inr.py` 中 `encode_timing_control_features()` 在 `raw_log` 模式下返回：

```python
[
    value / 1000.0,
    raw_log_timing_value(value, scale=log_scale),
]
```

- `src/model/integrated_pianoformer.py` 中 `_torch_timing_control_code()` 的 `raw_log` 分支也一致：

```python
[
    value / 1000.0,
    log1p((1000 / scale) * seconds),
]
```

当 `scale = 50` ms 时，第二维正好是 `log1p(20 * s)`。

本地 smoke check：

- `100 ms -> [0.1, 1.0986122886681096]`
- 其中 `1.0986122886681096 = log(3) = log1p(20 * 0.1)`

### 2. Timing 双 head 与双 loss

双 head 已完成，但 loss 权重还没有完全改到“直接相加”。

已完成部分：

- `src/train/train_inr.py` 中 `performance_dev_velocity_pedal4_binary_rows()` 在 `raw_log_deviation` 模式下输出 9 维 label：

```python
[
    log_ioi_dev,
    log_duration_dev,
    raw_ioi_dev_s,
    raw_duration_dev_s,
    velocity / 127,
    pedal4_binary...
]
```

- `src/model/integrated_pianoformer.py` 中 `_split_epr_mixture_params()` 对 `skew_normal` 拆出了两套 timing 参数：
  - `timing_log_loc / timing_log_log_scale / timing_log_alpha`
  - `timing_raw_loc / timing_raw_log_scale / timing_raw_alpha`

- `IntegratedContinuousDecoder` 对 SN timing head 的输出宽度也已经翻倍：
  - IOI: `per_feature_dim * 2`
  - Duration: `per_feature_dim * 2`

未完全符合的地方：

- `src/model/integrated_pianoformer.py` 中 loss 目前是：

```python
loss_ioi = loss_ioi_log + raw_lambda * loss_ioi_raw
loss_duration = loss_duration_log + raw_lambda * loss_duration_raw
```

- 当前新配置 [configs/inr0624_epr_sn_rawlog_sine.json](/home/sy/EPR/PianistTransformer/configs/inr0624_epr_sn_rawlog_sine.json:32) 里：

```json
"raw_timing_loss_lambda": 0.5
```

这表示现在实际是：

- `loss_ioi = log_loss + 0.5 * raw_loss`
- `loss_duration = log_loss + 0.5 * raw_loss`

因此“两个 loss 直接相加”这一点还没有完全完成。如果要严格符合原要求，`raw_timing_loss_lambda` 需要改为 `1.0`，或者直接去掉这个缩放系数。

### 3. `mo / md / ml` 不除 6，只保留原始值

已完成。

`src/train/train_inr.py` 中 `build_score_musical_rows()` 已经把原先的 `/ 6` 标量改成了原始值：

- continuous 分支：
  - `mo`
  - `md`
  - `raw_ml`

- musical51 分支：
  - `md`
  - `ml_eff`

- categorical62 / musical62 分支：
  - `o_scalar = mo`

速度归一化仍然保留为 `/ 127`，这和要求一致。

本地 smoke check 也确认了标量未再除 6：

- 第一条样例的 `md scalar = 2.0`
- 第一条样例的 `ml scalar = 4.0`

如果仍使用旧设计，这两个位置应该会分别看到约 `0.333` 和 `0.667`，但当前不是。

### 4. Head 改为 SN / MSN，不再用 MLN / logistic

代码主路径已支持并已提供 SN 配置，但仓库默认实验入口还没有整体切换完。

已完成部分：

- `src/model/integrated_pianoformer.py` 已新增：
  - `SN_DISTRIBUTIONS = {"sn", "skew_normal"}`
  - `_skew_normal_params()`
  - `_skew_normal_log_prob()`
  - `_skew_normal_nll()`
  - `_skew_normal_mean_or_sample()`

- `src/train/train_inr.py` 中 `create_model()` 已允许：
  - `"sn"`
  - `"skew_normal"`

- 新配置 [configs/inr0624_epr_sn_rawlog_sine.json](/home/sy/EPR/PianistTransformer/configs/inr0624_epr_sn_rawlog_sine.json:29) 已切到：

```json
"epr_distribution": "skew_normal"
```

当前仍未完全切换的地方：

- `script/launch_inr0624_epr_logscale_4gpu.sh` 仍默认使用：
  - `configs/inr0624_epr_mln3_cine_mslog.json`
  - `configs/inr0624_epr_mln3_sine_mslog.json`

- `script/README.md` 也仍主要引用 `mln3_*_mslog` 配置。

因此结论是：

- “SN 方案已实现并有新配置”是成立的。
- “仓库默认 EPR 启动入口已经完全从 MLN/MLN3 切换掉”目前还不成立。

### 5. Infer 只用 logscale 采样；raw dev 不做 infer 采样

已完成。

`src/model/integrated_pianoformer.py` 中 `_materialize_epr_prediction()` 在 `distribution in SN_DISTRIBUTIONS` 时，只使用：

- `params["timing_log_loc"]`
- `params["timing_log_log_scale"]`
- `params["timing_log_alpha"]`

它不会读取：

- `params["timing_raw_loc"]`
- `params["timing_raw_log_scale"]`
- `params["timing_raw_alpha"]`

也就是说，raw timing head 目前只参与训练 loss，不参与 infer materialization。

本地 smoke check：

- 构造固定的 SN raw decoder 输出后，只改 `timing_raw_*` 对应参数
- infer 前后输出完全一致：`torch.allclose(pred1, pred2) == True`

这与“训练时双 timing 表示，但 infer 时只用 logscale sample”的要求一致。

## 关于 dev 目标的核对

这部分已经符合要求。

当前 `raw_log_deviation` 路径中：

- log dev 采用：

```python
raw_log_timing_value(perf_ms) - raw_log_timing_value(score_ms)
```

- raw dev 采用：

```python
(perf_ms - score_ms) / 1000.0
```

这里已经没有旧版的：

- `+ 0.5`
- `clamp(0, 1)`

它们只还保留在旧的 `log_deviation` 兼容路径中，不在新 `raw_log_deviation` 路径里。

## 当前和旧文档的差异

当前主设计文档 [docs/INR0624.md](/home/sy/EPR/PianistTransformer/docs/INR0624.md:64) 仍主要描述旧的单 log-scale / `+0.5` / clamp / ALN-MLN 系表述，尚未反映本次 raw-log + SN 改动。

因此，这次新增本文档是必要的；它记录的是当前代码已经实现到哪一步，而不是旧设计文档中的历史状态。

## 最终判断

当前仓库对这 5 项修改的落地状态可以总结为：

1. timing 双输入：已完成
2. timing 双 head：已完成
3. timing 双 loss 直接相加：未完全完成，当前仍是 `raw_loss * 0.5`
4. `mo/md/ml` 去标准化：已完成
5. SN 头替代 MLN/logistic：代码与新配置已完成，但默认启动入口未完全切换
6. infer 只用 logscale sample：已完成

如果后续要把这组修改视为“完全完成”，最直接还差的两点是：

1. 把 `raw_timing_loss_lambda` 从 `0.5` 改成 `1.0` 或移除缩放
2. 把默认启动脚本和 README 从 `mln3_*_mslog` 切到新的 `skew_normal + raw_log` 配置

## 2026-07-08 晚间补充

晚间重新核对后，实验配置保持为：

```json
"raw_timing_loss_lambda": 0.5
```

这是按用户后续确认执行的最终口径，因此当晚正式启动的三组实验实际使用的是：

- `loss_ioi = log_loss + 0.5 * raw_loss`
- `loss_duration = log_loss + 0.5 * raw_loss`

也就是说：

- 仓库“全局默认值”仍未完全切换
- 这次正式实验也继续沿用 `0.5` 权重

## 2026-07-08 最终实验启动口径

当晚最终实际启动的三组实验，训练口径统一为：

- 只使用 `ASAP` 训练/验证，不再先跑全量数据再 adapt
- `num_train_epochs = 16`
- `per_device_train_batch_size = 32`
- `gradient_accumulation_steps = 2`
- 单实验单卡运行

对应启动批次为：

- `results/inr_epr_pipeline/launch_rawlog_3exp_20260708_235220`

三组实验分别是：

- `exp1_sine_tfmask50`
- `exp2_sine_nomus_tfmask50`
- `exp3_splitperf_tfmask50`
