所有实验都已经生成 MIDI。试听时主要进入各目录的 `sampling/midis/`；同名 MIDI 对应同一首曲子，可以直接横向比较。每组都是 19 首。

### 1. 原始 DLM 基线

- [Raw sampling MIDI](/home/sy/EPR/PianistTransformer/results/floorlog_dlm_2gpu/20260712_asap_test/k8-b256-veldlm/sampling/midis)
- [Deterministic MIDI](/home/sy/EPR/PianistTransformer/results/floorlog_dlm_2gpu/20260712_asap_test/k8-b256-veldlm/deterministic/midis)

### 2. Truncation

同一个原始 DLM checkpoint，仅改变推理采样。

- [trunc r=0.05](/home/sy/EPR/PianistTransformer/results/floorlog_trunc_sampling_4gpu/20260712_asap_test/trunc-r0p05/sampling/midis)
- [trunc r=0.10](/home/sy/EPR/PianistTransformer/results/floorlog_trunc_sampling_4gpu/20260712_asap_test/trunc-r0p1/sampling/midis)
- [trunc r=0.20](/home/sy/EPR/PianistTransformer/results/floorlog_trunc_sampling_4gpu/20260712_asap_test/trunc-r0p2/sampling/midis)
- [trunc r=0.30](/home/sy/EPR/PianistTransformer/results/floorlog_trunc_sampling_4gpu/20260712_asap_test/trunc-r0p3/sampling/midis)

### 3. Shrink

同一个原始 DLM checkpoint，仅做推理后处理。

- [linear s=0.25](/home/sy/EPR/PianistTransformer/results/floorlog_shrink_sweep_4gpu/20260712_floorlog_shrink_sweep/k8-b256-veldlm/linear-s0p25/sampling/midis)
- [linear s=0.50](/home/sy/EPR/PianistTransformer/results/floorlog_shrink_sweep_4gpu/20260712_floorlog_shrink_sweep/k8-b256-veldlm/linear-s0p50/sampling/midis)
- [tanh r=0.05](/home/sy/EPR/PianistTransformer/results/floorlog_shrink_sweep_4gpu/20260712_floorlog_shrink_sweep/k8-b256-veldlm/tanh-r0p05/sampling/midis)
- [tanh r=0.10](/home/sy/EPR/PianistTransformer/results/floorlog_shrink_sweep_4gpu/20260712_floorlog_shrink_sweep/k8-b256-veldlm/tanh-r0p10/sampling/midis)

Shrink 实验没有重复保存 deterministic MIDI，比较时使用上面的原始 DLM deterministic。

### 4. Scale / Tail loss

这些是从头训练 16 epoch 的模型。请使用有效的 `16ep_v2`，不要使用前面的无效旧目录。

- [重新训练的 DLM base](/home/sy/EPR/PianistTransformer/results/floorlog_dlm_constraints_4gpu/20260713_dlm_constraints16ep_v2/dlm-base/sampling/midis)
- [Scale only](/home/sy/EPR/PianistTransformer/results/floorlog_dlm_constraints_4gpu/20260713_dlm_constraints16ep_v2/dlm-scale-s001-s02/sampling/midis)
- [Tail loss only](/home/sy/EPR/PianistTransformer/results/floorlog_dlm_constraints_4gpu/20260713_dlm_constraints16ep_v2/dlm-tail-r005-l1/sampling/midis)
- [Scale + tail loss](/home/sy/EPR/PianistTransformer/results/floorlog_dlm_constraints_4gpu/20260713_dlm_constraints16ep_v2/dlm-scale-tail-r005-l1/sampling/midis)

每个实验的 `sampling/` 同级还有 `deterministic/midis/`，可以检查约束是否同时破坏了确定性中心预测。

### 5. Beta5

从头训练 16 epoch。

- [Beta5，仅 support 限制](/home/sy/EPR/PianistTransformer/results/floorlog_beta5_ln2_variance_4gpu/20260713_beta5_ln2_var16ep/beta5-support/sampling/midis)
- [Beta5 + variance loss](/home/sy/EPR/PianistTransformer/results/floorlog_beta5_ln2_variance_4gpu/20260713_beta5_ln2_var16ep/beta5-var-r005-l10/sampling/midis)

### 6. LN2

从头训练 16 epoch。

- [LN2，仅 support 限制](/home/sy/EPR/PianistTransformer/results/floorlog_beta5_ln2_variance_4gpu/20260713_beta5_ln2_var16ep/ln2-support/sampling/midis)
- [LN2 + variance loss](/home/sy/EPR/PianistTransformer/results/floorlog_beta5_ln2_variance_4gpu/20260713_beta5_ln2_var16ep/ln2-var-r005-l10/sampling/midis)

试听时建议固定同一首文件，优先比较：

1. Raw DLM
2. trunc 0.05
3. tanh 0.05
4. scale+tail
5. beta5-var
6. ln2-var
7. deterministic