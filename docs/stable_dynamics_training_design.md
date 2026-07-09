# Stable Dynamics Training for INR EPR

## 1. Motivation

The current DAgger-Lite model still fails to learn a stable full-AR timing stationary distribution. Although training already uses cached model-predicted decoder feedback, channel-wise prefix replacement, window curriculum, and k=1 rollout-aware validation, full closed-loop rollout remains unstable. Greedy decoding collapses timing variance, while stochastic sampling produces systematic rightward drift and IOI variance inflation. Timing-head diagnostics further show that the predicted IOI mean and standard deviation increase under self-conditioned history, indicating that the transition distribution itself changes under closed-loop feedback rather than merely suffering from sampling-tail noise.
Therefore, the next experiment should not continue to increase the complexity of DAgger / Scheduled Sampling. Instead, it should test whether explicitly constraining the local timing error dynamics can improve closed-loop stability.

The proposed method is **Stable Dynamics Training**. It introduces controlled continuous timing perturbations into the decoder feedback and adds a contraction-style loss that penalizes the model when a current timing deviation leads to an equal or larger next-step timing deviation.

---

## 2. Core Hypothesis

The observed full-AR failure suggests that the model has learned an unstable self-conditioned transition kernel:

```text
small timing deviation
→ shifted/widened timing head
→ larger sampled deviation
→ accumulated full-AR drift
```

The goal is not only to make the model robust to corrupted input, but to make it learn a recovery behavior:

```text
perturbed timing history
→ prediction whose timing error is smaller or at least not larger
```

This is different from dropout, feature masking, and DAgger-Lite:

| Method          | Corruption / Feedback Type                                    | Loss Target                  | Directly constrains error dynamics? |
| --------------- | ------------------------------------------------------------- | ---------------------------- | ----------------------------------- |
| PAD dropout     | Replace note embedding with PAD                               | GT target                    | No                                  |
| Feature mask    | Set selected features to 0 + mask embedding                   | GT target                    | No                                  |
| DAgger-Lite     | Replace feedback with cached model prediction                 | GT target                    | Weakly, only through NLL            |
| Stable Dynamics | Add continuous timing deviation matching observed drift modes | GT target + contraction loss | Yes                                 |

The key novelty is not the noise itself. The key is the additional loss that explicitly penalizes closed-loop error amplification.

---

## 3. Recommended Experimental Decision

Stable Dynamics should first be tested **as a standalone training objective**, not mixed with DAgger.

### Reason

The current DAgger-Lite setting already changes the decoder feedback distribution. If Stable Dynamics is immediately combined with DAgger, any improvement or failure becomes hard to interpret:

```text
improvement = better prefix coverage?
improvement = contraction loss?
failure = DAgger noise too strong?
failure = contraction loss ineffective?
```

Therefore, the first clean test should isolate the new idea.

### Recommended order

```text
Stage 1: Noise-only control
Stage 2: Stable-only training
Stage 3: Stable + DAgger, only if Stage 2 works
```

The main comparison should be:

```text
A. baseline / existing DAgger-Lite checkpoint
B. timing-noise augmentation without contraction
C. timing-noise augmentation with contraction
```

If B fails but C improves full-AR stationary metrics, then the effect comes from the contraction objective rather than from another form of noisy augmentation.

---

## 4. Method Definition

Let the timing target at note `t` be:

```text
y_t = [ioi_logdev_t, duration_logdev_t]
```

Let the original decoder feedback be:

```text
f_t = GT target7_t
```

Construct a corrupted timing feedback:

```text
f'_t = f_t
f'_t[ioi, duration] = f_t[ioi, duration] + epsilon_t
```

where `epsilon_t` is a controlled continuous perturbation in the same normalized/log-deviation space used by the timing target.

The model receives `f'` as decoder feedback but still predicts the original GT target:

```text
model input:  score + corrupted decoder feedback f'
target:       GT target7
```

The usual probabilistic loss remains unchanged:

```text
L_main = NLL(pred_distribution, GT target7)
```

Stable Dynamics adds a gated contraction loss on IOI and duration.

Define current feedback error:

```text
e_t = f'_t[timing] - GT_t[timing]
```

Define predicted next-step error using the predicted distribution mean:

```text
r_{t+1} = pred_mean_{t+1}[timing] - GT_{t+1}[timing]
```

The contraction loss is applied only at positions where timing feedback was actually perturbed:

```text
valid_t =
  attention_mask_t
  AND attention_mask_{t+1}
  AND corrupted_mask_t
  AND ||e_t|| > eps

L_contract = mean_{valid_t} max(0, ||r_{t+1}|| - alpha * ||e_t||)
```

where:

```text
alpha <= 1
```

If `alpha = 1.0`, the model is only penalized when the next-step error is larger than the current error.
If `alpha = 0.9`, the model is encouraged to reduce the deviation by at least 10%.

The final loss is:

```text
L = L_main + lambda_contract * L_contract
```

The gating is important. Without `corrupted_mask_t` and `||e_t|| > eps`, the contraction term degenerates into an extra mean-regression loss on clean teacher-forced positions, which can reduce sampling drift by making timing more rigid.

---

## 5. Noise Design

The perturbation should not mimic dropout or missing features. It should mimic the observed full-AR failure modes.

The cheap15 diagnostics show:

```text
sampling full AR:
IOI shifts right
duration shifts right
IOI variance inflates
timing head mean/std increase under closed-loop history
```

Therefore, the noise should be continuous and timing-specific.

### Recommended first noise modes

Use a mixture of three perturbation types:

```text
Mode A: zero-mean timing noise
Mode B: positive timing bias
Mode C: variance-inflation noise
```

Example configuration in log-deviation / normalized timing space:

```text
stable_noise_modes:
  zero_mean:
    prob: 0.50
    ioi_mu: 0.000
    ioi_sigma: 0.010
    dur_mu: 0.000
    dur_sigma: 0.010

  positive_bias:
    prob: 0.25
    ioi_mu: 0.003
    ioi_sigma: 0.010
    dur_mu: 0.003
    dur_sigma: 0.010

  variance_inflation:
    prob: 0.25
    ioi_mu: 0.000
    ioi_sigma: 0.025
    dur_mu: 0.000
    dur_sigma: 0.020
```

These values should be treated as starting points. The actual scale should be checked against empirical log-deviation standard deviation.

### Noise application rate

Do not corrupt 50% of all notes aggressively in the first run.

Recommended:

```text
stable_apply_prob: 0.30
stable_timing_only: true
stable_corrupt_channels: [ioi, duration]
```

The model should still see enough clean feedback to preserve normal EPR quality.

---

## 6. What Not To Do

### 6.1 Do not use PAD or mask embedding

Stable Dynamics should not replace values with PAD and should not add mask embeddings. The failure state during full AR is not a missing-value state. It is a wrong-but-plausible continuous state.

Wrong:

```text
replace note embedding with PAD
set timing feature to 0
add mask embedding
```

Correct:

```text
add continuous perturbation to IOI/duration feedback
keep the input semantically plausible
do not reveal an explicit corruption marker
```

### 6.2 Do not make the contraction too strong

If the contraction loss is too strong, the model may learn to suppress expressive timing variation. This would reduce sampling drift but worsen greedy-style rigidity.

Avoid:

```text
alpha too small
lambda_contract too large
all channels constrained at once
```

Start with:

```text
alpha: 1.0 or 0.9
lambda_contract: 0.01, 0.05, 0.1
channels: IOI + duration only
```

### 6.3 Do not combine with DAgger in the first test

The first goal is to test whether the contraction loss itself changes full-AR stationary behavior. Combining with DAgger immediately would obscure the result.

---

## 7. Training Variants

### Variant S0: Noise-only control

Purpose: test whether continuous timing noise alone helps.

```text
feedback = GT + timing noise
loss = NLL
dagger_prefix_training = false
stable_contract_loss = false
```

Expected outcome:

```text
If S0 improves, then simple continuous noise robustness already helps.
If S0 does not improve, noise alone is insufficient.
```

### Variant S1: Stable-only

Purpose: test the true Stable Dynamics idea.

```text
feedback = GT + timing noise
loss = NLL + lambda_contract * L_contract
dagger_prefix_training = false
stable_contract_loss = true
```

Expected outcome:

```text
If S1 improves over S0, the contraction loss is useful.
If S1 does not improve over S0, Stable Dynamics is probably not worth expanding.
```

### Variant S2: Stable + DAgger

Only run after S1 is positive.

```text
feedback = mix(GT, timing noise, cached model prediction)
loss = NLL + lambda_contract * L_contract
dagger_prefix_training = true
stable_contract_loss = true
```

Purpose:

```text
Test whether DAgger coverage and contraction regularization are complementary.
```

This should not be the first experiment.

---

## 8. Implementation Plan

### 8.1 Collator changes

Add an optional `stable_dynamics_training` branch after normal `decoder_feedback_continuous` is constructed.

Pseudo-flow:

```text
decoder_feedback_continuous = GT target7

if stable_dynamics_training:
    decoder_feedback_continuous, stable_feedback_mask = apply_timing_noise(
        decoder_feedback_continuous,
        channels=[ioi, duration],
        apply_prob=stable_apply_prob,
        noise_mode=sample_noise_mode()
    )

model_input = build_decoder_input(decoder_feedback_continuous)
```

Important:

```text
Do not change labels.
Do not change target7.
Do not add mask embedding.
Do not mark corrupted tokens.
```

### 8.2 Model output requirements

The contraction loss needs the predicted timing mean.

For `mixture_logistic_normal`, compute or approximate:

```text
pred_mean_ioi
pred_mean_duration
```

Use the distribution mean if already available. If the exact mean is inconvenient, use the mixture-weighted component location as a first approximation.

### 8.3 Loss computation

Inside the model/trainer loss function:

```text
L_main = existing EPR loss

if stable_dynamics_training:
    e_current = noisy_feedback[:, :-1, timing] - labels[:, :-1, timing]
    r_next = pred_mean[:, 1:, timing] - labels[:, 1:, timing]

    norm_current = sqrt(sum(e_current^2 over timing channels))
    norm_next = sqrt(sum(r_next^2 over timing channels))

    valid = (
        attention_mask[:, :-1]
        & attention_mask[:, 1:]
        & any(stable_feedback_mask[:, :-1, timing])
        & (norm_current > eps)
    )

    L_contract = mean_valid(relu(norm_next - alpha * norm_current))

    L = L_main + lambda_contract * L_contract
```

Mask invalid/padded positions.

### 8.4 Channel-wise version

The first cheap15 sample results show that IOI and duration prefer different
regularization strengths. The next version should therefore avoid one shared
timing-vector norm and compute channel-wise contraction by default:

```text
L_contract_ioi =
  mean_valid_ioi relu(|r_{t+1,ioi}| - alpha_ioi * |e_{t,ioi}|)

L_contract_duration =
  mean_valid_duration relu(|r_{t+1,duration}| - alpha_duration * |e_{t,duration}|)
```

Then:

```text
L_contract =
  lambda_ioi * L_contract_ioi
  + lambda_duration * L_contract_duration
```

Each channel is gated by its own corrupted feedback mask. This prevents a
duration perturbation from forcing IOI contraction, and vice versa.

Recommended first setting, based on the first S1a/S1b sample sweep:

```text
stable_contract_ioi_lambda = 0.05
stable_contract_duration_lambda = 0.01
stable_contract_ioi_alpha = 1.0
stable_contract_duration_alpha = 1.0
```

The legacy `stable_contract_lambda` path remains useful as a control, but it
uses the coupled timing norm and should no longer be the main experiment.

---

## 9. Evaluation Protocol

Do not judge this method primarily by teacher-forced validation loss.

The target metrics are full-AR stationary diagnostics.

### Required evaluation

Run the same cheap15 stationary diagnostics:

```text
greedy k-sweep
sampling k-sweep
temperature sweep
head stats under sampling
single-channel feedback ablation
```

### Primary success metrics

For sampling full AR:

```text
IOI shift ms ↓
duration shift ms ↓
IOI std ratio → 1
duration std ratio → 1
IOI W ↓
duration W ↓
```

For greedy full AR:

```text
IOI std ratio should not decrease further
duration std ratio should not decrease further
```

For head stats:

```text
full pred mean should not drift right from k=0
full pred std should not inflate from k=0
```

### Failure conditions

Stable Dynamics should be considered unsuccessful if:

```text
sampling drift decreases but greedy variance collapse worsens substantially
TF / k=0 EPR metrics degrade strongly
head std collapses instead of stabilizing
duration W remains high or worsens
```

The method should not simply convert sampling drift into deterministic rigidity.

---

## 10. Minimal Config Sketch

```yaml
stable_dynamics_training: true
stable_apply_prob: 0.30
stable_channels: ["ioi", "duration"]

stable_noise_modes:
  zero_mean:
    prob: 0.50
    ioi_mu: 0.000
    ioi_sigma: 0.010
    duration_mu: 0.000
    duration_sigma: 0.010

  positive_bias:
    prob: 0.25
    ioi_mu: 0.003
    ioi_sigma: 0.010
    duration_mu: 0.003
    duration_sigma: 0.010

  variance_inflation:
    prob: 0.25
    ioi_mu: 0.000
    ioi_sigma: 0.025
    duration_mu: 0.000
    duration_sigma: 0.020

stable_contract_loss: true
stable_contract_alpha: 0.9
stable_contract_lambda: 0.0
stable_contract_ioi_alpha: 1.0
stable_contract_duration_alpha: 1.0
stable_contract_ioi_lambda: 0.05
stable_contract_duration_lambda: 0.01

dagger_prefix_training: false
```

For the noise-only control:

```yaml
stable_dynamics_training: true
stable_contract_loss: false
dagger_prefix_training: false
```

For the later combined version:

```yaml
stable_dynamics_training: true
stable_contract_loss: true
dagger_prefix_training: true
```

---

## 11. Recommended Experiment Matrix

| Run           | DAgger | Timing noise | Contraction loss | Purpose                                   |
| ------------- | -----: | -----------: | ---------------: | ----------------------------------------- |
| Baseline-D    |    yes |           no |               no | Existing DAgger-Lite reference            |
| S0            |     no |          yes |               no | Test whether continuous noise alone helps |
| S1-α1.0-λ0.05 |     no |          yes |              yes | Weak contraction                          |
| S1-α0.9-λ0.05 |     no |          yes |              yes | Moderate contraction                      |
| S1-α0.9-λ0.1  |     no |          yes |              yes | Stronger contraction                      |
| S2            |    yes |          yes |              yes | Only if S1 improves                       |

The first decision should be made from S0 vs S1, not from S2.

---

## 12. Expected Interpretation

### Case 1: S1 improves over S0

This supports the hypothesis that the model lacks local recovery dynamics. Stable Dynamics is worth developing further.

Next steps:

```text
try learned/context-conditioned alpha
combine with DAgger
add head-stat dynamics matching
```

### Case 2: S0 and S1 both fail

This suggests the issue is not easily fixed by local timing recovery loss. The per-note AR formulation may be structurally unstable.

Next steps:

```text
move to action chunking / multi-output timing prediction
reduce per-note recursive feedback frequency
consider diffusion-policy-style chunk generation
```

### Case 3: S1 reduces sampling drift but worsens greedy collapse

The contraction is too strong or too global.

Next steps:

```text
increase alpha toward 1.0
reduce lambda_contract
apply contraction only under positive drift noise
exclude cadence / ritardando-like regions if identifiable
```

### Case 4: S1 improves IOI but not duration

The timing subsystem should be split.

Next steps:

```text
different noise scales
different lambdas
possibly condition duration recovery on articulation/context
```

---

## 13. Final Recommendation

Stable Dynamics should be treated as a small, falsifiable experiment, not as a new complicated training framework.

The first experiment should be:

```text
same continuous timing noise
NLL-only vs NLL + contraction
no DAgger
IOI + duration only
cheap15 full-AR stationary diagnostics
```

Only if the contraction loss improves full-AR stationary behavior should it be combined with DAgger-Lite. If it does not improve the stationary diagnostics, the project should stop expanding this line and move toward chunk-level / action-policy-style generation.
