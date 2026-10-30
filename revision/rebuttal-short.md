## Response to reviewer A: Thank you for constructive feedback on configuration, coherence, novelty, and target attribution

### A1: Configuration labels and matched results

The reviewer is correct that the submitted labels were inconsistent. Table 2's CINR value (PP HR 2.38) is from `extra/rep_slot_sum`, which sums six 768-dimensional typed slots. In the main text, we use the slot-sum configuration for both CINR and DINR, as it performs best overall among the tested encodings.

The DINR table note in the supplement was unclear, so we reorganized the results below across eight configurations defined by two factors: representation and coordinate use.

| Representation | PP HR | PP IOI | PP Dur. | PN HR | PN IOI | PN Dur. |
|---|---:|---:|---:|---:|---:|---:|
| slot128+mlp | 4.79 | 24.55 | 88.60 | 1.97 | 41.23 | 157.72 |
| slot128+mlp, coordinates | 4.64 | 24.30 | 73.20 | 1.95 | 41.95 | 148.84 |
| slot256+mlp | 4.46 | 20.72 | 58.79 | 1.94 | 39.53 | 142.83 |
| slot256+mlp, coordinates | 4.53 | 22.69 | 78.28 | 1.93 | 40.67 | 152.27 |
| slot-direct | 3.90 | 22.58 | 96.01 | 1.79 | 39.68 | 156.94 |
| slot-direct, coordinates | 3.85 | 24.04 | 92.92 | 1.78 | 41.33 | 155.95 |
| slot-sum | 3.78 | 20.88 | 102.99 | 1.79 | 37.82 | 160.82 |
| slot-sum, coordinates | **3.05** | **15.96** | **62.79** | **1.68** | **34.98** | **139.82** |

For DINR slot-sum, adding coordinates reduces PP HR from 3.78 to 3.05 and PN HR from 1.79 to 1.68. The table reports all six PP/PN metrics for the tested DINR configurations; we will identify the DINR setting explicitly.

### A2: Beat-level coherence study

PP Wasserstein is invariant to note order and PN is note-wise, so neither measures temporal coherence. We computed beat-level Pearson correlations for tempo $T$ and mean beat velocity $V$ against each human performance $h$:

$$
r_T^{(s,x,h)}=\operatorname{Corr}_{b\in\mathcal B_s}(T_{s,b}^{x},T_{s,b}^{h}),
\qquad
r_V^{(s,x,h)}=\operatorname{Corr}_{b\in\mathcal B_s}(V_{s,b}^{x},V_{s,b}^{h}).
$$

We macro-average across human performances, generated samples, and 19 test scores. CINR uses `bounded_2p5pct` for tempo and `dist_ln_k3` for velocity; DINR uses `rep_slot256_mlp` for tempo and `rep_slot_sum` for velocity.

| Method | Tempo Pearson | Velocity Pearson |
|---|---:|---:|
| Human | 0.303 | 0.760 |
| PT baseline | 0.120 | 0.505 |
| DINR | 0.119 | 0.285 |
| CINR | 0.135 | 0.341 |
| DExter | 0.003 | 0.220 |
| VirtuosoNet | 0.020 | -0.084 |
| ScorePerformer | 0.006 | 0.434 |

CINR's beat-level velocity correlation is modest despite low velocity Wasserstein distances (PP 6.18; PN 16.99). We will distinguish distributional fidelity from temporal coherence and narrow our musical-expressiveness claims. The listening-study plan is in C2.

### A3: Novelty and contribution

See B1. We will state the contribution as a combination for aligned EPR, not as the first one-step-per-note design.

### A4: Timing targets and attribution

The comparison with Pianist Transformer differs in architecture, pretraining, target, and output parameterization, so its full quality gap cannot be attributed to INR. We will narrow this claim. With the same CINR setup, we compared timing targets:

| CINR variant / timing target | PP HR | PP IOI | PP Dur. | PN HR | PN IOI | PN Dur. |
|---|---:|---:|---:|---:|---:|---:|
| Log-ratio residual (paper) | **2.38** | 11.87 | 47.67 | **1.59** | 32.93 | 119.11 |
| Linear residual, $[-1000,1000]$ ms | **3.26** | 13.68 | 84.36 | **1.77** | 37.17 | 155.70 |
| Raw ratio, $[0.135,2.718]$ | **3.27** | 13.73 | 62.36 | **1.69** | 33.38 | 126.53 |
| Serial-attribute CINR, 4 tokens/note | 3.27 | 11.69 | 68.54 | 1.59 | 32.24 | 112.81 |

The log-ratio target gives the lowest aggregate HR and PP timing distances within CINR. The earlier absolute-timing ablation increases PP IOI/duration from 11.87/47.67 to 272.30/480.38. This establishes the importance of target formulation, not an isolated INR effect. The serial row uses four attribute tokens per note and slot+MLP input fusion, whereas the main row uses slot-sum; it is evidence about the tested implementations, not a fully matched causal estimate of sequence compression.

The three relative targets are:

* Log-ratio residual: the paper target with nonzero-IOI support $[-2,1]$.
* Linear residual: $z=(t_{\mathrm{perf}}-t_{\mathrm{score}})/1000$, with support $[-1,1]$.
* Raw ratio: $z=t_{\mathrm{perf}}/\max(t_{\mathrm{score}},1\,\mathrm{ms})$, with nonzero-score support $[0.135,2.718]$.

For score IOI equal to zero, the raw ratio is undefined and uses support $[0,\exp(5)]$. A linear residual weights the same absolute error equally across score intervals; a raw ratio is proportional but not symmetric for doubling and halving; the log-ratio treats reciprocal speedups and slowdowns symmetrically.

## Response to reviewer B: Thank you for feedback on prior work, baseline coverage, and ablation scope

### B1: Relation to prior work and contributions

INSPIRE combines three task-specific design choices for aligned expressive piano rendering:

1. **Typed note representation:** separate encoders preserve the categorical, numerical, and binary types of aligned score and performance attributes, then fuse them into one vector per note.
2. **Attribute-matched outputs:** CINR predicts numerical timing, duration, and velocity with separate discretized-logistic distributions, while pedal uses binary outputs; DINR provides the categorical alternative.
3. **Score-relative log timing:** timing is represented as a log-ratio residual, giving reciprocal speedups and slowdowns symmetric target scales.

These choices are evaluated against relevant precedents:

| Method | Representation | Explicit numerical structure | Continuous per-attribute output distributions | Score-relative timing | Log-ratio timing residual | Output parameterization |
|---|---|:---:|:---:|:---:|:---:|---|
| Compound Word Transformer | Compound symbolic words | ✗ | ✗ | — | — | Categorical next-token probabilities |
| ScorePerformer | SPMuple: 8 score + 4 performance tokens per note | ✓ | ✗ | ✓ | ✗ | Categorical-token NLL decoder; MMD-VAE style latents |
| DExter | Symbolic expressive-performance representation | — | ✗ | — | — | Conditional diffusion over performance features |
| INSPIRE (CINR) | Six role-aware typed slots per note | ✓ | ✓ | ✓ | ✓ | Continuous inputs; per-attribute discretized-logistic distributions |

Compound Word Transformer groups categorical symbolic fields. ScorePerformer is the closest baseline: SPMuple groups score/performance fields and decodes by note, whereas CINR uses typed slot encoders, continuous MLPs for numerical attributes, separate discretized-logistic distributions, and an explicit score-relative log-ratio target. DExter uses conditional diffusion, while CINR uses autoregressive note-level decoding and attribute-specific heads. We will revise the related-work discussion accordingly.

### B2: Baseline coverage and inference cost

We evaluated public PianoFlow on all 19 test scores with three samples per score, recovered archived ScorePerformer metrics/timings, and evaluated a serial-attribute CINR under the 16-epoch budget.

| Method | PP HR | PP IOI | PP Dur. | PN HR | PN IOI | PN Dur. | Train GPU-h | Infer GPU-min (19 x 3) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SyMuPe/PianoFlow-base | 3.37 | 20.91 | 58.36 | 1.62 | 37.31 | 139.21 | — | 3.56 |
| ScorePerformer† | 5.88 | 26.64 | 101.05 | 2.85 | 105.30 | 160.12 | — | 165.93 |
| CINR (paper, slot-sum) | **2.38** | **11.87** | **47.67** | **1.59** | **32.93** | **119.11** | 0.62 | 6.0 |
| CINR (serial-attribute, 4 tokens/note) | 3.27 | 11.69 | 68.54 | 1.59 | 32.24 | 112.81 | 1.57 | 30.4 |

PianoFlow took 213.59 seconds on two RTX 3090 workers (3.56 GPU-minutes); it has no CC64 pedal controls, so pedal is treated as off. ScorePerformer archived timings cover 38 predictions and 110.62 GPU-minutes; 165.93 is scaled to three samples per score. Its paper does not report accelerator, wall-clock training time, or GPU-hours. The serial CINR took 2,829.8 seconds to train (1.57 GPU-hours) and about 30.4 GPU-minutes for 57 sampled predictions. The protocols and limitations will be stated with the table.

### B3: Scope of the ablations

See A4. We will label the timing-target ablation as isolated and the CINR/DINR comparison as joint.

## Response to reviewer C: Thank you for feedback on subjective evaluation and complementary metrics

### C1: Novelty

See B1. We will describe the contribution as task-specific formulation and evaluation, not a new Transformer architecture.

### C2: Subjective evaluation

The media supplement provides matched score excerpts, human performances, and generated examples. We do not yet have preliminary listening-test results. We plan a blinded study on eight ASAP test scores, with 30-second clips from CINR, DINR, Pianist Transformer, ScorePerformer, DExter, and VirtuosoNet. We will use the same synthesizer, loudness normalization, and excerpt boundaries, randomize clip order, and hide model identities. Twenty listeners will rank naturalness; human and Score MIDI will be labeled references, not ranked.

For $K=6$ model conditions, rank $r$ becomes the normalized Borda score

$$
b=(K-r)/(K-1),
$$

and we will report participant-level 95% bootstrap intervals, a repeated-measures Friedman test, and Holm-corrected pairwise Wilcoxon tests. We will limit current claims to distributional fidelity and report the listening study as planned validation.

### C3: Gain scope

The 13.8% figure is specifically the PP-HR difference (2.38 vs. 2.76), not a uniform gain across metrics or evidence of perceptual superiority.

### C4: KL/JS

We recomputed PP KL and JS on the 19 ASAP test scores using saved predictions and score-onset-span-normalized references. Timing uses 256 bins over $[-2,1]$ for duration/nonzero-score IOI and $[0,5]$ for zero-score IOI; velocity uses 128 bins over $[-0.5,127.5]$.

| Method | IOI KL | IOI JS | Dur. KL | Dur. JS | Vel. KL | Vel. JS |
|---|---:|---:|---:|---:|---:|---:|
| Human | 0.148 | 0.024 | 0.139 | 0.029 | 0.093 | 0.019 |
| Score MIDI | 3.046 | 0.549 | 5.056 | 0.673 | 5.695 | 0.506 |
| ScorePerformer | 5.023 | 0.419 | 2.028 | 0.303 | 4.644 | 0.354 |
| DExter | 7.243 | 0.410 | 2.403 | 0.294 | 1.662 | 0.182 |
| VirtuosoNet | 6.473 | 0.493 | 2.479 | 0.320 | 0.739 | 0.178 |
| Pianist Transformer | 5.315 | 0.233 | 0.697 | **0.068** | **0.114** | **0.025** |
| DINR (slot-sum) | 1.006 | 0.202 | **0.502** | **0.104** | 1.084 | 0.155 |
| CINR (slot-sum) | **0.349** | **0.089** | 0.598 | 0.145 | 0.341 | 0.044 |

CINR has the lowest IOI KL/JS among learned systems and lower velocity KL/JS than external EPR baselines; DINR has the lowest duration KL/JS, while Pianist Transformer has the lowest velocity KL/JS.

See A2 for the complementary beat-level correlation results.

## Response to reviewer D: Thank you for feedback on readability, modeling choices, data scale, and compute

### D1: Readability and notation

We will recast EPR as conditional sequence generation and simplify the music-specific wording. Let $X$ be a symbolic score and $Y$ its aligned performance, both with $N$ notes:

$$
p_{\theta}(Y\mid X)=\prod_{i=1}^{N}p_{\theta}(y_i\mid X,y_{<i}).
$$

The input is

$$
X=\{x_i\}_{i=1}^{N},\qquad x_i=(P_i,I_i^s,D_i^s,V_i^s,M_i),
$$

where $P_i$ is pitch, $I_i^s,D_i^s,V_i^s$ are score IOI, duration, and velocity, and $M_i$ contains musical onset, duration, note length, and six binary annotations (hand/staff, trill, grace note, staccato, stem-up, stem-down). The target is

$$
Y=\{y_i\}_{i=1}^{N},\qquad y_i=(P_i,I_i^p,D_i^p,V_i^p,\mathrm{Ped}_i).
$$

The output copies aligned pitch and predicts timing, dynamics, and four pedal states. Timing/duration use log-mapped 256-bin representations; velocity uses 128 values; pedal uses four binary snapshots (16 patterns). We will define symbols at first use, distinguish raw timing from log residuals, explain the slot roles, and clarify endpoint handling, pedal reconstruction, validation-window selection, and Figure 4's split after verifying the experiment protocol.

### D2: DINR

DINR provides a categorical baseline for comparing numerical representations. Coordinates reduce PP/PN HR from 3.78/1.79 to 3.05/1.68 in the slot-sum comparison. CINR performs better on most non-pedal metrics, but DINR remains useful for comparing categorical and continuous output parameterizations; both use BCE for pedal.

All DINR variants use the following head; coordinates modify value prototypes, not the head:

$$
\begin{aligned}
\ell_i^a &= H_a(h_i)C_T^\top+b_a,\qquad a\in\{I,D\},\\
\ell_i^V &= H_V(h_i)C_V^\top+b_V,
\end{aligned}
\qquad
p_i^a=\operatorname{softmax}(\ell_i^a),\quad a\in\{I,D,V\}.
$$

$h_i$ is the decoder state; $H_I,H_D,H_V$ are query MLPs; $C_T$ is shared by IOI/duration; $C_V$ is the velocity table; each row is a candidate value; $b_I,b_D,b_V$ are class biases. Training uses

$$
\mathcal{L}_{\mathrm{DINR}}=\sum_{a\in\{I,D,V\}}\mathrm{CE}(\ell_i^a,q_a(y_i^a)) + \mathrm{BCE}(\hat p_i,p_i).
$$

Softmax is used for sampling and BCE for the four pedal states.

### D3: Modeling contribution

The Transformer encoder-decoder and autoregressive training are standard. We position the contribution at aligned EPR formulation: typed attribute embedding, attribute-matched output distributions, and score-relative log-timing targets. The relative-versus-absolute comparison isolates the target; CINR/DINR jointly changes input encoding and output parameterization. We will state this scope and avoid presenting the work as a new Transformer architecture.

### D4: Data and compute

ASAP is the shared aligned benchmark used by prior EPR comparisons. PianoCoRe-A combines several transcription sources with fewer Disklavier recordings, so its distribution differs. We trained slot-sum INSPIRE on the full PianoCoRe-A training portion:

| Training setting | Works | Performances | Score notes | Performance hours | Window examples | Epochs | Training cost |
|---|---:|---:|---:|---:|---:|---:|---|
| PianoCoRe-A full training | 1,780 | 141,710 | 3,184,866 | 10,826.1 | 781,846 | 4 | 14.36 h wall-clock; 28.7 GPU-h on 2 x RTX 3090 |
| ASAP benchmark training | 188 | 887 | 505,584 | 75.4 | 6,475 | 16 | 0.6 GPU-h reported in Table 2 |

Four PianoCoRe-A epochs process 48,868 optimizer steps; 16 ASAP epochs process about 1,632. For CINR and Pianist Transformer, timing used two RTX 3090 GPUs, bfloat16, global batch size 64, node-cache size 16, eight sampling workers, 19 scores, and three samples per score. GPU-minutes are summed across workers. Public PianoFlow and archived ScorePerformer timings use separate protocols. Full-baseline reproduction on PianoCoRe-A remains future work.

## Response to reviewer E (AI reviewer): Thank you for feedback on stability, metric scope, efficiency, and revisions

### E1: Stability

The values 2.38, 2.54, and 2.08 are from different configurations, not repeated runs. Table 2 uses `extra/rep_slot_sum`; Table 3 uses canonical slot+MLP CINR (DLM k1, 5% bound, T5 8+4); Figure 4 uses the logistic slot+MLP model. We measured training- and sampling-seed variation on 19 scores with three samples per score:

| Source of variation | Three runs included | PP HR | PN HR |
|---|---|---:|---:|
| Training seed | Original Table 2 checkpoint + 2 independently trained checkpoints; fixed sampling seed | 2.374 $\pm$ 0.072 | 1.578 $\pm$ 0.020 |
| Sampling seed | Original Table 2 sample set + 2 independent sampling seeds; fixed checkpoint | 2.329 $\pm$ 0.041 | 1.576 $\pm$ 0.010 |

Training-seed variation is larger, especially for PP HR, but both variations remain near Table 2. Neither explains 2.54 or 2.08; those are separate ablation/sampling configurations.

### E2: Metric limits

See A2. PP/PN support distributional-fidelity claims only; we will not infer phrasing or perceptual quality from them.

### E3: Novelty

See B1. We will acknowledge the prior note-level and compound-token designs and state the narrower combination we contribute.

### E4: Attribution

See A4. We will not attribute the full quality difference to INR without a matched token-decoder control.

### E5: Zero-score IOI

For $I_i^s=0$, $g(I_i^s)=0$ and

$$
\Delta I_i=\log(\max(I_i^p,1\,\mathrm{ms})).
$$

Both variants use a separate zero-IOI support $[0,5]$ with 256 values; at inference, $\hat I_i^p=\exp(\Delta I_i)\,\mathrm{ms}$, representing about 1--148 ms. The nonzero-score support $[-2,1]$ does not apply here. On the test split, 30,735/65,039 notes (47.3%) have zero score IOI; among 197,891 aligned targets, 612 (0.31%) exceed $[0,5]$. CINR maps these to the highest bin; DINR masks them from IOI cross-entropy. We will document this handling and truncation.

### E6: Efficiency

See D4. The controlled sequence-length test shows 6.0 GPU-minutes for one step per note versus 30.4 for four steps (5.1$\times$); we will qualify the end-to-end 20$\times$ claim because $K^2$ describes self-attention only.

### E7: Other revisions

We will organize the paper around three hypotheses: note integration reduces decoding cost, numerical parameterization supplies ordinal structure, and score-relative timing simplifies the target. We will add the role-by-slot table and clarify continuous inputs versus discretized likelihood outputs, finite-support parameters ($s_a^{min}$, $s_a^{max}$, the 5% bound, and endpoint tails), score-pitch copying, pedal-to-CC64 reconstruction, validation-window selection, and the split used by Figure 4. Full-baseline comparisons on PianoCoRe-A, a second corpus, stratified analysis by chord density/ornamentation/alignment quality, and a joint timing/velocity/pedal output distribution are valuable follow-up studies beyond this rebuttal.
