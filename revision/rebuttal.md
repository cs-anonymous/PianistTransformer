## Response to R1: temporal coherence

PP and PN Wasserstein distances measure distributional fidelity. They do not directly measure temporal coherence. PP is invariant to note order. PN evaluates notes independently.

We therefore computed beat-level Pearson correlations. For score $s$, beat $b$, and performance $x$, let $T_{s,b}^{x}$ be the beat-period tempo and $V_{s,b}^{x}$ the mean velocity of notes in that beat. For each generated sample and each human performer $h$, we compute

$$
r_T^{(s,x,h)}=\operatorname{Corr}_{b\in\mathcal B_s}(T_{s,b}^{x},T_{s,b}^{h}),
\qquad
r_V^{(s,x,h)}=\operatorname{Corr}_{b\in\mathcal B_s}(V_{s,b}^{x},V_{s,b}^{h}).
$$

We macro-average over human performers, generated samples, and the 19 test scores. For each column, CINR and DINR use their strongest already-evaluated configuration. CINR uses `bounded_2p5pct` for Tempo and `dist_ln_k3` for Velocity. DINR uses `rep_slot256_mlp` for Tempo and `rep_slot_sum` for Velocity.

| Method | Tempo Pearson | Velocity Pearson |
|---|---:|---:|
| Human ceiling | 0.303 | 0.760 |
| PT baseline | 0.120 | 0.505 |
| DINR | 0.119 | 0.085 |
| CINR | 0.135 | 0.141 |
| DExter | 0.003 | 0.220 |
| VirtuosoNet | 0.020 | -0.084 |
| ScorePerformer | 0.006 | 0.434 |

CINR has low velocity Wasserstein distances (PP = 6.18; PN = 16.99), but only modest beat-level velocity correlation. The two measures capture different properties. We will distinguish distributional fidelity from temporal coherence in the revision. We will also limit claims about musical expressiveness. The current velocity objective does not fully model long-range dynamics. A controlled listening study remains future work.

## Response to R1: novelty and relation to prior note-level methods

We thank the reviewer for identifying these precedents. VirtuosoNet already predicts expressive parameters note by note. It uses regression to produce a specific value for each parameter. Compound-token methods also place several symbolic attributes at one sequence position. MusicBERT uses OctupleMIDI for masked-token pretraining, and MIDI2Score converts performance MIDI to a score. These works establish important precedents for note-level prediction and compound representations. INSPIRE studies aligned, score-conditioned generation of expressive performance MIDI, where the model predicts distributions over timing, velocity, and pedal controls.

ScorePerformer is a close note-level precedent; CINR differs in its typed MLP-based numerical inputs, per-attribute distribution heads, and explicit score-relative log-ratio target.

We will revise the paper to describe our contribution more precisely. For CINR, the main design has three components:

1. **Typed attribute representation and embedding.** We identify each score and performance attribute as categorical, continuous, or binary. Separate encoders preserve these types and fuse them into one vector per aligned note. In CINR, numerical timing, duration, and velocity values enter through continuous MLP encoders rather than token embeddings.
2. **Attribute-matched output parameterizations and distribution modeling.** CINR predicts IOI, duration, and velocity with separate single-component discretized-logistic distributions. Pedal is modeled with four binary logits. These heads provide a distribution for each expressive numerical attribute.
3. **Score-relative log-timing targets.** We predict timing changes relative to the score on a log scale. This gives a compact target for expressive timing. The absolute-timing ablation shows substantially worse timing metrics.

Our empirical study evaluates these choices on ASAP. CINR performs better than DINR on most non-pedal metrics. Its note-level decoder also reduces autoregressive steps and inference cost relative to flat token decoders.

Together, these components form a task-specific design for aligned EPR. We will revise the abstract, introduction, and related work to present this scope and acknowledge prior note-level and compound-token work.

## Response to R1: configuration and result inconsistency

Thank you for identifying this inconsistency. The submitted labels were incorrect. The CINR PP HR of 2.38 in Table 2 came from `extra/rep_slot_sum`, which directly sums six 768-dimensional typed slot vectors. The documented `cinr/default_dlm_k1_bounded5` configuration uses slot+MLP fusion. The DINR row came from `dinr/rep_slot_direct`, whose configuration disables coordinates. The table description incorrectly called it coordinate-augmented. We will correct the configuration labels and method description.

We rechecked the archived checkpoints, prediction manifests, Wasserstein summaries, and score-span-normalized Human anchors. The main comparison used 19 test scores and three samples per score. We will identify each configuration in the main table and supplement. The table below reports the matched DINR representation-by-coordinate matrix and four CINR representation runs. For DINR, “coordinates” enables all four numerical coordinate inputs and tables. “w/o coordinates” disables them. All rows use the same split, sampling settings, and four-feature HR definition. Lower is better.

| Representation | PP IOI | PP Dur. | PP Vel. | PP Ped. | PP HR | PN IOI | PN Dur. | PN Vel. | PN Ped. | PN HR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DINR: slot128+mlp | 24.55 | 88.60 | 19.06 | 0.290 | 4.79 | 41.23 | 157.72 | 23.43 | 0.434 | 1.97 |
| DINR: slot128+mlp, coordinates | 24.30 | 73.20 | 21.82 | 0.236 | 4.64 | 41.95 | 148.84 | 24.42 | 0.412 | 1.95 |
| DINR: slot256+mlp | 20.72 | 58.79 | 23.65 | 0.244 | 4.46 | 39.53 | 142.83 | 25.83 | 0.423 | 1.94 |
| DINR: slot256+mlp, coordinates | 22.69 | 78.28 | 20.85 | 0.234 | 4.53 | 40.67 | 152.27 | 23.54 | 0.420 | 1.93 |
| DINR: slot-direct | 22.58 | 96.01 | 12.80 | 0.174 | 3.90 | 39.68 | 156.94 | 18.47 | 0.398 | 1.79 |
| DINR: slot-direct, coordinates | 24.04 | 92.92 | 10.01 | 0.220 | 3.85 | 41.33 | 155.95 | 16.87 | 0.411 | 1.78 |
| DINR: slot-sum | 20.88 | 102.99 | 8.53 | 0.245 | 3.78 | 37.82 | 160.82 | 16.35 | 0.449 | 1.79 |
| DINR: slot-sum, coordinates | **15.96** | **62.79** | 9.47 | 0.222 | **3.05** | **34.98** | **139.82** | 16.85 | 0.432 | **1.68** |
| CINR: slot-sum | **11.87** | **47.67** | **6.18** | 0.218 | **2.38** | **32.93** | **119.11** | **16.99** | 0.427 | **1.59** |
| CINR: slot-direct | 13.10 | 50.29 | 12.95 | 0.235 | 3.07 | 34.28 | 123.61 | 19.96 | 0.427 | 1.69 |
| CINR: slot+MLP | 11.48 | 49.38 | 8.88 | 0.225 | 2.62 | 32.17 | 117.88 | 18.54 | 0.425 | 1.61 |
| CINR: slot256+MLP | 12.59 | 51.03 | 10.27 | 0.242 | 2.86 | 33.43 | 121.17 | 18.70 | 0.428 | 1.64 |

The coordinate effect depends on the slot representation. For slot-sum, coordinates reduce PP HR from 3.78 to 3.05 and PN HR from 1.79 to 1.68. Table 2 reports CINR `extra/rep_slot_sum` and DINR `dinr/rep_slot_direct`. We will label both configurations explicitly. The CINR PP HR of 2.38 is the result of the slot-sum model used in the main comparison. It is 13.8% lower than Pianist Transformer's PP HR of 2.76. We will update the method description to match these configurations. The canonical slot+MLP model and other representation variants will remain identified separately in the configuration study.

## Response to R1W4: contribution of INR and relative timing targets

We thank the reviewer for pointing out that our comparison with Pianist Transformer does not isolate INR. The two systems differ in architecture, pretraining, target formulation, and output parameterization. We agree that the full quality difference cannot be attributed to INR alone. We will narrow the claim and distinguish evidence about timing targets from evidence about note-level decoding.

To measure the effect of the timing target, we compared three targets with the same CINR setup: the paper's log-ratio residual, a linear residual, and a raw ratio. All three used the same score-relative timing inputs, model configuration, 19 test scores, and three samples per score. We also include the serial-attribute CINR variant in the same table because it uses the paper's log-ratio target and tests a different decoder serialization. The results are:

| CINR variant / timing target | PP IOI | PP Dur. | PP Vel. | PP Ped. | PP HR | PN IOI | PN Dur. | PN Vel. | PN Ped. | PN HR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Log-ratio residual (paper) | 11.87 | 47.67 | 6.18 | 0.218 | **2.38** | 32.93 | 119.11 | 16.99 | 0.427 | **1.59** |
| Linear residual, $[-1000,1000]$ ms | 13.68 | 84.36 | 9.47 | 0.241 | **3.26** | 37.17 | 155.70 | 17.90 | 0.422 | **1.77** |
| Raw ratio, $[0.135,2.718]$ | 13.73 | 62.36 | 15.05 | 0.187 | **3.27** | 33.38 | 126.53 | 20.72 | 0.417 | **1.69** |
| Serial-attribute CINR, 4 tokens/note | 11.69 | 68.54 | 13.50 | 0.232 | 3.27 | 32.24 | 112.81 | 18.20 | 0.434 | 1.59 |

The log-ratio target gives the lowest aggregate HR and the lowest PP timing distances in this comparison. This supports the target choice within CINR; it does not establish that INR is the cause of CINR's advantage over Pianist Transformer. The raw-ratio target is undefined for notes with zero score IOI, so these notes use the separate support $[0,\exp(5)]$.

The earlier absolute-timing ablation makes the same point at a larger scale: PP IOI and duration increase from 11.87 and 47.67 with the log-ratio target to 272.30 and 480.38 with absolute timing. This shows that target formulation is a major factor in the observed timing results. It does not provide an INR-versus-token comparison, because both target settings were evaluated within the INR framework.

The serial-attribute row decodes four attribute tokens per note and uses the same log-relative timing target and CINR output distributions. It uses slot+MLP input fusion, whereas the main row uses slot-sum. We therefore treat it as evidence about the tested note-level versus serial implementations, rather than as a fully isolated estimate of decoder sequence length.

Taken together, these experiments show that the log-ratio target improves results within CINR and that the tested serial-attribute CINR does not match the main slot-sum CINR on PP HR. They do not establish a general quality advantage for INR over a token-based model using the same target and output distributions. We will revise the paper to make this scope explicit and avoid attributing the full comparison with Pianist Transformer to INR.

## Response to R2: matched comparison of relative timing targets

Our original ablation combined the relative target with the log transformation. We now compare three timing targets with the same CINR setup. They represent additive correction, multiplicative correction, and multiplicative correction in log coordinates:

* **Log-ratio residual (paper target):** the existing CINR timing target, with nonzero-IOI support $[-2,1]$.
* **Linear residual:** $z=(t_{\mathrm{perf}}-t_{\mathrm{score}})/1000$, with support $[-1,1]$, corresponding to an absolute residual of $[-1000,1000]\ \mathrm{ms}$.
* **Raw ratio:** $z=t_{\mathrm{perf}}/\max(t_{\mathrm{score}},1\,\mathrm{ms})$, with nonzero-score support $[0.135,2.718]$.

The raw ratio is undefined when score IOI is zero, as it is for some notes within a chord. We use a separate support of $[0,\exp(5)]$ for these notes.

The targets encode different assumptions. A linear residual gives the same weight to a 100-ms error for both short and long score intervals. A raw ratio represents proportional change, but doubling and halving are not equally spaced around one. The log-ratio treats reciprocal speedups and slowdowns symmetrically. A fixed multiplicative change produces a fixed change in log space. This comparison tests the log transform while keeping the score-relative target.

All three runs use 19 test scores and three generated samples per score. The resulting comparison is:

| CINR timing target | PP IOI | PP Dur. | PP Vel. | PP Ped. | PP HR | PN IOI | PN Dur. | PN Vel. | PN Ped. | PN HR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Log-ratio residual (paper) | 11.87 | 47.67 | 6.18 | 0.218 | **2.38** | 32.93 | 119.11 | 16.99 | 0.427 | **1.59** |
| Linear residual, $[-1000,1000]$ ms | 13.68 | 84.36 | 9.47 | 0.241 | **3.26** | 37.17 | 155.70 | 17.90 | 0.422 | **1.77** |
| Raw ratio, $[0.135,2.718]$ | 13.73 | 62.36 | 15.05 | 0.187 | **3.27** | 33.38 | 126.53 | 20.72 | 0.417 | **1.69** |

The log-ratio target has the lowest PP and PN HR, as well as the lowest PP/PN timing and velocity distances. The linear residual has a lower PP velocity distance than the raw ratio, but larger duration distances. The raw ratio improves several PN measures over the linear residual, but has higher velocity distances and higher aggregate HR than the log-ratio target. These results support the log-ratio target for this ASAP setting. Further comparisons are needed to assess other relative-timing formulations.

## Response to R2W1: positioning relative to prior work

We thank the reviewer for asking us to clarify how INR relates to Compound Word Transformer, ScorePerformer, and DExter. The table marks the specific design features relevant to our method. A check indicates that the feature is used; a cross indicates that it is not part of the method; a dash indicates that the feature does not apply to its generation setup.

| Method | Representation | Multi-attribute note tuples | One autoregressive output step per note | Explicit numerical structure | Hierarchical style latents | Stochastic generation | Continuous per-attribute output distributions | Score-relative timing | Log-ratio timing residual | Output parameterization |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---|
| Compound Word Transformer | Compound symbolic words | ✓ | ✓ | ✗ | ✗ | ✓ | ✗ | — | — | Categorical next-token probabilities |
| ScorePerformer | SPMuple: 8 score + 4 performance tokens per note | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✗ | Categorical-token NLL decoder; MMD-VAE style latents |
| DExter | Symbolic expressive-performance representation | — | — | — | ✗ | ✓ | ✗ | — | — | Conditional diffusion over performance features |
| INSPIRE (CINR) | Six role-aware typed slots per note | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ | ✓ | Continuous inputs; per-attribute discretized-logistic distributions |

**Compound Word Transformer.** It groups categorical symbolic fields into compound words for music generation. CINR applies note-wise grouping to aligned EPR and distinguishes categorical, continuous, and binary inputs. For timing, duration, and velocity, CINR uses continuous numerical inputs and predicts per-attribute discretized-logistic distributions.

**ScorePerformer.** It is the closest comparison. SPMuple groups score and performance fields into note tuples, and its decoder advances by note. CINR uses typed slot encoders, sends numerical timing, duration, and velocity values through continuous MLPs rather than token embeddings, and predicts them with separate discretized-logistic distributions rather than categorical logits. CINR also uses an explicit score-relative log-ratio target, while ScorePerformer uses discrete relative timing features and local tempo tokens. Both methods decode at the note level.

**DExter.** It generates score-conditioned expressive performances with a diffusion model. CINR uses autoregressive note-level decoding and attribute-specific output heads for numerical timing, duration, velocity, and binary pedal. The methods differ in both generation mechanism and output parameterization.

Together, these comparisons position CINR around three design choices: typed attribute representation and embedding, attribute-matched output distributions, and score-relative log-timing targets. The baseline results provide empirical context, while our ablations test selected CINR design choices. We will revise the related-work discussion to state these distinctions clearly.

## Response to R2: baseline coverage and inference efficiency

We thank the reviewer for suggesting SyMuPe/PianoFlow and ScorePerformer efficiency results. We evaluated the public PianoFlow model on the 19 test scores and generated three samples per score. We also recovered archived ScorePerformer metrics and timings, and evaluated a serial-attribute CINR variant under the paper's 16-epoch training budget. The table reports PP/PN Wasserstein distances, training GPU-hours where measured, and total inference GPU-minutes across workers. Lower metric values are better.

| Method | PP IOI | PP Dur. | PP Vel. | PP Ped. | PP HR | PN IOI | PN Dur. | PN Vel. | PN Ped. | PN HR | Train GPU-h | Infer GPU-min (19 × 3) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SyMuPe/PianoFlow-base | 10.91 | 58.36 | 5.19 | 0.527 | 3.37 | 27.31 | 139.21 | 14.04 | 0.527 | 1.62 | — | **3.56** |
| ScorePerformer† | 26.64 | 101.05 | 20.08 | 0.525 | 5.88 | 105.30 | 160.12 | 27.09 | 0.524 | 2.85 | — | **165.93** |
| CINR (paper, slot-sum) | 11.87 | 47.67 | 6.18 | 0.218 | **2.38** | 32.93 | 119.11 | 16.99 | 0.427 | **1.59** | — | **about 6.0** |
| CINR (serial-attribute, 4 tokens/note) | 11.69 | 68.54 | 13.50 | 0.232 | 3.27 | 32.24 | 112.81 | 18.20 | 0.434 | 1.59 | **1.57** | **about 30.4** |

PianoFlow took 213.59 seconds across two NVIDIA RTX 3090 workers, or 3.56 GPU-minutes. We score-span-normalized each of its three generated samples for all 19 test scores and evaluated them against score-span-normalized human references. Its outputs contain no CC64 pedal controls, so pedal is treated as off at all four note positions. This limitation accounts for its high pedal distance and affects its HR. ScorePerformer† quality values match the main-paper results. Its archived timings cover 38 predictions and total 110.62 GPU-minutes. The table scales this average to three samples per score, so 165.93 GPU-minutes is an estimate. The ScorePerformer paper reports 70,000 training iterations with batch size 128, but does not report accelerator type, wall-clock training time, or GPU-hours. CINR's approximately 6 GPU-minutes is the paper-reported cost under its original protocol. The serial-attribute CINR was trained for 16 epochs on two RTX 3090 GPUs; its measured training runtime was 2,829.8 seconds, or 1.57 GPU-hours. Its 57 sampled predictions took 15 minutes 12 seconds on two GPUs, or approximately 30.4 GPU-minutes. Training GPU-hours are unavailable for the pretrained baselines and archived paper CINR run, so these entries are marked —.

All quality metrics in this table use score-span-normalized timing. The serial-attribute model matches the paper CINR on PN HR (1.59) but has a higher PP HR (3.27 vs. 2.38), so this sequence-compressed output variant does not improve the overall distributional match. We identify the different sample counts and timing protocols for ScorePerformer and CINR; the ScorePerformer inference cost is scaled from two to three samples per score.

## Response to R4: accessibility of the problem description

We thank the reviewer for noting that the problem description assumes too much music-specific knowledge. We will introduce EPR as a sequence-generation task, then define its musical attributes in simpler language. The following summary clarifies the notation in Table 1.

Expressive piano rendering (EPR) is a conditional sequence-generation task:

$$
p_{\theta}(Y\mid X)=\prod_{i=1}^{N}p_{\theta}(y_i\mid X,y_{<i}).
$$

Here, $X$ is a symbolic score and $Y$ is a performance of that score. The notes are aligned: $x_i$ and $y_i$ refer to the same note, and both sequences have length $N$. A score can have many valid performances, so the model predicts a distribution over possible performances. Figure 1 shows the score MIDI, a human performance, and the two note representations.

We represent the input score as

$$
X=\{x_i\}_{i=1}^{N},\qquad x_i=(P_i,I_i^s,D_i^s,V_i^s,M_i),
$$

Here, $P_i$ is pitch. $I_i^s$ and $D_i^s$ are score IOI and duration, and $V_i^s$ is score velocity. The superscript $s$ marks values from the mechanically rendered score MIDI. $M_i$ contains score-derived musical features: musical onset, musical duration, note length, and six binary annotations for hand/staff, trill, grace note, staccato, stem-up, and stem-down. The first three values are measured in quarter lengths. These features supply metrical and notation cues that are absent from MIDI note events.

The target performance sequence is

$$
Y=\{y_i\}_{i=1}^{N},\qquad y_i=(P_i,I_i^p,D_i^p,V_i^p,\mathrm{Ped}_i),
$$

Here, $I_i^p$, $D_i^p$, and $V_i^p$ are performed IOI, duration, and velocity. $\mathrm{Ped}_i$ contains four pedal states for the aligned note. The output retains score pitch and adds performed timing, dynamics, and pedal.

Table 1 gives each attribute's type, range, and number of modeled values. Pitch has 128 categorical values. IOI and duration are continuous values, log-mapped from 0–8000 ms to [0, 9] and represented with 256 bins. Velocity ranges from 0 to 127 and uses 128 values. Pedal has four binary snapshots, taken at 0%, 25%, 50%, and 75% of each note, for 16 possible patterns. The musical slot combines three categorical score features, each with 145 values, and six binary annotations, for 64 annotation patterns.

## Response to R4: why include DINR, and notation in its output head

DINR helps us study how to represent numerical attributes in EPR. A standard categorical table learns a separate embedding for each value. DINR can add a numerical coordinate to each embedding. This gives the discrete values information about their order and scale. In the slot-sum ablation, coordinates reduce PP HR from 3.78 to 3.05 and PN HR from 1.79 to 1.68. The effect varies with the slot representation. CINR models the same attributes with continuous distributions. In Table 2, it performs better on most non-pedal metrics. These results compare categorical values, coordinate-aware values, and continuous modeling. Both variants use BCE for pedal, so pedal results do not distinguish their numerical representations.

We agree that the DINR output-head notation needs clearer definitions. All DINR variants use this head. Coordinate augmentation changes the value prototypes. The head itself stays the same. At note step $i$, it computes logits as

$$
\begin{aligned}
\ell_i^a &= H_a(h_i)C_T^\top+b_a,\qquad a\in\{I,D\},\\
\ell_i^V &= H_V(h_i)C_V^\top+b_V,
\end{aligned}
\qquad
p_i^a=\operatorname{softmax}(\ell_i^a),\quad a\in\{I,D,V\}.
$$

Here, $h_i$ is the decoder hidden-state vector. The learned MLPs $H_I$, $H_D$, and $H_V$ map it to separate query vectors. $C_T$ is the timing value table shared by IOI and duration. $C_V$ is the velocity value table. Each row represents one candidate value. The query scores each row, and $b_I$, $b_D$, and $b_V$ add class-specific biases.

During training, cross-entropy takes the logits and target bin as input. It applies log-softmax internally. The loss is

$$
\mathcal{L}_{\mathrm{DINR}}=\sum_{a\in\{I,D,V\}}\mathrm{CE}(\ell_i^a,q_a(y_i^a)) + \mathrm{BCE}(\hat p_i,p_i).
$$

At inference, softmax converts each logit vector into a categorical distribution for sampling. The four pedal states use BCE. We will add these definitions next to the DINR equations in the paper.
