## Response to R1: temporal coherence

We agree that PP/PN Wasserstein distances measure distributional fidelity but do not directly measure temporal coherence: PP is invariant to note-order permutations, and PN evaluates notes independently.

We therefore computed beat-level Pearson correlations. For score $s$, beat $b$, and performance $x$, let $T_{s,b}^{x}$ be the beat-period tempo and $V_{s,b}^{x}$ the mean velocity of notes in that beat. For each generated sample and each human performer $h$, we compute

$$
r_T^{(s,x,h)}=\operatorname{Corr}_{b\in\mathcal B_s}(T_{s,b}^{x},T_{s,b}^{h}),
\qquad
r_V^{(s,x,h)}=\operatorname{Corr}_{b\in\mathcal B_s}(V_{s,b}^{x},V_{s,b}^{h}).
$$

The results are macro-averaged over human performers, generated samples, and the 19 test scores. CINR and DINR use the strongest already-evaluated configuration separately for each column: CINR uses bounded_2p5pct for Tempo and dist_ln_k3 for Velocity; DINR uses rep_slot256_mlp for Tempo and rep_slot_sum for Velocity.

| Method | Tempo Pearson | Velocity Pearson |
|---|---:|---:|
| Human ceiling | 0.303 | 0.760 |
| PT baseline | 0.120 | 0.505 |
| DINR | 0.119 | 0.085 |
| CINR | 0.135 | 0.141 |
| DExter | 0.003 | 0.220 |
| VirtuosoNet | 0.020 | -0.084 |
| ScorePerformer | 0.006 | 0.434 |

CINR still has a low velocity Wasserstein distance (PP = 6.18; PN = 16.99), but its beat-level velocity correlation is modest. This shows that Wasserstein distance and temporal correlation capture different properties. We will revise the manuscript to distinguish distributional fidelity from temporal coherence and will avoid interpreting lower Wasserstein distance as direct evidence of better musical expressiveness. The current velocity objective does not fully model long-range dynamic structure; a controlled listening study remains important future work.

## Response to R1: novelty and relation to prior note-level methods

We thank the reviewer for pointing out these precedents. We agree that one step per note is not, by itself, a novel contribution. VirtuosoNet already predicts expressive parameters note by note. Its regression outputs specific parameter values, rather than a distribution over possible performances. Compound-token methods also combine several symbolic attributes at one sequence position. The cited compound-token works, however, do not use this representation to generate aligned performance MIDI from a score. MusicBERT uses OctupleMIDI for masked-token pretraining, while MIDI2Score converts performance MIDI to a score.

We will revise the paper to state the following contributions. The first three describe the modeling design. The last summarizes the empirical evidence.

1. **A typed note representation for aligned EPR.** We organize score and performance attributes by type: categorical, continuous, or binary. Separate encoders preserve these types, then fuse them into one vector per aligned note. We do not claim that one vector per note is new. Our contribution is applying typed fusion to aligned EPR, where score inputs and performance feedback must be represented together.
2. **Score-relative log-timing targets.** We predict timing changes relative to the score on a log scale. This gives a compact target for expressive timing. Our absolute-timing ablation shows that this target choice substantially improves timing metrics.
3. **Attribute-matched output parameterizations.** DINR tests categorical value tables, including numerical-coordinate augmentation. CINR models IOI, duration, and velocity with conditional distributions. In the canonical CINR setting, these are single-component discretized logistic distributions, with robust bounds on timing scales. Pedal remains a binary prediction task. Unlike point-value regression, these output heads represent uncertainty over expressive controls.
4. **An empirical quality--efficiency study.** On ASAP, we compare the discrete and continuous variants with prior renderers and human references. The results show that coordinate information can improve discrete value representations, while CINR gives stronger results on most non-pedal metrics. The note-level decoder also reduces autoregressive steps and inference cost. These are findings from our evaluation, not claims that the underlying components are individually new.

Thus, our contribution is not the first note-level model or the first compound-token representation. It is a task-specific EPR design that combines typed note fusion, score-relative timing, and output parameterizations matched to expressive controls, together with an empirical study of quality and efficiency. We will revise the abstract, introduction, and related work to make this scope clear.

## Response to R1: configuration and result inconsistency

Thank you for identifying this inconsistency. The reviewer is correct that the submitted labels were wrong. The CINR PP HR of 2.38 in Table 2 came from extra/rep_slot_sum (six 768-dimensional typed slot vectors added directly), not the documented cinr/default_dlm_k1_bounded5 slot+MLP run. The original DINR row came from dinr/rep_slot_direct, whose configuration had coordinates disabled, despite the table describing it as coordinate-augmented. We had mismatched the result rows with the model descriptions and will correct those references.

We rechecked the archived checkpoints, prediction manifests, raw Wasserstein summaries, and the score-span-normalized Human anchors. The main comparison used 19 test scores and three samples per score. We will revise the representation descriptions and make the selected configurations explicit in both the main table and the supplement. The table below combines the new matched DINR representation-by-coordinate matrix with the four CINR representation runs. For DINR, “coordinates” enables all four numerical coordinate inputs/tables; “w/o coordinates” disables all four. All rows use the same split, sampling settings, and four-feature HR definition. Lower is better.

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

The matched results show that the coordinate effect depends on representation: for slot-sum, adding coordinates improves PP HR from 3.78 to 3.05 and PN HR from 1.79 to 1.68. We clarify that the reported CINR result in Table 2 is `extra/rep_slot_sum` (the six 768-dimensional typed slot vectors are summed directly), and the reported DINR result is `dinr/rep_slot_direct`; the submitted prose did not identify these configurations correctly. The 2.38 CINR PP HR is therefore the result of the slot-sum model actually evaluated in the main comparison, corresponding to a 13.8% reduction relative to Pianist Transformer's PP HR of 2.76. We will make the mapping from each reported row to its representation and coordinate setting explicit in the paper and supplement, and revise the method description so it does not imply that the main-table result used the slot+MLP configuration. The canonical slot+MLP and other representation variants remain separately reported in the configuration study.

## Response to R2: matched comparison of relative timing targets

We agree that the original experiments did not isolate the log transformation itself. The purpose of this ablation is to separate three assumptions about the relationship between score time and performance time: additive correction, multiplicative correction, and multiplicative correction represented in log coordinates. Using the paper's CINR configuration, we compare:

* **Log-ratio residual (paper target):** the existing CINR timing target, with nonzero-IOI support $[-2,1]$.
* **Linear residual:** $z=(t_{\mathrm{perf}}-t_{\mathrm{score}})/1000$, with support $[-1,1]$, corresponding to an absolute residual of $[-1000,1000]\ \mathrm{ms}$.
* **Raw ratio:** $z=t_{\mathrm{perf}}/\max(t_{\mathrm{score}},1\,\mathrm{ms})$, with nonzero-score support $[0.135,2.718]$.

For raw ratios, score IOI can be zero for chord-internal notes, making the ratio undefined. We use a separate zero-score support of $[0,\exp(5)]$ for those notes.

These targets test different inductive biases. The linear residual treats a 100-ms timing error as equally important regardless of whether the score interval is 100 ms or 1,000 ms. The raw ratio expresses proportional deviation directly but is asymmetric in the sense that doubling and halving are not equally spaced around one. The log-ratio makes reciprocal slowdowns and speedups symmetric: multiplying the performance time by a fixed factor adds a fixed amount to the target, while the score-dependent scale is removed. The comparison therefore tests the log operation itself while keeping the underlying score-relative formulation.

All three runs use 19 test scores and three generated samples per score. The resulting comparison is:

| CINR timing target | PP IOI | PP Dur. | PP Vel. | PP Ped. | PP HR | PN IOI | PN Dur. | PN Vel. | PN Ped. | PN HR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Log-ratio residual (paper) | 11.87 | 47.67 | 6.18 | 0.218 | **2.38** | 32.93 | 119.11 | 16.99 | 0.427 | **1.59** |
| Linear residual, $[-1000,1000]$ ms | 13.68 | 84.36 | 9.47 | 0.241 | **3.26** | 37.17 | 155.70 | 17.90 | 0.422 | **1.77** |
| Raw ratio, $[0.135,2.718]$ | 13.73 | 62.36 | 15.05 | 0.187 | **3.27** | 33.38 | 126.53 | 20.72 | 0.417 | **1.69** |

The archived log-ratio target remains best overall: it has the lowest PP HR and PN HR, and the lowest PP/PN timing and velocity distances. The linear residual has a lower PP velocity distance than the raw ratio, but its duration distances are substantially larger. The raw ratio is closer on PN timing, duration, and pedal than the linear residual, yet its velocity distances are higher and its aggregate HR remains above the log-ratio result. Thus, neither alternative improves the original log-ratio target. The result supports the log-ratio target for this aligned ASAP setting, while the limited scope of the ablation does not establish universal superiority over every relative-timing representation.

## Response to R2: baseline coverage and inference efficiency

We thank the reviewer for pointing out the missing continuous baseline and efficiency comparison. We evaluated the publicly available SyMuPe/PianoFlow model on the same 19 test scores, generating three samples per score (57 MIDI files total). We also recovered the archived ScorePerformer quality metrics and inference timings. The table uses the paper's full PP/PN Wasserstein layout; lower is better. The final column reports the sum of elapsed inference time across GPUs, so it is GPU-minutes rather than parallel wall-clock time.

| Method | PP IOI | PP Dur. | PP Vel. | PP Ped. | PP HR | PN IOI | PN Dur. | PN Vel. | PN Ped. | PN HR | Infer GPU-min (19 × 3) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SyMuPe/PianoFlow-base | 85.86 | 241.41 | 5.19 | 0.527 | 9.34 | 94.66 | 298.51 | 14.04 | 0.527 | 2.89 | **3.56** |
| ScorePerformer† | 26.64 | 202.10 | 20.08 | 0.525 | 7.12 | 105.30 | 320.25 | 27.09 | 0.524 | 3.39 | **165.93** |
| CINR (paper, slot-sum) | 11.87 | 47.67 | 6.18 | 0.218 | **2.38** | 32.93 | 119.11 | 16.99 | 0.427 | **1.59** | **about 6.0** |

PianoFlow's measured total is 213.59 seconds across the two NVIDIA RTX 3090 workers, or 3.56 GPU-minutes. Its PP/PN values are computed from the generated MIDI's own tick-to-time conversion, using the tempo encoded in each output file and the same binary-pedal evaluator; no score tempo map is applied. ScorePerformer† quality metrics come from the archived score-span evaluation (19 scores, two generated samples per score). Its archived timings cover 38 predictions and total 110.62 GPU-minutes; the 165.93 GPU-minute value scales the observed per-sample average to three samples per score, so this cost is an estimate. CINR's approximately 6 GPU-minutes is the paper-reported cost under its original evaluation protocol.

The PianoFlow row has an evaluation limitation: its generated MIDI files contain no CC64 pedal controls, so predicted pedal is zero under the binary-pedal convention. The row is therefore useful for the requested efficiency comparison and as an absolute-time audit, but its HR should be interpreted with this pedal limitation. The ScorePerformer and CINR rows retain their archived evaluation protocols; sample count and protocol differences are stated explicitly rather than hidden in the table.

## Response to R4: accessibility of the problem description

We thank the reviewer for highlighting this. The paper defines the relevant concepts. We agree that we have not connected them clearly enough to the general sequence-learning view, and that some sentences are too complex. We will simplify the wording and present the task from this general perspective before introducing the music-specific features. Below, we restate the problem formulation through the feature schema in Table 1.

Expressive piano rendering (EPR) is a conditional sequence-generation task:

$$
p_{\theta}(Y\mid X)=\prod_{i=1}^{N}p_{\theta}(y_i\mid X,y_{<i}).
$$

Here, $X$ is a symbolic score and $Y$ is a performance of the same notes. The two sequences are aligned note by note: $x_i$ and $y_i$ refer to the same note, and both sequences follow the same note order and have length $N$. A score can have multiple valid performances, so the model predicts a distribution over possible $Y$ given $X$. Figure 1 illustrates this mapping: a mechanically rendered score MIDI provides the score-side note sequence, while a human performance MIDI provides one expressive realization; the figure also contrasts token-based note encoding with our integrated note-level representation.

We represent the input score as

$$
X=\{x_i\}_{i=1}^{N},\qquad x_i=(P_i,I_i^s,D_i^s,V_i^s,M_i),
$$

where $P_i$ is pitch, $I_i^s$ and $D_i^s$ are the score inter-onset interval (IOI) and duration, $V_i^s$ is score velocity, and $M_i=(O_i^m,D_i^m,L_i^m,A_i^m)$ contains score-derived musical features. The superscript $s$ denotes quantities extracted from mechanically rendered score MIDI. Following the score-feature design of MIDI2Score, $O_i^m$, $D_i^m$, and $L_i^m$ denote musical onset, musical duration, and note length in quarter-length units. $A_i^m$ contains six binary annotations: hand/staff, trill, grace note, staccato, stem-up, and stem-down. These features add metrical and notational information that is not explicit in the note events. We omit voice and accidental indicators because they do not provide additional information for performance expression.

The target performance sequence is

$$
Y=\{y_i\}_{i=1}^{N},\qquad y_i=(P_i,I_i^p,D_i^p,V_i^p,\mathrm{Ped}_i),
$$

where $I_i^p$, $D_i^p$, and $V_i^p$ are the performed IOI, duration, and velocity, and $\mathrm{Ped}_i$ gives the pedal state associated with the same aligned note. Thus, each output note retains the score pitch and adds the performed timing, dynamics, and pedal controls.

Table 1 summarizes the feature types, ranges, and representation sizes, where “Values” denotes the categorical support size or numerical discretization used by the model. Pitch is categorical with 128 values (0–127). Score and performed IOI and duration are continuous, span 0–8000 ms, and are log-mapped to [0, 9] with 256 modeled values each. Score and performed velocity are continuous over [0, 127] with 128 values. Pedal consists of four binary snapshots, sampled at 0%, 25%, 50%, and 75% of the note duration, giving 16 possible patterns. The musical slot contains musical onset, musical duration, and note length, each categorical over 0–6 quarter lengths with 145 values, and six binary annotations (hand/staff, trill, grace note, staccato, stem-up, and stem-down), giving 64 combinations. INR fuses these four score-derived musical features into one musical slot.

## Response to R4: why include DINR, and notation in its output head

DINR is included to study how numerical attributes should be represented in EPR. A standard categorical table learns a separate embedding for each value. DINR lets us test whether numerical coordinates give these discrete values a useful order and scale. In the slot-sum ablation, adding coordinates lowers PP HR from 3.78 to 3.05 and PN HR from 1.79 to 1.68. The effect varies with the slot representation. CINR then tests continuous modeling of the same numerical attributes. It performs better than DINR on all metrics in Table 2. Together, these results support a progression of inductive biases for EPR: typed attribute slots, coordinate-aware discrete values, and continuous numerical modeling. DINR is a diagnostic baseline for this comparison, not a claim that discrete modeling is better. Both variants use the same BCE head for pedal, so pedal results do not distinguish their numerical representations.

We agree that the notation for the DINR output head was not adequately defined. This head is used by all DINR variants. Coordinate augmentation changes the value prototypes, not the form of the head. For note step $i$, the head computes logits as

$$
\begin{aligned}
\ell_i^a &= H_a(h_i)C_T^\top+b_a,\qquad a\in\{I,D\},\\
\ell_i^V &= H_V(h_i)C_V^\top+b_V,
\end{aligned}
\qquad
p_i^a=\operatorname{softmax}(\ell_i^a),\quad a\in\{I,D,V\}.
$$

Here, $h_i$ is the decoder hidden-state vector. $H_I$, $H_D$, and $H_V$ are learned MLPs. Each maps $h_i$ to a query vector. $C_T$ is the timing value table shared by the IOI and duration heads. $C_V$ is the separate velocity value table. Each row represents one candidate value. The query scores each row, and the bias vectors $b_I$, $b_D$, and $b_V$ adjust the logits. During training, categorical cross-entropy takes the logits and target bin as input. It applies log-softmax internally, so the loss is

$$
\mathcal{L}_{\mathrm{DINR}}=\sum_{a\in\{I,D,V\}}\mathrm{CE}(\ell_i^a,q_a(y_i^a)) + \mathrm{BCE}(\hat p_i,p_i).
$$

At inference, softmax converts each logit vector into a categorical distribution for sampling. The four pedal states use BCE during training. We will add these definitions next to the DINR equations in the paper.
