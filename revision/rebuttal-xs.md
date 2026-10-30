## Response to reviewer A: configuration, coherence, novelty, and target attribution

### A1: Configurations
Table 2's CINR PP HR=2.38 is slot-sum (sum of six 768-d typed slots), not slot+MLP. Main text uses slot-sum for both CINR/DINR; it performs best overall among tested encodings. The supplement's DINR table note was unclear; below are its 8 configurations across 4 representations, each with coordinates off/on.

|Repr.|Coord.|PP HR|PP IOI|PP Dur.|PN HR|PN IOI|PN Dur.|
|---|:---:|---:|---:|---:|---:|---:|---:|
|slot128+mlp|✗|4.79|24.55|88.60|1.97|41.23|157.72|
||✓|4.64|24.30|73.20|1.95|41.95|148.84|
|slot256+mlp|✗|4.46|20.72|58.79|1.94|39.53|142.83|
||✓|4.53|22.69|78.28|1.93|40.67|152.27|
|slot-direct|✗|3.90|22.58|96.01|1.79|39.68|156.94|
||✓|3.85|24.04|92.92|1.78|41.33|155.95|
|slot-sum|✗|3.78|20.88|102.99|1.79|37.82|160.82|
||✓|3.05|15.96|62.79|1.68|34.98|139.82|

For DINR slot-sum, coordinates reduce PP/PN HR 3.78/1.79→3.05/1.68. Table reports all six PP/PN metrics; we will identify the DINR setting.

### A2: Beat-level coherence
PP Wasserstein is order-invariant; PN is note-wise; neither measures temporal coherence. For tempo $T$, mean beat velocity $V$, and human performance $h$, beat-level Pearson correlations are:
$$r_T^{(s,x,h)}=\operatorname{Corr}_{b\in\mathcal B_s}(T_{s,b}^{x},T_{s,b}^{h}),\quad r_V^{(s,x,h)}=\operatorname{Corr}_{b\in\mathcal B_s}(V_{s,b}^{x},V_{s,b}^{h}).$$
Macro-average over human performances, generated samples, and 19 test scores.

|Method|Tempo Pearson|Velocity Pearson|
|---|---:|---:|
|Human|0.303|0.760|
|PT baseline|0.120|0.505|
|DINR|0.119|0.285|
|CINR|0.135|0.341|
|DExter|0.003|0.220|
|VirtuosoNet|0.020|0.084|
|ScorePerformer|0.006|0.434|

CINR velocity correlation is modest despite low velocity Wasserstein (PP=6.18; PN=16.99). We will separate distributional fidelity from coherence and narrow expressiveness claims; listening plan: C2.

### A3: Novelty
See B1. We will claim a combination for aligned EPR, not the first one-step-per-note design.

### A4: Timing targets and attribution
Pianist Transformer differs in architecture, pretraining, target, and output parameterization; its full quality gap cannot be attributed to INR. Same-CINR timing-target results:

|CINR variant/target|PP HR|PP IOI|PP Dur.|PN HR|PN IOI|PN Dur.|
|---|---:|---:|---:|---:|---:|---:|
|Log-ratio residual (paper)|2.38|11.87|47.67|1.59|32.93|119.11|
|Linear residual, $[-1000,1000]$ ms|3.26|13.68|84.36|1.77|37.17|155.70|
|Raw ratio, $[0.135,2.718]$|3.27|13.73|62.36|1.69|33.38|126.53|
|Serial-attribute CINR, 4 tokens/note|3.27|11.69|68.54|1.59|32.24|112.81|

Log-ratio gives lowest aggregate HR and PP timing distances within CINR. Absolute timing raises PP IOI/duration 11.87/47.67→272.30/480.38: evidence for target formulation, not isolated INR; we will narrow attribution. Serial row uses 4 attribute tokens/note and slot+MLP; main row uses slot-sum, so this is not a fully matched sequence-compression estimate.

Targets: log-ratio residual, paper target, nonzero-IOI support $[-2,1]$; linear $z=(t_{perf}-t_{score})/1000$, support $[-1,1]$; raw ratio $z=t_{perf}/\max(t_{score},1\,ms)$, nonzero-score support $[0.135,2.718]$. At zero score IOI raw ratio is undefined; support $[0,\exp(5)]$. Linear residual equally weights absolute error across score intervals; raw ratio is proportional but not symmetric for doubling/halving; log-ratio makes reciprocal speedups/slowdowns symmetric.

## Response to reviewer B: prior work, baselines, and ablations

### B1: Prior work and contribution
INSPIRE combines (1) typed note representation: separate encoders preserve categorical/numerical/binary types of aligned score/performance attributes and fuse one vector/note; (2) attribute-matched outputs: CINR predicts numerical timing/duration/velocity with separate discretized-logistic distributions and binary pedal; DINR is categorical alternative; (3) score-relative log timing: log-ratio residual gives reciprocal speedups/slowdowns symmetric target scales.

|Method|Representation|Explicit numerical structure|Continuous per-attribute output distributions|Score-relative timing|Log-ratio timing residual|Output parameterization|
|---|---|:---:|:---:|:---:|:---:|---|
|Compound Word Transformer|Compound symbolic words|✗|✗|—|—|Categorical next-token probabilities|
|ScorePerformer|SPMuple: 8 score+4 performance tokens/note|✓|✗|✓|✗|Categorical-token NLL decoder; MMD-VAE style latents|
|DExter|Symbolic expressive-performance representation|—|✗|—|—|Conditional diffusion over performance features|
|INSPIRE (CINR)|Six role-aware typed slots/note|✓|✓|✓|✓|Continuous inputs; per-attribute discretized-logistic distributions|

Compound Word Transformer groups categorical symbolic fields. Closest baseline, ScorePerformer: SPMuple groups score/performance fields and decodes by note; CINR uses typed slot encoders, continuous numerical MLP inputs, separate discretized-logistic outputs, and explicit score-relative log-ratio target. DExter uses conditional diffusion; CINR autoregressive note-level decoding and attribute-specific heads. We will clarify these distinctions in related work.

### B2: Baselines and cost
Public PianoFlow evaluated on all 19 test scores, 3 samples/score; archived ScorePerformer metrics/timings recovered; serial-attribute CINR evaluated within 16-epoch budget.

|Method|PP HR|PP IOI|PP Dur.|PN HR|PN IOI|PN Dur.|Train GPU-h|Infer GPU-min (19x3)|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|SyMuPe/PianoFlow-base|3.37|20.91|58.36|1.62|37.31|139.21|—|3.56|
|ScorePerformer†|5.88|26.64|101.05|2.85|105.30|160.12|—|165.93|
|CINR (paper, slot-sum)|2.38|11.87|47.67|1.59|32.93|119.11|0.62|6.0|
|CINR (serial, 4 tokens/note)|3.27|11.69|68.54|1.59|32.24|112.81|1.57|30.4|

PianoFlow: 213.59 s on 2 RTX 3090 workers=3.56 GPU-min; no CC64, pedal off. ScorePerformer archived timing: 38 predictions, 110.62 GPU-min; 165.93 scales to 3 samples/score. Its paper gives 70,000 iterations, batch 128, but no accelerator, wall time, or GPU-h. Serial CINR: 2,829.8 s=1.57 GPU-h training; 57 samples=15m12s on 2 GPUs=about 30.4 GPU-min. Training GPU-h unavailable for pretrained baselines and archived paper CINR. Protocol/count differences and limits will be stated.

### B3: Ablation scope
See A4. We will label timing-target ablation isolated and CINR/DINR comparison joint.

## Response to reviewer C: subjective evaluation and complementary metrics

### C1: Novelty
See B1. Contribution is task-specific formulation/evaluation, not a new Transformer architecture.

### C2: Subjective evaluation
Media supplement has matched score excerpts, human performances, and generated examples; no preliminary listening results. Plan: blinded study, 8 ASAP test scores; 30-s clips from CINR, DINR, Pianist Transformer, ScorePerformer, DExter, VirtuosoNet; same synthesizer, loudness normalization, excerpt boundaries; randomized order, hidden model IDs. 20 listeners rank naturalness; human and Score MIDI are labeled references, not ranked. For $K=6$, rank $r$ maps to normalized Borda $b=(K-r)/(K-1)$. Report participant-level 95% bootstrap intervals, repeated-measures Friedman test, Holm-corrected pairwise Wilcoxon tests. Claims limited to distributional fidelity; listening study is planned validation.

### C3: Gain scope
13.8% is PP-HR difference only (2.38 vs. 2.76), not uniform across metrics or evidence of perceptual superiority.

### C4: KL/JS
Recomputed PP KL/JS on 19 ASAP test scores from saved predictions and score-onset-span-normalized references. Timing: 256 bins, $[-2,1]$ for duration/nonzero-score IOI and $[0,5]$ for zero-score IOI; velocity: 128 bins, $[-0.5,127.5]$.

|Method|IOI KL|IOI JS|Dur. KL|Dur. JS|Vel. KL|Vel. JS|
|---|---:|---:|---:|---:|---:|---:|
|Human|0.148|0.024|0.139|0.029|0.093|0.019|
|Score MIDI|3.046|0.549|5.056|0.673|5.695|0.506|
|ScorePerformer|5.023|0.419|2.028|0.303|4.644|0.354|
|DExter|7.243|0.410|2.403|0.294|1.662|0.182|
|VirtuosoNet|6.473|0.493|2.479|0.320|0.739|0.178|
|Pianist Transformer|5.315|0.233|0.697|0.068|0.114|0.025|
|DINR (slot-sum)|1.006|0.202|0.502|0.104|1.084|0.155|
|CINR (slot-sum)|0.349|0.089|0.598|0.145|0.341|0.044|

CINR: lowest IOI KL/JS among learned systems and lower velocity KL/JS than external EPR baselines. DINR: lowest duration KL/JS. Pianist Transformer: lowest velocity KL/JS. Complementary beat correlations: A2.

## Response to reviewer D: readability, modeling, data scale, and compute

### D1: Readability and notation
We will recast EPR as conditional sequence generation and simplify music-specific wording. Let score $X$ and aligned performance $Y$ each have $N$ notes:
$$p_\theta(Y|X)=\prod_{i=1}^N p_\theta(y_i|X,y_{<i}).$$
Input:
$$X=\{x_i\}_{i=1}^N,\quad x_i=(P_i,I_i^s,D_i^s,V_i^s,M_i),$$
where $P_i$=pitch; $I_i^s,D_i^s,V_i^s$=score IOI, duration, velocity; $M_i$=musical onset, duration, note length, and 6 binary annotations (hand/staff, trill, grace note, staccato, stem-up, stem-down). Target:
$$Y=\{y_i\}_{i=1}^N,\quad y_i=(P_i,I_i^p,D_i^p,V_i^p,Ped_i).$$
Output copies aligned pitch and predicts timing, dynamics, 4 pedal states. Timing/duration: log-mapped 256 bins; velocity: 128 values; pedal: 4 binary snapshots (16 patterns). We will define symbols at first use; distinguish raw timing/log residuals; explain slot roles; clarify endpoint handling, pedal reconstruction, validation-window selection, and Fig. 4 split after protocol verification.

### D2: DINR
DINR is a categorical baseline for numerical representations. Slot-sum coordinates reduce PP/PN HR 3.78/1.79→3.05/1.68. CINR is better on most non-pedal metrics; DINR still compares categorical vs continuous output parameterizations; both use pedal BCE. All DINR variants use this head; coordinates change value prototypes, not head:
$$\ell_i^a=H_a(h_i)C_T^\top+b_a,\ a\in\{I,D\};\quad\ell_i^V=H_V(h_i)C_V^\top+b_V,$$
$$p_i^a=\operatorname{softmax}(\ell_i^a),\quad a\in\{I,D,V\}.$$
$h_i$=decoder state; $H_I,H_D,H_V$=query MLPs; $C_T$=shared IOI/duration value table; $C_V$=velocity table; rows=candidate values; $b_I,b_D,b_V$=class biases.

### D3: Modeling contribution
Transformer encoder-decoder and AR training are standard. Contribution: aligned-EPR formulation—typed attribute embedding, matched output distributions, score-relative log-timing. Relative-vs-absolute comparison isolates target; CINR/DINR jointly changes input encoding and output parameterization. We will clarify scope and not claim a new Transformer architecture.

### D4: Data and compute
ASAP is shared aligned benchmark in prior EPR comparisons. PianoCoRe-A combines transcription sources and fewer Disklavier recordings; distributions differ. Full PianoCoRe-A slot-sum INSPIRE training:

|Training setting|Works|Performances|Score notes|Performance hours|Window examples|Epochs|Training cost|
|---|---:|---:|---:|---:|---:|---:|---|
|PianoCoRe-A full|1,780|141,710|3,184,866|10,826.1|781,846|4|14.36 h wall; 28.7 GPU-h, 2×RTX 3090|
|ASAP benchmark|188|887|505,584|75.4|6,475|16|0.6 GPU-h (Table 2)|

4 PianoCoRe-A epochs=48,868 optimizer steps; 16 ASAP epochs≈1,632. CINR/PT timing: 2 RTX 3090, bfloat16, global batch 64, node-cache 16, 8 sampling workers, 19 scores×3 samples; GPU-min summed across workers. Public PianoFlow/archived ScorePerformer use separate protocols. Full-baseline PianoCoRe-A reproduction remains future work.

## Response to reviewer E (AI reviewer): stability, metrics, efficiency, and revisions

### E1: Stability
2.38, 2.54, 2.08 are different configurations, not repeated runs: Table 2=`extra/rep_slot_sum`; Table 3=canonical slot+MLP CINR (DLM k1, 5% bound, T5 8+4); Fig. 4=logistic slot+MLP. Seed variation on 19 scores×3 samples:

|Variation|Three runs|PP HR|PN HR|
|---|---|---:|---:|
|Training seed|Table 2 checkpoint+2 independent checkpoints; fixed sampling seed|2.374±0.072|1.578±0.020|
|Sampling seed|Table 2 samples+2 independent sampling seeds; fixed checkpoint|2.329±0.041|1.576±0.010|

Training-seed variation larger, especially PP HR; both remain near Table 2. Neither explains 2.54/2.08; these are separate ablation/sampling configurations.

### E2: Metric limits
See A2. PP/PN support distributional-fidelity claims only, not phrasing/perceptual quality.

### E3: Novelty
See B1. We acknowledge prior note-level/compound-token designs and claim the narrower combination.

### E4: Attribution
See A4. We will not attribute the full quality gap to INR without a matched token-decoder control.

### E5: Zero-score IOI
For $I_i^s=0$, $g(I_i^s)=0$ and
$$\Delta I_i=\log(\max(I_i^p,1\,ms)).$$
Both variants: separate zero-IOI support $[0,5]$, 256 values; inference $\hat I_i^p=\exp(\Delta I_i)\,ms$≈1–148 ms; nonzero support $[-2,1]$ does not apply. Test: 30,735/65,039 notes (47.3%) have zero score IOI; 612/197,891 aligned targets (0.31%) exceed $[0,5]$. CINR maps them to highest bin; DINR masks them from IOI CE. We will document handling/truncation.

### E6: Efficiency
See D4. Controlled sequence test: 1 step/note 6.0 GPU-min vs 4 steps 30.4 (5.1×). We will qualify end-to-end 20× claim; $K^2$ describes self-attention only.

### E7: Other revisions
Organize around 3 hypotheses: note integration reduces decoding cost; numerical parameterization supplies ordinal structure; score-relative timing simplifies target. Add role-by-slot table; clarify continuous inputs vs discretized likelihood outputs; finite-support parameters ($s_a^{min},s_a^{max}$, 5% bound, endpoint tails); score-pitch copying; pedal-to-CC64 reconstruction; validation-window selection; Fig. 4 split. Full-baseline PianoCoRe-A, second corpus, stratification by chord density/ornamentation/alignment quality, and joint timing/velocity/pedal output distribution remain valuable follow-up studies.
