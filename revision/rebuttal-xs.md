# Response to reviewer A: configuration, coherence, novelty, and target attribution

## A1: Configurations
Table 2's CINR PP HR=2.38 is slot-sum (sum of six 768-d typed slots), not slot+MLP. Main text uses slot-sum for both CINR/DINR; it performs best overall among tested encodings. The supplement's DINR table note was unclear; below are its 8 configurations across 4 representations, each with coordinates off/on.

|Representation|Coord.|PP HR|PP IOI|PN HR|PN IOI|
|---|:---:|---:|---:|---:|---:|
|slot128+mlp|✗|4.79|24.55|1.97|41.23|
||✓|4.64|24.30|1.95|41.95|
|slot256+mlp|✗|4.46|20.72|1.94|39.53|
||✓|4.53|22.69|1.93|40.67|
|slot-direct|✗|3.90|22.58|1.79|39.68|
||✓|3.85|24.04|1.78|41.33|
|slot-sum|✗|3.78|20.88|1.79|37.82|
||✓|3.05|15.96|1.68|34.98|

For DINR slot-sum, coordinates reduce PP/PN HR 3.78/1.79→3.05/1.68. Table reports PP/PN HR and IOI across all DINR configurations; we will identify the slot-sum setting.

## A2: Beat-level coherence
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
|VirtuosoNet|0.020|-0.084|
|ScorePerformer|0.006|0.434|

CINR velocity correlation is modest despite low velocity Wasserstein (PP=6.18; PN=16.99). We will separate distributional fidelity from coherence and narrow expressiveness claims; listening plan: C2.

## A3: See B1
## A4: Timing targets and attribution
Pianist Transformer differs in architecture, pretraining, target, and output parameterization; its full quality gap cannot be attributed to INR. Same-CINR timing-target results:

|CINR variant/target|PP HR|PP IOI|PN HR|PN IOI|
|---|---:|---:|---:|---:|
|Log-ratio residual (paper)|2.38|11.87|1.59|32.93|
|Linear residual, $[-1000,1000]$ ms|3.26|13.68|1.77|37.17|
|Raw ratio, $[0.135,2.718]$|3.27|13.73|1.69|33.38|
|Serial-attribute CINR, 4 tokens/note|3.27|11.69|1.59|32.24|

Log-ratio gives lowest aggregate HR and PP timing distances within CINR. Absolute timing raises PP IOI/duration 11.87/47.67→272.30/480.38: evidence for target formulation, not isolated INR; we will narrow attribution. Serial row uses 4 attribute tokens/note and slot+MLP; main row uses slot-sum, so this is not a fully matched sequence-compression estimate.

Targets: log-ratio residual, paper target, nonzero-IOI support $[-2,1]$; linear $z=(t_{perf}-t_{score})/1000$, support $[-1,1]$; raw ratio $z=t_{perf}/\max(t_{score},1\,ms)$, nonzero-score support $[0.135,2.718]$. At zero score IOI raw ratio is undefined; support $[0,\exp(5)]$. Linear residual equally weights absolute error across score intervals; raw ratio is proportional but not symmetric for doubling/halving; log-ratio makes reciprocal speedups/slowdowns symmetric.

# Response to reviewer B: prior work, baselines, and ablations

## B1: Prior work and contribution
INSPIRE combines (1) typed note representation: separate encoders preserve categorical/numerical/binary types of aligned score/performance attributes and fuse one vector/note; (2) attribute-matched outputs: CINR predicts numerical timing/duration/velocity with separate discretized-logistic distributions and binary pedal; DINR is categorical alternative; (3) score-relative log timing: log-ratio residual gives reciprocal speedups/slowdowns symmetric target scales.

- **Compound Word Transformer:** groups categorical symbolic fields; CINR targets aligned score-conditioned performance and models numerical controls with attribute-specific distributions.
- **ScorePerformer:** SPMuple groups 8 score+4 performance tokens/note and decodes by note; CINR uses typed slots, continuous numerical MLP inputs, separate discretized-logistic outputs, and log-ratio timing.
- **DExter:** conditional diffusion over expressive features; CINR autoregressive note decoding with attribute-specific heads.

We will clarify these distinctions in related work.

## B2: Baselines and cost
Public PianoFlow evaluated on all 19 test scores, 3 samples/score; archived ScorePerformer metrics/timings recovered; serial-attribute CINR evaluated within 16-epoch budget.

|Method|PP HR|PP IOI|PN HR|PN IOI|Train GPU-h|Infer GPU-min (19x3)|
|---|---:|---:|---:|---:|---:|---:|
|SyMuPe/PianoFlow-base|3.37|20.91|1.62|37.31|—|3.56|
|ScorePerformer†|5.88|26.64|2.85|105.30|—|165.93|
|CINR (paper, slot-sum)|2.38|11.87|1.59|32.93|0.62|6.0|
|CINR (serial, 4 tokens/note)|3.27|11.69|1.59|32.24|1.57|30.4|

PianoFlow: 213.59 s on 2 RTX 3090 workers=3.56 GPU-min; no CC64, pedal off. ScorePerformer archived timing: 38 predictions, 110.62 GPU-min; 165.93 scales to 3 samples/score. Its paper gives 70,000 iterations, batch 128, but no accelerator, wall time, or GPU-h. Serial CINR: 2,829.8 s=1.57 GPU-h training; 57 samples=15m12s on 2 GPUs=about 30.4 GPU-min. Training GPU-h unavailable for pretrained baselines and archived paper CINR. Protocol/count differences and limits will be stated.

## B3: See A4

# Response to reviewer C: subjective evaluation and complementary metrics

## C1: See B1
## C2: Subjective evaluation
Media supplement has matched score excerpts, human performances, and generated examples; no preliminary listening results. Plan: blinded study, 8 ASAP test scores; 30-s clips from CINR, DINR, Pianist Transformer, ScorePerformer, DExter, VirtuosoNet; same synthesizer, loudness normalization, excerpt boundaries; randomized order, hidden model IDs. 20 listeners rank naturalness; human and Score MIDI are labeled references, not ranked. For $K=6$, rank $r$ maps to normalized Borda $b=(K-r)/(K-1)$. Report participant-level 95% bootstrap intervals, repeated-measures Friedman test, Holm-corrected pairwise Wilcoxon tests. Claims limited to distributional fidelity; listening study is planned validation.

## C3: KL/JS
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

CINR: lowest IOI KL/JS among learned systems and lower velocity KL/JS than external EPR baselines. DINR: lowest duration KL/JS (0.502/0.104). Pianist Transformer: lowest velocity KL/JS. Complementary beat correlations: A2.

# Response to reviewer D: readability, modeling, data scale, and compute

## D1: Readability and notation
EPR maps a symbolic piano score $X$ to its aligned expressive performance $Y$. For $N$ aligned notes, the first equation states autoregressive conditional generation: each note output $y_i$ depends on the score and previous outputs $y_{<i}$. We will introduce EPR before this equation and define its terms at first use; the remaining notation and concepts will be defined in the paper.
$$p_\theta(Y\mid X)=\prod_{i=1}^{N}p_\theta(y_i\mid X,y_{<i}).$$

## D2: DINR
For the DINR head, $h_i$ is the decoder state; $H_I,H_D,H_V$ are query MLPs; $C_T$ is the shared IOI/duration value table; $C_V$ is the velocity value table; each table row represents a candidate value; $b_I,b_D,b_V$ are class biases. The logits and probabilities are:
$$\ell_i^a=H_a(h_i)C_T^\top+b_a,\quad a\in\{I,D\};\qquad \ell_i^V=H_V(h_i)C_V^\top+b_V,$$
$$p_i^a=\operatorname{softmax}(\ell_i^a),\quad a\in\{I,D,V\}.$$

## D3: Modeling contribution
Transformer encoder-decoder and AR training are standard. Contribution: aligned-EPR formulation—typed attribute embedding, matched output distributions, score-relative log-timing. Relative-vs-absolute comparison isolates target; CINR/DINR jointly changes input encoding and output parameterization. We will clarify scope and not claim a new Transformer architecture.

## D4: Data and compute
ASAP is shared aligned benchmark in prior EPR comparisons. PianoCoRe-A combines transcription sources and fewer Disklavier recordings; distributions differ. Full PianoCoRe-A slot-sum INSPIRE training:

|Training setting|Works|Performances|Score notes|Performance hours|Window examples|Epochs|Training cost|
|---|---:|---:|---:|---:|---:|---:|---|
|PianoCoRe-A|1,780|141,710|3,184,866|10,826.1|781,846|4|14.36 h wall; 28.7 GPU-h, 2×RTX 3090|
|ASAP|188|887|505,584|75.4|6,475|16|0.6 GPU-h (Table 2)|

4 PianoCoRe-A epochs=48,868 optimizer steps; 16 ASAP epochs≈1,632. CINR/PT timing: 2 RTX 3090, bfloat16, global batch 64, node-cache 16, 8 sampling workers, 19 scores×3 samples; GPU-min summed across workers. Full-baseline PianoCoRe-A reproduction remains future work.

# Response to reviewer E (AI reviewer): stability, metrics, efficiency, and revisions

## E1: Stability
2.38, 2.54, 2.08 are different configurations, not repeated runs. Seed variation (3 independent seed) on 19 scores×3 samples:

|Variation|PP HR|PN HR|
|---|---:|---:|
|training seed|2.374±0.072|1.578±0.020|
|sampling seed|2.329±0.041|1.576±0.010|

Training-seed variation larger, especially PP HR; both remain near Table 2. Neither explains 2.54/2.08; these are separate ablation/sampling configurations.

## E2: See A2
## E3: See B1
## E4: See A4
## E5: Zero-score IOI
For $I_i^s=0$, $g(I_i^s)=0$ and
$$\Delta I_i=\log(\max(I_i^p,1\,ms)).$$
Both variants: separate zero-IOI support $[0,5]$, 256 values; inference $\hat I_i^p=\exp(\Delta I_i)\,ms$≈1–148 ms; nonzero support $[-2,1]$ does not apply. Test: 30,735/65,039 notes (47.3%) have zero score IOI; 612/197,891 aligned targets (0.31%) exceed $[0,5]$. CINR maps them to highest bin; DINR masks them from IOI CE. We will document handling/truncation.

## E6: See D4
## E7: Other revisions
Organize around 3 hypotheses: note integration reduces decoding cost; numerical parameterization supplies ordinal structure; score-relative timing simplifies target. Add role-by-slot table; clarify continuous inputs vs discretized likelihood outputs; finite-support parameters ($s_a^{min},s_a^{max}$, 5% bound, endpoint tails); score-pitch copying; pedal-to-CC64 reconstruction; validation-window selection; Fig. 4 split. Full-baseline PianoCoRe-A, second corpus, stratification by chord density/ornamentation/alignment quality, and joint timing/velocity/pedal output distribution remain valuable follow-up studies.
