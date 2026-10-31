# Response to reviewer A: configuration, coherence, novelty, and target attribution

## A1: Configurations
The paper's configuration labels are incorrect: both reported CINR and DINR results use slot-sum (six 768-d typed slots). The supplement's DINR note was unclear; below are all 6 configurations.

|Representation|Coord.|PP HR|PP IOI|PN HR|PN IOI|
|---|:---:|---:|---:|---:|---:|
|slot128+mlp|✗|4.79|24.55|1.97|41.23|
||✓|4.64|24.30|1.95|41.95|
|slot-direct|✗|3.90|22.58|1.79|39.68|
||✓|3.85|24.04|1.78|41.33|
|slot-sum|✗|3.78|20.88|1.79|37.82|
||✓|3.05|15.96|1.68|34.98|

Among tested DINR encodings, slot-sum performs best overall; adding coordinates improves all three reported metrics. We will correct the paper's configuration labels.

## A2: Beat-level coherence
To measure coherence beyond order-invariant PP and note-wise PN, we define beat-level Pearson correlations of tempo $T$ and mean beat velocity $V$ against human performance $h$:
$$r_T^{(s,x,h)}=\operatorname{Corr}_{b\in\mathcal B_s}(T_{s,b}^{x},T_{s,b}^{h}),\quad r_V^{(s,x,h)}=\operatorname{Corr}_{b\in\mathcal B_s}(V_{s,b}^{x},V_{s,b}^{h}).$$
Macro-average over human performances, generated samples, and 19 test scores.

|Method|Tempo Pearson|Velocity Pearson|
|---|---:|---:|
|Human|0.303|0.760|
|ScorePerformer|0.006|0.434|
|DExter|0.003|0.220|
|VirtuosoNet|0.020|0.084|
|PT|0.120|0.505|
|DINR|0.119|0.285|
|CINR|0.135|0.341|


The beat-level correlations are consistent with the paper's PN Wasserstein results. INSPIRE correlates reasonably on tempo (CINR 0.135; DINR 0.119), but less strongly on velocity (0.341; 0.285). We plan a subjective evaluation in C2.

## A3: contribution see B1
## A4: Timing targets and attribution
To clarify attribution, we ran a timing-target ablation with the same CINR setup:

|CINR variant/target|PP HR|PP IOI|PN HR|PN IOI|
|---|---:|---:|---:|---:|
|Log residual (paper)|2.38|11.87|1.59|32.93|
|Linear residual, $[-1000,1000]$ ms|3.26|13.68|1.77|37.17|
|Raw ratio, $[0.135,2.718]$|3.27|13.73|1.69|33.38|

Log residual yields the best aggregate HR and PP timing results; absolute timing raises PP IOI/duration from 11.87/47.67 to 272.30/480.38. This supports the log-residual target.

# Response to reviewer B: prior work, baselines, and ablations

## B1: Prior work and contribution
INSPIRE's contributions for aligned EPR:
1. **Typed note fusion:** separate encoders preserve categorical/numerical/binary score and performance attributes in one vector/note.
2. **Attribute-matched outputs:** CINR uses separate discretized-logistic distributions for numerical timing/duration/velocity and binary pedal; DINR provides categorical heads.
3. **Score-relative log residual:** makes reciprocal speedups/slowdowns symmetric.

Differences from prior work:
- **Compound Word Transformer:** groups categorical symbolic fields; CINR handles aligned score-conditioned performance with numerical attribute-specific distributions.
- **ScorePerformer:** SPMuple groups 8 score+4 performance tokens/note and decodes by note; CINR uses typed slots, continuous numerical MLP inputs, separate discretized-logistic outputs, and log-residual timing.
- **DExter:** conditional diffusion; CINR autoregressive note decoding with attribute-specific heads.

## B2: Baselines and cost
|Method|PP HR|PP IOI|PN HR|PN IOI|Train GPU-h|Infer GPU-min|
|---|---:|---:|---:|---:|---:|---:|
|SyMuPe/PianoFlow-base|3.37|20.91|1.62|37.31|—|7.12|
|ScorePerformer†|5.88|26.64|2.85|105.30|—|165.93|
|CINR (paper, slot-sum)|2.38|11.87|1.59|32.93|0.62|6.0|
|CINR (serial, 4 tokens/note)|3.27|11.69|1.59|32.24|1.57|30.4|

Inference GPU-min sums computation across both RTX 3090s. PianoFlow lacks CC64, so pedal is off; ScorePerformer timing is estimated by scaling its archived run to the shared sample count, and its GPU setup and training time are unreported. CINR achieves the best HR and inference efficiency in this comparison.

## B3: ablation see A4

# Response to reviewer C: subjective evaluation and complementary metrics

## C1: contribution see B1
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
EPR maps a symbolic piano score $X$ to its aligned expressive performance $Y$. For $N$ aligned notes, the first equation states autoregressive conditional generation: each note output $y_i$ depends on the score and previous outputs $y_{<i}$. The remaining notation and concepts are already defined in the main text. Our feature definitions follow MIDI2Score (Beyer and Dai, 2024); to improve readability, we will define the adopted symbols more explicitly and use shorter sentences.
$$p_\theta(Y\mid X)=\prod_{i=1}^{N}p_\theta(y_i\mid X,y_{<i}).$$

## D2: DINR
For the DINR head, $h_i$ is the decoder state; $H_I,H_D,H_V$ are query MLPs; $C_T$ is the shared IOI/duration value table; $C_V$ is the velocity value table; each table row represents a candidate value; $b_I,b_D,b_V$ are class biases. The logits and probabilities are:
$$\ell_i^a=H_a(h_i)C_T^\top+b_a,\quad a\in\{I,D\};\qquad \ell_i^V=H_V(h_i)C_V^\top+b_V,$$
$$p_i^a=\operatorname{softmax}(\ell_i^a),\quad a\in\{I,D,V\}.$$

## D3: contribution see B1

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

## E2: metric see A2
## E3: contribution see B1
## E4: ablation see A4
## E5: Zero-score IOI
For $I_i^s=0$, $g(I_i^s)=0$ and
$$\Delta I_i=\log(\max(I_i^p,1\,ms)).$$
Both variants: separate zero-IOI support $[0,5]$, 256 values; inference $\hat I_i^p=\exp(\Delta I_i)\,ms$≈1–148 ms; nonzero support $[-2,1]$ does not apply. Test: 30,735/65,039 notes (47.3%) have zero score IOI; 612/197,891 aligned targets (0.31%) exceed $[0,5]$. CINR maps them to highest bin; DINR masks them from IOI CE. We will document handling/truncation.

## E6: efficiency see D4
## E7: Other revisions
Organize around 3 hypotheses: note integration reduces decoding cost; numerical parameterization supplies ordinal structure; score-relative timing simplifies target. Add role-by-slot table; clarify continuous inputs vs discretized likelihood outputs; finite-support parameters ($s_a^{min},s_a^{max}$, 5% bound, endpoint tails); score-pitch copying; pedal-to-CC64 reconstruction; validation-window selection; Fig. 4 split. Full PianoCoRe-A, second corpus, stratification by chord density/ornamentation/alignment quality, and joint timing/velocity/pedal output distribution remain valuable follow-up studies.
