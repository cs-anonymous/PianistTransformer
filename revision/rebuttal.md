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

The matched results show that the coordinate effect depends on representation: for slot-sum, adding coordinates improves PP HR from 3.78 to 3.05 and PN HR from 1.79 to 1.68, while the best overall DINR result is slot-sum with coordinates. We will use this as the corrected DINR configuration. For CINR, slot-sum has the best aggregate PP and PN HR among the four tested representations. The corrected CINR PP HR remains 2.38, a 13.8% reduction from the Pianist Transformer PP HR of 2.76. The table makes the model-to-result mapping and the representation comparisons explicit.
