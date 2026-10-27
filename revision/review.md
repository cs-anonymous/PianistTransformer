## INSPIRE: INtegrated Score-to-Performance InterpRetation for Expressive Piano Rendering

### [Ye Sun](https://openreview.net/profile?id=~Ye_Sun6), [Lei Shi](https://openreview.net/profile?id=~Lei_Shi13)

20 Jul 2026 (modified: 21 Aug 2026)AAAI 2027 Conference SubmissionConference, Area Chairs, Senior Program Committee, Program Committee, Authors[Revisions](https://openreview.net/revisions?id=OrhTToE46Y)[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

**TL;DR:** INSPIRE represents each note as a single Transformer step, improving expressive timing and piece-level human relevance while reducing inference cost by over 20×.

**Abstract:**

Expressive piano rendering maps a symbolic score to a human-like performance by predicting note-level timing, dynamics, and pedal behavior. Existing symbolic models expand each score note into multiple autoregressive MIDI tokens. This mechanism increases sequence length and represents ordered controls such as inter-onset interval, duration, and velocity as categorical tokens, discarding their explicit ordinal and numerical structure. We propose INSPIRE, a Score-to-Performance Interpretation framework based on Integrated Note Representation (INR). Using discrete or continuous parameterizations, INR encodes the six typed attribute slots of each aligned note within a single Transformer timestep and autoregressively predicts its expressive attributes. On the ASAP benchmark, INSPIRE improves piece-level human relevance by 13.8% over the strongest existing method while retaining competitive note-level accuracy. It also reduces inference time by more than 20$\times$ while requiring substantially less training. Case studies show that INSPIRE preserves rhythmic organization while allowing local expressive timing.

**Primary Topic:** AUD: Music Information Retrieval & Generation

**Secondary Topics:** ML: Representation Learning

**Country of Institutions:** China

**Reproducibility Checklist:** [pdf](https://openreview.net/attachment?id=OrhTToE46Y&name=reproducibility_checklist "Download Reproducibility Checklist")

**Technical Supplement:** [pdf](https://openreview.net/attachment?id=OrhTToE46Y&name=technical_supplement "Download Technical Supplement")

**Media Supplement:** [zip](https://openreview.net/attachment?id=OrhTToE46Y&name=media_supplement "Download Media Supplement")

**Code and Data Supplement:** [zip](https://openreview.net/attachment?id=OrhTToE46Y&name=code_and_data_supplement "Download Code And Data Supplement")

**Submission Policies Acknowledgement:** I confirm that all authors have up-to-date OpenReview profiles, including their full publication name, current position, institution-affiliated email address, and DBLP URL if available. I understand that submissions with incomplete author profiles will be subject to desk rejection., I confirm that neither this manuscript nor a substantially similar version of it is currently under review at another archival venue., I confirm that any other relevant simultaneous submissions by any author of this manuscript are cited in this manuscript in anonymous form as "under review" work.

**Reciprocal Reviewer Nomination:** [Lei Shi](https://openreview.net/profile?id=~Lei_Shi13)

**Reciprocal Reviewer Nomination Confirmation:** We have nominated a qualified author (see above)

**Submission Number:** 24778

Filter by reply type...

Filter by author...

Sort: Newest FirstSort: Oldest First

_6 / 6 replies shown_

Add:

#### **Review for INSPIRE**

Official Reviewby Program Committee X8Vp24 Oct 2026, 01:34 (modified: 24 Oct 2026, 20:23)Program Chairs, Area Chairs, Senior Program Committee, Program Committee X8Vp, Authors, Program Committee Submitted[Revisions](https://openreview.net/revisions?id=FDhPiASFcD)

**Paper Summary:**

The paper proposes INSPIRE for expressive piano rendering. Its Integrated Note Representation (INR) encodes pitch, IOI, duration, velocity, pedal, and musical information as typed slots fused into one embedding per note, so each note takes a single Transformer timestep instead of K>=4 tokens. An encoder–decoder Transformer predicts score-relative log-timing deviations, velocity, and pedal, using either categorical heads (DINR) or bounded discretized-logistic heads (CINR). Trained only on ASAP, CINR is reported to improve piece-level Human Relevance over Pianist Transformer by 13.8% while reducing inference cost by more than 20x.

**Strengths:**

1. Well-motivated and simple design. Representing each aligned note as one timestep with typed slots and attribute-specific heads (Sec. 4, Fig. 3) is a natural fit for aligned score-to-performance rendering.

2. The score-relative log-timing residual is a sensible and effective target. The absolute-timing ablation (PP IOI/Dur. 272.30/480.38 vs. 11.87/47.67) shows this choice matters a lot.

3. Large efficiency gains. About 6 GPU-minutes of inference and 0.6 GPU-hours of training, compared with 137.3 GPU-minutes and >=100 GPU-hours for Pianist Transformer.


**Weaknesses:**

1. Inconsistency between the main results and the supplement (needs clarification). The Table 2 "CINR" numbers match extra/rep_slot_sum in Supplementary Table 7 (768-dim slots, sum fusion), whereas Sec. 4 and the code report describe slot+MLP (cinr/default_dlm_k1_bounded5) as the canonical model; the latter corresponds to PP HR ≈ 2.49 (~9.8% better than Pianist Transformer, rather than 13.8%). Similarly, the "DINR" row, described as coordinate-augmented, matches dinr/rep_slot_direct, whose config has coordinates disabled; the all-coordinate variant gives PP HR 4.63 vs. 4.69 without coordinates. I may be missing something, but the authors should clarify which configurations are reported, since this affects the headline gain and the coordinate-augmentation claim.

2. The metrics cannot measure musical coherence. PP Wasserstein is invariant to note order: randomly permuting the notes of a human performance leaves its PP distance unchanged. PN Wasserstein is computed independently per note. Neither captures phrasing or tempo/dynamic shaping, yet the main claims rest on them. In the authors' own case studies, VirtuosoNet (PP HR 4.32) often sounds more convincing to me than CINR (2.38). Metrics of temporal structure (e.g., correlation of beat-level tempo and velocity curves) or a small listening test are needed.

3. The novelty of "one timestep per note" is overstated. VirtuosoNet already predicts continuous per-note parameters with one step per note, and compound tokens (CP Word, OctupleMIDI, MIDI2Score) already fuse attributes into one step. The contribution is better described as a combination of typed slot fusion, score-relative log residuals, and bounded discretized-logistic heads.

4. The gains are confounded with the prediction target. The paper attributes its timing advantage to the score-relative residual, and absolute timing degrades results by >10x. Since the same target could be used by a token-based model, the comparison with Pianist Transformer does not isolate the contribution of INR.


**Justification of Recommendation:**

The note-level formulation is practical, the residual timing target is effective, and the efficiency gains are real. However, the Table 2 numbers appear inconsistent with the supplement, which needs clarification as it affects the headline gain and the coordinate-augmentation claim. In addition, the main metric is invariant to note order, so it cannot show that the generated performances are more musically coherent. Together with the overstated novelty and the confound with the residual target, I consider the paper marginally below the acceptance threshold in its current form.

I would rate the paper higher if the authors clarify the reported configurations, add metrics of temporal structure or a small listening test, and isolate INR from the residual timing target.

**Specific Points of Feedback for Rebuttal:**

1. Which configurations do the CINR and DINR rows in Table 2 correspond to? Please report Table 2 and the abstract numbers with the canonical configuration described in Sec. 4.
2. How do the methods compare on metrics of temporal structure, such as correlation of beat-level tempo and velocity curves with the human references?
3. Can the authors separate the contribution of INR from that of the residual target, e.g., with a token-based baseline trained on the same target?
4. How does INR differ from note-level models such as VirtuosoNet and from compound-token representations?

**Rating:** 5: Marginally below acceptance threshold

**Confidence:** 4: The reviewer is confident but not absolutely certain that the evaluation is correct

Add:

#### **Promising efficient rendering, but the representation’s novelty and contribution attribution need clearer evidence**

Official Reviewby Program Committee f3Yx18 Oct 2026, 22:50 (modified: 24 Oct 2026, 20:23)Program Chairs, Area Chairs, Senior Program Committee, Program Committee f3Yx, Authors, Program Committee Submitted[Revisions](https://openreview.net/revisions?id=dko5x1CLKh)

**Paper Summary:**

This paper presents INSPIRE, an expressive piano rendering framework that integrates note attributes into a single representation per Transformer step. It predicts score-relative timing deviations, velocity, and pedal states through attribute-specific heads, with discrete and continuous-parameter variants. Experiments on ASAP evaluate rendering quality, computational efficiency, and key design choices.

**Strengths:**

1. Practical and coherent design. Note-level decoding with typed encoders and attribute-specific heads addresses the computational cost of expressive piano rendering.
2. Promising empirical results. The reported improvements in timing metrics and inference speed suggest a useful quality–efficiency tradeoff.
3. Useful experimental analysis. Comparisons cover several modeling approaches, while ablations examine numerical encoding, timing targets, and architectural choices.

**Weaknesses:**

W1. Clearer positioning against prior work. The relationship to Compound Word Transformer, ScorePerformer, and DExter could be clarified, particularly regarding attribute grouping, numerical encoding, and performance parameterization. This would help readers identify INR’s specific contributions. W2. Incomplete baseline and efficiency comparisons. Including the publicly available SyMuPe/PianoFlow would strengthen comparisons with continuous representations. Reporting ScorePerformer’s inference time would also clarify the efficiency advantage over an existing compact note-tuple baseline. W3. Limited isolation of key design choices. The ablations are useful, but a matched serial-attribute versus note-level comparison would better isolate sequence compression. DINR and CINR also change both input encoding and output heads, leaving their individual contributions unclear. Finally, comparison with an existing relative-timing representation would clarify the benefit of the proposed log-time residuals.

**Justification of Recommendation:**

The paper offers a practical design with promising timing and efficiency results. I assign 5/10 because clearer differentiation from prior work, more complete baseline comparisons, and better isolation of key design contributions are needed to support a higher rating.

**Specific Points of Feedback for Rebuttal:**

1. Could you clarify the main differences between INR and the related designs in Compound Word Transformer, ScorePerformer, and DExter, highlighting the intended contribution of your combination?
2. Could you explain any practical constraints behind the omission of SyMuPe/PianoFlow and ScorePerformer’s inference-time results?
3. Could you clarify which design choices are isolated by the existing ablations and which are evaluated jointly? In particular, how should readers interpret the DINR–CINR and relative-versus-absolute timing comparisons?

**Rating:** 5: Marginally below acceptance threshold

**Confidence:** 3: The reviewer is fairly confident that the evaluation is correct

Add:

#### **Phase 1 update**

Official Commentby Program Chairs25 Sept 2026, 06:39 (modified: 25 Sept 2026, 06:39)Program Chairs, Program Committee, Senior Program Committee, Area Chairs, Secondary Senior Program Committee, Authors[Revisions](https://openreview.net/revisions?id=s37f1PSdEi)

**Comment:**

This paper remains in the AAAI-27 review process. Reviews for this paper will be provided at the conclusion of the Phase 2 reviewing period, before opening for Rebuttal.

#### **Well-executed formulation for expressive piano rendering with objective gains**

Official Reviewby Program Committee nxJE29 Aug 2026, 13:56 (modified: 24 Oct 2026, 20:23)Program Chairs, Area Chairs, Senior Program Committee, Program Committee nxJE, Authors, Program Committee Submitted[Revisions](https://openreview.net/revisions?id=aQPpUeXMeC)

**Paper Summary:**

INSPIRE addresses the numerical quantization and loss of ordinal structure in expressive piano rendering by introducing Integrated Note Representation (INR), which fuses heterogeneous score and performance attributes into a single Transformer timestep per note rather than expanding them into multi-token sequences. Implemented via an asymmetric Transformer encoder-decoder, the model autoregressively predicts score-relative log-timing deviations, velocities, and pedal states using either coordinate-augmented categorical heads (DINR) or continuous discretized logistic distribution heads (CINR). Evaluated on the ASAP benchmark, CINR achieves lower Wasserstein distance against human performance distributions than previous methods across temporal metrics and improves piece-level Human Relevance by 13.8% over the Pianist Transformer, demonstrating that modeling continuous, score-relative residuals better preserves expressive timing and dynamic variations.

**Strengths:**

Clear and Well-Organized Writing:

The manuscript is clearly written and structured logically, making the motivation, methodology, and design choices easy to follow.

Rigorous and Logical Experimental Pipeline:

The experimental design is systematic and well-reasoned, establishing a coherent progression from problem formulation to model implementation and evaluation.

Objective Evaluation Demonstrating Stronger Performance:

The paper presents a principled, objective evaluation framework that effectively shows the proposed method achieves better performance than established work, such as the Pianist Transformer, across key target metrics under this evaluation protocol

**Weaknesses:**

Limited Technical and Architectural Novelty:

The core methodology builds on standard Transformer encoder-decoder blocks for a well-established task (expressive piano rendering). The main innovation largely boils down to reforming the output representations and coordinate parameterization rather than introducing substantial architectural breakthroughs, though the simplicity of the approach yielding strong results is appreciated.

Absence of Rigorous Subjective Evaluation:

For expressive piano rendering, subjective evaluation (such as listening tests) is essential to assess musical naturalness and human expressiveness. Although the rendered samples in the media supplement sound convincing, relying solely on objective Wasserstein distance is insufficient to guarantee perceptual quality, as marginal distribution alignment does not inherently reflect human auditory preferences or musical phrasing.

Modest Perceptual and Objective Gains Over Pianist Transformer:

While the proposed method shows improvements in specific temporal metrics, the objective gains over the strong baseline (Pianist Transformer) remain somewhat modest on several fronts. Furthermore, from a subjective listening perspective on the provided audio supplements, the perceptual enhancement over Pianist Transformer is not readily distinguishable, making the lack of formal subjective validation all the more prominent.

**Justification of Recommendation:**

I recommend an accept, driven primarily by the paper's clear writing, logically sound experimental design, and principled objective evaluation showing convincing gains over baselines like the Pianist Transformer on key expressive metrics. While the technical novelty is somewhat limited by relying on standard Transformer components to reformulate output representations, the simplicity of the approach yielding strong empirical results is commendable. What keeps me from assigning a higher rating is the lack of a formal subjective listening test; although the supplementary audio sounds promising, validating whether the objective Wasserstein distance gains truly reflect human auditory perception is essential for expressive music rendering.

**Specific Points of Feedback for Rebuttal:**

Plan or Preliminary Results for Subjective Evaluation:

As discussed in the weaknesses, human listening tests are critical for expressive music rendering. While conducting a full-scale user study during the short rebuttal period may be challenging, could the authors provide any existing preliminary listening test results.

Reporting Complementary Distributional Metrics (KL / JS Divergence):

While the authors justified using the 1D Wasserstein distance to preserve ground distance, prior expressive performance rendering works frequently report KL or JS divergences on binned/smoothed attribute distributions. Since binning and smoothing can be applied consistently across all baselines under unified discretization setups (e.g., matching the 256-bin resolution used in the model), could the authors report the corresponding KL or JS divergence results on the test split to provide a more comprehensive and directly comparable picture alongside Wasserstein distances?

**Rating:** 7: Good paper, accept

**Confidence:** 3: The reviewer is fairly confident that the evaluation is correct

Add:

#### **Review of INSPIRE**

Official Reviewby Program Committee P1Ud29 Aug 2026, 05:10 (modified: 24 Oct 2026, 20:23)Program Chairs, Area Chairs, Senior Program Committee, Program Committee P1Ud, Authors, Program Committee Submitted[Revisions](https://openreview.net/revisions?id=ywv2qtgwMQ)

**Paper Summary:**

The paper presents two models to generate piano music from symbolic score. The crux of the article lies in the problem representation that enables an encoder-decoder pair of transformers to be employed for the solution. Two variants of the problem formulation are discussed. The corresponding models are trained in a typical autoregressive manner with little modifications in the input and output layers to fit the representation. Experimental results are present and compared against the existing models.

**Strengths:**

1. The problem is certainly interesting and significant from the perspective of generative music.
2. The problem formulation is quite elegant that exploits the existing modeling strategies rather cleverly.
3. The Related Works section is especially helpful to describe the domain.

**Weaknesses:**

The reviewer understands the fact that the problem is very much application-specific and may require a significant domain knowledge. Unfortunately, the reviewer is strictly from a deep learning background and even though they are familiar with audio/music foundation models the same cannot be said about their proficiency in theory and practices of a symbolic representation (say MIDI) in a musical instrument like Piano. Given the paper assumes such a deep domain knowledge is pre-existing the evaluation and readability becomes more difficult than it needs to be. The long, often convoluted, complex sentence structure does not help the cause. So, the primary suggestion would be to represent the problem in a more generic learning-focused way and accordingly modify the writing style of the paper focusing on simplicity. For example, Table 1 and the initial problem formulation somewhat help to clear the idea.

Given CINR should be clearly more powerful due to the absence of quantization through a dictionary, why would DINR be an option? This is much more evident from the comparison in Table 2 where CINR except Ped. Clearly outperforms DINR. Interestingly, Ped is the common factor that is treated by Binary Cross Entropy in both variants.

The reviewer failed to find the definition of the notation in the equations following the “DINR uses typed categorical heads. Each field-specific query is projected from the decoder hidden state and compared with the corresponding value table”.

The overall modelling is somewhat traditional, i.e., only the input and output layers are customized to the problem. The autoregressive training is fairly straight-forward and requires no problem-specific modification for the transformer-based encoder-decoder translation-type modelling. Thus, there are not much problem-dependent architectural contributions. The novelty in this case translates to the problem representation that now enables a better exploitation of the existing modelling strategies. This is not necessarily a weakness but some discussions are necessary to better highlight the novelty and contribution in the context.

Given the full-scale PianoCoRe-A training is left for future work and only a subset ASAP is used for training, the question comes is the paper is a simple proof-of-concept rather than a complete work? Is this a conventional practice for the field? Can the authors please compare the full PianoCoRe-A data and the ASAP subset in terms of both the scale and training time. The training GPU-hours (which are strictly less than 1) in Table 2 for the proposed methods are actually pretty low compared to the current standard which often goes to 1-2 weeks.

The reviewer assumes the same GPU setup has been used to compare the training and inference time. However, a bit more details about the number and variant of the common GPU setup may help.

There are several notations with missing and conflicting definitions throughout the article. This is a major readability issue.

**Justification of Recommendation:**

The score is CLEAR REJECT which is primarily driven by three factors. First, the severe readability issues in the article both in terms of a heavy domain-specific problem and the notational hazards, with some convoluted writing. Second, the possible incompleteness of the paper citing feasibility concerns even when the training time appears to be very low. Third, the novelty only lies in the problem representation and even there one formulation clearly outperforms the other, making the latter rather irrelevant.

**Specific Points of Feedback for Rebuttal:**

The authors are encouraged to submit a rebuttal addressing the reviewer’s concerns to the best of their ability. The reviewer is willing to update the score if the major readability constraint is lifted and the novelty is highlighted.

**Rating:** 3: Clear rejection

**Confidence:** 4: The reviewer is confident but not absolutely certain that the evaluation is correct

Add:

#### **AI Review**

AI Reviewby AI Reviewer ()27 Jul 2026, 19:59 (modified: 24 Oct 2026, 20:23)Program Chairs, AI Reviewer, Area Chairs, Senior Program Committee, Authors, Program Committee Submitted[Revisions](https://openreview.net/revisions?id=6pr5UttTUd)

**Review:**

**Title:** INSPIRE: INtegrated Score-to-Performance InterpRetation for Expressive Piano Rendering

**Synopsis of the paper**

INSPIRE addresses expressive piano rendering by replacing multi-token performance decoding with one typed, fused note representation per Transformer step. It provides discrete coordinate-augmented and continuous distributional variants that predict score-relative timing, velocity, and pedal behavior. Experiments on ASAP compare distributional similarity and computation against prior renderers and analyze representation, distribution, architecture, and sampling choices.

**Summary of Review**

The paper addresses the computational cost and weak numerical inductive bias of categorical MIDI-token decoding with a coherent note-level formulation. Its distinctive elements are continuous or coordinate-aware expressive controls and score-relative log timing, while one-step compound prediction has relevant precedents. The central empirical concern is that nominally identical CINR settings yield PP HR values whose internal range exceeds the headline improvement, without uncertainty estimates. Moreover, the marginal evaluation metrics cannot validate rhythmic organization or perceptual human-likeness, and no controlled token-level baseline isolates INR from residual timing and other system differences. The evidence therefore supports improved marginal timing-distribution distances and fewer autoregressive steps on ASAP, but not all broader causal and qualitative claims.

**Strengths**

- The score-relative timing target has a concrete empirical rationale. Figure 2 shows that log-ratio residuals are concentrated on a compact support, and Section 5 reports that absolute-timing prediction sharply degrades PP timing distances; this identifies a musically meaningful target transformation rather than relying only on architectural complexity.

- The evaluation distinguishes piece-level distributional diversity from score-conditioned note-level fidelity. Section 5 defines separate PP and PN Wasserstein protocols, while Figure 4 demonstrates their opposing responses to sampling truncation; this exposes a genuine trade-off that a single aggregate metric would conceal.

- The method directly addresses a concrete systems bottleneck. Section 5 relates one-step note decoding to reducing the autoregressive sequence from $KN$ to $N$ positions, and Table 2 reports inference costs under a common test workload, making computational efficiency part of the empirical evaluation rather than only a theoretical motivation.


**Weaknesses**

- The headline PP improvement is not stable across the paper’s own reports. Table 2 gives CINR PP HR of 2.38; Table 3 defines the canonical configuration as DLM k1, 5% scale, slot+MLP, and T5 8+4, yet its T5 8+4 row gives 2.54; Figure 4 gives 2.08 at the reported temperature 1.0 and top-$p$ 0.9. This 0.46 internal range exceeds the 0.38 gain over Pianist Transformer, while no training- or sampling-level uncertainty is reported.

- The evaluation does not substantiate broad claims of rhythmic organization or human-like rendering. CINR’s PP HR is 2.38 versus the human anchor of 1.00, and every PP attribute distance is at least twice its human counterpart; moreover, PP discards temporal order and PN evaluates attributes marginally. These metrics cannot measure phrase structure, beat preservation, cross-attribute coherence, or pedal continuity, so the quantitative evidence supports distribution matching rather than the broader musical claims.

- The architectural novelty is narrower than the framing suggests. Compound Word Transformer already groups typed musical fields into one outer autoregressive step and uses type-specific prediction heads (Hsiao et al., 2021), while Jeong et al. (2019) embed score inputs and predict expressive features at note level. The paper cites both works but does not engage these specific similarities, making the clearer novelty the combination of aligned EPR, numerical parameterization, and score-relative timing rather than one-step typed prediction itself.

- The quality gains cannot be attributed specifically to INR. Table 2 compares systems differing in architecture, pretraining, targets, and output distributions, without a same-backbone multi-token decoder; meanwhile, Section 5 reports that replacing residual timing with absolute timing increases PP IOI and duration from 11.87 and 47.67 to 272.30 and 480.38. Thus, the paper’s own evidence indicates that target formulation may dominate the quality improvement, while INR’s independent quality effect remains unmeasured.

- The IOI formulation leaves chord-note behavior operationally unresolved. For $I^s=0$, the definition $\Delta I=g(I^p)-g(I^s)$ and support $[-2,1]$ cap representable positive IOI at approximately $e^1$ ms under the stated transform; Figure 2 excludes zero-score-IOI notes, and Section 3 does not state whether such targets are masked or how they are reconstructed. This affects a common component of polyphonic piano performance despite the alignment problem being acknowledged in Section 6.

- The reported efficiency ratio is not a controlled systems comparison. Table 2 does not specify accelerator type, precision, batching, caching, or whether baseline costs were remeasured under the same implementation, while the claimed $K^2$ reduction concerns only self-attention. Consequently, the twenty-fold inference difference cannot be attributed solely to INR.


**Suggestions for Improvement**

- The exposition would be clearer if it organized the contribution around separate hypotheses: note integration reduces decoding cost, numerical parameterization supplies an ordinal inductive bias, and score-relative timing simplifies the prediction target. Section 5 already attributes the strongest timing result to the last mechanism, whereas the introduction foregrounds INR as a unified explanation.

- Evaluation on the full PianoCoRe-A collection or a second aligned corpus would test whether the findings generalize beyond the nineteen-score ASAP test split. Analysis by chord density, ornamentation, and alignment quality would be particularly informative given the failure modes identified in Section 6.

- A joint within-note output distribution could preserve one outer Transformer step while modeling dependencies among timing, velocity, and pedal more directly. Section 4 currently uses parallel heads conditioned on one hidden state, and Section 6 identifies cross-attribute interference; structured coupling could address both expressive coherence and optimization interference.

- A role-by-slot table should state each encoder value, decoder-feedback value, prediction target, transformation, support, and parameter-sharing choice. Figure 3 currently requires readers to reconstruct how absolute score timing, performance feedback, and timing deviations occupy the shared schema.


**Additional Details and Minor Issues**

- Figure 1 says CINR has “no quantization,” whereas Section 4 uses discretized likelihoods with 256 timing bins and 128 velocity bins; the wording should distinguish continuous parameters from discretized outputs.

- Section 4 should specify endpoint-tail handling for the finite-support discretized logistic and define $s_a^{min}$, $s_a^{max}$, and the “5%” bound in Table 3.

- Section 3 includes pitch in $y_i$, but Section 4 has no pitch head; stating that aligned pitch is copied from the score would remove ambiguity.

- The paper should state how overlapping note-level pedal snapshots are converted into one globally valid MIDI controller trajectory.

- Section 5 should clarify whether validation windows were selected by piece before overlapping segmentation and whether Figure 4’s sweep used validation rather than test data.


**References**

Hsiao, W.-Y., Liu, J.-Y., Yeh, Y.-C., & Yang, Y.-H. (2021). Compound Word Transformer: Learning to compose full-song music over dynamic directed hypergraphs. _Proceedings of the AAAI Conference on Artificial Intelligence, 35_(1), 178–186. doi:10.1609/aaai.v35i1.16091. ([ojs.aaai.org](https://ojs.aaai.org/index.php/AAAI/article/view/16091))

Jeong, D., Kwon, T., Kim, Y., & Nam, J. (2019). Graph neural network for music score data and modeling expressive piano performance. In _Proceedings of the 36th International Conference on Machine Learning_ (pp. 3060–3070). Proceedings of Machine Learning Research, 97. ([proceedings.mlr.press](https://proceedings.mlr.press/v97/jeong19a.html))

Add:

[About OpenReview](https://openreview.net/about)

[Contact](https://openreview.net/contact)

[FAQ](https://docs.openreview.net/getting-started/frequently-asked-questions)

[Hosting a Venue](https://openreview.net/group?id=OpenReview.net/Support)

[Sponsors](https://openreview.net/sponsors)

[Terms of Use](https://openreview.net/legal/terms) / [Privacy Policy](https://openreview.net/legal/privacy)

[All Venues](https://openreview.net/venues)

[**Donate**](https://openreview.net/donate)

[News](https://openreview.net/group?id=OpenReview.net/News&referrer=[Homepage]\(/\))

[OpenReview](https://openreview.net/about) is a long-term project to advance science through improved peer review with legal nonprofit status. We gratefully acknowledge the support of the [OpenReview Sponsors](https://openreview.net/sponsors). © 2026 OpenReview

INSPIRE: INtegrated Score-to-Performance InterpRetation for Expressive Piano Rendering | OpenReview
