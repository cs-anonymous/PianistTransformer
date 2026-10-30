# sample 1

We thank the reviewers for their constructive feedback. We have carefully addressed all concerns as below:
To ALL
Motivation and novelty: 1) we use GDM to generate molecules, and RL to optimize their performance; 2) MoE is not a simple addition but a core mechanism that dynamically routes information to reduce variance. Motivated by the success of MoE in LLMs, we concretized it in molecular graph generation and found it to be effective.
Substructure extraction: All substructures are automatically extracted via the BRICS algorithm to ensure valid bond-breaking rules. Classification are fully unsupervised and based solely on structural similarity, avoiding any data leakage or label dependence. Table 2 also demonstrates that the diffusion process ensures high validity of such structure.
To NE1C

R1W2: Compared with experiments in most other papers on the same topic (DiffLinker@NMI, DiGress@NeurIPS, etc.), we believe our experiments are on par with them on both scale and breadth. . SchNet offers an effective trade-off between accuracy and computational efficiency.
R1W3: Will address in the camera-ready.
R1Q1: Table 2 demonstrates that the generated molecules are realistic, valid, and novel.
R1Q2: The number of experts selected is consistent across different datasets, but the optimal number varies. Increasing the number of experts increases the learning difficulty of the gating network and leads to load imbalance, resulting in decreased performance.
To xM9y
R2W2: Will add more discussion of previous work in the camera-ready.
R2Q3: The core is the gated routing mechanism, and we observed in experiments that the variance decreases with iterations.
We believe the deficiency on motivation and related work are not key to a technical paper, and we have answered all your questions. We sincerely hope you can consider to raise the score.
To Sg9H
R3W1&W2: Refer to To ALL for the motivation and substructure sections. Will add discussion of other RL methods and provide a thorough explanation.
To AMV2
R4: Will provide more detailed and clear discussion.
To kJxF
R5Q2&6&7: Will add explanations for this section.
R5Q3&4&5: GDM operates on 2D molecular graphs. SMILES are used for data input, preserving the equivalence to 2D. 3D results are for visualization only, and the results of different tasks are for reference only.

# sample 2 (SIGMOD'23) 

Thank you for constructive comments. Below we address most concerns, with new results & revision plan for the extra page. We hope you could consider the novelty aspect over writing issues in our 2-round review model. To save space, “Will do”=“We will do in the revision”.

All: Will fix minor issues as suggested (grammar, reference, etc.)

==R1==

O1: Will discuss GF’s sensitivity to primary author detection. In concept level, it is important as GF is designed to capture only the author’s primary contribution w/o minor co-authorships. This can be remedied by high-recall (lower-precision) algorithms, making the true GF a subgraph of computed GF. For award-inference task, it is not clear-cut, as a few research fields also value non-primary-author contribution.

O2/O3: Will rewrite Sec. 5.3 by: 1) compare GF & citation metrics; 2) better illustrate the value of GF for impact evaluation.

Sketch of Sec. 5.3:
* 2 authors with similar # of paper (awarded vs. not), their GFs differ in: # of high-citation papers (node outline in red)
Link 1: geneticflow.actvis.cn/idx/2
Link 2: [same_link_above]/1

* Similar # of citation, diff.: research time span & width
L1: …/3
L2: …/38

* Similar h-index, diff.: focus on the same or many research fields (node fill color)
L1: …/32
L2: …/41

Other comments:
* Color in Vis.: we add an interaction to assist color-impaired. In any links above, hover the topic bar at the bottom will highlight all the papers on that topic.
* Parameters of advisor-advisee detection: will provide. Now calibrated by the small-scale exp. in Sec. 4.2, will refine by the large-scale exp. (see R3-O4)

==R2==

O1: Thank you for pointing out this issue. As outsiders to the community, our writing style may not communicate well to the audience. In the revision, will make fundamental changes in writing to fit our community’s taste. We hope to continuously receive your guidance and suggestions. Will also upgrade the case studies (See R1-O2/O3).

O2: 1) Will remove the “novel” claim of Historiography (H). This is unrelated to our novelty. 2) Will change “structured info.” to “relational info.”, then H exploits citation relationship among papers. 3) A highly-cited paper may not be totally relevant to the scientific impact of the author, as s/he may not have the primary contribution. 4) Will change “of evaluation” to “of impact evaluation for academic users”

O3: We use a GNN to classify 50 awardees and 150 others in each CS sub-field, with their GFs as input, via supervised learning. Will state this explicitly and expand the intro. of GNN.

O4: Will discuss the limitation of primary author detection. Our main proposal is the GF framework, better detection algorithm can be plugged in (see R1-O1). CS fields with alphabetic author order have been excluded (e.g., TCS). For fields with a few such papers, the pros/cons to detect&remove them will be discussed.

O5: In a larger-scale exp., recall=87.5%, precision=88.3%. (refer to R3-O4)

O6: There are many existing studies on citation type classification ([47-49][34]). They all list extend-type citations as one class with similar definitions. E.g., [47] defines it as "the author uses cited work as basis or starting point". When an author extends the work of itself, it may be an evolution of the author's research.

O7: We see ourselves outsider to the community (rarely submit to SIGMOD as primary author). This work stems from the idea to visualize personal idea flows (visualization, part of graphics). The application area is Scientometrics (outside CS). From all aspects, we find ourselves outsiders.

==R3==

Track: GFs are small, but mining GFs is data-intensive & requires processing big scholar data (e.g., MAG has 0.5B papers+authors). Tasks such as merging MAG&ARC require mapping each ARC author/paper to every MAG entry. We are researchers outside in all aspects (see R2-O7). CFP requires “researcher OR practitioner”.

O1: Kindly refer to R2-O4, R1-O1.

O2: Will discuss our view on these concepts (with reference). E.g., “scientific impact”=“the scope/degree/structure of scientific contribution and its community recognition”. Award-receiving largely overlaps with scientific impact but is not identical. Award factors could include community service and may bias towards minorities.

O3: actvis.cn:8020/GeneticFlow

Vis. tool: geneticflow.actvis.cn (use “scholar list”, better accessed after Sept. 10)

O4: Previously, we use 74 PhDs of 3 scholars as positive labels, w/o negative label. Only recall can be computed. Now, we conduct a new evaluation with OpenReview (OR) data. For a top scholar on NLP, we crawl its papers/co-authors from OR. Some co-authors provide their PhD advisor at OR. If it matches the top scholar, the tie is used as a positive label. For a co-author, if during his/her every co-authored paper’s year, s/he has a different PhD advisor (not the top scholar), it is used as a negative label.

We have processed top-100 NLP scholars and obtained 520 labels (P=351, N=169). Applying our detection algorithm, recall=81.8%, precision=88.3%. Note: 23 advisees only have one co-authored paper with their PhD advisor. After removing these labels, recall=>87.5%. Will extend the exp. to more scholars/fields. 

O5: Will expand annotation details. We only label 133 citations with high extend prob. from 2k data. Annotators first learn from 89 extend citations in literature for consistency. Two coders agree on every label (drop if not).

O6: We suppose, GF works better with high-quality data (ARC+MAG@NLP), worse with MAG-only data (non-NLP, some with marginal perf.) To verify this, we add a new exp. with MAG-only data@NLP for direct comparison. Initial results support our claim:
* GFs: full-graph=.694+-.014 (.776+-.015@MAG+ARC), core-graph=.674+-.012 (.732+-.016@MAG+ARC)
* best indicator=.691+-.018 (.683+-.021@MAG+ARC)
* best bib. network=.547+-.031 (.525+-.031@MAG+ARC)

Full exp. will be reported. GFs could have sig. advantage in all fields with fulltext+citation data.


# sample 3 (KDD)

Response to all reviewers
Dear reviewers:

Thank you all for the valuable time and effort to lend your constructive comments. Below we first post the replies to some of the common questions raised in your comments. Later, in the succeedingreply to each of your reviews, we provide one-by-one responses to all your comments, predominately according to the “paper weakness” section of each review. All your comments and our responses will be carefully incorporated to the camera-ready version of our paper if accepted.
“Evolution-rich” requirement 
We thank the reviewer for pointing out this ambiguous, less-represented requirement. The “evolution-rich” requirement in our work does not primary mean the standard citation impact evolution as exemplified by the citation count time series chart in Google Scholar (line 95-97 in the paper), e.g., https://scholar.google.com/citations?view_op=view_citation&hl=en&user=qc6CJjYAAAAJ&citation_for_view=qc6CJjYAAAAJ:qyhmnyLat1gC . Though, these citation count time series can be easily incorporated as node attributes of GF profile for either GNN representation or node attribute visualization.
By “evolution-rich” requirement for GF profiling, we actually mean, as already briefly stated in line 519-531 of the original paper, to reveal the evolution of a scholar’s research ideas and scientific contribution. The evolution, together with the citation impact of each paper within the evolution, can be important to his/her scientific impact. The GF profile faithfully accomplishes this “evolution-rich” requirement by modeling the profile as a temporal graph (best revealed in Figure 1(d).I). We believe, satisfying the “evolution-rich” requirement plays a key role in the success of GF profiling method. In the case study of Sec. 5.3, the discovery of patterns of elite scholars’ GF profiles, i.e., well-connected, sufficiently-sized, spanning many years continuously, are mostly through representing the evolution of a scholar’s research work in a time-oriented manner.
This “evolution-rich” requirement can not be met by individual impact indicators or classical bibliometric networks, as the time dimension is not explicitly modeled there. The citation count time series of Google Scholar shows the evolution of a single paper’s impact, but not the evolution among research works.
We will expand these details on “evolution-rich” requirement and its fulfillment by GF profiles in the revised paper.
On the GF profile of ordinary/normal scholars
It is true that the extend-type citations with high probability could be rare among all citation links, making the possible GF profile very small or sparsely connected. We already have some initial treatment on this issue. In computing the GF profile, there is a parameter, i.e., the lower-bound threshold of extend-type citation probability, which controls the number of extend-type citations appearing in the GF graph profile. For scholars with fewer high-probability extend-type citations, this threshold could be lowered to allow more extend-type citation links. This mechanism is better perceived in the video demo attached within the paper: https://vimeo.com/795348791/
We also note that, the rate of extend-type citation is not necessarily linked to the impact-level of the scholar. An extra experiment has been carried out that the percentage of extend-type citation is 13.3% among elite scholars (top-100 in NLP ranked by #papers), and is 13.8% among ordinary scholars (ranked 901~1000 in NLP). There is no significant difference as the extend-type citation is mainly influenced by the nature of the research but not by the impact of the research. Afterall, the number of extend-type citations could be much smaller in ordinary scholars’ GF profile because the number of total citations there is already smaller. In comparing these ordinary scholars, we could apply a same but lower threshold on extend-citation probability, so that the displayed GF profiles are more comprehensive while still comparable.

GeneticFlow authors
Response to Reviewer jHXz
Thank you for the constructive comments.
1.We actually mean “the drawback of the third-type indicators such as i10-index …”. We do not expect to criticize Google Scholar’s approach. “They do not consider timeliness that older papers could accumulate more citations …” - we actually want to say that “the third-type indicators such as i10-index do not consider the dynamic evolution of paper citation counts, which could be important for a scholar’s impact …”. We will incorporate these in the revised paper.
2.Equation (2) is used to estimate the probability of a paper having all its authors ordered exactly by the alphabetical order, under the condition that this paper is not intentionally ordered alphabetically per the requirement of certain fields. That is, even if it is not intentionally ordered alphabetically, there is chance that the author order is exactly the same with the alphabetical order. This chance (probability) is calculated by Equation (2). For a paper with k authors, there is only one alphabetical author order, while there are k! possible author orders. This chance 1/k!, is further weighted by the ratio of papers with k authors (N_k/N), which gives Equation (2). We will expand the illustration of Equation (2) in the revised paper.
For alphabetically ordered papers in the preserved fields, it is important to analyze whether they are intentionally alphabetically ordered or only by chance. For the latter case, these alphabetically ordered papers will not distort GF profile as the advisor-advisee relationship detected on these papers will be correct. According to Ref. [1], the tendency to apply alphabetical authorship is largely influenced by field-specific traditions. It implies that, if we exclude those fields with largely biased percentage of alphabetically ordered papers, the preserved fields will have a small portion of intentionally alphabetical-ordered papers.
Further, these intentionally alphabetically-ordered papers do not necessarily distort the GF profile much. There are two cases. First, they could lead to false-positive core papers, which may not be quite harmful. The core GF profile is just augmented with a few irrelevant papers, while the correct core GF profile can still be perceived/utilized in the visualization/inference. Second, there could be false-negative core papers, i.e., some true core papers are removed from the GF profile, which may be influential. This drawback can be alleviated by introducing an alternative order-irrelevant indicator – the percentage of scholar’s students in a paper’s author list. A larger value of this indicator will help to infer the scholar’s core paper on the alphabetically-ordered paper set. The determination of the new indicator's threshold will need more discussion. 
[1] Fernandes and Cortez, "Alphabetic order of authors in scholarly publications: a bibliometric study for 27 scientific fields", Scientometrics 125, 3 (2020), 2773–2792.
3.We truly understand your concern. Our definition of advisor-advisee relationship in line 473-477 is made up of four criteria. We believe all criteria are necessary, but we do not want to make these priors too strong to exclude many actual advisor-advisee relationship. So in practice, we make each criterion parameterized so that it could be flexible to detect real-world advisor-advisee relationship.
Our definition is designed to include not only formal advisor-advisee relationship, but also informal mentor-mentee ties. However, labeled dataset on informal relationship is not available. We tried our best to obtain the largest-possible labeled advisor-advisee dataset from OpenReview (Sec. 5.2&Supplement B), which mainly contains formal advisor-advisee relationship and sometimes informal relationship. The parameters of each criterion are calibrated using this labeled data to optimize the advisor-advisee detection (Supplement B). We also compare our new algorithm with existing benchmark of TPFG model, and demonstrate our superiority (Sec. 5.2). Our method achieves sufficiently good metrics (ACC/F1 >= 0.83) to be applied in real world.
The four criteria in our definition are also applied in very similar way by TPFG [54] and Shifu2 [38] in the literature. In TPFG, the Kulczynski measure reflects D2&D3, while IR evaluates D1. Shifu2 encodes D2&D3 as edge attributes of their academic network.
4.Please refer to "Response to All Reviewers": #2 item. Your comment is also related to our long-thought improvement for GF. The core links in GF profile could go beyond the extend-citation links, and include the links between papers with very similar topic, even if there is no citation link between them. This improvement will alleviate the issue of missing citation links due to low data quality. It also helps when the authors do not explicitly cite themselves for various reasons (e.g., papers published in the same year).As the first work on GF, we want to keep the method principled, without too many work-arounds. This could be a future work.
Response to Reviewer iPhr
Thank you for the constructive comments.
1. Generality issue: In this reply, we demonstrate that the proposed profiling method can work well and robustly beyond the computer science (CS) domains. In particular, we have conducted extra experiments in the larger field of Physics, whose bibliographic data is also available in the MAG dataset (a paper count of 11.7M, total citation count of 101M). Again, in the physics field, we consider the highest-level scientific award, including the Nobel prize and the Wolf prize. We rank all scholars in physics (#paper in physics >100) by their h-index and obtain the top 5000 scholars. Within these 5000 scholars, there are 42 Nobel/Wolf prize winners (90% are winners after 1960 as older winners may not have many physics papers/citations recorded in MAG). Then we randomly sample 158 non-award-winners within the same h-index range as the 42 Nobel/Wolf prize winners. This sampling mechanism mimics the sampling method used in our paper on CS sub-domains. The comparative result of the award inference task in the physics domain is summarized as the below table. It is shown that, the F1 scores using the full GF profile are significantly better than all alternative methods of author-level impact indicators and bibliometric networks. This result is consistent with those in CS sub-domains. Note that we do not have inference result by the co-authorship network (CA) on the physics dataset. It is because the co-authorship networks of the selected 200 top scholars grow too large to apply and evaluate GNN models on the network. In average, these CA networks have XXX nodes and XXX edges, XX times larger than the CA networks in the CS-NLP sub-field. (*** add these XXX figures, in the below table, OOM => N/A)
Table 1: The performance of award inference task in the physics field.
	GeneticFlow	SVM	XGB	RF	MLP	CC	BC	CA
F1	0.631±0.011(p=0.05)	0.550±0.021	0.553±0.024	0.582±0.023	0.547±0.028	0.574±0.024	0.526±0.022	OOM
Here the GF model applies the recommended configuration in the paper, i.e., ARMA model without graph coarsening, using bi-directional graph setting. There is no additional parameter tunning for GF method. In fact, all alternative methods in comparison have applied the best parameter setting after considerable tunning effort.
2. Scalability issue: Note that line 734-740 already briefly justify the choice of small sample size used in the evaluation. To demonstrate the scalability of GF profiling method, we extend the sample size to 1000 in the NLP sub-field. There are now 80 award recipients vs. 920 non-awardees. The award inference experiment is conducted using the same setting with that for the 200-sample dataset. The result is summarized in the below table. It can be shown that the advantage of GF method holds as the data sample size scales.
Table 2: The performance of the award inference task in NLP sub-field of Top-1000 scholars.
	GeneticFlow	SVM	XGB	RF	MLP	CC	BC	CA
F1	0.661±0.020(p=0.006)	0.580±0.016	0.591±0.016	0.574±0.015	0.557±0.010	0.553±0.024	0.546±0.030	0.578±0.041
Note that over the MAG data, it is computationally efficient to apply GF profiling method. Extracting and computing full and core GF profiles for the top-1000 scholars in the NLP sub-field only spent 1 hour in a single workstation with 8-core CPU, 64 GB of memory.
3. “Evolution-rich” requirement: We thank the reviewer for pointing out this ambiguous, less-represented requirement. Please refer to "Response to All Reviewers": #1 item.
4. MAG dataset: We thank the reviewer for pointing out this issue. It is true that the MAG service by Microsoft has sunsetted. However, its open-source successor, OpenAlex [41], has inherited most of MAG data. All required bibliographic data to construct GF profile is available in OpenAlex, so that the deployment of GF approach will not be made harder without MAG. We will mention this in the revised paper.
Response to Reviewer oitM
Thank you for the constructive comments. They will be carefully incorporated in our revised paper.
1.“Evolution-rich” requirement: We thank the reviewer for pointing out this ambiguous requirement. In fact, the core citation detection algorithm applies a few temporal features (line 590 in the paper) - cross-correlations between citation time series of cited/citing papers. A scholar’s accumulated impact over time on each paper is also represented by the total citation count of this individual paper (an node attribute of GF profile). However, these are not the key temporal information intended to be delivered by GF profile. Please refer to "Response to All Reviewers": #1 item for more details.
2.On comparison with hybrid profiling method: In fact, the alternative method of bibliometric-network profilings (co-citation, bibliographic coupling, co-authorship network) are already the so-called hybrid profiling methods. To make fair comparison with GF profiling methods, the node attributes on GF profile (e.g., paper citation count) are also included in bibliometric-network profilings in our evaluation. We will make this setting more explicitly in the revised paper.
We have conducted an extra experiment here, to compare between the hybrid profiling method with the raw bibliometric-networks without node attributes (impact indicators). The results below show that the raw bibliometric-networks perform even worse than the hybrid method already reported in the paper.
Table 1: The performance of raw bibliometric-networks.
	GeneticFlow	CC(hybrid)	BC(hybrid)	CA(hybrid)	CC(raw)	BC(raw)	CA(raw)
NLP-ARC	0.762±0.016(p<1e-4)	0.531±0.030	0.578±0.021	0.473±0.034	0.386±0.020	0.392±0.025	0.385±0.014
Database	0.638±0.016(p=0.012)	0.550±0.021	0.588±0.012	0.501±0.035	0.408±0.008	0.425±0.012	0.418±0.008
DM	0.653±.020 (p=0.007)	0.563±0.022	0.569±0.019	0.476±0.020	0.415±0.011	0.422±0.023	0.413±0.005
3.Cases on GF profiling of normal scholars: We actually have these case studies on less-impactful scholars, which are finally excluded due to the page limit. In the following two figures, we study a visualization researcher known to us. Using the full GF profile below, we only get to know that the scholar has worked on multiple topics and these topics are related by many self-citations. However, when the core GF profile is extracted and visualized as below, it is immediately clear that the scholar focused on the ochre-colored topic during 2009-2016, as shown by the temporal GF profile graph, the side-bar time-series chart, and the beneath-graph bar charts. The ochre-colored topic is actually “text visualization”. As far as we know about the scholar, s/he is indeed known in the community for “text visualization” (awarded IEEE fellow on this!).
Full GF profile: https://raw.githubusercontent.com/visdata/GeneticFlow/main/Full_GF.png
Core GF profile: https://raw.githubusercontent.com/visdata/GeneticFlow/main/Core_GF.png
On this topic, please also refer to "Response to All Reviewers": #2 item.Comparison with ACM Author Profile and AMiner Profile: We will add detailed discussions in comparison to these well-known profiling practices in the revised paper. An outline of the discussion is given below:
a) the ACM author profiles and AMiner scholar profiles provide more scholar-centric information than the already discussed Google Scholar profile. For example, ACM author profile: more topic keywords/map, altmetrics (paper downloads, etc.); AMiner profile: topic timelines, more impact indicators, patents, education backgrounds.
b) the proposed GF profile is not designed to replace Google Scholar or ACM/AMiner profiles, but to supplement existing methods with the new impact-oriented scholar-centric evolution graph of one’s research work/impact. We believe, the GF profile that exploits the structured evolution of one’s research work/impact, is not available in all existing profiling methods/practices.
c) In fact, the GF profile can be inserted into ACM author profile or AMiner scholar profile page as a sub-view. Similarly, the novel ingredients of ACM/AMiner profiles (altmetrics, extended impact indicators, etc.) can also be incorporated in GF profiling to further improve its performance/capability.
