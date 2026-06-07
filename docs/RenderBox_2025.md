# RenderBox: Expressive Performance Rendering with Text Control  

Huan Zhang1 Akira Maezawa2 Simon Dixon1 1Centre for Digital Music, Queen Mary University of London 2MINA Lab, R&D Division, Yamaha Corporation huan.zhang@qmul.ac.uk  

#文摘

富有表现力的音乐表演包括用时间、动态、发音和乐器特定技术的变化来解释象征性的乐谱，从而使表演能够捕捉音乐的情感意图。我们介绍了RenderBox，这是一个统一的框架，用于跨多种乐器生成文本分数控制的音频性能，通过自然语言描述应用粗级控制，使用乐谱应用粒度级控制。基于扩散变压器架构和交叉注意联合条件反射，我们提出了一种基于课程的范式，从简单的综合训练到表达性表现，逐步纳入速度、错误和风格多样性等可控因素。与基线模型相比，RenderBox在FAD和CLAP等关键指标上实现了高性能，并且在不同提示任务下也实现了节奏和音调精度。主观评价进一步表明，RenderBox能够产生可控的表达性能，听起来自然，音乐上引人入胜，与提示和意图很好地对齐。

# 1简介

一个训练有素的音乐家可以用自己的方式来诠释一段音乐，通过微妙地改变表演参数来塑造和改变作品的情感表达。参数维度包括时间、动态、发音和特定于乐器的技术，如弦的弓弦、风的呼吸控制或钢琴的踏板使用[Cancino-Chacón, 2018；帕默,1996]。

长期以来，音乐家、教育工作者和研究人员对这种表达模式的研究一直很感兴趣，它提出了一个令人信服的问题，即探索这种复杂的表达是否可以被计算系统准确地封装和复制[Hashida等人，2008]。作为一项任务，它可以应用于以技术为媒介的音乐教育和娱乐和创造性协助的互动音乐系统。

![](images/181a5bf22bdf9b58d92da4592e1de70f2872e9a4c5c343c4c2a4b693dcbf0874.jpg)  
Figure 1: An overview of the performance space proposed by our paradigm, progressively from strict to variant relative to the input MIDI score.  

Just as human speech varies in terms of accent, tone and pace, the space of performances is diverse, ranging from the mistake-prone playing of amateur players to highly personal interpretations of virtuosi. While there have been attempts to condition performance generation with controls on tempo [Borovik and Viro, 2023; Rhyu et al., 2022] and perceptual features [Zhang et al., 2024a], enforcing flexible, multidimensional control via natural language has not been addressed in the performance rendering task. Here, we present RenderBox, a unified performance rendering framework that bridges symbolic music scores and natural language descriptions to generate expressive and controllable audio performances across multiple instruments. Our contributions are as follows:  

1. We propose the first text-and-score controlled audio performance generation model that supports multiple instruments, enforcing coarse (text-based expressive direction) and granular (MIDI (Musical Instruments Digital Interface) score) controls into the expressive perfor  

mance realm.  

2. We present a training method that enables the model to generate performance audio signals of varying "performance variability," such as playing styles, speed, ornamentation and inclusion of mistakes. This is achieved by applying curriculum learning with carefully designed stages of performance variability, by feeding data with progressive difficulty during training.  

pursuit of expressive performance (which focuses on Western Classical and Jazz) involves much more complex, polyphonic compositions and requires outputs that are faithful to the score notes but not necessarily time-aligned. Thus, we approach the score conditioning with MIDI tokens directly, retaining the details such as exact note onset and track instrumentation, similar to a synthesis approach [Hawthorne et al., 2022].  

3. Besides demonstrating superior performance in both objective and subjective evaluations, our model learnt interpretable similarity space regarding to the performance and composition styles in classical piano literature.  

# 2 Related Work  

# 2.1 Controllability in Music Generation  

To align better with real-world music-making applications, enhancing the controllability of music generation has been a central topic since the introduction of large generative models. Recent music generation models [Copet et al., 2023; Agostinelli et al., 2023; Melechovsky et al., 2024] are prominently based on text controls, which enables descriptions such as genre, instrumentation and mood. As discussed by Lin et al. [2024], the coarse and high-level text controls are intrinsically limited, and music creators would need them to be combined with lower-level content-based controls such as melody, chord and rhythm. A similar idea is used in AudioBox [team, 2023], in which style descriptions (pace, voice type) and text transcripts jointly condition the audio outputs.  

Various conditioning mechanisms are used to control music generation. Large generative models like MusicGen [Copet et al., 2023], MusicLM [Agostinelli et al., 2023] and Riffusion are end-to-end systems that are trained with conditioning signals as input. JASCO [Tal et al., 2024] features multiple conditions of text, chords, melody and drums, each passing through their own representation projections that are aligned in time. Diff-A-Riff [Nistal et al., 2024] is a latent diffusion model based on consistency autoencoder input as well as CLAP [Wu et al., 2023] conditioning that is inserted via FiLM [Perez et al., 2018].  

Motivated by transferability and resource considerations, some approaches focus on trainable control modules that augment large, pre-trained models. Coco-Mulla [Lin et al., 2024] employs parameter efficient fine-tuning (PEFT) by inserting adaptor joint embeddings into the self-attention layers of the MusicGen decoder. Similar module-based control is used for music editing [Zhang et al., 2024c], building on a pre-trained music generation model. Music ControlNet [Wu et al., 2024] adopts the ControlNet [Zhang et al., 2023] mechanism, inserting control injection layers into a pre-trained U-Net spectrogram generator.  

Extracting their controlled information from chromagrams or piano roll matrices, models like MusicGen-Melody, CocoMulla and Music ControlNet are conditioned with monophonic melody and generating variations (or accompaniments) that are time-aligned with the melody. However, our  

# 2.2 Performance Rendering  

Traditional performance rendering is the task of generating a piece of natural, human-like performance given an input score, and it has been mostly applied in the symbolic domain [Bresin et al., 2002; Maezawa et al., 2019; Jeong et al., 2019; Zhang et al., 2024a], with a few attempts in the audio domain [Dong et al., 2022; Renault et al., 2022]. Audio performance rendering bears much resemblance to the more popular textto-speech (TTS) task, with the goal of faithfully reproducing a piece of transcript (text or symbolic music events) in the audio realm, while retaining acoustic or style cues such as voice timbre, pace or tempo so that the audio sounds natural. Compared to text, music is much more challenging in terms of time alignment, due to its polyphonic nature, as techniques such as the phoneme duration model [Le et al., 2023] that allows fine-grained alignment control in speech synthesis are harder to transfer into a music setting.  

Due to the complexity of the task, the variance in the music performance space has not been explored much. A piece can be played in numerous ways, ranging from masterful interpretations by virtuosi to student performances that reflect varying levels of expertise. Zhang et al. [2025; 2024b] highlight the importance of performance understanding and the scarcity of annotated performance data. Our current work situates the performance rendering task within this diverse interpretative space, demonstrating effective control through text-based inputs.  

# 2.3 Curriculum Learning and Continual Learning  

In our formulation of the performance variation space, a spectrum of sub-tasks with varying difficulty naturally emerges as shown in Figure 1. It ranges from the “strict easy” synthesis task that does not involve any alteration of the pitch-time relationship from MIDI conditioning, to the “deviated hard” performance tasks that are subject to variation in tempo and timing, and even addition and removal of pitches. By bootstrapping the tasks on a gradual difficulty scale, simulating the performance variation space can be posited in the framework of continual learning with a curriculum structure.  

Curriculum learning [Wang et al., 2022] enables machine learning model training to progress gradually from easy examples to harder ones, and continual learning (CL) enables models to learn new knowledge and preserve knowledge acquired previously from a data stream with a continuously changing distribution. While CL is frequently applied to classification tasks [Wang et al., 2019; Bhatt et al., 2024; Faber et al., 2024], the application of CL in generative models is less prominent. Suffering from potential challenges such as catastrophic forgetting, strategies such as rehearsals, replay and regularization are employed, but often at the cost of training additional discriminators. In our case, we take a dataincremental curriculum, and we are also the first to apply CL in incremental development of control for music generation.  

![](images/ea90715d37b6879658e48b25ce6207ccf3c07dd1506ecb64a3332e53191d2ec8.jpg)  
Table 1: Aligned datasets of scores and performances, organized by curriculum level. The dataset duration is computed with regards to the total length of input MIDI segments, and also subject to the availability and validity of accurate alignment.  

# 3 Methodology  

# 3.1 Base Architecture  

We base our conditioning model on the text-to-audio architecture from Stable-audio-open [Evans et al., 2024]: an autoencoder with a latent rate of 21.5Hz to preprocess the raw waveform sampled at 44.1kHz, a T5-based text embedding for text conditioning, and a diffusion transformer (DiT) that operates in the latent space of the autoencoder with latent sequences of length 1024 (47 seconds).  

During training, we initialize the autoencoder, text embedding module and diffusion transformer with the pre-trained weights of the original text-to-audio model. The DiT is trained to predict a noise increment from noised ground-truth latents, following the standard formulation of the v-objective. Given a latent zt at time t, perturbed with noise ϵt according to a variance schedule, the model predicts the velocity vt = αtϵt 一 σtzt, where αt and σt are determined by the variance-preserving diffusion process. During inference, we use the DPM-Solver++ for 100 steps with classifier-free guidance (scale of 7.0).  

# MIDI Conditioning  

The text and duration (a float input with positional embedding that is used to control output length) conditioning signals are incorporated into the base architecture via cross-attention. We also need to incorporate MIDI score input to instruct the rendering. The input MIDI data is first converted to a MIDILike tokenization sequence via note-seq‡. We use the same note event vocabulary as MT3 [Hawthorne et al., 2022], which is based on the MIDI protocol. Specifically, there are events for Instrument (128 values), Note (128 values), On/Off (2 values), Time (512 values), End Tie Section (1 value), and EOS (1 value). The MIDI conditioning module consists of an embedding layer that projects token vocabularies into a 768- dimensional space, to which a sinusoidal positional encoding is added. Our input MIDI window is 10 seconds, which could include 300–2000 tokens depending on the note density of the piece.  

![](images/04bcba3e65b2e3735b32a1ffc8da2bbb379ed583365ef4bbd1f738fec0714610.jpg)  
Figure 2: ControlNet conditioning (left) and concatenative crossattention conditioning (right), with color highlighting the initialization of modules and their optimization in our experiments.  

To subsequently insert the MIDI embeddings into the DiT, we experimented with two methods:  

Exp.1: Concatenative cross-attention conditioning. The MIDI embedding is concatenated with the T5 text embedding and the duration embedding along the sequence dimension. Besides the new MIDI embedding module, all other modules and layers are initialized with the stable-audio-open weights and receive full fine-tuning (except for T5 which is frozen). Exp.2: ControlNet conditioning based on the implementation of stable-audio-controlnet§ [Ciranni et al., 2025]. With the main DiT and the text and duration embeddings frozen with pretrained weights, the MIDI embeddings go through the ControlNet branch which is a reduced version of DiT with around 20% of the depth. Only the ControlNet branch and MIDI embedder are optimized. See Fig. 2 for an illustration of the conditioning.  

Stage 3 - Mistake-corrupted performance (4k steps). The target is performance MIDI corrupted with artificial mistakes, such that the model learn possible pitch or rhythm deviation from the score (compared to the deviationfree performance in the previous stage) under the performance category of less-experienced player. Stage 4 - Style-directed performance (10k steps). This stage also trains with performance data, but augments the text prompt with available style directions, including performer names or expressivity annotations (i.e. calm, passionate).  

# 3.2 Curriculum Training Scheduler  

Curriculum learning sequences the learning process in a curriculum of increasing complexity tasks, which allows learning on large data collections that otherwise would be impossible to learn from scratch. As shown in Table 1, we arrange the datasets and training targets by stages of difficulty, distributing to five stages:  

Stage 0 - Synthesis, which enforces the same text prompt (20k steps) ’Synthesis’. This stage forces the model to direct its attention away from the text information to the MIDI tokens, focusing entirely on mapping MIDI events precisely to audio events.   
Stage 1 - Synthesis with speed change (10k steps). As human performers naturally vary tempo, this stage is an important bridge to train the model to map a stretched / squeezed time series of MIDI events in response to speed prompts (e.g., twice as fast).   
Stage 2 Expressive performance (15k steps). This stage introduces performance recordings as training target (in contrast to mechanically synthesized audio in previous stages), which involves not only global tempo change (indicated by the text prompt) but also local timing variations (such as rubato).  

For ablation, we also trained a version of MIDI concatenative cross-attention conditioning without the curriculum learning by mixing all stages of data (Exp.3).  

# 3.3 Experiments  

All of our experiments are performed on four 80GB A100 GPUs, using 16-bit precision training. For training Exp.1 and Exp.3, we use the AdamW optimizer, with a base learning rate of 1e-5 and a scheduler including exponential ramp-up and decay. With a batch size of 108, Exp.1 trains for a total of 59k steps as stages scheduled in the previous section, and Exp.3 (no stages) is trained for 60k steps. Other training techniques such as EMA are applied, following Evans et al. [2024]. For training Exp.2 of the ControlNet, we have trained 30k steps with a batch size of 108, and a learning rate of 1e-4.  

# 4 Datasets  

To train our model, we have aggregated a large number of publicly available performance datasets with aligned score and audio performances, across a range of instruments. That includes (n)ASAP [Peter et al., 2023], MusicNet [Maman and Bermano, 2022], GAPS [Riley et al., 2024], BachViolin [Dong et al., 2022], ATEPP [Zhang et al., 2022], Con Espressione [Cancino-Chacón et al., 2020], and Vienna 4x22¶. Their size, repertoire, annotations for prompts, augmentation and scheduling are listed in Table 1. In the expressive performance training of stage 2, we also utilized an in-house recorded saxophone dataset (copyrighted) of funk and swing Jazz standards from The Real Book.  

# 4.1 Augmentations  

Speed augmentation: For stage 1 training, we apply speed augmentation to induce the model’s ability to synthesize the MIDI score timing into proportionally faster or slower audio, before learning the more flexibly varied performance timing. As shown in Table 2, each score in the (n)ASAP, MusicNet and GAPS datasets is augmented with 5 tiers of speed ranges, in which the speed ratio is randomly sampled within the tier’s range, and the text prompt is augmented with a corresponding speed keyword of common terminology. The speed-controlled re-synthesized score audio is served as the training target. We choose to not speed-augment the performances, since the artist’s interpretation of timing and phrasing would be influenced by tempo [Repp, 1996; Repp, 1995]. While the speed ranges include significant tempo shifts, they intend to help the model generalize across different expressive timing variations, rather than imply that all music would be performed at such extreme tempos.  

Mistake augmentation: For the stage 4 training with mistake instruction, we followed the mistake taxonomy proposed by Morsi et al. [2024] and implemented several types of mistakes to augment the ASAP dataset: mistouch, asynchrony (delay or anticipation), substitution, ghost note and reorientation (time block removal). The procedure of applying the mistakes to each segment is detailed in Algorithm 1. Note that our mistake corruption does not involve adding shifts or silence on the time axis, since that would change the scoreaudio segment alignments.  

# Algorithm Mistake Augmentation on Piece Level  

1: Input: A sequence of notes N with total duration sec  
onds, each with properties: pitch πn, velocity vn, star   
time sn, and end time en.   
2: Definitions:   
3: Puniform(a, b): A random value uniformly sampled from   
the interval [a, b]   
4: U (0, 1): A uniform distribution over the interval [0, 1]   
5: for each note n do   
6: Mistouch:   
7: if U (0, 1) < 0.05 then   
8: Generate a new note n′:   
9: πn′ ← πn + choice({−1, 1})   
10: vn′ ↑ 0.8 vn   
11: sn′ ← sn + Puniform(0.02, 0.1)   
12: en′ ↑ sn′ + Puniform(0.1, 0.3)   
13: Add n′ to   
14: Asynchrony:   
15: if (0, 1) < 0.2 then   
16: Shift sn and en by Puniform(−0.7, 0.7)   
17: Ensure sn M 0 and en ≥ sn   
18: Substitution:   
19: if U (0, 1) < 0.05 then   
20: πn ← πn + choice({−1, 1})   
21: Ghost Notes:   
22: if (0, 1) < < 0.05 then   
23: Remove n from   
24: Time Block Removal:   
25: for k = 0 to T5 ⌋ do   
26: tstart ↑ 5k + Puniform(0, 5)   
27: tend ↑ tstart + Puniform(0.2, 0.5)   
28: Remove all n EN where tstart ≤ sn < tend  

# 4.2 Prompt Preparation  

Our prompts, aligning with the tasks to provide multiple stages of instructions, may include the following fields of information: sonification type ‘synthesis’ or ‘performance’ (all stages), speed keyword (stage 1 and after), title, composer, instrumentation, mistake (stage 3), performer ID (stage 4), expression label (stage 4), subject to the availability of this information in the metadata as shown in Table 1.  

The speed keywords (Table 2) are first utilized in stage 1 with the speed-augmented synthesis as described in section 4.1. For the later stages of performance data, we incorporate a tempo prompt by estimating the length ratio of the aligned performance window to the reference score. For example, if the aligned performance window is 17 seconds of a given 10-second MIDI score, we simplify the tempo ratio as 1.7 and supply a prompt keyword of Considerably slower. For our fixed-length (10s) input window, the performance ratios in our datasets are roughly between 0.4 and 2.2. For other available metadata such as title, composer and instrumentation, we also choose to optionally include them (with random dropout of 0.5) as these labels would facilitate rendering by giving extra context about the MIDI piece.  

Table 2: Speed prompt keywords arranged in tiers.   

![](images/ab2a327ab8dbcfb8f9645acb7f28989d1f97a10b1df44ed20b3b17da1208e8f0.jpg)  

The prompt field values are specified in a comma-separated list in any order (e.g. “a bit slower, expressive performance, Bach, Piano” in stage 2, or “expressive performance, style of Vladimir Ashkenazy, Etude Op.25 No.11, Chopin, notably faster” in stage 4).  

# 5 Evaluation  

We compare our work with the following three models. MusicGen-melody is the melody-conditioned version of MusicGen [Copet et al., 2023]. Given that the melody conditioning is implemented via an audio input, we synthesize the MIDI score|| for MusicGen-melody as score conditioning. Coco-Mulla is a MusicGen-based conditioning model which enforces more external controls such as drum track, chord symbols and reference MIDI. In our usage, we supply null values for the drum track and chord symbols, taking it as a MIDI-and-text conditioned generation. MIDI-DDSP is an expressive synthesizer framework dedicated to string and wind instruments that is able to control aspects such as brightness and vibrato. Evaluations are conducted in stage 0, 2, and 4 with respect to each stage’s testing set and their prompting.  

![](images/fc3cb6fda38f6e3b3aa5a7c22bdbf7a5e7b9f21ee75317181c16684e92a93d1d.jpg)  
Table 3: Comparison of models across the three main stages, using FAD, CLAP, Chroma, and Tempo scores as metrics. Each stage features different prompting, MIDI conditioning and ground truth data as shown in Table 1. For MIDI-DDSP, the evaluation is only performed on the MusicNet and BachViolin subsets as the model is restricted to chamber instruments.  

![](images/38fd3b7d7038ebb63e6c57a9508e9abfbf2ce01466c1688ad7f015ce73ef732f.jpg)  
Figure 3: MOS score of the subjective evaluation on the four dimensions, separated by participant’s experience.  

# 5.1 Objective Evaluation  

For the audio metrics, we follow the previously established metrics [Evans et al., 2024] implemented in stable-audiometrics, including the Fréchet audio distance (FAD) on OpenL3 embeddings [Cramer et al., 2019] between the output and ground truth distributions, and distance in LAIONCLAP space [Wu et al., 2023] between the text prompt and output. Given the goal of performance rendering, it is crucial to enforce that the correct piece is played. To evaluate pitch-wise accuracy, we compute the chroma similarity, inspired by the evaluation approach used for MusicGen-Melody [Copet et al., 2023]. As the output and ground truth audio are not necessarily time-aligned, we employ dynamic time warping (DTW) on the chromagrams to estimate frame correspondences for a rough audio-level alignment. Following this alignment, frame-wise cosine similarity is calculated between the aligned chroma features. Additionally, the DTW alignment cost is incorporated as a penalty term, scaled by a weighting factor λ = 10−3.  

We also measure tempo deviation to evaluate whether the output tempo is within the prompted tempo tier as specified in Table 2. Given the expected tempo ratio range (for stage 0 synthesis we expect tempo not to change), the tempo deviation is computed as the percentage difference between the estimated output tempo (using madmom) and the score tempo adjusted by the prompt.  

# 5.2 Subjective Evaluation  

Besides the aforementioned baselines, we compare with two additional symbolic-output performance rendering models: VirtuosoNet [Jeong et al., 2019] and DExter [Zhang et al., 2024a]. The performances are synthesized from MIDI\*\*, as the focus of the subjective evaluation is on the musical content rather than audio quality. Given the conditioning score and text prompt, test participants were asked to rate examples on a 100-point numeric scale on text alignment, music score alignment, expressivity and skill. We included nine questions, spanning six instruments as well as all the task (prompt) types in Section 3.2. Responses were collected from 23 participants, 10 of whom have more than 5 years’ experience of instrument playing.  

The results, illustrated in Fig. 3, reveal that RenderBox achieves the highest overall scores across most dimensions for both experienced and inexperienced participants. RenderBox significantly (p < 0.05) outperforms the conditioned audio generation models Coco-Mulla and MusicGen-Melody in all four dimensions. While symbolic models (those bypassing note event prediction: DExter, VirtuosoNet, and MIDIDDSP), achieve comparably positive feedback in score alignment, they lack the balanced performance across dimensions demonstrated by RenderBox.  

In general, experienced listeners exhibit a larger difference across the models, due to their greater sensitivity to musical nuances. However, text correspondence does not seems to significantly influence listeners overall perception. Although RenderBox is the only model capable of speed control based on the text prompt, this feature alone does not seem to heavily impact MOS score compared to the output of the symbolic models as long as they sound musically correct.  

![](images/b59ce874893ec5ae8247e1cc6322e0d1ff777631ad2c13fab7b91fdeef132de6.jpg)  
Figure 4: Input MIDI piano rolls and output spectrograms with respect to different text prompting. All visualizations are 20-second windows.  

# 5.3 Results and Ablation  

As shown in table 3, RenderBox largely outperforms the compared baselines in most metrics, except the CLAP score and tempo deviations from the synthesis stage. [to add] Comparison with RenderBox-no-CL demonstrates the effectiveness of the curriculum learning approach: Despite training with the same data with the same number of iterations, RenderBox-no-CL does not perform as well as the CL version from any of the stages. The forgetting phenomenon is also observed in the evaluations: For the RenderBox models trained up to stage 2 and stage 4, their performance on the stage 0 synthesis task decreased dramatically, as they have been fitted with increasingly diverse distributions.  

The controlnet experiment, trained with full set of data in a non-bootstrapped manner, could not outperform the main RenderBox experiments’ result. The zero-convolutioninserted MIDI tokens does not yield a strong impact on the output audio compared to the text conditioning.  

# 5.4 The Performer-Piece Embedding Space  

In stage 4, we directed the model to learn highly specific performance styles by fitting the model on ATEPP, a collection of virtuoso piano recordings. Although evaluating a pianist’s unique style is inherently challenging due to the expertise required, the outputs of our model received very positive feedback during informal interviews with students and professors from conservatories. ††  

In figure 5 we surveyed 380 test data pieces, generating with prompt “In the style of pianist X,” where X is one of ten famous pianists. We plot the final step’s denoised latent as a t-SNE reduction (perplexity=30). Some pianists’ styles clustered prominently regardless of the piece like Argerich, Cortot and Gould. Composers also form clusters, indicating a greater consistency across interpretations that transcends individual performer styles, such as the Debussy cluster, Bach cluster and Mozart cluster at the bottom. Within the composer clusters, we also witnessed patterns of style proximity such as Gilels and Richter, who are often close regardless of piece, which can be explained by the fact that they are both Russian Silver Age pianists and Neuhaus’s pupils [Razumovskaya, 2018]. Modern pianists such as Yuja Wang feature much more spread-out interpretations according to the model.  

![](images/b8860957109e156138f5c62eeef79b5d59c12ddd38e51c01c2cdbe32f9a0e5cf.jpg)  
Figure 5: t-SNE visualization of generation with testing data subset, colored by performers and shaped by composers.  

# 6 Conclusion and Future Work  

We introduced RenderBox, the first model capable of generating expressive performances with text-based control. Combining coarse language descriptions with fine- grained MIDI conditioning, RenderBox achieves flexible control for speed, mistakes, and style diversity with multiple instruments.  

While RenderBox demonstrates robust performance, limitations persist in its handling of acoustic quality due to the lack of detailed annotations for instrument timbres in the training data. Still, as the model bridges symbolic scores and audio, future creative applications can leverage annotated datasets to support instrument transfer, orchestral arrangements from symbolic scores, refinement of MIDI inputs with mistakes into polished audio outputs, and even generation with specific performance techniques.  

References   
[Agostinelli et al., 2023] Andrea Agostinelli, Timo I. Denk, Zalán Borsos, Jesse Engel, Mauro Verzetti, Antoine Caillon, Qingqing Huang, Aren Jansen, Adam Roberts, Marco Tagliasacchi, Matt Sharifi, Neil Zeghidour, and Christian Frank. MusicLM: Generating Music From Text. arXiv preprint arXiv:2301.11325, 2023.   
[Bhatt et al., 2024] Ruchi Bhatt, Pratibha Kumari, Dwarikanath Mahapatra, Abdulmotaleb El Saddik, and Mukesh Saini. Characterizing Continual Learning Scenarios and Strategies for Audio Analysis. Arxiv preprint arXiv:2407.00465, jun 2024.   
[Borovik and Viro, 2023] Ilya Borovik and Vladimir Viro. ScorePerformer : Expressive Piano Performance Rendering with Fine-grained Control. In Proceeding of the 24th International Society on Music Information Retrieval (ISMIR), Milan, Italy, 2023.   
[Bresin et al., 2002] Roberto Bresin, Anders Friberg, and Johan Sundberg. Director Musices : The KTH Performance Rules System. Special Interest Group on Music and Computer(SIGMUS) - Kyoto, pages 43–48, 2002.   
[Cancino-Chacón et al., 2020] Carlos Cancino-Chacón, Silvan Peter, Shreyan Chowdhury, Anna Aljanaki, and Gerhard Widmer. On the Characterization of Expressive Performance in Classical Music: First Results of the Con Espressione Game. In Proceedings of the 21st International Society for Music Information Retrieval Conference (ISMIR), 2020.   
[Cancino-Chacón, 2018] Carlos Eduardo Cancino-Chacón. Computational Modeling of Expressive Music Performance with Linear and Non-linear Basis Function Models. PhD thesis, Johannes Kepler University Linz, 2018.   
[Ciranni et al., 2025] Ruben Ciranni, Emilian Postolache, Giorgio Mariani, Michele Mancusi, Luca Cosmo, and Emanuele Rodolà. COCOLA: Coherence- Oriented Contrastive Learning of Musical Audio Representations. In Proceeding of the IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2025.   
[Copet et al., 2023] Jade Copet, Felix Kreuk, Itai Gat, Tal Remez, David Kant, Gabriel Synnaeve, Yossi Adi, and Alexandre Défossez. Simple and Controllable Music Generation. In Proceedings of the Conference on Neural Information Processing Systems (NeurIPS), 2023.   
[Cramer et al., 2019] Jason Cramer, Ho Hsiang Wu, Justin Salamon, and Juan Pablo Bello. Look, Listen, and Learn More: Design Choices for Deep Audio Embeddings,. In In Proceedings of the ICASSP 2019 - 2019 IEEE International Conference on Acoustics, Speech and Signal (ICASSP), 2019.   
[Dong et al., 2022] Hao-Wen Dong, Cong Zhou, Taylor Berg-Kirkpatrick, and Julian Mcauley. Deep Performer: Score-to-audio Music Performance Synthesis. Proceeding of the IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2022.   
[Evans et al., 2024] Zach Evans, Julian D. Parker, CJ Carr, Zack Zukowski, Josiah Taylor, and Jordi Pons. Stable Audio Open. Arxiv preprint arXiv:2407.14358, jul 2024.   
[Faber et al., 2024] Kamil Faber, Dominik Zurek, Marcin Pietron, Nathalie Japkowicz, Antonio Vergari, and Roberto Corizzo. From MNIST to ImageNet and back: benchmarking continual curriculum learning. Machine Learning, 113(10), mar 2024.   
[Hashida et al., 2008] M Hashida, M Nakra, H Katayose, T Murao, K Hirata, K Suzuki, and T Kitahara. Rencon: Performance Rendering Contest for Automated Music Systems. In Proceedings of the 10th International Conference on Music Perception and Cognition (ICMPC)., Sapporo, 2008.   
[Hawthorne et al., 2022] Curtis Hawthorne, Ian Simon, Adam Roberts, Neil Zeghidour, Josh Gardner, Ethan Manilow, and Jesse Engel. Multi-instrument Music Synthesis with Spectrogram Diffusion. In Proceeding of the International Society on Music Information Retrieval (ISMIR), Bengaluru, India, 2022.   
[Jeong et al., 2019] Dasaem Jeong, Taegyun Kwon, Yoojin Kim, Kyogu Lee, and Juhan Nam. VirtuosoNet: A Hierarchical RNN-based System for Modeling Expressive Piano Performance. In Proceedings of the 20th International Society for Music Information Retrieval Conference (ISMIR), Delft, Netherlands, 2019.   
[Le et al., 2023] Matthew Le, Apoorv Vyas, Bowen Shi, Brian Karrer, Leda Sari, Rashel Moritz, Mary Williamson, Vimal Manohar, Yossi Adi, Jay Mahadeokar, and Wei Ning Hsu. Voicebox: Text -Guided Multilingual Universal Speech Generation at Scale. In Advances in Neural Information Processing Systems, 2023.   
[Lin et al., 2024] Liwei Lin, Gus Xia, Junyan Jiang, and Yixiao Zhang. Content-based Controls for Music Large Language Modeling. In Proceeding of the 25t International Society on Music Information Retrieval (ISMIR), 2024.   
[Maezawa et al., 2019] Akira Maezawa, Kazuhiko Yamamoto, and Takuya Fujishima. Rendering music performance with interpretation variations using conditional variational RNN. Proceedings of the 20th International Society for Music Information Retrieval Conference (ISMIR), 2019.   
[Maman and Bermano, 2022] Ben Maman and Amit H. Bermano. Unaligned Supervision for Automatic Music Transcription in-the-Wild. In Proceedings of Machine Learning Research, volume 162, 2022.   
[Melechovsky et al., 2024] Jan Melechovsky, Zixun Guo, Deepanway Ghosal, Navonil Majumder, Dorien Herremans, and Soujanya Poria. Mustango: Toward Controllable Text-to-Music Generation. In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, NAACL, volume 1, 2024.   
[Morsi et al., 2024] Alia Morsi, Huan Zhang, Akira Maezawa, Simon Dixon, and Xavier Serra. Simulating Piano Performance Mistakes for Music Learning. In Proceedings of the Sound and Music Computing Conference (SMC), 2024.   
[Nistal et al., 2024] Javier Nistal, Marco Pasini, Cyran Aouameur, Maarten Grachten, and Stefan Lattner. DiffA-Riff: Musical Accompaniment Co-creation via Latent Diffusion Models. Proceeding of the 25th International Society on Music Information Retrieval (ISMIR), 2024.   
[Palmer, 1996] Caroline Palmer. Anatomy of a performance: Sources of musical expression. Music Perception, 13(3):433–453, 1996.   
[Perez et al., 2018] Ethan Perez, Florian Strub, Harm De Vries, Vin cent Dumoulin, and Aaron Courville. FiLM: Visual reasoning with general conditioning layer. In 32nd AAAI Conference on Artificial Intelligence, AAAI 2018, New Orleans, USA, 2018.   
[Peter et al., 2023] Silvan David Peter, Carlos Eduardo Cancino-chacón, Francesco Foscarin, Florian Henkel, and Gerhard Widmer. Automatic Note-Level Alignments in the ASAP Dataset. Transactions of the International Society for Music Information Retrieval (TISMIR), 2023.   
[Razumovskaya, 2018] Maria Razumovskaya. Heinrich Neuhaus: A Life beyond Music. Boydell & Brewer, NED new edition, 2018.   
[Renault et al., 2022] Lenny Renault, Rémi Mignot, and Axel Roebel. Differentiable Piano Model for MIDI-toAudio Performance Synthesis. In Proceedings of the International Conference on Digital Audio Effects, DAFx, volume 3, pages 232–239, 2022.   
[Repp, 1995] Bruno H Repp. Quantitative Effects of Global Tempo on Expressive Timing in Music Performance: Some Perceptual Evidence. Music Perception, 13(1):39– 57, 1995.   
[Repp, 1996] Bruno H Repp. Pedal Timing and Tempo in Expressive Piano Performance: A Preliminary Investigation. Psychology of Music, 24(2):199–221, 1996.   
[Rhyu et al., 2022] Seungyeon Rhyu, Sarah Kim, and Kyogu Lee. Sketching the Expression: Flexible Rendering of Expressive Piano Performance with Self-Supervised Learning. In Proceeding of the International Society on Music Information Retrieval (ISMIR), Bengaluru, India, 2022.   
[Riley et al., 2024] Xavier Riley, Zixun Guo, Drew Edwards, and Simon Dixon. GAPS: A Large and Diverse Classical Guitar Dataset and Benchmark Transcription Model. In Proceeding of the 25th Inter national Society on Music Information Retrieval (ISMIR), aug 2024.   
[Tal et al., 2024] Or Tal, Alon Ziv, Itai Gat, Felix Kreuk, and Yossi Adi. Joint Audio and Symbolic Conditioning for Temporally Controlled -to-Music Generation. In Proceeding of the 25t International Society on Music Information Retrieval (ISMIR), jun 2024.   
[team, 2023] AudioBox team. Audiobox: Unified Audio Generation with Natural Language Prompts. Arxiv preprint arXiv:2312.15821, 2023.   
[Wang et al., 2019] Zhepei Wang, Cem Subakan, Efthymios Tzinis, Paris Smaragdis, and Laurent Charlin. Continual learning of new sound classes using generative replay. In IEEE Workshop on Applications Signal Processing to Audio and Acoustics, volume 2019 -Octob, jun 2019.   
[Wang et al., 2022] Xin Wang, Yudong Chen, and Wenwu Zhu. A Survey on Curriculum Learning. IEEE Transactions on Pattern Analysis and Machine Intelligence, 44(9), oct 2022.   
[Wu et al., 2023] Yusong Wu, Ke Chen, Tianyu Zhang, Yuchen Hui, Taylor Berg-Kirkpatrick, and Shlomo Dubnov. Large-Scale Contrastive Language-Audio Pretraining with Feature Fusion and Keyw ord-to-Caption Augmentation. In ICASSP, IEEE International Conference on Acoustics, Speech and Signal Processing Proceedings, 2023.   
[Wu et al., 2024] Shih Lun Wu, Chris Donahue, Shinji Watanabe, and Nicholas J. Bryan. Music ControlNet: Multiple Time-Varying Controls for Music Generation. IEEE/ACM Transactions on Audio Speech and Language Processing, 2024.   
[Zhang et al., 2022] Huan Zhang, Jingjing Tang, Syed Rafee, Simon Dixon, and George Fazekas. ATEPP: A Dataset of Automatically Transcribed Expressive Piano Performance. In Proceedings of the International Society for Music Information Retrieval Conference (ISMIR), Bengaluru, India, 2022.   
[Zhang et al., 2023] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding Conditional Control to Text-to- Image Diffusion Models. In Proceedings of the IEEE International Conference on Computer Vision, 2023.   
[Zhang et al., 2024a] Huan Zhang, Shreyan Chowdhury, Carlos Eduardo Cancino-Chacón, Jinhua Liang, Simon Dixon, and Gerhard Widmer. DExter: Learning and Controlling Performance Expression with Diffusion Models. Applie d Sciences, 14( 15), 2024.   
[Zhang et al., 2024b] Huan Zhang, Jinhua Liang, Liang, and and Simon Dixon. From Audio Encoders to Piano Judges: Benchmarking Performance Understanding for Solo Piano. In Proceeding of the 25t International Society on Music Information Retrieval (ISMIR), 2024.   
[Zhang et al., 2024c] Yixiao Zhang, Yukara Ikemiya, Woosung Choi, Naoki Murata, Marco A. MartínezRamírez, Liwei Lin, Gus Xia, Wei-Hsiang Liao, Yuki Mitsufuji, and Simon Dixon. Instruct-MusicGen: Unlocking Text-to-Music Editing for Music Language Models via Instruction Tuning. In Proceeding of the International Joint Conference on Artificial Intelligence (IJCAI), 2024.   
[Zhang et al., 2025] Huan Zhang, Vincent Cheung, Hayato Nishioka, Simon Dixon, and Shinichi Furuya. LLaQo: Towards a Query-Based Coach in Expressive Music Performance Assessment. In Proceedings of the IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2025.  