# DISENTANGLING SCORE CONTENT AND PERFORMANCE STYLE FOR JOINT PIANO RENDERING AND TRANSCRIPTION  

Wei Zeng, Junchuan Zhao, Ye Wang∗   
National University of Singapore   
{w.zeng, junchuan}@u.nus. edu; wangye@comp nus. edu. sg  

# ABSTRACT  

表达性演奏呈现（EPR）和钢琴自动转录（APT）是音乐信息检索的基本任务，但两者是相反的：EPR从符号乐谱中生成表达性演奏，而APT从演奏中恢复乐谱。尽管它们具有双重性质，但先前的工作已经独立地解决了它们。在本文中，我们提出了一个统一的框架，通过从成对和非成对数据中分离音符级分数内容和全局表现风格表示，来联合建模EPR和APT。我们的框架建立在基于转换器的序列到序列（Seq2Seq）体系结构上，并且只使用序列对齐的数据进行训练，不需要细粒度的笔记级对齐。为了使渲染过程自动化，同时确保与乐谱的风格兼容性，我们引入了一个独立的基于扩散的性能风格推荐（PSR）模块，该模块直接从乐谱内容中生成风格嵌入。这个模块化组件既支持样式转换，也支持跨一系列表达风格的灵活呈现。客观和主观评估的实验结果表明，我们的框架在EPR和APT任务上实现了具有竞争力的性能，同时实现了有效的内容风格解纠缠、可靠的风格转移和风格适当的呈现。 Demos are available at https://jointpianist.github.io/epr-apt/.  

# 1 INTRODUCTION  

音乐以多种形式存在，尤其是象征性的乐谱和富有表现力的录音。这些音乐模式之间的转换对于使机器学习模型能够跨符号和音频域进行推理至关重要，从而支持从艺术创作到音乐教育的广泛应用（Cancino-Chaco ' n et al., 2023; Chaco ' n et al., 2023）。例如，在现场音乐会中，钢琴家将书面乐谱呈现为富有表现力的表演，并在时间、动态和发音上添加个性化的细微差别。相反，出于分析、重新表演或存档等目的，需要转录以将表演的音频记录转换回符号表示。这两个过程对应于音乐信息检索（MIR）中的两个核心任务：表现力表现呈现（EPR），它从符号乐谱中生成表演MIDI（捕捉表现力时间、动态和发音的MIDI） （Chac´on等人，2018），以及自动钢琴转录（APT），它从表演MIDI中预测符号乐谱（Desain & Honing, 1989）。

先前的工作将EPR和APT作为两个独立的任务进行研究（Maezawa等人，2019；Jeong等人，2019；Rhyu等人，2022；Borovik和Viro, 2023； Liu等人，2022；Cogliati等人，2016；Nakamura等人，2018；Shibata等人，2021）。然而，如图1左上角所示，这两个任务本质上是相连的，表示符号形式和表达形式之间的逆转换。在渲染中，表演既反映了作曲家的意图，也反映了圆周率
安尼斯特的诠释风格；在转录中，系统应该过滤掉这些表达元素，以恢复潜在的分数。

自动语音识别（ASR）和文本到语音（TTS）等语音任务中的联合建模显示出了互惠互利，并实现了弱监督训练(Ren et al., 2019; Peyser et al.，

2022年)。基于此，我们提出了一个统一的基于变压器的框架，该框架通过建模两个因素来共同学习EPR和APT:(a)音符级别的乐谱内容表示，它捕获音高和节奏等符号结构；(b)一个全局的表演风格表征，它封装了表演的高级艺术特征（例如，“重”或“放松”），并作为一个条件反射信号，指导解码器生成细粒度的表达细节。这种解纠缠的表示允许跨任务共享信息，同时保留呈现过程的可解释性和可控性。此外，使用统一的Seq2Seq架构使我们的模型能够仅使用序列对齐数据进行训练，从而消除了大多数EPR系统所需的注释级对齐的需要（Rhyu等人，2022；Borovik和Viro, 2023； Tang等人，2023；Jeong等人，2019；Zhang等人，2024）。

为了实现灵活和真实的性能呈现，区分在解纠缠表示中编码的信息类型是至关重要的。我们将风格定义为乐谱的表现力实现（例如，Widmer等人（2003）的“Horowitz因素”），流派定义为作品的潜在结构和和声特征。虽然两者都是全局属性，但它们捕获了不同的音乐方面。受乐谱分类最新进展的启发（Ji等人，2021；Pasquale等人，2020），我们假设，为了使演奏听起来自然，所选择的风格应该理想地与潜在的流派保持一致。这表明，风格上合适的表演可以直接从乐谱内容中推断出来，类似于熟练的钢琴家如何解读作品。此外，现有的EPR模型通常依赖于作曲家标签（Jeong等人，2019；Tang等人，2023）或需要手动控制表达参数（Borovik和Viro, 2023； Rhyu等人，2022），这限制了非专业用户的可访问性。基于这些观察结果，我们提出了一个性能风格推荐（PSR）模块，该模块仅根据分数产生不同的风格嵌入。

我们使用客观和主观指标来评估我们的框架。在标准基准测试中，我们的联合模型实现了EPR和APT的竞争性能。主观评价证实了EPR生成性能的自然性。通过风格转移和潜在空间可视化来验证解缠。此外，我们发现学习风格嵌入编码了表演者和作曲家的信息，其中作曲家的特征更占主导地位。最后，对PSR模块的评估表明，它能够仅从内容中生成风格适当的嵌入。
综上所述，本文做出了以下三点贡献：

一个统一的基于变压器的联合EPR和APT模型，它分离了得分内容和表现风格表示，并利用两个任务之间的对偶性进行相互监督。这种联合形式使音乐的象征性和表现性形式之间的双向建模成为可能。
一个基于扩散的表演风格推荐（PSR）模块，它直接从分数内容中生成多样化和适当的风格嵌入。这个模块模仿钢琴家的能力，从书面乐谱中推断出合适的表达风格，并使可控和非专家驱动的性能呈现。
Seq2Seq的EPR公式，没有笔记级对齐，这消除了对精细对齐的训练数据的需要，并且仅使用序列级监督即可实现可扩展的学习。尽管有这种宽松的监管，我们的模型与依赖于对齐的基线相比取得了具有竞争力的表现。
# 2 RELATED WORK  

# 2.1 EXPRESSIVE PIANO PERFORMANCE RENDERING  

EPR的早期工作依赖于基于规则的系统（Widmer & Goebl, 2004; Chac ' on等人，2018;Kirke & Miranda, 2013）。最近的方法利用深度学习，包括基于RNN和lstm的模型（Maezawa等人，2019；Jeong等人，2019），以及基于变压器的架构（Rhyu等人，2022；Borovik和Viro, 2023； Renault等人，2023；Tang等人，2023）。EPR的一个核心挑战是生成适当反映乐谱内容的演奏风格。现有的方法通常需要明确的作曲家或表演者标签（Jeong等人，2019；Tang等人，2023），或者依赖于对表达参数的手动控制（Borovik和Viro, 2023； Rhyu等人，2022），限制了非专业用户的可用性。一种基于扩散的模型已经被引入，依靠手工制作的音符级风格特征，直接从乐谱中生成表达性控制（Zhang et al., 2024）。然而，这种音符级的方法需要复杂的、细粒度的调整，并且在具有不同音乐结构的作品之间提供有限的灵活性。

当前模型的另一个关键限制（Rhyu等人，2022；Borovik和Viro, 2023； Tang等人，2023；Jeong等人，2019；Zhang等人，2024）是它们对注释对齐数据集的依赖，这些数据集通常需要使用对齐工具进行预处理（Nakamura等人，2017）。这种依赖阻碍了灵活性，特别是对于像颤音和颤音这样的表达技巧，它们会带来时间上的模糊性。已经提出了一种基于无监督gan的方法来绕过对齐（Renault et al., 2023），但它的性能不如有监督的方法。在这项工作中，我们通过将EPR表述为Seq2Seq任务并引入用于自动样式生成的PSR模块来解决这些限制。

# 2.2 AUTOMATIC PIANO TRANSCRIPTION  

Automatic piano transcription (APT) methods can be categorized by their input and output modalities. Input formats include raw audio signals (e.g., waveforms or spectrograms) and symbolic representations such as MIDI. Output targets are typically note-level sequences (Hawthorne et al., 2018; Kim & Bello, 2019; Kong et al., 2021; Toyama et al., 2023; Hawthorne et al., 2021) or notation- level formats resembling human-readable sheet music (Rom´an et al., 2019; Alfaro-Contreras et al., 2024; Rom´an et al., 2018; Zeng et al., 2024; Hiramatsu et al., 2021; Liu et al., 2021; 2022; Shibata et al., 2021; Beyer & Dai, 2024). This work focuses on symbolic-to-symbolic transcription, where the model maps expressive performance MIDI to corresponding score sheet representations.  

Early APT approaches relied on signal processing heuristics (Raphael, 2001) and probabilistic models such as Hidden Markov Models (HMMs) (Cogliati et al., 2016; Shibata et al., 2021). Recent advances leverage deep neural networks (Liu et al., 2022; Beyer & Dai, 2024; Suzuki, 2021), which have demonstrated substantial improvements in accuracy and generalization. Particularly, (Beyer & Dai, 2024) proposed a Seq2Seq framework that eliminates the need for note-aligned supervision while achieving state-of-the-art performance. Building on this insight, we adopt a similar Seq2Seq framework to model score content features within our unified system.  

# 2.3 DISENTANGLED REPRESENTATION LEARNING  

解纠缠表示学习（Disentangled representation learning， DRL）旨在学习分离观测数据变化的潜在因素的表示（Wang et al., 2024）。它在计算机视觉（Dupont, 2018； Yang等人，2021；Chen等人，2016；Karras等人，2020）和自然语言处理（He等人，2017；Bao等人，2019；Cheng等人，2020；Wu等人，2020）中得到了广泛的研究，其中将内容与风格或语义分离可以提高通用性和可控制性。

在音乐信息检索（MIR）中，DRL最近被用于分离音乐内容和风格以支持生成和操作（Tan & Herremans, 2020； Wang等人，2020；Yang等人，2019；Zhao等人，2024）。一项密切相关的研究（Zhang & Dixon, 2023）以无监督的方式从富有表现力的表演中学习内容和风格表征，从而实现音乐分析和风格转移。相比之下，我们的工作侧重于从符号乐谱中生成富有表现力的表演，这是基于drl的音乐建模的一个较少探索但重要的方向。

# 3 METHODOLOGY  

# 3.1 DATA REPRESENTATION FOR INPUT AND OUTPUT  

根据Peyser等人（2022b）的研究，我们将分数和性能输入都表示为长度大致相等的音符级序列，从而使联合编码器能够学习分数内容的域无关表示。每个序列按开始时间和音高的顺序包含N个音符，每个音符表示为K个离散符号属性的元组，详见附录a .2。我们分别用x和y表示得分和表现序列。对于分数输入，每个音符包含K = 7个属性，而性能输入包含K = 4个属性。最后的注释嵌入是通过对其组成属性的嵌入求和得到的，得到Ex， Ey RN×D，其中D表示嵌入维数。

![](images/00a7739102f767bce06d33fb99b2ad5272f2b76249be44d18ea502eaceb9faac.jpg)  
图1:EPR与APT之间的关系（左上）以及拟议框架的概述。该模型包括一个用于EPR和APT的基于变压器的联合架构，以及一个基于扩散的性能风格推荐（PSR）模块。联合训练4个任务：掩码重构、掩码性能重构、表现力性能呈现（EPR）和自动性能转录（APT）。分数内容特征zx和zy，分别从分数和表现输入中提取，被鼓励对齐。学习全局风格特征zs作为解纠缠因子，支持风格迁移。PSR模块经过独立训练，仅从乐谱内容生成z，模拟钢琴家选择适当演奏风格的能力。

对于分数预测（x´），我们采用Beyer & Dai（2024）中引入的表示方案。对于性能预测（y -），我们最初应用了与输入表示中使用的相同的标记化，但观察到它降低了生成质量。由于我们的Seq2Seq模型不需要笔记级对齐，因此我们采用了Huang & Yang（2020）提出的结构化性能表示（通过MidiTok库实现）（Fradet et al., 2021）。

# 3.2 UNIFIED MODELING OF EPR AND APT  

We consider two domains of symbolic musical sequences: score sequences x E X and performance sequences y . These two domains are connected by two inverse processes: expressive performance rendering (EPR), mapping scores to performances (X → ), and automatic performance transcription (APT), mapping performances to scores (Y → ). Both domains share a latent content space Zc, capturing note-level attributes such as pitch and rhythm. In contrast, Y additionally depends on a style space Zs, serving as a conditioning signal for the high-level summary of its overall expressive interpretation. Our framework supports training on both paired and unpaired data.  

Paired setting Given paired data (x, y), we define content encoders fc,X 七 → Zc and fc, : Y → Zc, along with a style encoder fs, Y → Zs, producing:  

We perform the EPR and APT tasks by decoding from these latent representations:  

where 田 denotes broadcasted addition of the global style vector to each time step in zx. Both decoders are optimized via cross-entropy losses:  

Unpaired setting To incorporate unpaired data, we adopt a masked reconstruction objective inspired by masked autoencoders (He et al., 2022). Specifically, we define x˜ = MASK(x) and y˜ = MASK(y), where MASK(·) randomly replaces a subset of input tokens with a special ⟨MASK⟩ token during encoding. The model is then trained to reconstruct the full original sequence:  

We encourage disentanglement between the content space Zc and the style space Zs through both training objectives and architectural design. From a training perspective, The content encoders fc,X (·) and fc,Y (·) are supervised to capture score-relevant information via losses from APT, EPR, and masked reconstruction tasks. Architecturally, We represent content and style at distinct levels: zc encodes fine-grained, note-level attributes such as pitch and rhythm as a sequence of latent vectors, while zs summarizes the overall expressive style as a single latent vector.  

To regularize the style space and promote smoothness, we impose a Kullback-Leibler divergence penalty between the posterior over zs and a standard Gaussian prior:  

The total training objective integrates three components: supervised losses from EPR and APT on paired data, reconstruction losses from masked inputs on unpaired data, and KL regularization on the style representation:  

# 3.4 MODELING OF PERFORMANCE STYLE RECOMMENDATION  

After training the joint model with disentangled representations, we introduce an independent performance style recommendation (PSR) module that generates style embeddings conditioned solely on score content. This setup mimics the behavior of a pianist who selects an expressive style based on the music score alone. The goal is to model the distribution of plausible performance styles for a given score x, enabling flexible and automated expressive rendering.  

Training Given a paired sample (x, y), the ground-truth style embedding zs = fs,Y (y) is extracted from our frozen, pre-trained joint model. A separate score encoder fg, (·) concurrently extracts a global content representation eg = fg, (x). We then adopt a denoising diffusion probabilistic model (DDPM) (Ho et al., 2020) to learn the conditional distribution p(zs eg), jointly training the diffusion denoiser and fg,X (·). The forward process perturbs the style vector by adding Gaussian noise:  

and the reverse process learns to denoise zts conditioned on eg and the diffusion step t. The style generator gs(·) is trained to predict the added noise and is optimized using the following objective:  

Inference At inference time, given x, a style embedding zˆs is generated by sampling from a standard Gaussian prior and iteratively denoising it using the trained model, conditioned on eg = fg, (x). The resulting pair (x, zˆs) is passed to the decoder gY(·) to synthesize the expressive performance yˆ.  

# 3.5 MODEL ARCHITECTURE  

Joint model of EPR and APT As illustrated in Figure 1, the joint model consists of five transformerbased components: Score Encoder, Performance Encoder, Style Encoder, Score Decoder, and Performance Decoder. Each component adopts a standard transformer architecture (Vaswani et al., 2017) with six layers and eight attention heads, selected for their ability to model long-range dependencies and scale effectively to large symbolic music datasets. We employ rotary positional encodings (Su et al., 2024), pre-layer normalization (Brown et al., 2020), and SwiGLU activations (Shazeer, 2020), with a feed-forward hidden dimension of 3072. Decoder outputs are projected to token distributions via parallel linear layers where applicable. To obtain global style embedding, we follow the BERT architecture (Devlin et al., 2019) in the Style Encoder by prepending a special ⟨CLS⟩ token to the input sequence and taking the final hidden state corresponding to this token as the style vector.  

Performance style recommendation A separate transformer encoder, architecturally aligned with the Style Encoder, is used to extract a global score representation. A ⟨CLS⟩ token is prepended to the input score sequence, and its final hidden state is used as the global content embedding eg, which conditions the style generation process.  

During training, a ground -truth style vector zs, obtained from the joint model, is perturbed using a forward diffusion process. The diffusion timestep is encoded using sinusoidal positional embeddings and concatenated with eg and the noisy style vector zts. This combined representation is passed through a feed-forward network (FCN) to predict the injected noise ϵ. The model is trained using a mean squared error (MSE) loss between the predicted and true noise.  

# 4 EXPERIMENTS  

# 4.1 DATASETS  

We use the ASAP dataset (Foscarin et al., 2020) for both paired training and evaluation, as it provides aligned annotations between musical scores and expressive performances. We select 967 high-quality performances and split them into training, validation, and test sets with an 8:1:1 ratio, same as Beyer & Dai (2024). To enable unpaired training, we curate an unpaired score dataset consisting of 75,913 public-domain MusicXML files collected from MuseScore1 We also compile an unpaired performance dataset by sourcing piano cover videos from YouTube and transcribing the audio into performance MIDI using a state-of-the-art audio-to-MIDI transcription model2. The model is selected based on a pilot study demonstrating strong accuracy in both note and pedal transcription. To evaluate the generalization of disentangled representations in out-of-distribution (OOD) settings, we additionally use the ATEPP dataset (Zhang et al., 2022), which contains 11,674 performances by 49 pianists spanning 25 composers, with explicit annotations of both composer and performer identities.  

# 4.2 TRAINING SETUP  

The joint model is trained on 3 NVIDIA A5000 GPUs with a total batch size of 144 sequences, each containing 256 notes. Each training step comprises 36 sequences for EPR, APT, score reconstruction, and performance reconstruction, respectively. Optimization is performed using AdamW (Loshchilov & Hutter, 2019) for 40,000 steps, with a cosine decay learning rate schedule and linear warmup over the first 4,000 steps, peaking at 5 × 10−5. The PSR model is trained separately on a single GPU with a batch size of 48, using the same schedule but with a peak learning rate of 1 X 10−4.  

# 4.3 METRICS  

APT We evaluate APT using two widely adopted metrics: MUSTER (Nakamura et al., 2018; Hiramatsu et al., 2021) and ScoreSimilarity (Suzuki, 2021; Cogliati & Duan, 2017). MUSTER assesses high-level transcription accuracy with a focus on rhythmic structure, including sub-metrics such as pitch edit distance (Ep), missing notes (Emiss), extra notes (Eextra), onset deviation (Eonset), and offset deviation (Eoffset). ScoreSimilarity also captures pitch-level edit distances (Emiss, Eextra), with additional metrics for stem direction (Estem), pitch spelling (Espell), and staff assignment (Estaff).  

EPR We use both objective and subjective evaluations. Objectively, we compare the generated performance to its human reference and compute three metrics: alignment rate, insertion rate, and missing rate. Besides, we conduct objective statistics using three metrics (Tang et al., 2023; Zhang et al., 2024): per-note variance of onset, duration, and velocity; KL divergence from human distributions; and note-aligned mean absolute error (MAE) relative to human references. Subjectively, we conduct a listening test with eleven participants trained in music performance. We randomly sample five pieces from Bach, Rachmaninoff, Schubert, Scriabin, and Ravel to cover a range of genres and styles. Each participant rates the outputs in randomized order on a 5-point Likert scale (1–5) across four dimensions: dynamics, tempo, style, and overall human- likeness.  

Table 1: APT results on the ASAP dataset. Lower values indicate better performance across all metrics. The best results are shown in bold, and the second-best are underlined.   

![](images/9ab494584eefd4a2cf76a5c69592207b90f6389e50645731c733a00db85653f6.jpg)  

Table 2: Objective evaluation of EPR results. We compare variance (σ2), KL divergence, and MAE for onsets (O), durations (D), and velocities (V ). For σ2, values closer to the Human reference are better. For all other metrics, lower is better. Best results are in bold; second-best are underlined.   

![](images/4c01e9d90eb64908136b922c1db7dac670beaf2c47418a8af21b780e36d09f14.jpg)  

Table 3: Objective evaluation of EPR accuracy on test samples using alignment (Align), insertion (Insert), and missing (Miss) rates.   

![](images/78a8bfa86d181af5f880e2322259e41ff9a5774f3ded394fd0d183bd3692cbee.jpg)  

Table 4: Performer (Perf) and composer (Comp) identification accuracy based on performance style (Style) and score content (Cont).   

![](images/b6f42341971c9858916232ca94634548b66fe842eec839b4fca7cd1ca7165f26.jpg)  

# 5 RESULTS  

# 5.1 EPR AND APT PERFORMANCE  

APT As shown in Table 1, our model achieves performance comparable to the state-of-the-art APT system, indicating that the learned score representations capture key musical attributes such as pitch, rhythm, and structure. Our alignment-free Seq2Seq formulation achieves competitive results without requiring explicit note-level alignment. In contrast, methods such as Liu et al. (2022) and Shibata et al. (2021) attain lower pitch errors by relying on note-aligned data, which simplifies pitch and onset prediction, but limits flexibility in musically complex, one-to-many contexts (e.g. ornaments, trills, or expressive deviations).  

EPR We compare against two strong alignment-based baselines: VirtuosoNet Jeong et al. (2019) and DExter Zhang et al. (2024). Our method is evaluated under two conditions: with extracted target styles (Ours–Target) and with PSR-generated styles (Ours–PSR). We also take score MIDI (Score) as a baseline model; it is shaded in gray in Table 2 and Table 3 to indicate that it is not an EPR model and serves only as a comparison anchor.  

The objective statistics in Table 2 indicate that our models exhibit duration and velocity variances that closely match those of human performances, reflecting natural variability. While DExter shows even larger velocity variance (326.33), this does not translate to better quality, as listening tests suggest it results from unstable dynamics rather than meaningful expressiveness. Moreover, our models achieve lower KL and MAE scores than most baselines (especially Ours–Target), confirming that they faithfully replicate the fine-grained expressive details found in human renditions.  

![](images/81c68dc983b468f8553ce2fa7e77744ea061d663ed59b7a65e7e508ef11faf79.jpg)  
(a) Subjective ratings of PSR outputs across musical (b) Breakdown of the overall subjective ratings by comattributes (dynamics, tempo, and style). posers.  

![](images/cd902068242a5762ac123bf0edeb5da867582b75b90345271b3fbe29b795eaaf.jpg)  

![](images/c348d904ba8064cdc5dda5569d6bc8add32cf9bf894c5cb591e55073a6de3d13.jpg)  
Figure 2: Subjective evaluation of expressive piano rendering performance across different systems, including human renditions, direct-from-score, baselines, and our proposed models.   
(b) Two-dimensional projection of style embeddings, colored by performer clusters.  

(a) Two-dimensional projection of style embeddings, colored by composer clusters.  

Figure 3: Two-dimensional visualization of performance style representations from real performances, with colors indicating clusters by composer or performer.  

The accuracy evalution in Table 3 shows that Ours (PSR) achieves the highest alignment rate (92.27%) and the lowest insertion rate (3.77%), demonstrating the effectiveness of our alignment-free sequenceto-sequence formulation. Subjective results in Figure 2 show that Ours (Target) achieves the highest ratings across all attributes and styles, with Ours (PSR) closely following and outperforming baseline systems. Both variants perform strongly across composers, particularly on Bach and Scriabin.  

# 5.2 REPRESENTATION DISENTANGLEMENT  

Performer/composer identification To further analyze the structure of the learned representations, we perform performer and composer identification using score content and performance style representations on the ATEPP dataset Zhang et al. (2022), which is split into training, validation, and test sets with an 8:1:1 ratio. We evaluate four model configurations: using either the score content or performance style representation as input, and predicting either the composer or performer as the target. Each performance MIDI is segmented into 256-note chunks and processed by the trained joint model to extract latent representations, which are then averaged across chunks to obtain a single representation per piece. For visualization, we insert a 2D bottleneck layer before the classification head and project the resulting embeddings onto 2D plane. The classification results and visualization are presented in Table 4 and Figure 3, respectively.  

The results in Table 4 demonstrate the effectiveness of the disentangled representations. Classifiers using the style representation zs achieve substantially higher composer and performer accuracy than those using the content representation zc, confirming successful disentanglement of performance style from score content. While zc primarily encodes pitch and rhythmic structure, it is expected to preserve performance-independent musical characters (e.g. composer-specific information). This explains why the composer classifier using zc (Cont→Comp) still achieves a non -trivial accuracy of 29.99%. Notably, the composer classifier using zc (Style→Comp) shows much higher accuracy (77.46%). Beyond the effective disentanglement, we attribute this result to two other factors: first, as a global embedding, zs is better suited for capturing high-level stylistic features than the note-level zc; second, professional pianists often align their performance style with the composer’s stylistic conventions, thereby encoding composer information directly into their expression.  

![](images/37cc6402d2a7202f3cb459e63312f9d95cf3836a1bc8816b3df850f5d74f5db0.jpg)  
Figure 4: Two-dimensional visualization of style representations across historical eras. Colored regions denote era -specific clusters with centroids marked by black crosses; yellow arrows indicate temporal progression of musical styles.  

Two-dimensional projection of style embed- (b) Two-dimensional projection of style embeddings extracted from actual performances using dings generated by the PSR model from correthe joint model. sponding scores.  

The visualization in Figure 3 further supports our findings, with style embeddings forming clear clusters by composer and performer. We also observe that embeddings from human performances contain information about both the artist and the composition. This further supports our assumption that skilled pianists adapt their style to the piece, validating the motivation behind our PSR module.  

Style transfer evaluation To further evaluate the disentanglement of content and style, we conducted a subjective listening test on style transfer between pieces from distinct genres. For each test case, listeners rated generated outputs on two criteria: style similarity to a reference performance and overall listening quality. We compared three conditions for the rendered style: the original (Original), the transferred reference style (Target), and an interpolation of both (Mean) to study the learned style feature space. As shown in Figure 5, the Target condition achieves the highest style similarity ratings in Samples 1 and 3, indicating successful transfer. Notably, this improvement does not compromise overall quality. The Mean condition yields consistently strong quality across all samples, suggesting that the style space is well-structured and supports smooth interpolation.  

# 5.3 EFFECTIVENESS OF PSR  

To evaluate the styles generated by the PSR model, we collect 5,003 performances from the ATEPP dataset with aligned scores. For each performance, we obtain two style vectors: one extracted directly from the performance using the joint model, and one generated from the corresponding score using the PSR model. Each piece is assigned to one of four historical eras— Baroque, Classical, Romantic, or Modern— based on title and composer metadata parsed using GPT-4o mini (Achiam et al., 2023).  

![](images/ebf9063cb8d7dad17c5e12036ccbf2c12dc579deacc9c0319ac2dc682ee9f81e.jpg)  
Figure 5: Subjective ratings for three generated samples using different style settings. Listeners rated each output on style similarity and overall listening quality.  

We project the style vectors into 2D using the classifier from Section 5.2. As shown in Figure 4, the PSR-generated styles (right) closely mirror those extracted from real performances (left), exhibiting similar clustering structure, era-wise separation, and centroid locations. This alignment, together with the subjective results in Figure 2, supports the PSR model’s ability to synthesize stylistically meaningful embeddings from score content alone.  

# 6 CONCLUSION  

In this paper, we present a unified framework for expressive piano performance rendering (EPR) and automatic performance transcription (APT), built upon disentangled latent representations of score content and performance style. To enable flexible style-aware rendering, we introduce a DDPM-based Performance Style Recommendation (PSR) module that generates expressive styles directly from score content. Evaluated through objective metrics, subjective listening tests, and representation visualizations, our approach achieves performance on par with state-of-the-art methods across both EPR and APT tasks. Our findings demonstrate that: (a) the joint model effectively learns disentangled representations of content and style; (b) EPR can be formulated as a sequence-to-sequence task without requiring note-level alignment; (c) the model supports flexible style transfer; and (d) the PSR module produces stylistically appropriate outputs conditioned solely on the score. As future work, we aim to extend this framework to popular music, which presents greater stylistic diversity and practical relevance than classical music.  

# ETHICS STATEMENT  

The authors have reviewed and conformed in every respect with the ICLR Code of Ethics https: //iclr. cc/public/CodeOfEthics. The human study in our experiment is based on online crowdsourcing, which bears minimum risk. Participants are informed that participation in our study is enstirely voluntary and that they may choose to stop participating at any time without any negative consequences. No personally identifying information is collected in the human study.  

# REPRODUCIBILITY STATEMENT  

We introduce our dataset and experimental settings in Section 4.1 and ection 4.2, respectively. We also provide details of model architectures necessary for reproduction in Appendix B. The code will be released upon acceptance with sufficient instructions for reproducing the model architecture and training pipeline using public datasets such as ASAP and ATEPP.  

# REFERENCES  

Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.   
Marı´a Alfaro-Contreras, Rı´os-Vila, Jose J. Valero-Mas, and Jorge Calvo-Zaragoza. A transformer approach for polyphonic audio-to-score transcription. In IEEE International Conference on Acoustics, Speech and Signal Processing, ICASSP 2024, Seoul, Republic of Korea, April 14-19, 2024, pp. 706–710. IEEE, 2024. doi: 10.1109/ICASSP48485.2024.10447162.   
Yu Bao, Hao Zhou, Shujian Huang, Lei Li, Lili Mou, Olga Vechtomova, Xin-Yu Dai, and Jiajun Chen. Generating sentences from disentangled syntactic and semantic spaces. In Anna Korhonen, David R. Traum, and Lluı´s Ma\`rquez (eds.), Proceedings of the 57th Conference of the Association for Computational Linguistics, ACL 2019, Florence, Italy, July 28- August 2, 2019, Volume 1: Long Papers, pp. 6008–6019. Association for Computational Linguistics, 2019. doi: 10.18653/V1/ P19-1602.   
Tim Beyer and Angela Dai. End-to-end piano performance-midi to score conversion with transformers. In Blair Kaneshiro, Gautham J. Mysore, Oriol Nieto, Chris Donahue, Cheng-Zhi Anna Huang, Jin Ha Lee, Brian McFee, and Matthew McCallum (eds.), Proceedings of the 25th International Society for Music Information Retrieval Conference, Retrieval Conference, ISMIR 2024, ISMIR 2024, San Francisco, California, USA and Online, November 10-14, 2024, pp. 319–326, 2024. 10.5281/ZENODO.14877339.   
Ilya Borovik and Vladimir Viro. Scoreperformer: Expressive piano performance rendering with fine-grained control. In Augusto Sarti, Fabio Antonacci, Mark Sandler, Paolo Bestagini, Simon Dixon, Beici Liang, Gae¨l Richard, and Johan Pauwels (eds.), Proceedings of the 24th International Society for Music Information Retrieval Conference, ISMIR 2023, Milan, Italy, November 5-9, 2023, pp. 588–596, 2023. doi: 10.5281/ZENODO.10265355.   
Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners. In Hugo Larochelle, Marc’Aurelio Ranzato, Raia Hadsell, Maria-Florina Balcan, and Hsuan-Tien Lin (eds.), Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual, 2020.   
Carlos Cancino-Chaco´n, Silvan David Peter, Emmanouil Karystinaios, Francesco Foscarin, Maarten Grachten, and Gerhard Widmer. Partitura: A python package for symbolic music processing. arXiv  

preprint arXiv:2206.01071, 2022.  

Carlos Cancino-Chac´on, Silvan Peter, Patricia Hu, Emmanouil Karystinaios, Florian Henkel, Francesco Foscarin, Nimrod Varga, and Gerhard Widmer. The accompanion: Combining reactivity, robustness, and musical expressivity in an automatic piano accompanist. arXiv preprint arXiv:2304.12939, 2023.  

Carlos Cancino Chac´on, Silvan Peter, Patricia Hu, Emmanouil Karystinaios, Florian Henkel, Francesco Foscarin, and Gerhard Widmer. The accompanion: Combining reactivity, robustness, and musical expressivity in an automatic piano accompanist. In Proceedings of the Thirty-Second International Joint Conference on Artificial Intelligence, IJCAI 2023, 19th-25th August 2023, Macao, SAR, China, pp. 5779–5787. ijcai.org, 2023. doi: 10.24963/IJCAI.2023/641.   
Carlos Eduardo Cancino Chac´on, Maarten Grachten, Werner Goebl, and Gerhard Widmer. Computational models of expressive music performance: A comprehensive and critical review. Frontiers Digit. Humanit., 5:25, 2018. doi: 10.3389/FDIGH.2018.00025. URL https: //doi.org/10.3389/fdigh.2018.00025.   
Xi Chen, Yan Duan, Rein Houthooft, John Schulman, Ilya Sutskever, and Pieter Abbeel. Infogan: Interpretable representation learning by information maximizing generative adversarial nets. In Daniel D. Lee, Masashi Sugiyama, Ulrike Ulrike von von Luxburg, Isabelle Guyon, and Roman Garnett (eds.), Advances in Neural Information Processing Systems 29: Annual Conference on Neural Information Processing Systems 2016, December 5-10, 2016, Barcelona, Spain, pp. 2172–2180, 2016.   
Pengyu Cheng, Martin Renqiang Min, Dinghan Shen, Christopher Malon, Yizhe Zhang, Yitong Li, and Lawrence Carin. Improving disentangled text representation learning with informationtheoretic guidance. Jurafsky, Joyce Chai, Natalie Schluter, and Joel R. Tetreault (eds.), Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, ACL 2020, Online, July 5-10, 2020, pp. 7530–7541. Association for Computational Linguistics, 2020. doi: 10.18653/V1/2020.ACL-MAIN.673.   
Andrea Cogliati and Zhiyao Duan. A metric for music notation transcription accuracy. In Sally Jo Cunningham, Zhiyao Duan, Xiao Hu, and Douglas Turnbull (eds.), Proceedings of the 18th International Society for Music Information Retrieval Conference, ISMIR 2017, Suzhou, China, October 23-27, 2017, pp. 407–413, 2017.   
Andrea Cogliati, David Temperley, and Zhiyao Duan. Transcribing human piano performances into music notation. In Michael I. Mandel, Johanna Devaney, Douglas Turnbull, and George Tzanetakis (eds.), Proceedings of the 17th International Society for Music Information Retrieval Conference, ISMIR 2016, New York City, United States, August 7-11, 2016, pp. 758–764, 2016.   
Peter Desain and Henkjan Honing. The quantization of musical time: A connectionist approach. Computer Music Journal, 13(3):56–66, 1989.   
Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: pre-training of deep bidirectional transformers for language understanding. In Jill Burstein, Christy Doran, and Thamar Solorio (eds.), Proceedings of the 2019 Conference of the North North American American Chapter of the Association for Computational Linguistics: Human Language Technologies, NAACL-HLT 2019, Minneapolis, MN, USA, June 2-7, 2019, Volume 1 (Long and Short Papers), pp. 4171–4186. Association for Computational Linguistics, 2019. doi: 10.18653/V1/N19-1423.   
Emilien Dupont. Learning disentangled joint continuous and discrete representations. In Samy Bengio, Hanna M. Wallach, Hugo Larochelle, Kristen Grauman, Nicolo\` Cesa-Bianchi, and Roman Garnett (eds.), Advances in Neural Information Processing Systems 31: Annual Conference on Neural Information Processing Systems 2018, NeurIPS 2018, December 3-8, 2018, Montr´eal, Canada, pp. 708–718, 2018.   
Francesco Foscarin, Andrew McLeod, Philippe Rigaux, Florent Jacquemard, and Masahiko Sakai. ASAP: a dataset of aligned scores and performances for piano transcription. In Julie Cumming, Jin Ha Lee, Brian McFee, Markus Schedl, Johanna Devaney, Cory McKay, Eva Zangerle, and Timothy de Reuse (eds.), Proceedings of the 21th International Society for Music Information Retrieval Conference, ISMIR 2020, Montreal, Canada, October 11-16, 2020, pp. 534–541, 2020.  

Nathan Fradet, Jean-Pierre Briot, Fabien Chhel, Amal El Fallah Seghrouchni, and Nicolas Gutowski. MidiTok: python package for MIDI file tokenization. In Extended Abstracts for the Late-Breaking Demo Session of the 22nd International Society for Music Information Retrieval Conference, 2021.  

Curtis Hawthorne, Erich Elsen, Jialin Song, Adam Roberts, Ian Simon, Colin Raffel, Jesse H. Engel, Sageev Oore, and Douglas Eck. Onsets and frames: Dual-objective piano transcription. In Emilia Go´mez, Xiao Hu, Eric Humphrey, and Emmanouil Benetos (eds.), Proceedings of the 19th International Society for Music Information Retrieval Conference, ISMIR 2018, Paris, France, September 23-27, 2018, pp. 50–57, 2018.   
Curtis Hawthorne, Ian Simon, Rigel Swavely, Ethan Manilow, and Jesse H. Engel. Sequence-tosequence piano transcription with transformers. In Jin Ha Lee, Alexander Lerch, Zhiyao Duan, Juhan Nam, Preeti Rao, Peter van Kranenburg, and Ajay Srinivasamurthy (eds.), Proceedings of the 22nd International Society for Music Information Retrieval Conference, ISMIR 2021, Online, November 7-12, 2021, pp. 246–253, 2021.   
Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Doll´ar, and Ross B. Girshick. Masked autoencoders are scalable vision learners. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2022, New Orleans, LA, USA, June 18-24, 2022, pp. 15979–15988. IEEE, 2022. doi: 10.1109/CVPR52688.2022.01553.   
Ruidan He, Wee Sun Lee, Hwee Tou Ng, and Daniel Dahlmeier. An unsupervised neural attention model for aspect extraction. In Regina Barzilay and Min-Yen Kan (eds.), Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics, ACL 2017, Vancouver, Canada, July 30 August 4, Volume 1: Long Papers, pp. 388–397. Association for Computational Linguistics, 2017. doi: 10.18653/V1/P17-1036. Hiramatsu, Eita Nakamura, and Kazuyoshi Yoshii. Joint estimation of note values and voices for audio-to-score piano transcription. In Jin Ha Lee, Alexander Lerch, Zhiyao Duan, Juhan Nam, Preeti Rao, Peter van Kranenburg, and Ajay Srinivasamurthy (eds.), Proceedings of the 22nd International Society for Music Information Retrieval Conference, ISMIR 2021, Online, November 7-12, 2021, pp. 278–284, 2021.   
Jonathan Ho, Ajay Ajay Jain, Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In Hugo Larochelle, Marc’Aurelio Ranzato, Raia Hadsell, Maria-Florina Balcan, and Hsuan-Tien Lin (eds.), Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual, 2020.   
Yu-Siang Huang and Yi-Hsuan Yang. Pop music transformer: Beat-based modeling and generation of expressive pop piano compositions. In Chang Wen Chen, Rita Cucchiara, Xian-Sheng Hua, Guo-Jun Qi, Elisa Ricci, Zhengyou Zhang, and Roger Zimmermann (eds.), MM ’20: The 28th ACM International Conference on Multimedia, Virtual Event Seattle, WA, USA, October 12-16, 2020, pp. 1180–1188. ACM, 2020. doi: 10.1145/3394171.3413671.   
Dasaem Jeong, Taegyun Kwon, Yoojin Kim, Kyogu Lee, and Juhan Nam. Virtuosonet: A hierarchical rnn-based system for modeling expressive piano performance. In Arthur Flexer, Geoffroy Peeters, Juli´an Urbano, and Anja Volk (eds.), Proceedings of the 20th International Society for Music Information Retrieval Conference, ISMIR 2019, Delft, The Netherlands, November 4-8, 2019, pp. 908–915, 2019.   
Kevin Ji, Daniel Yang, and Timothy Tsai. Piano sheet music identification using marketplace fingerprinting. In Jin Ha Alexander Lerch, Zhiyao Duan, Juhan Nam, Preeti Rao, Peter van Kranenburg, and Ajay Srinivasamurthy (eds.), Proceedings of the 22nd International Society for Music Information Retrieval Conference, ISMIR 2021, Online, November 7-12, 2021, pp. 326–333, 2021.   
Tero Karras, Samuli Laine, Miika Aittala, Janne Hellsten, Jaakko Lehtinen, and Timo Aila. Analyzing and improving the image quality of stylegan. In 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2020, Seattle, WA, USA, June 13-19, 2020, pp. 8107–8116. Computer Vision Foundation IEEE, 2020. doi: 10.1109/CVPR42600.2020.00813.  

Jong Wook Kim and Juan Pablo Bello. Adversarial learning for improved onsets and frames music transcription. In Arthur Flexer, Geoffroy Peeters, Julia´n Urbano, and Anja Volk (eds.), Proceedings of the 20th International Society for Music Information Retrieval Conference, ISMIR 2019, Delft, The Netherlands, November 4-8, 2019, pp. 670–677, 2019.  

Alexis Kirke and Eduardo R. Miranda (eds.). Guide to Computing for Expressive Music Performance.Springer, 2013. ISBN 978-1-4471-4122-8. doi: 10.1007/978-1-4471-4123-5.  

Qiuqiang Kong, Bochen Li, Xuchen Song, Yuan Wan, and Yuxuan Wang. High-resolution piano transcription with pedals by regressing onset and offset times. IEEE ACM Trans. Audio Speech Lang. Process., 29:3707–3717, 2021. doi: 10.1109/TASLP.2021.3121991.  

Lele Liu, Veronica Morfi, and Emmanouil Benetos. Joint multi-pitch detection and score transcription for polyphonic piano music. In IEEE International Conference on Acoustics, Speech and Signal Processing, ICASSP 2021, Toronto, ON, Canada, June 6-11, 2021, pp. 281–285. IEEE, 2021. doi: 10.1109/ICASSP39728.2021.9413601.  

Lele Liu, Qiuqiang Kong, Veronica Morfi, and Emmanouil Benetos. Performance midi-to-score conversion by neural beat tracking. In Preeti Rao, Hema A. Murthy, Ajay Srinivasamurthy, Rachel M. Bittner, Rafael Caro Repetto, Masataka Goto, Xavier Serra, and Marius Miron (eds.), Proceedings of the 23rd International Society for Music Information Retrieval Conference, ISMIR 2022, Bengaluru, India, December 4-8, 2022, pp. 395–402, 2022.  

Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In 7th International Conference on Learning Representations, ICLR 2019, New Orleans, LA, USA, May 6-9, 2019. OpenReview.net, 2019.  

Akira Maezawa, Kazuhiko Yamamoto, and Takuya Fujishima. Rendering music performance with interpretation variations using conditional variational RNN. In Arthur Flexer, Geoffroy Peeters, Juli´an Urbano, and Anja Volk (eds.), Proceedings of the 20th International Society for Music Information Retrieval Conference, ISMIR 2019, Delft, The Netherlands, November 4-8, 2019, pp. 855–861, 2019.  

MakeMusic, Inc. Finale version 27. https://www. finalemusic. com, 1988. Accessed: 2024-02-28.  

MuseScore. Musescore: Free music composition and notation software. https://musescore. org, 2002. Accessed: 2024-02-28.  

Eita Nakamura, Kazuyoshi Yoshii, and Haruhiro Katayose. Performance error detection and postprocessing for fast and accurate symbolic music alignment. In Sally Jo Cunningham, Zhiyao Duan, Xiao Hu, and Douglas Turnbull (eds.), Proceedings of the 18th International Society for Music Information Retrieval Conference, ISMIR 2017, Suzhou, China, October 23-27, 2017, pp. 347–353, 2017.  

Eita Nakamura, Emmanouil Benetos, Kazuyoshi Yoshii, and Simon Dixon. Towards complete polyphonic music transcription: Integrating multi-pitch detection and rhythm quantization. In 2018 IEEE International Conference on Acoustics, Speech and Signal Processing, ICASSP 2018, Calgary, AB, Canada, April 15-20, 2018, pp. 101–105. IEEE, 2018. doi: 10.1109/ICASSP.2018.8461914.  

Giuseppe De Pasquale, Blerina Spahiu, Pietro Ducange, and Andrea Maurino. Towards automatic classification of sheet music. In Maristella Agosti, Maurizio Atzori, Paolo Ciaccia, and Letizia Tanca (eds.), Proceedings of the 28th Italian Symposium on Advanced Database Systems, Villasimius, Sud Sardegna, Italy (virtual due to Covid-19 pandemic), June 21-24, 2020, volume 2646 of CEUR Workshop Proceedings, pp. 266–277. CEUR-WS.org, 2020.  

Cal Peyser, W. Ronny Huang, Andrew Rosenberg, Tara N. Sainath, Michael Picheny, and Kyunghyun Cho. Towards disentangled speech representations. In Hanseok Ko and John H. L. Hansen (eds.), 23rd Annual Conference of the International Speech Communication Association, Interspeech 2022, Incheon, Korea, September 18-22, 2022, pp. 3603–3607. ISCA, 2022a. doi: 10.21437/ INTERSPEECH.2022-30.  

Cal Peyser, W. Ronny Huang, Andrew Rosenberg, Tara N. Sainath, Michael Picheny, and Kyunghyun Cho. Towards disentangled speech representations. In Hanseok Ko and John H. L. Hansen (eds.), 23rd Annual Conference of the International Speech Communication Association, Interspeech 2022, Incheon, Korea, September 18-22, 2022, pp. 3603–3607. ISCA, 2022b. doi: 10.21437/ INTERSPEECH.2022-30.  

Christopher Raphael. Automated rhythm transcription. In ISMIR 2001, 2nd International Symposium on Music Information Retrieval, Indiana University, Bloomington, Indiana, USA, October 15-17, 2001, Proceedings, 2001.   
Yi Ren, Xu Tan, Tao Qin, Sheng Zhao, Zhou Zhao, and Tie-Yan Liu. Almost unsupervised text to speech and automatic speech recognition. In Kamalika Chaudhuri and Ruslan Salakhutdinov (eds.), Proceedings of the 36th International Conference on Machine Learning, ICML 2019, 9-15 June 2019, Long Beach, California, USA, volume 97 of Proceedings of Machine Learning Research, pp. 5410–5419. PMLR, 2019.   
Lenny Renault, R´emi Mignot, and Axel Roebel. Expressive piano performance rendering from unpaired data. In International Conference on Digital Audio Effects (DAFx23), pp. 355–358, 2023.   
Seungyeon Rhyu, Sarah Kim, and Kyogu Lee. Sketching the expression: Flexible rendering of expressive piano performance with self-supervised learning. In Preeti Rao, Hema A. Murthy, Ajay Srinivasamurthy, Rachel M. Bittner, Rafael Caro Repetto, Masataka Goto, Xavier Serra, and Marius Miron (eds.), Proceedings of the 23rd International Society for Music Information Retrieval Conference, ISMIR 2022, Bengaluru, India, December 4-8, 2022, pp. 178–185, 2022.   
Miguel A. Rom´an, Antonio Pertusa, and Jorge Calvo-Zaragoza. An end-to-end framework for audio-to-score music transcription on monophonic excerpts. In Emilia G´omez, Xiao Hu, Eric Humphrey, and Emmanouil Benetos (eds.), Proceedings of the 19th International Society for Music Information Retrieval Conference, ISMIR 2018, Paris, France, September 23-27, 2018, pp. 34–41, 2018.   
Miguel A. Rom´an, Antonio Pertusa, and Jorge Calvo-Zaragoza. A holistic approach to polyphonic music transcription neural networks. In Arthur Flexer, Geoffroy Peeters, Julia´n Urbano, and Anja Volk (eds.), Proceedings of the 20th International Society for Music Information Retrieval Conference, ISMIR 2019, Delft, The Netherlands, November 4-8, 2019, pp. 731–737, 2019.   
Tim Salimans and Jonathan Ho. Progressive distillation for fast sampling of diffusion models. In The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022. OpenReview.net, 2022.   
Noam Shazeer. Glu variants improve transformer. arXiv preprint arXiv:2002.05202, 2020.   
Kentaro Shibata, Eita Nakamura, and Kazuyoshi Yoshii. Non-local musical statistics as guides for audio-to-score piano transcription. Inf. Sci., 566:262–280, 2021. doi: 10.1016/J.INS.2021.03.014.   
Jianlin Su, Murtadha H. M. Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063, 2024. doi: 10.1016/J.NEUCOM.2023.127063.   
Masahiro Suzuki. Score transformer: Generating musical score from note-level representation. In Changwen Chen, Helen Huang, Jun Zhou, Tatsuya Harada, Jianfei Cai, Wu Liu, and Dong Xu (eds.), MMAsia ’21: ACM Multimedia Asia, Gold Coast, Australia, December 1 - 3, 2021, pp. 31:1–31:7. ACM, 2021. doi: 10.1145/3469877.3490612.   
Hao Hao Tan and Dorien Herremans. Music fadernets: Controllable music generation based on high-level features via low-level feature modelling. In Julie Cumming, Jin Ha Lee, Brian McFee, Markus Schedl, Johanna Devaney, Cory McKay, Eva Zangerle, and Timothy de Reuse (eds.), Proceedings of the 21th International Society for Music Information Retrieval Conference, ISMIR 2020, Montreal, Canada, October 11-16, 2020, pp. 109–116, 2020.   
Jingjing Tang, Geraint Wiggins, and Gyorgy Fazekas. Reconstructing human expressiveness in piano performances with a transformer network. arXiv preprint arXiv:2306.06040, 2023.  

David Temperley. What’s key for key? the krumhansl-schmuckler key-finding algorithm reconsidered. Music Perception, 17(1):65–100, 1999.  

Keisuke Toyama, Taketo Akama, Yukara Ikemiya, Yuhta Takida, Yuhta Takida, Wei-Hsiang Wei-Hsiang Liao, and Yuki Mitsufuji. Automatic piano transcription with hierarchical frequency-time transformer. In Augusto Sarti, Fabio Antonacci, Mark Sandler, Paolo Bestagini, Simon Dixon, Beici Liang, Ga¨el Richard, and Johan Pauwels (eds.), Proceedings of the 24th International Society for Music Information Retrieval Conference, ISMIR 2023, Milan, November 5-9, 2023, pp. 215–222, 2023. doi: 10.5281/ZENODO .10265261.   
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Isabelle Guyon, Ulrike von Luxburg, Samy Bengio, Hanna M. Wallach, Rob Fergus, S. V. N. Vishwanathan, and Roman Garnett (eds.), Advances in Neural Information Processing Systems 30: Annual Conference on Neural Information Processing Systems 2017, December 4-9, 2017, Long Beach, CA, USA, pp. 5998–6008, 2017.   
Xin Wang, Hong Chen, Si’ao Tang, Zihao Wu, and Wenwu Zhu. Disentangled representation learning. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2024.   
Yixin Wang, David Blei, and John P Cunningham. Posterior collapse and latent variable nonidentifiability. Advances in neural information processing systems, 34:5443–5455, 2021.   
Ziyu Wang, Dingsu Wang, Yixiao Zhang, and Gus Xia. Learning interpretable representation for controllable polyphonic music generation. In Julie Cumming, Jin Ha Lee, Brian McFee, Markus Schedl, Johanna Devaney, Cory McKay, Eva Zangerle, and Timothy de Reuse (eds.), Proceedings of the 21th International Society for Music Information Retrieval Conference, ISMIR 2020, Montreal, Canada, October 11-16, 2020, pp. 662–669, 2020.   
Gerhard Widmer and Werner Goebl. Computational models models of expressive of expressive music performance: The state of the art. Journal of new music research, 33(3):203–216, 2004.   
Gerhard Widmer, Simon Dixon, Werner Goebl, Elias Pampalk, and Asmir Tobudic. In search of the horowitz factor. AI Mag., 24(3):111–130, 2003. doi: 10.1609/AIMAG.V24I3.1722.   
Jiawei Wu, Xiaoya Li, Xiang Ao, Yuxian Meng, Fei Wu, and Jiwei Li. Improving robustness and generality of nlp models using disentangled representations. arXiv preprint arXiv:2009.09587, 2020.   
Mengyue Yang, Furui Liu, Zhitang Chen, Xinwei Shen, Jianye Hao, and Jun Wang. Causalvae: Disentangled representation learning via neural structural causal models. IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2021, virtual, June 19-25, 2021, pp. 9593–9602. Computer Vision Foundation IEEE, 2021. doi: 10.1109/CVPR46437.2021.00947.   
Ruihan Yang, Dingsu Wang, Ziyu Wang, Tianyao Chen, Junyan Jiang, and Gus Xia. Deep music analogy via latent representation disentanglement. In Arthur Flexer, Geoffroy Peeters, Juli´an Urbano, and Anja Volk (eds.), Proceedings of the 20th International Society for Music Information Retrieval Conference, ISMIR 2019, Delft, The Netherlands, November 4-8, 2019, pp. 596–603, 2019.   
Wei Zeng, Xian He, and Ye Wang. End-to-end real-world polyphonic piano audio-to-score transcription with hierarchical decoding. Proce edings of the Thirty-Third International Joint Conference on Artificial Intelligence, IJCAI 2024, Jeju, South Korea, August 3-9, 2024, pp. 7788–7795. ijcai.org, 2024.   
Huan Zhang and Simon Dixon. Disentangling the horowitz factor: Learning content and style from expressive piano performance. In IEEE International Conference on Acoustics, Speech and Signal Processing ICASSP 2023, Rhodes Island, Greece, June 4-10, 2023, pp. 1–5. IEEE, 2023. doi: 10.1109/ICASSP49357.2023.10095009.   
Huan Zhang, Jingjing Tang, Syed Rm Rafee, Simon Dixon, George Fazekas, and Geraint A. Wiggins. ATEPP: A dataset of automatically transcribed expressive piano performance. In Preeti Rao, Hema A. Murthy, Ajay Srinivasamurthy, Rachel M. Bittner, Rafael Caro Repetto, Masataka Goto, Xavier Serra, and Marius Miron (eds.), Proceedings of the 23rd International Society for Music Information Retrieval Conference, ISMIR 2022, Bengaluru, India, December 4-8, 2022, pp. 446–453, 2022.   
Huan Zhang, Shreyan Chowdhury, Carlos Eduardo Cancino-Chac´on, Jinhua Liang, Simon Dixon, and Gerhard Widmer. Dexter: Learning and controlling performance expression with diffusion models. Applied Sciences, 14(15):6543, 2024.   
Jingwei Zhao, Gus Xia, Ziyu Wang, and Ye Wang. multi-track accompaniment arrangement via style prior modelling. In Amir Globersons, Lester Mackey, Danielle Belgrave, Angela Fan, Ulrich Paquet, Jakub M. Tomczak, and Cheng Zhang (eds.), Advances in Neural Information Processing Systems 38: Annual Conference on Neural Information Processing Systems 2024, NeurIPS 2024, Vancouver, BC, Canada, December 10 - 15, 2024, 2024.  

# APPENDICES  

The appendix is structured into 6 main parts. Appendix A specifies the data processing details involved in the paper. Appendix B presents implementation details of our proposed methods. Appendix C provides subjective listening test details. Appendix D presents supplementary experimental results on GPT-4o results verification, diversity analysis of EPR, and ablation studies. In Appendix E, we provide several examples of expressive piano rendering (EPR) and automatic piano transcription (APT). Finally, we disclose the use of LLMs in Appendix F.  

# A DATA PROCESSING DETAILS  

# A.1 DATA FILTERING  

To construct a clean and consistent symbolic dataset from MuseScore, we apply a series of rule-based filters to exclude low -quality or incompatible piano scores. A score is retained only if it satisfies all of the following criteria:  

Staff structure: The score must contain exactly two staves, conforming to standard piano notation.   
Note count: The total number of notes must be at least 100.   
• Bar count: The score must span at least 10 bars.   
Note density: No individual bar may contain more than 64 notes, to avoid overly dense notation.   
Time signature: The time signature must fall within a musically plausible range: the number of beats per measure must be between 1 and 16, and the beat type must belong to the set {2, 4, 8, 16, 32}.   
Key signature: The notated key signature, expressed as the number of fifths, must lie within [−7, 7]. In addition, the mean distance between the notated and estimated keys (Temperley, 1999; Cancino-Chaco´n et al., 2022) must not exceed 1.  

To compute key signature distance, we segment each score into contiguous regions with a constant notated key signature. For each segment, we estimate the key and compare it to the notated key. Let ki ∈ [−7, 7] denote the notated key signature and kˆi E [−7, 7] the estimated key. The key distance is defined as:  

accounting for circularity in the circle of fifths. The final mean key distance is computed as:  

where N is the number of key-stable segments. Only scores with D ≤ 1 are retained.  

Table 5: Vocabulary size and value ranges of input and output parameters for music score.   

![](images/b463667d836434934f128adccaca2eed31fe0d35065d4e94202646f6073fd1a5.jpg)  

Table 6: Vocabulary size and value ranges of input and output parameters for performance MIDI.   

![](images/5f2c350b8cc6f951eb7a064010c02598399873005ac6c1b7f23ac2974a2517ee.jpg)  

# A.2 DATA REPRESENTATION DETAILS  

Score The score representation captures structural and timing information relevant for expressive rendering. The input encodes performance-related features, while the output is extended to include additional notation-specific attributes necessary for producing readable sheet music.  

Time-based features, including inter-onset interval (IOI), onset-in-bar, note value, and downbeat, are discretized into consistent vocabularies spanning 0 to 6 quarter lengths, each with 145–146 bins. Boolean-valued attributes, such as grace note and hand/staff assignment, are encoded as binary values. The score output additionally predicts symbolic formatting elements such as voice number, articulation markings (e.g., trill, staccato), and engraving-specific cues including stem direction and accidentals (e.g., double flats and sharps). All features are treated as discrete classification targets using small, well-defined vocabularies summarized in Table 5.  

Performance MIDI The performance representation captures expressive aspects of human execution, including timing, articulation, and dynamics. At the input level, we extract four note-level features: Pitch (MIDI number), IOI (inter-onset interval in seconds), Duration (extended by pedal usage), and Velocity (loudness). IOI and Duration are quantized into 200 bins, while Velocity is coarsely grouped into 8 bins for robustness.  

For output, we adopt a structured token-based representation (Huang & Yang, 2020), implemented using the miditok library (Fradet et al., 2021). The model generates discrete token sequences that include Note-On, Duration, Velocity, and Time-Shift events, enabling expressive sequence generation without explicit note-level alignment. Special tokens such as BOS (beginning of sequence) and PAD are also used to facilitate training and formatting. Table 6 provides the vocabulary sizes and ranges for all input and output features.  

# B IMPLEMENTATION DETAILS  

# B.1 JOINT MODEL  

Our joint model is implemented in PyTorch Lightning and trained via multi-task learning to simultaneously handle EPR, APT, and masked reconstruction from unpaired data. This section outlines the training tasks, loss formulation, optimization strategy, and implementation setup.  

Training tasks Each training step involves four supervised or self-supervised subtasks:  

APT The score decoder reconstructs symbolic score tokens from the performance content encoder.   
EPR The performance decoder generates MIDI tokens conditioned on the score content encoder and a style embedding.   
Score Reconstruction The score encoder is trained using random masking to reconstruct full sequences from partially masked inputs.   
MIDI Reconstruction The performance content encoder and decoder reconstruct MIDI sequences from masked inputs in a similar fashion.  

Additionally, a Kullback-Leibler (KL) regularization term is applied to the style embedding to encourage compactness and diversity in the latent style space.  

Training loss Let LAPT, LEPR, Lrec,X , and Lrec, denote the cross-entropy losses for APT, EPR, score reconstruction, and MIDI reconstruction, respectively. The total training objective is given by:  

where λrec = 0.2 and λKL = 0.1. We apply a 50% masking rate to encoder inputs during reconstruction, and a lighter masking rate of 10–20% to decoder inputs to improve robustness and mitigate overfitting.  

Optimization We use AdamW optimizers (Loshchilov & Hutter, 2019) with a learning rate of 5 10−5, following a cosine learning rate schedule with 4,000 warm-up steps and 40,000 total steps. Gradient updates are manually scheduled, and training is performed using mixed precision (fp16).  

Batching and scheduling Each training step processes 144 sequences (each of length 256 notes), evenly divided among the four subtask types: APT, EPR, unpaired score, and unpaired MIDI. Data loaders for each subset are interleaved and sampled in parallel. KL regularization is computed once per batch using the mean and variance of the predicted style embeddings.  

Implementation notes All model components use a unified embedding dimension of d = 512, with task-specific embedding layers. Attention masks are dynamically modified during training to simulate incomplete inputs, following masked language modeling strategies. The system is trained on 3 NVIDIA A5000 GPUs using batch-level data parallelism.  

# B.2 PERFORMANCE STYLE RECOMMENDATION (PSR)  

The performance style recommendation (PSR) module is designed to generate expressive style embeddings directly from symbolic scores, enabling performance rendering without requiring paired expressive data at inference time. The overall architecture is illustrated in Figure 6.  

Overview The PSR model comprises two components: (1) a transformer-based score encoder that extracts a global content embedding from a symbolic score sequence, and (2) a denoising diffusion probabilistic model (DDPM) that generates a style vector conditioned on this content embedding. This pipeline enables sampling stylistically coherent vectors from Gaussian noise, guided by the structure of the input score.  

![](images/1ed92ce349c4844f25b4f0d85d56413a78dd8fbd566e7c3003b2de0911b807ff.jpg)  
Figure 6: Architecture of the performance style recommendation (PSR) module. Given a symbolic score, we extract a global content embedding using a transformer encoder and train a diffusion model to predict the style embedding from noise.  

Score encoder We adopt a transformer encoder fg,X (x) to process the input score sequence. Following the BERT-style design (Devlin et al., 2019), a special [CLS] token is prepended to the sequence, and its final-layer hidden state as the global score content representation eg RD.  

Diffusion network We employ a DDPM (Ho et al., 2020) with velocity prediction (Salimans & Ho, 2022) to model the conditional distribution over style embeddings given the content vector. During training, the model learns to recover a ground -truth style vector zs, extracted from human performances via the joint model, from a noisy version zts produced by the forward diffusion process. A sinusoidal timestep embedding et is concatenated with the projected content embedding e′g and the noisy style vector zts, and passed through a multi-layer perceptron (MLP) to predict the velocity target vtarget. The model is optimized with the following mean squared error loss:  

Inference At inference time, a style vector is initialized from a standard Gaussian distribution and iteratively denoised using the exponential moving average (EMA) version of the MLP denoising network. The resulting style embedding zˆs can be combined with the score content to condition the expressive rendering model. This one-to-many mapping enables diverse, plausible, and stylistically appropriate generation from symbolic input alone.  

# C SUBJECTIVE LISTENING TEST INSTRUCTIONS  

# C.1 OVERVIEW  

We conduct our subjective evaluation using a Google Form 3, structured into two sections: (1) evaluation of performance style recommendation (PSR), and (2) evaluation of style transfer. Each participant completes both sections, with an average completion time of approximately 32 minutes. Figure 7 shows sample survey pages along with participant instructions. Detailed descriptions of the survey structure are provided below.  

# C.2 SURVEY STRUCTURE  

Part I: Overall Evaluation Participants are presented with 4 music clips, each accompanied by 6 audio renditions generated by different EPR models. Each rendition is rated along the following four dimensions:  

![](images/ac5b00afa75c5d13ff45120e48f8407b0228165997b18b85feac7f02b094c4a2.jpg)  

(a) Overall evaluation of EPR.  

(b) Style transfer evaluation.  

Figure 7: Screenshots of survey pages and instructions of our online survey.  

Dynamics: Naturalness and expressiveness of loudness variation.   
Tempo: Naturalness and expressiveness of tempo fluctuations over time.   
Performance Style: Appropriateness of the performance’s character, mood, and interpretation.   
Overall Human-Likeness: How convincingly the performance resembles that of a human.  

Ratings are provided on 5-point Likert scale ranging from (Very Poor) to 5 (Very Good).  

Part II: Style Similarity Participants are presented with 3 examples. Each example consists of:  

A reference performance.   
Three test renditions generated by different models, with varied content but intended to share the same performance style.  

Each test rendition is rated on:  

Performance Style Similarity: The extent to which the style (e.g., rhythm, dynamics, pedal usage) matches the reference, independent of pitch content. Overall Human-Likeness: Perceived expressiveness and realism of the performance.  

All ratings are again provided on a 5-point Likert scale.  

# C.3 ADDITIONAL NOTES  

Participants are instructed to evaluate variation and human-likeness, rather than personal preference or audio fidelity.  

Table 7: Agreement matrices between human annotators and GPT-4o. Cohen’s κ values: Annotator 1 (A1) v.s. Annotator 2 (A2) = 0.89; Annotator (A1) v.s. GPT-4o = 0.85; Annotator 2 (A2) v.s. GPT-4o = 0.89. B = Baroque, C = Classical, R = Romantic, T = Contemporary.   

![](images/9d4f0b539fb9cbb56cc9897ee82e931118223ac6434849309d5b6b85962942bf.jpg)  

Table 8: Average pairwise MAEs for human renditions and model outputs.   

![](images/998f6371bd26299c32e5beb3dad3d95f5755accdbed3fc62d374c44eeec5d3b8.jpg)  

Table 9: Pairwise MAEs among 7 human renditions.   
(b) Velocities   

![](images/fe52bc5d917edb5e24467a9584b81826c23a8eaf39edffbc187cd69a877074cc.jpg)  

(a) Durations   

![](images/f9c2e7949c9299a6175f41f1c62c31be33a2be18e8ec787a9207b2644b4b08b8.jpg)  

All audio sources are anonymized; both the order of clips and model outputs are randomized to reduce potential bias.   
Participants are encouraged to use headphones in a quiet environment for optimal listening conditions.   
The total duration of the survey is approximately 20–25 minutes. No personal data is collected.  

# D SUPPLEMENTARY EXPERIMENTAL RESULTS  

# D.1 HUMAN VERIFICATION OF GPT-4O OUTPUTS  

To assess the reliability of GPT-4o predictions in Section 5.3, we conducted a human verification study on 100 randomly sampled movements, independently annotated by two professionally trained pianists into four eras (Baroque, Classical, Romantic, Contemporary). Agreement was measured using Cohen’ s κ = po−pe , where po is the observed agreement and pe is the expected agreement by 1−pe   
chance. As shown in Table 7, inter-annotator agreement was high (κ = 0.89), and GPT-4o showed similarly strong consistency with both annotators (κ = 0.85 and κ = 0.89). Most disagreements occurred in transitional works between Classical and Romantic eras, where stylistic boundaries are ambiguous. For example, Piano Sonata No. 26 in E-flat, Op. 81a “Les adieux”: II. Abwesenheit (Andante espressivo) was annotated as Classical by both human experts but labeled as Romantic by GPT-4o. Such cases are reasonable given the transitional nature of the repertoire. Overall, these results confirm that GPT-4o aligns closely with expert judgment and can be used as a reliable reference for PSR evaluation. Table 11: APT results on different proportions of paired/unpaired data. Lower is better for all metrics.   
The best results are shown in bold, and the second-best are underlined.  

![](images/959650c9b471b0f50e8bee9db1f87352489d835c5a920e600b1ad2af32736192.jpg)  

Table 10: Pairwise MAEs among 7 model outputs.   

![](images/d55a1dfd4c2c671e33f65ec0dceb6e27717a7aa8241eee99954378f1abeca037.jpg)  

Table 12: Performer (Perf) and composer (Comp) identification under two data settings: paired + 0% unpaired and paired + 100% unpaired. Boldface is kept only for Style→Perf and Style →Comp to highlight the effect of adding unpaired data. The rightmost block reports the per-metric gain ∆ (100% unpaired 一 0% unpaired).   

![](images/38ca23b565ae4a12380441e341840b4a8f9dbd5caf8a78e4d657a19f499257bc.jpg)  

Table 13: Ablation of KL weight on KL divergence, active units (AU), and classification accuracy (CA).   

![](images/4fafc7ed0a0c608a5775198b07323eebab30f6c4f4712ac3008ba09a36bcb330.jpg)  

# D.2 DIVERSITY ANALYSIS OF EPR  

To verify that the model captures one-to-many expressive variation rather than collapsing to an averaged output, we analyzed diversity on a score from ASAP with 7 human performances and 7 model outputs generated via top-k sampling (k = 5). Pairwise note-aligned MAEs were computed for durations and velocities. As summarized in Table 8, the average human MAEs were 0.06 (duration) and 11.62 (velocity), while the model achieved 0.08 and 8.01, respectively. Detailed pairwise matrices (Table 9, Table 10) show that model outputs exhibit meaningful internal variation, following the diversity observed in human renditions. This demonstrates that the proposed model captures distributional expressiveness in performance generation rather than regressing to a mean output.  

# D.3 ABLATION STUDIES  

Ablations on unpaired data To evaluate the impact of unpaired data, we conduct an ablation study by varying the ratio of unpaired data used in training. We train four model variants using 0% (paired data only), 25%, 50%, and 100% of our curated unpaired datasets, while keeping all other hyperparameters constant. The APT results in Table 11 show that incorporating unpaired data generally enhances performance. Adding just 25% of the unpaired data provides a consistent improvement over the baseline model trained only on paired data, while using the full 100% unpaired dataset achieves the best overall performance.  

Furthermore, to study the influence of unpaired data on representation disentanglement, we conduct performer and composer identification in Section 5.2. As shown in Table 12, introducing unpaired data significantly enhances the quality of the style representation. For both performer (Style→Perf) and composer (Style→Comp) identification, all metrics see a substantial improvement, with classification accuracy increasing by +8.31% and +8.39%, respectively. In contrast, the classification performance using the content representation remains almost unchanged. These results indicate that our model effectively leverages unpaired data to enrich the style embedding while successfully preserving the disentanglement between performance style and score content.  

KL divergence analysis We evaluate latent informativeness across different KL weights for the KL divergence loss introduced in Section 3.3 using three metrics (Wang et al., 2021): (i) KL divergence between posterior and prior, (ii) Active Units (AU) measuring the number of latent dimensions with sample variance > 0.01, and (iii) style classification accuracy (CA) using zs and ground-truth era labels from Section 5.3. As shown in Table 13, stronger KL regularization reduces both KL divergence and classification accuracy, while the number of active units remains consistently high (512). This indicates that although some information compression occurs, the latent representation does not undergo full posterior collapse, and still preserves musically meaningful information.  

# E EXAMPLES OF EPR AND APT  

EPR Demos are available at https://jointpianist. github .io/epr-apt/. The page includes two sections: (1) rendering results from various models, including ours, on five music pieces from different composers; and (2) style transfer results on three music pieces, showcasing the flexibility of our method.  

APT Three examples of APT are shown from Figure 8 to Figure 13. Specifically, ground truth and transcription of Piano Sonata No.5, Op.10 No.1, by Ludwig van Beethoven are shown in Figure 8 and Figure 9; ground truth and transcription of Piano Sonata No.12 in F Major, K 332, by Wolfgang Amadeus Mozart are shown in Figure 10 and Figure 11; ground truth and transcription of Impromptu Op.90 D.899, by Franz Schubert are shown in Figure 12 and Figure 13.  

# F THE USE OF LARGE LANGUAGE MODELS (LLMS)  

In accordance with the ICLR policy, we disclose the use of Large Language Models (LLMs) as assistive tools in the preparation of this manuscript. The specific applications are detailed below:  

Data annotation: We employed an LLM to assist in the annotation of our dataset. The detailed methodology and human verification have been introduced in Section 5.3 and Appendix D.1.   
Literature search: LLMs were used as a tool to aid in the initial search and summarization of relevant prior work.   
Writing and polishing: We utilized an LLM for proofreading and language refinement.  

All authors have carefully reviewed and edited the manuscript. We take full responsibility for all content of this paper, including the final research ideas, experimental results, and the accuracy and integrity of the text.  

![](images/9359ede749abbf9a7a932d59b87c9ffb451d29669287f1586b9aac9a46778cdb.jpg)  
Figure 8: Ground truth score from Piano Sonata No.5, Op.10 No.1, by Ludwig van Beethoven (APT sample 1).  

![](images/3b8914497ca35835562ab9ec55cb9dca67ed758edfa9b4346a9e278169187a73.jpg)  

![](images/c37054f958e44d52203f347b4a024cd92f4b390eea3ec0e803a8adc696a01b1d.jpg)  

Figure 9: Transcription results from Piano Sonata No.5, Op.10 No.1, by Ludwig van Beethoven (APT sample 1).  

![](images/eaceab5ecbbfa0271562fb5768a4eb5d52a151c00751d9f765a671d4be35fe21.jpg)  
Figure 10: Ground truth score from Piano Sonata No.12 in F Major, K 332, by Wolfgang Amadeus Mozart (APT sample 2).  

![](images/44a2d596c7c422f8455bbe63e50425072344a5336f426d27d6fbcc9c0d4c6ffe.jpg)  
Figure 11: Transcription results from Piano Sonata No.12 in F Major, K 332, by Wolfgang Amadeus Mozart (APT sample 2).  

![](images/7ce7a1888400e8277872389ff7c2a4472a569a56d455ed3ae678f8724316816c.jpg)  

![](images/91d1d57fc094ccb1197997d0b30fbdaf2329ffe0084fa779b3430c67b50ebaf2.jpg)  

![](images/8eaf4517326f6d8406c931f09ecf6f82b0a5a5fb51ce12fc6dba216814643137.jpg)  

![](images/776ef183a1f0f0839cd3dfe6aa3de88a6028382501ac871bb8ff6a849da57ad0.jpg)  

![](images/35143378a96520e0324634731e9695061307a58948d21cfca3754fda1d37d153.jpg)  
Figure 12: Ground truth score from Impromptu Op.90 D.899, by Franz Schubert (APT sample 3).  

![](images/b110e0d007ebd0bc362ed42cc7b55c683219e63dc6e48231b03275a874d0267d.jpg)  

![](images/c2a87f7a497ce06cf79eb75f9d672c9dfa9824424124e6f0be226522053c7508.jpg)  

![](images/2a7742363e427a52f8fa5b69bda2c940e6a65f3b0b588fbeda3769f654ab7e93.jpg)  

![](images/1aff409e754e9486317926f4c5dbd36d904e5b65b796a245e5bb9b26ee0bad2d.jpg)  

![](images/7e710902da21776a1d32cc4da00c077a43d6fadc9d717eb42fe857429794e964.jpg)  
Figure 13: Transcription results from Impromptu Op.90 D.899, by Franz Schubert (APT sample 3).  