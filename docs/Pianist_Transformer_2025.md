# Pianist Transformer: Towards Expressive Piano Performance Rendering via Scalable Self-Supervised Pre-Training  

Hong-Jie You1,2, Jie-Jing Shao1, Xiao-Wen Yang1,2, Lin-Han Jia1, Lan-Zhe Guo1,3, Yu-Feng Li1,2†  

1State Key Laboratory of Novel Software Technology, Nanjing University, Nanjing 210023, China   
2School of Artificial Intelligence, Nanjing University, Nanjing 210023, China   
3School of Intelligence Science and Technology, Nanjing University, Suzhou 215163, China  

{youhj, shaojj, yangxw, jialh, guolz}@lamda.nju.edu.cn liyf@nju.edu.cn  

Corresponding Author  

# Abstract  

现有的表达音乐表演渲染方法依赖于对小标记数据集的监督学习，这限制了数据量和模型大小的缩放，尽管有大量未标记的音乐，如视觉和语言。为了解决这一差距，我们引入了钢琴家转换器，它有四个关键贡献：1)一个统一的乐器数字接口（MIDI）数据表示，用于学习音乐结构和表达的共享原则，而无需显式注释；2)高效的非对称架构，在不牺牲渲染质量的情况下实现更长的上下文和更快的推理；3)具有10B个令牌和135m参数模型的自监督预训练管道，解锁数据和模型缩放优势，实现表现力的性能渲染；4)最先进的绩效模型，实现了强大的客观指标和人类水平的主观评分。总的来说，piano Transformer在音乐领域建立了一个可扩展的人类表演合成路径。

Github: https://github. com/yhj137/PianistTransformer HuggingFace: https://huggingface. co/collections/yhj137/pianist-transformer 四 Project Page: https://yhj137.github. io/pianist-transformer-demo/  

# 1. Introduction  

表现性表演渲染旨在从象征性的乐谱中自动生成类似人类的音乐表演。这项任务不仅仅是音高和节奏的准确性，还要捕捉时间、动态、发音和踏板上的细微变化，这些变化塑造了音乐表达。核心挑战在于如何通过计算建模，将乐谱的基础音乐结构（如旋律和和声）与这些表达选择之间的复杂映射联系起来。几十年来，从概率模型（Teramura等人，2008年）到使用rnn的现代深度学习方法（Jeong等人，2019b）和变形金刚（Borovik和Viro， 2023年）的研究主要依赖于监督范式。然而，这种模式面临着一个持续的瓶颈：对齐的分数-性能监督数据集通常是劳动密集型的，并且扩展成本很高。

为了最大限度地利用有限的数据集，现有的作品通常采用不对称的、专门的表示，在分数方面注入丰富的结构描述子（例如，测量，仪表）（Jeong等人，2019b； Maezawa等人，2019;Borovik & Viro, 2023）。这提高了标签效率，因为每个标记的示例都提供了明确的结构线索，而不是强迫模型推断它们。然而，这些描述符需要标记分数和分数-性能一致性。相比之下，性能MIDI通常是从数字钢琴录音捕获或由人工智能转录产生，因此由一连串的音符事件组成，没有明确的措施，仪表，或可用的速度地图。因此，这些方法无法计算大量未对齐的仅性能的MIDI语料库所需的结构特征，使得它们不适合大规模地利用无监督数据（图1，左）。雷诺等人（2023）探索了一种对抗性的、循环一致的架构，该架构将乐谱内容与表演风格分开，并且乐谱到音频生成器学习从未对齐的数据中呈现具有表现力的钢琴音频，以对抗现实主义鉴别器。然而，由于复杂的训练动态，对抗性训练的规模具有挑战性，并且迄今为止生成的质量有限。需要一个更稳定和可扩展的范例，以真正利用大量未开发的、未对齐的MIDI性能语料库的潜力（图1，右）。

![](images/158708b45387742796d168a07458e7824eedb321bbf8a361baf5164dabe22d18.jpg)  
图1：表现性钢琴演奏呈现的范式转换。（左）先前监督范式：现有系统在严格监督的管道下运行，依赖于稀缺的对齐数据集（≈100小时），不能利用大量的野外MIDI语料库（100小时）。这种对显式结构特性的依赖从根本上限制了可伸缩性。（右）我们可扩展的自我监督范式：钢琴家变压器通过使大规模的自我监督学习成为可能，从而改变了这种范式。通过统一的MIDI表示，该模型可以对超过100K小时的未对齐的MIDI进行预训练，获得丰富的音乐先验，然后通过监督微调进行有效的泛化。

在本文中，我们提出了piano Transformer，这是一个使用大规模未标记的MIDI语料库训练的具有表现力的性能渲染模型。我们的主要贡献如下：

统一数据表示：我们引入了一个单一的、细粒度的MIDI标记化，它在相同的离散事件词汇表中编码标记分数和表达性能。通过缩小这些模式之间的表示差距，这种共享公式使未对齐的，仅性能的MIDI直接可用于预训练，缩放到10B MIDI令牌，而无需显式的分数-性能对齐，同时保持数据多样性。在这个统一的空间里，模型不仅可以学习音乐的“语法”，还可以学习乐谱层次结构和表达控制（时间、动态、发音和踏板）之间的统计联系。

高效的音乐建模架构：我们设计了一个非对称编码器-解码器架构，具有音符级序列压缩，将固定的每个音符事件束合并到一个令牌中，将编码器的自注意成本降低了64倍。这将计算集中在单个并行通道中，缓解了解码瓶颈，并产生更长的上下文覆盖范围和更快的推理，并具有较强的渲染质量。与对称架构相比，它提供了2.1倍的推理速度，在不牺牲表达质量的情况下满足实际使用的低延迟要求。

可扩展的训练管道：我们采用自监督的预训练方案，为模型提供初始化，内化常见的音乐规律和表达模式。因此，在下游监督微调期间，模型从一个更强的表示开始，收敛得更快，并且比从头开始训练的相同模型的损失低得多，并且实现了更强的客观指标。特别是，刮痕模型在更高的损失下趋于平稳，产生更弱的表达分布，而预训练模型从一个更好的基础开始，并在整个微调过程中不断改进。为了在模型生成和实际音乐制作工作流程之间架起桥梁，我们引入了表达速度映射，这是一种后处理算法，可将模型输出转换为可编辑的速度映射。这产生了一种适合现实世界使用的可编辑格式，同时保留了表达时间。

结果性能模型：整个训练配方产生一个最先进的表现性能模型。在客观指标上，它的表现优于强劲的基线。在全面的听力研究中，它的输出在统计上与人类钢琴家没有区别，并且更受欢迎。

# 2. Related Work  

钢琴演奏渲染。表演渲染的目标是从符号乐谱中合成一种富有表现力的、类似人的表演。该领域已经从早期基于规则的（Sundberg等人，1983年）和统计模型（Teramura等人，2008年；Flossmann等人，2013年；Kim等人，2013年）发展到基于rnn （Cancino-Chacón & Grachten， 2016年）、变分自编码器（Maezawa等人，2019年）、图神经网络（Jeong等人，2019b）和Transformers的深度学习架构。例如，scoreformer （Borovik & Viro, 2023）等最近的作品专注于细粒度风格控制，这是对我们专注于可扩展预训练的补充。尽管有这些架构上的创新，但监督式学习范式的发展却成为了瓶颈；它需要的小而昂贵的对齐数据集不足以让模型学习从音乐结构到表达细微差别的复杂映射。这种依赖限制了模型的可伸缩性和泛化。虽然最近的工作已经探索了对非配对数据进行对抗性训练以绕过对齐（Renault等人，2023），但训练稳定性和质量方面的挑战仍然存在。这强调了需要一个强大的范例，可以有效地利用大量的、不一致的数据，我们通过大规模的自我监督预训练提出了这一点。

音乐的自我监督学习。自我监督预训练是NLP （Devlin et al., 2019; Brown et al., 2020）和计算机视觉（Chen et al., 2020; He et al., 2022）中的主流范式，也已被用于音乐。在符号领域，早期的研究如MusicBERT （Zeng et al., 2021）将掩模语言建模应用于MIDI来理解任务。这种方法最近得到了显著的扩展：Bradshaw等人（2025）在大型钢琴语料库上进行预训练，以完成旋律延续等任务，而Moonbeam （Guo & Dixon, 2025）等基础模型已经在数十亿个标记上进行了训练，用于不同的条件生成。对于使用对比或重建目标的原始音频，也存在类似的努力（Spijkervet & Burgoyne, 2021； Hawthorne等人，2022）。然而，将自我监督的预训练应用于具体的表现性表演渲染任务，在很大程度上尚未得到探索。虽然现有的自监督模型擅长学习高级音乐语义，如生成或分类等任务，但性能渲染是一个独特的、细粒度的挑战，集中在建模微妙的表达细节上。大规模自我监督预训练的好处能否成功转移到这个微妙的、性能水平的领域，是一个悬而未决的问题，激励着我们的工作。

# 3. The Pianist Transformer Framework  

我们的目标是开发强大的钢琴演奏呈现系统，该系统可以通过自我监督的预训练范式利用大规模的未标记数据。本节详细介绍了我们的方法，从我们方法的核心开始：一个统一的数据表示，可以进行大规模的预训练。然后，我们描述了基于transformer的体系结构和基于此表示构建的两阶段训练策略。最后，我们引入了一个新的后处理步骤，以确保模型的输出对音乐家是实用的。
# 3.1. Unified MIDI Representation  

将自我监督学习应用于表演呈现的一个基本挑战是结构化分数数据和表达性表现数据之间的差异。具体来说，分数代表音乐的符号，韵律的时间（例如，四分音符，八分音符）和分类的动态（例如，p, mf, f），而表演被捕获为事件流，以毫秒为单位的绝对时间和连续的速度值。为了克服这个问题，我们提出了一个统一的、基于事件的令牌表示，它将两种格式相同地对待，使它们能够混合在一个单一的、大规模的预训练语料库中。 

![](images/66e6878f02b50720df24d22ebf119e5d94f0ec7d6d4a2780e289657bfd1f71f7.jpg)  
图2：钢琴家Transformer的整体架构和工作流程。我们的框架通过统一的Tokenizer处理所有MIDI数据，支持两阶段的训练过程。其核心模型是一个具有编码器序列压缩的非对称转换器，用于高效处理长乐谱。工作流由三个阶段组成：(1)预训练：模型通过屏蔽去噪目标从大量未标记的语料库中学习基础音乐上下文，其中它将屏蔽令牌序列作为输入并预测原始序列。(2) SFT：监督式微调（Supervised Fine-Tuning）使模型使用对准的乐谱-演奏对将音乐上下文映射到有表现力的细微差别，其中它将乐谱标记作为输入并预测相应的演奏标记。(3)推理：模型接受一个分数输入，然后生成一个性能，然后通过我们的Expressive Tempo Mapping算法使其可用于daw编辑。

我们将每个音符表示为八个符号的序列。这个序列从前一个音符中捕获音符的音调、速度、持续时间和间隔时间（IOI），计时信息以毫秒为单位量化。为了模拟细微的踏板控制，我们包括四个额外的踏板令牌，它们表示音符IOI窗口内采样点的持续踏板状态。

至关重要的是，这种表示避免依赖于高水平的音乐概念，如节拍或节拍，开启了对未对齐的MIDI的大规模预训练，并使模型能够通过统计规律揭示音乐原则，从旋律轮廓到和声进程。

# 3.2. Architecture for Efficient Long-Sequence Modeling  

我们使用了一个编码器-解码器转换器，但其标准的O（N2）自关注复杂性对于通常超过数千个符号的长音乐序列来说是一个关键瓶颈。为了实现高效的渲染，我们引入了两个协同的架构修改：编码器序列压缩和非对称层分配。

编码器序列压缩。利用每个音符固定的8个符号结构，我们压缩编码器的输入序列。我们不处理原始标记嵌入，而是首先投影，然后将单个音符的八个嵌入聚合到一个合并向量中。这种笔记级聚合将序列长度减少了8倍，从而将自注意计算成本从0 （N2）减少到O((N/8)2)，减少了64倍。因此，编码器可以有效地处理更长的序列，捕获呈现所必需的全局上下文。

非对称编码器-解码器架构。我们故意采用非对称架构，采用深度10层编码器和轻量级2层解码器（因此为10-2），以最大限度地提高效率。这种设计与我们的序列压缩协同，将大部分计算集中到一个单一的，高度并行化的编码通道中。这大大加快了训练速度，减少了训练和推理的内存开销。在生成过程中，浅解码器是自回归任务的主要瓶颈，它以最小的延迟和内存占用运行，同时以编码器的强大表示为条件。这种架构代表了计算效率和模型性能之间的一种有意识的权衡，我们将在第4.5节的消融研究中定量分析这种平衡。

# 3.3. Two-Stage Training for Expressive Rendering  

我们的训练范例直接解决了表达性呈现的核心挑战：在乐谱的音乐结构和人类表演的细微差别之间建立复杂的依赖关系。为了达到这个目的，我们的训练分两个阶段进行：第一，学习理解音乐背景，第二，学习将音乐背景转化为富有表现力的表演。

# 3.3.1. SELF-SUPERVISED PRE-TRAINING ON MUSICAL PRINCIPLES  

最初的预训练阶段建立了对指导人类表达的内隐语境的理解。我们在大量未标记的MIDI语料库上使用了自监督掩码去噪目标。通过学习从被破坏的背景中重建原始音乐片段，该模型被迫内化深层结构线索，如和声功能和旋律方向，这些线索为演奏选择提供了信息。

他的目标是最小化原始令牌在掩码位置的负对数似然：

其中M是掩码令牌的索引集，p（xi|Xcorr， X<i）是在给定损坏的输入和基真前缀X<i的情况下预测原始令牌xi的概率。

3.3.2. SUPERVISED FINE-TUNING FOR EXPRESSIVE RENDERING  

有了一个理解音乐背景的模型，我们然后执行监督微调（SFT）来教它如何将这种理解转化为表演。这个阶段学习从潜在的结构线索到微妙的、连续的人类表达参数的明确映射。

SFT的框架是一个序列到序列的学习任务对对齐的分数-表现对。编码器处理分数的标记序列，而解码器通过最小化标准交叉熵损失来训练自回归生成相应的性能序列。这个微调阶段基于模型在预训练期间获得的深度音乐理解来表达决策，例如时间和动态的变化。

# 3.4. Post-processing: Expressive Tempo Mapping  

实际应用的一个关键挑战是原始模型输出，计时以绝对毫秒为单位，缺乏与标准音乐软件的兼容性。这些表演不与数字音频工作站（DAW）的格律网格对齐，阻碍了可编辑性。为了弥合人工智能生成和现代音乐制作工作流程之间的差距，我们引入了一种新的后处理算法，表达速度映射。

该算法（详见附录B）智能地将表现的时间偏差转换为动态节奏图。然后，它重新排列所有的音符和踏板事件的音乐网格由这个新的节奏曲线。该过程保留了生成性能的声音细微差别，同时恢复了编辑和集成所必需的结构对齐。最后的输出是一个MIDI文件，既具有音乐表现力，又可以在任何标准的DAW中完全编辑。

# 4. Experiments  

我们进行了一套全面的实验来评估我们提出的钢琴家变压器。我们的评估以三个核心问题为指导。首先，大规模自监督预训练在多大程度上有助于渲染模型的最终性能？其次，当客观指标和主观的人类评估来判断时，piano Transformer与现有方法相比表现如何？第三，什么样的架构选择会影响模型的有效性，以及它在不同音乐背景下的表现有多稳健？下面的部分将依次解决这些问题。

# 4.1. Experimenta Setup  

我们在从几个公共MIDI数据集聚合的100亿个令牌语料库上预训练我们的模型。对于监督微调和评估，我们使用ASAP数据集（Foscarin et al., 2020）进行严格的分段分割。我们的piano Transformer与强大的基线进行比较，包括VirtuosoNet-HAN (Jeong et al., 2019a), VirtuosoNet-ISGN （Jeong et al., 2019b）和scoreformer (Borovik & Viro, 2023)，以及无表现力的Score MIDI和ground truth Human表演。

我们使用一套客观和主观的措施来评估所有的模型。客观地，我们使用Jensen-Shannon (JS) Divergence and Intersection Area在四个关键表达维度上评估了与人类表演的分布相似性：速度、持续时间、IOI和Pedal。主观上，我们进行了全面的听力研究，以评估人类的相似性和整体偏好。关于数据集、基线实现和评估方案的全面细节见附录A和附录C。
# 4.2. Pre-training Substantially Improves Performance  

To quantify the impact of large-scale self-supervised pre-training, we perform a controlled ablation comparing our full Pianist Transformer with an identical model trained from scratch (w/o PT). This setup isolates the effect of pre-training and reveals its crucial role in expressive performance modeling.  

![](images/8f19f6d04cba033c14981c5c1de0c798c0e17c1ce812f9daa5e7298087542cb2.jpg)  
图3：大规模自监督预训练的深远影响。我们将钢琴师变压器与从头开始训练的相同模型（无PT）进行比较。（a, b）预训练导致衡量与人类表现的分布相似性的客观指标的显著改善。(c)这是基于更好的学习基础，因为预先训练的模型收敛得更快，并且在微调期间损失大大降低。

如图3所示，预训练在目标度量和学习动态方面都产生了实质性的改进。预训练显著地减少了JS的发散，并增加了所有表达维度上的交集面积（图3a， 3b），这表明它与人类的表现分布明显更接近。微调曲线进一步显示，预训练模型收敛更快，达到d监督损失低得多，突出了初始化的优越性（图3c）。

表中的定量结果进一步强化了这一差距。整个交叉区域从0.6032 （scratch）提高到0.8501（预训练），相对增益为40.9%。预训练还可以大幅度降低JS发散速度（66.3%）、持续时间（65.2%）、IOI（37.6%）和踏板（61.2%）。这些在所有表达维度上的一致改进表明，仅在有限的监督数据上训练的模型很难捕捉到人类音乐能力的复杂、高方差分布。

表1:ASAP测试集的客观评价结果。我们使用JS发散区和交叉区比较钢琴师变压器和基线。对于JS Div，越低越好（↓）。对于交集，越高越好（↑）。“总体”列报告了四个表达维度的平均得分。在生成模型中，我们的模型在大多数指标上实现了最佳性能，优于先前的SOTA，并展示了预训练的深远影响。 

![](images/6529ee39e6bc31cdc5f4099742c31e0823e0e9956c1606bdca55b4cc0601ea92.jpg)  

Together, these results validate that large-scale self-supervised pre-training is essential, as it provides the broad musical priors that limited supervised data alone cannot supply.  

# 4.3. Pianist Transformer Achieves State-of-the-Art Results  

我们现在将我们的钢琴师变压器与以前最先进的模型进行比较。如表1所示，我们的模型全面展示了优越的性能。

我们的模型在所有生成模型中在8个指标中的6个指标和两个总体平均分数上取得了最好的分数。值得注意的是，《钢琴师变形》大大缩小了与人类真实的差距。例如，它的总体JS散度为0.1634，比最佳基线VirtuosoNet-ISGN（0.2791）有了实质性的改进。这表明，我们的模型生成的速度、持续时间和时间的分布比以前的方法更像人类。

仔细观察每个维度的结果，就会发现我们的模型在音乐时间建模方面的优势。最显著的收获是持续时间和IOI，它们控制着音乐的节奏和时间感。我们的模型在这些维度上的JS Divergence得分（分别为0.1879和0.1740）明显低于最佳基线得分。这表明大规模的预训练赋予了模型对音乐节拍和乐句的复杂理解，可能是从数据的深层结构背景中学习到的。

值得注意的是，VirtuosoNet-ISGN在Pedal指标上取得了更好的分数，这可能归因于其专门的架构。然而，我们的模型仍然产生高质量的踏板。它的JS散度为0.1111，与最先进的（0.0829）相比具有竞争力，并且优于其他基线。这表明，虽然在这个特定维度上不是最优的，但我们的通用预训练方法产生了强大的综合性能。

# 4.4. Subjective Evaluations Reveal Human-Level Performance  

虽然客观指标量化了统计相似性，但它们往往无法捕捉到真正音乐体验的整体品质。因此，我们进行了一项全面的主观聆听研究，以进行明确的、以人为本的评估。

# 4.4.1. STUDY DESIGN  

我们设计了一个严格的主观聆听研究，以确保我们的发现的可靠性和公正性。我们招募了57名来自不同音乐背景的参与者，经过严格的注意力检查和完成时间筛选，保留了39个高质量的回答。参与者对五个匿名的表现版本(我们的模型，两条基线，

乐谱和人类)为六个15秒的音乐节选跨越巴洛克到现代流行风格。为了减轻偏见，所有表演的呈现顺序对每个参与者都是完全随机的。综合方法，包括参与者人口统计、刺激选择和我们严格的数据验证协议，详见附录C。
![](images/26037d509909808345192e867b8d01add8eda91374ceb2e24020abf5bafb515a.jpg)  
4.4.2. MAIN RESULTS: OVERALL PREFERENCE AND HUMAN-LIKENESS   
图4：主观偏好排序结果。评价包括海顿（P1）、贝多芬（P2）、肖邦（P3）、巴赫（P4）的作品。(a)我们的钢琴师变压器的平均排名在统计上与人类的表现没有区别，并且明显优于所有基线。(b)我们的模型获得的第一名得票率略高于人类钢琴家，显示出强烈的听众吸引力。

听力研究结果如图4所示，显示出对Pianist Transformer的明确且一致的偏好。最直接的质量衡量标准是听众的偏好，我们的模型一直被评为所有生成系统中最好的。

如图4b所示，钢琴家变形金刚的第一名得票率（32.7%）不仅大大高于基线（7.7%和14.7%），甚至略高于人类钢琴家的得票率（30.8%）。这表明它的渲染不仅真实，而且对听众很有吸引力。

坚持不懈地坚持不懈，坚持不懈，坚持不懈。[2] [2] [5] [5] [3] [4] [4] [4] [3] [5] [5] [6] [5] [5] [6] [5] [6]采用一系列双侧配对t检验，对两组数据进行分析。结果证实，我们的模型的评级明显优于VirtuosoNet-ISGN (p < 0.001), VirtuosoNet-Han （p < 0.001）和Score基线（p < 0.001）。虽然我们观察到的模型优于人类表现的优势在统计上并不显著（p = 0.21），但这一结果提供了强有力的证据，表明我们的模型不仅达到了与人类艺术家相当的质量水平，而且与人类艺术家极具竞争力。

# 4.4.3. MULTI-DIMENSIONAL QUALITY AND STYLISTIC ROBUSTNESS  

为了理解这种强烈的听众偏好背后的原因，我们分析了多维评分和模型在不同音乐风格下的表现。

如雷达图（图5）所示，我们的钢琴师变压器富有表现力的轮廓与人类的表现密切相关，表明在所有评级方面都有良好的平衡，高质量的呈现。从数量上看，我们的模型不仅在节奏和计时（3.44 vs. 3.21）和发音（3.38 vs. 3.24）方面的平均得分高于人类钢琴家，而且在最关键的全局指标——人类相似性（3.43 vs. 3.29）方面也高于人类钢琴家。这个显著的结果表明，我们的模型产生的表演被认为是人类的，甚至可能是理想化的版本，没有任何单个人类录音中存在的小缺陷或特殊选择。

![](images/a9517565612c679d54ed4ea0d989f36b7bb77328d00aea02cb86ea66804c960e.jpg)  

![](images/3bb2eb9448aa18e0e45a8e40472542eee466e28e71712d7d47486b5723f79237.jpg)  
图5：多维主观评分（标准化）。一个雷达图，将四个表现维度的平均得分以5分制可视化。钢琴师变压器展示了一个侧面，密切反映了人类的表现，表明在各个方面都有良好的平衡和高质量的渲染。我们的基线覆盖的面积比所有其他基线都大得多。
图6：不同音乐风格的人类相似度得分分析。小提琴图显示了每个模型的人类相似度的分布，按历史时期分组。（a, b）对于巴洛克和古典音乐，基线模型的性能显著下降，有时低于Score基线。(c)虽然基线在浪漫音乐上表现更好，但钢琴师变压器在所有风格上都保持一贯的高水平表现。

此外，预训练的好处在模型跨越历史时期的风格稳健性中是显而易见的，如图6所示。虽然基线模型表现出强烈的风格依赖性，但它们的性能在巴洛克和古典作品中显着下降，但钢琴家变压器在所有风格中保持一致的高水平的人类相似性，接近人类水平。我们将这种鲁棒性归因于在大规模预训练期间获得的各种音乐知识，这可以防止过度拟合微调数据集的特定风格偏差。附录C.3提供了一个关于流行音乐领域外泛化的案例研究。

# 4.5. Analysis of Scaling Effects and Architecture  

在最后的分析中，我们对扩展效应进行了初步探索，以了解性能、扩展和架构选择之间的关系。结果如图7所示，验证了我们的框架是可伸缩的，同时也揭示了告知我们设计权衡的关键瓶颈。

模型缩放和解码器瓶颈。我们首先分析模型大小的影响。如图7a所示，将模型参数从15M增加到65M可以获得显著的性能提升。然而，从65M到135M，曲线明显变平，表明性能饱和。我们假设这个瓶颈源于我们轻量级的2层解码器。为了测试这一点，我们训练了对称的6层编码器，6层解码器（6-6）变体和更强大的解码器（用星号标记）。与我们的135M模型的1.260相比，它实现了1.230的显著低损耗，证实了浅解码器确实是模型容量扩展的主要瓶颈。

![](images/74e14ce271c8a30e84c49c1b89e1f1585a2ce2565e751c3a3c7275623c608854.jpg)  
图7：尺度效应的初步探索。预训练验证损失作为模型大小和数据量的函数。(a)在我们的非对称10-2架构中增加参数持续提高性能，尽管在135M处观察到饱和。星号标志着对称6-6变体实现的较低损耗，突出了解码器作为瓶颈。(b)增加数据产生可观的收益，最多可达1B代币，之后性能趋于平稳，表明135M模型的容量成为限制因素。

数据扩展和模型容量瓶颈。接下来，我们将研究数据规模的影响。图7b显示了将数据扩展到1B令牌时损失的显著下降。然而，当数据量增加10倍至10B （1.199 vs. 1.195）时，性能再次饱和。为了确定这是否也是一个解码器问题，我们利用了更强大的6-6模型。即使使用这种更强大的架构，10B数据的损失也只是略微下降到1.168。这表明，对于一个庞大的10b令牌数据集，无论编码器-解码器层分配如何，整体模型容量（大约135M个参数）本身都成为主要瓶颈。

体系结构权衡。虽然对称6-6模型实现了较低的预训练损失，证实了浅解码器是一个容量瓶颈，但我们在附录F中的详细评估显示，这一优势并没有转化为优越的下游性能。这表明，充分利用深度解码器的潜力可能需要比目前使用的更广泛的训练或数据扩展。因此，我们优先考虑非对称的10-2架构。如表10中的效率分析所示，10-2模型提供了相当的渲染质量，同时在CPU推断期间速度提高了约2.1倍，这对于实际应用程序来说是一种更为实际的折衷。

# 5. Conclusion  

在这项工作中，我们引入了钢琴家变形，通过大规模的自我监督预训练，建立了一种新的钢琴表演表现技术。通过从具有统一表示的100亿个标记的MIDI语料库中学习，我们的模型克服了阻碍先前方法学习音乐结构和表达之间复杂映射的数据稀缺性。我们的实验为这种范式转变提供了令人信服的证据：piano Transformer不仅在客观指标上表现出色，而且在主观评估方面达到了与人类艺术家在统计上没有区别的质量水平，其中其渲染有时更受欢迎。此外，我们的模型在不同的音乐风格中表现出强大的性能，这是其预训练基础的直接好处。最终，piano Transformer证明了规模化的自我监督学习是一条很有前途的道路，可以产生真正具有人类艺术水平的音乐，为未来的计算音乐表演研究建立一个有效和可扩展的范式。

# 6. Limitations and Future Work  

虽然piano Transformer表现出强大的性能，但它有几个限制，表明了未来的方向。首先，我们的高效轻量级解码器是扩展的性能瓶颈，激励研究更强大但更高效的解码器架构。其次，我们对钢琴独奏的关注将我们的自我监督模式扩展到多乐器和管弦乐设置。最后，超越基于分数的渲染的限制，从自然语言等直观输入中进行可控生成仍然是一个有前途的前沿。

References   
Borovik, I. V. Scoreperformer: Expressive piano performance rendering with fine-grained control. In Proceedings of the 24th International Society for Music Information Retrieval Conference, pp. 588–596, 2023.   
Bradshaw, L. and Colton, S. Aria-midi: A dataset of piano MIDI files for symbolic music modeling. In The 13th International Conference on Learning Representations, 2025.   
Bradshaw, L., Fan, H., Spangher, A., Biderman, S., and Colton, S. Scaling self-supervised representation learning for symbolic piano performance. CoRR, abs/2506.23869, 2025.   
Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J. D., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., Agarwal, S., Herbert-Voss, A., Krueger, G., Henighan, T., Child, R., Ramesh, A., Ziegler, D., Wu, J., Winter, C., Hesse, C., Chen, M., Sigler, E., Litwin, M., Gray, S., Chess, B., Clark, J., Berner, C., McCandlish, S., Radford, A., Sutskever, I., and Amodei, D. Language models are few-shot learners. In Advances in Neural Information Processing Systems, pp. 1877–1901, 2020.   
Cancino-Chacón, C. E. and Grachten, M. The basis mixer A computational romantic pianist. In Late-Breaking Demos of the 17th International Society for Music Information Retrieval Conference, 2016.   
Chen, T., Kornblith, S., Norouzi, M., and Hinton, G. A simple framework for contrastive learning of visual representations. In Proceedings of the 37th International Conference on Machine Learning, pp. 1597–1607, 2020.   
Chou, Y., Chen, I., Chang, C., Ching, J., and Yang, Y. Midibert-piano: Midibert-piano: Large-scale pre- -scale pre-training for symbolic music understanding. CoRR, abs/2107.05223, 2021.   
Devlin, J., Chang, M., Lee, K., and Toutanova, K. BERT: pre-t raining of deep bidirectional transformers for language understanding. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pp. 4171–4186, 2019.   
Flossmann, S., Grachten, M., and Widmer, G. Expressive performance rendering with probabilistic models. Guide to Computing for Expressive Music Performance, pp. 75–98, 2013.   
Foscarin, F., McLeod, A., Rigaux, P., Jacquemard, F., and Sakai, M. ASAP: a dataset of aligned scores and performances for piano transcription. In Proceedings of the 21th International Society for Music Information Retrieval Conference, pp. 534–541, 2020.   
Guo, Z. and Dixon, S. Moonbeam: A MIDI foundation model using both absolute and relative music attributes. CoRR, abs/2505.15559, 2025.   
Hawthorne, C., Simon, I., Roberts, A., Zeghidour, N., Gardner, J., Manilow, E., and Engel, J. H. Multiinstrument music synthesis with spectrogram diffusion. In Proceedings of the 23rd International Society for Music Information Retrieval Conference, pp. 598–607, 2022.   
He, K., Chen, X., Xie, S., Li, Y., Dollár, P., and Girshick, R. B. Masked autoencoders are scalable vision learners. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 15979–15988, 2022.   
Hsiao, W., Liu, J., Yeh, Y., and Yang, Compound word transformer: Learning to compose full- song music over dynamic directed hypergraphs. In The 35th AAAI Conference on Artificial Intelligence, pp. 178–186, 2021.   
Huang, Y. and Yang, Y. Pop music transformer: Beat-based modeling and generation of expressive pop piano compositions. In The 28th ACM International Conference on Multimedia, pp. 1180–1188. ACM, 2020.   
Jeong, D., Kwon, T., Kim, Y., Lee, K., and Nam, J. Virtuosonet: A hierarchical rnn-based system for modeling expressive piano performance. Proceedings of the 20th International Society for Music Information Retrieval Conference, pp. 908–915, 2019a.  

Jeong, D., Kwon, T., Kim, Y., and Nam, J. Graph neural network for music score data and modeling expressive piano performance. In Proceedings of the 36th International Conference on Machine Learning, pp. 3060–3070, 2019b.  

Kim, T. H., Fukayama, S., Nishimoto, T., and Sagayama, S. Statistical approach to automatic expressive   
rendition of polyphonic piano music. In Guide to Computing for Expressive Music Performance, pp. 145–179. 2013.   
Kong, Q., Li, B., Chen, J., and Wang, Y. Giantmidi-piano: A large-scale MIDI dataset for classical piano music. Transactions of the International Society for Music Information Retrieval, 5(1):87–98, 2022. P., Novack, Z., Berg-Kirkpatrick, T., and McAuley, J. J. PDMX: A large-scale public domain musicxml dataset for symbolic music processing. In 2025 IEEE International Conference on Acoustics, Speech and Signal Processing, pp. 1–5, 2025.   
Loshchilov, I. and Hutter, F. Decoupled weight decay regularization. In The 7th International Conference Learning Representations, 2019.   
Maezawa, A., Yamamoto, K., and Fujishima, T. Rendering music performance with interpretation variations using conditional variational RNN. In Proceedings of the 20th International Society for Music Information Retrieval Conference, pp. 855–861, 2019.   
Nakamura, E., Yoshii, K., and Katayose, H. Performance error detection and post-processing for fast and accurate symbolic music alignment. In Proceedings of the 18th International Society for Music Information Retrieval Conference, pp. 347–353, 2017.   
Oore, S., Simon, I., Dieleman, S., Eck, D., and Simonyan, K. and Simonyan, K. This time This time with feeling: learning expressive musical performance. Neural Comput. Appl., 32(4):955–967, 2020.   
Renault, L., Mignot, R., and Roebel, A. Expressive piano performance rendering from unpaired data. In International Conference on Digital Audio Effects, pp. 355–358, 2023.   
Spijkervet, J. and Burgoyne, J. A. Contrastive learning of musical representations. In Proceedings of the 22nd International Society for Music Information Retrieval Conference, pp. 673–681, 2021.   
Sundberg, J., Askenfelt, A., and Frydén, L. Musical performance: A synthesis-by-rule approach. Computer Music Journal, 7(1):37–43, 1983.   
Teramura, K., Okuma, H., Taniguchi, Y., Makimoto, S., and ichi Maeda, S. Gaussian process regression for rendering music performance. In Proceedings of 10th International Conference on Music Perception   
and Cognition, pp. 167–172, 2008.   
Wang, Z., Chen, K., Jiang, J., Zhang, Y., Xu, M., Dai, S., and Xia, G. POP909: A pop-song dataset for music arrangement generation. In Proceedings of the 21th International Society for Music Information Retrieval Conference, pp. 38–45, 2020.   
Warner, B., Chaffin, A., Clavié, B., Weller, O., Hallström, O., Taghadouini, S., Gallagher, A., Biswas, R., Ladhak, F., Aarsen, T., Adams, . T., Howard, J., and Poli, I. Smarter, better, faster, longer: A modern bidirectional encoder for fast, memory efficient, and long context finetuning and inference. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics, pp. 2526–2547, 2025.   
Zeng, Wang, R., Ju, Z., Qin, T., and Liu, T. Musicbert: Symbolic music understanding with large-scale pre-training. In Findings of the Association for Computational Linguistics, pp. 791–800, 2021.   
Zhang, B., Moiseev, F., Ainslie, J., Suganthan, P., Ma, M., Bhupatiraju, S., Lebrón, F., Firat, O., Joulin, A., and Dong, Z. Encoder-decoder gemma: Improving the quality-efficiency trade-off via adaptation. CoRR, abs/2504.06225, 2025.  

#  

Appendix Experimental Setup & Implementation Details 15 A.1 Dataset Details 15151515161717171718 A.2 Model Architecture and Tokenizer A.2.1 Model Architecture A.2.2 Unified MIDI Representation A.2.3 Comparison with Prior MIDI Tokenization Schemes A.3 Training Procedure A.3.1 Self-Supervised Pre-training A.3.2 Supervised Fine-Tuning A.4 Calculation of Objective Metrics A.5 Baseline Implementation   
B Expressive Tempo Mapping Algorithm 18 Subjective Listening Study Details 19 C.1 Participant Demographics 19 C.2 Musical Excerpts for Evaluation 20 C.3 Case Study: Generalization to Out-of-Domain Popular Music 20 C.4 Reliability and Consistency Analysis 21   
D Ablation Study of the Masking Ratio in Pre-training 21   
E Efficiency Analysis of Sequence Compression and Asymmetric Architecture 22   
F Detailed Discussion about Architectural Trade-off. 22   
G Inference Strategy 23   
H Ethics Statement 24  

# A. Experimental Setup & Implementation Details A.1. Dataset Details  

The pre-training corpus is constructed from the following sources. We applied specific preprocessing steps to ensure data quality and diversity.  

Aria-MIDI (Bradshaw & Colton, 2025): A large collection of over 1.1 million MIDI files transcribed from solo piano recordings. To ensure high fidelity, we only included segments with a transcription quality score above 0.95. Due to the coarse quantization in the original transcriptions, we applied random augmentations to the Velocity, Duration, and IOI values of these files to better simulate performance nuances.  

GiantMIDI-Piano (Kong et al., 2022): A dataset of over 10,000 unique classical piano works transcribed from live human performances using a high-resolution system. These files retain fine-grained expressive details, including velocity, timing, and pedal events.  

PDMX (Long et al., 2025): A diverse dataset of over 250,000 musical scores, originally in MusicXML format. We used their MIDI conversions to provide our model with clean, score-based MIDI data. To filter out overly simplistic or empty files, we only included MIDI files larger than 7 KB.  

POP909 (Wang et al., 2020): A dataset of 909 popular songs. We extracted the piano accompaniment tracks to include non-classical and accompaniment-style patterns.  

Pianist8 (Chou et al., 2021): A collection of 411 pieces from 8 distinct artists, consisting of audio recordings paired with machine-transcribed MIDI files.  

For SFT and evaluation, we use the ASAP dataset (Foscarin et al., 2020), a collection of aligned score-performance pairs of classical piano music.  

Before training, we first normalize all MIDI files so they can be tokenized under a consistent format. For multi-track scores, we merge all tracks into single time-ordered event stream and remove duplicate notes created during merging. We also convert every file to a fixed tempo of 120 BPM and rescale all onset times and durations. After this processing, all MIDI files share the same temporal scale and event structure, allowing reliable and uniform tokenization across the entire corpus.  

To ensure precise note-level correspondence between scores and performances, we refined the provided alignments. We first employed an HMM-based note alignment tool (Nakamura et al., 2017) to establish a direct mapping for each note. For localized mismatches where a few notes could not be paired, we applied an interpolation algorithm to infer the correct alignment based on the surrounding context. Finally, segments with large, contiguous blocks of unaligned notes were filtered out and excluded from our training and evaluation sets to maintain high data quality. We create a strict piece-wise split by randomly holding out 10% of the pieces for our test set. The remaining 90% are used for fine-tuning.  

# A.2. Model Architecture and Tokenizer  

A.2.1. MODEL ARCHITECTURE  

Our Pianist Transformer employs an asymmetric encoder-decoder architecture based on the T5-Gemma framework (Zhang et al., 2025). The encoder is designed to be substantially deeper than the decoder, with 10 layers, to efficiently process long input sequences and build a rich contextual representation. The decoder, with only 2 layers, is lightweight to ensure fast and efficient autoregressive generation during inference. This design strikes a balance between expressive power and practical utility. Key hyperparameters for our 135M model are detailed in Table 2.  

# A.2.2. UNIFIED MIDI REPRESENTATION  

Central to our approach is a unified, event-based token representation that treats both score and performance MIDI identically. Each musical note is represented as a fixed-length sequence of eight tokens, capturing its core attributes and nuanced pedal information. The sequence order is: [Pitch, IOI, Velocity, Duration, Pedal1, Pedal2, Pedal3, Pedal4].  

The vocabulary is structured as follows:  

Table 2: Key hyperparameters for the Pianist Transformer.   

![](images/4c27ac10a0647d22fd1eb8110b1b5e783f7a3bf0acd33a2776478fe790fd075a.jpg)  

Pitch: MIDI pitch values are mapped directly to 128 tokens (range 0 to 127).  

• Velocity: MIDI velocity values are mapped to 128 tokens (range 0 to 127). Timing (IOI & Duration): The Inter-Onset Interval (IOI) and Duration are quantized at a 1ms resolution and share a common vocabulary of 5000 tokens. The Duration can utilize the full range (0 to 4999), while the IOI is restricted to a smaller range (0 to 4990) to avoid a known artifact in transcribed MIDI where durations frequently saturate at the maximum value. Pedal: Four pedal tokens represent the sustain pedal state sampled at four equidistant points within the interval leading to the next note. Each pedal value is mapped to one of 128 tokens (range 0 to 127). While this representation supports continuous (half-pedal) values, our pre-training data predominantly contained binary pedal events (0 or 127), effectively training the model to generate on/off pedal control.  

Special tokens including [PAD], [MASK], [BOS], [EOS], and a special [PLAY] unused, resulting in a total vocabulary size of 5389. Figure 8 provides a concrete example of how a short musical phrase, under 120 BPM, is mapped into our 8-token event representation capturing pitch, timing, velocity, and pedal states.  

![](images/c2c34d64e770d4e2589bdb2dfe85b922f3a44110985022c70609e0fc10db454f.jpg)  
Figure 8: An Example of the Unified MIDI Representation  

A.2.3. COMPARISON WITH PRIOR MIDI TOKENIZATION SCHEMES Table 3 provides a comparison between our MIDI representation and several prior tokenization schemes. The design of our representation is guided by the specific requirements of the expressive performance rendering task, particularly the need to leverage large-scale MIDI data without structural features.  

The core advantages of our representation are as follows:  

Unlocks Self-Supervised Pre-training. By using time shift for temporal representation, our method does not require structural features like bars or beats. This is a crucial advantage because it allows us to pre-train on massive datasets of performance MIDI, which lack this structure and thus cannot be used by other methods.  

Table 3: Comparison of MIDI tokenization schemes. Our representation is tailored for scalable, selfsupervised expressive performance rendering.  

![](images/73e25194044aab94089bc4607b4ef68aa193943239ca2dd7dbae9ec58efa4951.jpg)  

Designed for Performance Rendering. Our note-centric approach treats each note and its properties as a single unit. This design is a natural fit for the rendering task, as it simplifies the process of matching an input score to an output performance and makes our model highly efficient.  

Captures Essential Piano Acoustics. Our representation explicitly encodes the sustain pedal, a crucial factor in producing the rich resonance characteristic of real piano performances. Incorporating pedal information enables the model to generate more expressive and realistic renderings.  

# A.3. Training Procedure  

A.3.1. SELF-SUPERVISED PRE-TRAINING  

The pre-training phase is designed to build a foundational understanding of musical structure and expression from our large-scale, unlabeled MIDI corpus. We employ a masked denoising objective, similar to T5, where the model learns to reconstruct corrupted segments of the input token sequences. In this stage, following recent mature practices in NLP for masked denoising objectives (Warner et al., 2025) we adopt masking ratio of 0.3 with tokens randomly masked. The model was trained for 40,000 steps using the AdamW optimizer (Loshchilov & Hutter, 2019). Key hyperparameters for this stage are detailed in Table 4.  

# A.3.2. SUPERVISED FINE-TUNING  

The fine- tuning process ran for 2 epochs. We adopted a slightly higher learning rate than in pre- training, which empirically led to faster convergence and a lower final loss. The learning rate followed a cosine decay schedule without a warmup phase. The global batch size was set to 32. All other settings remained consistent with the pre-t raining stage. A side-by-side comparison of pre-training and SFT hyperparameters is provided in Table 4.  

Table 4: Comparison of hyperparameters for Pre-training and SFT stages.   

![](images/1bfe7d35f68649d1da824de913ac7c92cc361ee17514d16cd2c7a6655a79727f.jpg)  

# A.4. Calculation of Objective Metrics  

To measure how closely the generated performances resemble human playing, we compare the global token distributions of the model outputs with those of the human performances across the entire test set. For Velocity, Duration, and IOI, we aggregate the tokens of each type from all generated pieces and compute their distributions, which we then compare with the corresponding human distributions using JS Divergence and Intersection Area. For Pedal, where the corpus mainly contains binary values, we binarize both model outputs and human data and evaluate the distribution of the 16 possible joint configurations formed by the four pedal tokens of each note.  

To obtain a human baseline, for every piece, we treat one human performance as the candidate and use the remaining human performances as the reference set, applying the same distributional comparison. This provides a measure of the natural stylistic variation among human performers.  

# A.5. Baseline Implementation  

For VirtuosoNet-HAN and VirtuosoNet-ISGN, we used the official implementations and pre-trained weights, followed their recommended inference procedures, and selected the composer-style configurations that best matched the pieces in our test set.  

ScorePerformer was evaluated under the same score-only setting. Since it is designed to operate with fine-grained style vectors derived from reference performances, which are not available in our setup, we adopted the unconditional generation mode recommended in the original paper, where style vectors are sampled from the prior distribution.  

All generated MIDI files from all models were rendered to audio using the same high-quality piano soundfont to ensure a fair subjective listening study.  

# B. Expressive Tempo Mapping Algorithm  

To make our model’s output compatible with standard music production software, we introduce the Expressive Tempo Mapping algorithm. This process converts the generated performance, which has timing in absolute milliseconds, into a standard MIDI file where expressive timing is encoded as a dynamic tempo map. This makes the performance fully editable within any DAW. The procedure is outlined in Algorithm 1.  

# Algorithm Expressive Tempo Mapping  

1: Input: Score MIDI Mscore, Performance MIDI Mper f   
2: Output: DAW-friendly expressive MIDI MDAW   
3: Extract notes Nscore, Nper f and pedal events CCper f from input files.   
4: Estimate a dynamic tempo curve Tchanges based on timing deviations.   
5: Initialize empty lists for aligned events: Naligned, CCaligned.   
6: for each corresponding note pair (nscore, nper f ) do   
7: Create a new note nnew where:   
8: pitch is from pitch of nscore   
9: velocity is from velocity of nper f   
10: onset in ticks is converted from nper f ’s onset in milliseconds using Tchanges.   
11: duration in ticks is converted from nper f ’s duration in milliseconds using Tchanges.   
12: Append nnew to Naligned.   
13: end for   
14: for each control event cc in CCper f do   
15: Convert cc’s timestamp from milliseconds to ticks using Tchanges to get tnew.   
16: Create a new control event ccnew with value from cc and time from tnew.   
17: Append ccnew to CCaligned.   
18: end for   
19: Assemble MDAW by combining Tchanges, Naligned, and CCaligned.   
20: return MDAW  

The algorithm executes in three main stages:  

1. Tempo Estimation (Line 4): First, we compare the timing of note onsets between the score MIDI (Mscore) and the generated performance MIDI (Mperf). The differences in timing are used to calculate a local tempo (BPM) for each segment of the piece. This sequence of tempo changes forms dynamic tempo curve, Tchanges, which captures all the expressive timing (rubato) of the performance.  

2. Event Remapping (Lines 6-18): Next, we create a new set of notes and pedal events. Each new note uses the pitch from the original score and the velocity from the generated performance. The crucial step is converting the onset time and duration of every note and pedal event from absolute milliseconds into musical ticks. This conversion is done using the tempo curve Tchanges estimated in the previous step. This aligns all events to a musical grid while preserving their expressive timing.  

3. Final Assembly (Line 19): Finally, the newly created tempo curve (Tchanges), the remapped notes (Naligned), and the remapped pedal events (CCaligned) are combined into a single, standard MIDI file (MDAW). The resulting file sounds identical to the original performance but is now fully editable in a DAW, with all timing nuances represented in the tempo track.  

# C. Subjective Listening Study Details  

To conduct a definitive, human-centric evaluation of our model’s performance, we designed and carried out a comprehensive subjective listening study. This appendix provides a detailed account of the study’s design, participants, materials, and procedures.  

# C.1. Participant Demographics  

Our subjective listening study’s validity rests on the quality and diversity of its participant pool. We initially recruited 57 individuals; after a rigorous screening for attentiveness and completion quality, 39 responses were retained for the final analysis. This section details the demographic composition of this group, providing evidence for its suitability for the nuanced task of evaluating musical expression.  

The detailed distributions of participants’ musical experience and listening habits are visualized in Figure 9. Several key characteristics of the group bolster the credibility of our findings:  

Balanced Expertise Spectrum (Figure 9a). The participants’ formal music training is not skewed towards one extreme. The pool includes a substantial proportion of listeners with no formal training (28.2%), ensuring that our model’s appeal is not limited to musically educated ears. Concurrently, the presence of highly experienced individuals (15.4% with > 10 years of training) guarantees that subtle expressive details are also being critically evaluated. This heterogeneity mitigates potential bias and strengthens the generalizability of our preference results.  

Representative Listening Habits (Figure 9b). The distribution of classical piano listening frequency reflects a general audience rather than a niche group of connoisseurs. The largest segment listens “Monthly" (46.2%), suggesting that the superior performance of Pianist Transformer is perceptible and appreciated even by those who are not deeply immersed in the genre daily.  

Competent and Calibrated Self-Assessment (Figure 9c). The self-assessed ability to discern music quality is centered around “Moderate" (46.2%), with a healthy portion rating themselves as “High" (20.5%). This distribution suggests a group that is confident in their judgments without being overconfident, indicating that the participants were well-suited for the evaluation task.  

In summary, the participant pool is intentionally diverse, comprising a mix of novices, enthusiasts, and experts. This composition ensures that our findings are robust, reliable, and reflective of a broad range of listener perceptions.  

![](images/560f0d1d0d1e5213a7c4c30d2bbcc0886df956d431766fc4b53f6e81556a3403.jpg)  
Figure 9: Demographic distribution of the 39 participants in the listening study. The plots show (a) the duration of formal music training, (b) the frequency of listening to classical piano music, and (c) self-assessed ability to discern piano music quality on a 1-5 scale. This diverse composition validates the generalizability of our study’s findings.  

# C.2. Musical Excerpts for Evaluation  

The listening study was based on six musical excerpts, each approximately 15 seconds long. To ensure an unbiased comparison, all excerpts were systematically taken from the beginning of each piece. The selection was also deliberately curated for stylistic breadth, featuring works from the Baroque, Classical, and Romantic periods, as well as modern pop style. This diversity provides a rigorous testbed for evaluating the models’ generalization abilities across varied musical contexts. The specific pieces are detailed in Table 5.  

Table 5: Musical excerpts selected for the subjective listening study, highlighting their stylistic diversity.   

![](images/fd62164ab7107f439c6a7fd2d43177af9257eb3ab22f89d734900de8b255de74.jpg)  

# C.3. Case Study: Generalization to Out-of-Domain Popular Music  

To rigorously probe the generalization capabilities of our model, we included a musical excerpt from a modern popular song. This piece is stylistically distinct from the primarily classical ASAP dataset used for fine-tuning, thereby serving as a challenging out-of-domain test. The goal was to assess whether the robust musical understanding gained during pre-training would translate effectively to genres beyond the immediate scope of the fine-tuning data.  

The results of this case study, summarized in Figure 10, reveal a nuanced dynamic. We observe that VirtuosoNet-ISGN delivers a highly competitive performance on this slow, lyrical piece. Its multidimensional ratings (Figure 10a) and average rank (Figure 10b) are nearly on par with our Pianist Transformer, suggesting that the expressive patterns it learned are well-suited for this particular style of song-like playing.  

However, a crucial distinction emerges from the first-place vote rate (Figure 10c). Despite the close average scores, listeners chose Pianist Transformer as the single best performance by a dominant margin. This finding highlights a key advantage of our approach. While other systems may produce competent or even good performances on stylistically favorable pieces, Pianist Transformer is significantly more likely to generate a truly exceptional rendering that listeners perceive as the definitive best. We attribute this superior appeal to the fine-grained nuances and avoidance of subtle AI artifacts learned during large-scale pre-training, which ultimately translates to a more compelling and preferred musical experience.  

![](images/bac1b1c91ca7c014857d2afd2c70585ca916b69b65818235626f4af73bc60705.jpg)  
Figure 10: Case Study on an Out-of-Domain Popular Music Excerpt. Subjective evaluation results for a slow, lyrical pop piece. (a, b) VirtuosoNet-ISGN performs competitively in average ratings and rankings. (c) However, our Pianist Transformer secures a dominant share of first-place votes, indicating superior overall appeal and quality.  

# C.4. Reliability and Consistency Analysis  

To ensure the robustness and impartiality of our subjective evaluation, we embedded an internal consistency check within the study. For one musical excerpt (Liszt’s work), two identical audio clips from the same model’s performance were presented to each participant as if they were distinct versions. The analysis of ratings for these duplicates, shown in Table 6, provides crucial insights into the study’s validity.  

First, the analysis confirms the experiment’s impartiality. The Mean Error (Bias) between the ratings for the duplicate clips is negligible, and a paired t-test showed these differences to be statistically insignificant (all p > 0.6). This result demonstrates that our experimental design successfully mitigated systematic biases, such as those arising from presentation order or listener fatigue. Furthermore, the Pearson correlation (r) between the paired ratings is positive but modest. This is an expected outcome, reflecting the inherent variability and noise in the subjective human perception of music. The presence of this natural perceptual uncertainty makes our main findings, the clear and statistically significant preference for Pianist Transformer, even more compelling. It indicates that the perceived quality difference between our model and its counterparts was strong and consistent enough to overcome this noise, thereby solidifying the significance and reliability of our conclusions.  

Table 6: Intra-rater reliability analysis on duplicate audio stimuli. We report the Mean Error (Bias) and Pearson Correlation (r) between ratings for two identical audio clips presented to the same user. The low, statistically non-significant bias confirms the experiment’s impartiality.   

![](images/f5f6e14564091e6a1041e8c5a7feb4a8156cdc8dfaad1104e81b63c0e26aecce.jpg)  

# D. Ablation Study of the Masking Ratio in Pre-training  

To analyze the effect of the masking ratio on downstream performance, we conducted an ablation study by pre-training models with three different ratios: 15%, 30% and 45% and then fine-tuning them on our rendering task. The results are presented in Table 7.  

The results indicate that our main setting of 30% outperforms the 15% ratio, while the 45% ratio yields slightly better overall performance. This suggests that the optimal masking strategy for symbolic music may differ from common practices in NLP, an interesting direction for future work.  

However, all three pre-trained models significantly outperform the supervised-only baselines, showing that the gains from self-supervised pre-training do not depend on any specific masking setting.  

Table 7: Ablation study on the pre-training masking ratio. While the 45% ratio achieves the best overall scores, all pre-trained variants significantly outperform supervised -only baselines.   

![](images/386ecfb6fb58750c7fb192d9c895b127fcd4324d1a05b68087002e93a05a88d7.jpg)  

# E. Efficiency Analysis of Sequence Compression and Asymmetric Architecture  

To investigate how our efficiency-related components influence the overall training efficiency, we conduct detailed analysis of note-level sequence compression and the asymmetric architecture. Each component individually reduces computational cost, but their combination leads to a substantially larger improvement than using either one alone.  

Table 8: Synergistic Efficiency Analysis of Sequence Compression and the Asymmetric Architecture. We report relative metrics where the baseline (6-6, Uncompressed) is set to 1.00x. Lower is better for VRAM, higher is better for Speed.   

![](images/a0090dc21aee8a72fc1d9cae7f5e5bbcf4823b8b234ff9cb7bbf57cad2eef3c2.jpg)  

As shown in table 8, compression accelerates training by 1.81× on the 6–6 architecture, and the 10–2 architecture improves speed by 1.07× without compression. However, when both components are applied together, the overall training speed reaches 3.13×, substantially exceeding the product of their individual gains. A similar synergistic effect appears in VRAM reduction, where compression and architectural asymmetry jointly amplify memory savings.  

These findings confirm that the proposed efficiency components are not merely additive but interact in a way that significantly amplifies their benefits, effectively achieving a "1 + 1 > 2" efficiency outcome.  

# F. Detailed Discussion about Architectural Trade-off.  

We chose an asymmetric 10-2 architecture to balance performance and efficiency. To validate this choice, we compare i against a symmetric 6-6 baseline with a similar parameter count.  

Table 9 shows the final rendering performance of both models after fine-tuning. While the symmetric 6-6 model achieves a slightly lower loss during pre -training, this does not translate to superior performance on the downstream rendering task. Our 10-2 model performs comparably to the 6-6 variant.  

While the final rendering quality is highly comparable between the two architectures, the choice is justified by the significant gains in computational efficiency, detailed in Table 10. Our 10-2 model is  

Table 9: Objective evaluation results on the ASAP test set, comparing the asymmetric 10-2 architecture with a symmetric 6-6 baseline. Despite the 6-6 model having a deeper decoder, our 10-2 model achieves comparable or even slightly better performance on the final rendering task.  

![](images/c695dfb874bc00d6054ec1e8ce9a4004eeb232dfe3953770d34eadcf755f256c.jpg)  

over 2x faster in CPU inference and substantially more resource-efficient during training, requiring 40% less VRAM and achieving 60% faster throughput.  

Table 10: Efficiency comparison between the asymmetric (10-2) and symmetric (6-6) architectures. The 6-6 model is set as the baseline for relative speed and memory usage. Our design significantly accelerates both training and inference while reducing memory footprint.   

![](images/192da1279b6b4cd108c675ec370d31c08702eb1ec591b9cfa9dd3e21a73b7e07.jpg)  

In conclusion, our asymmetric 10-2 architecture provides a superior trade-off, delivering state-of-the-art rendering quality while being significantly more efficient for both training and deployment. This makes it a more practical solution.  

# G. Inference Strategy  

During inference, our goal is to generate expressive performances that remain strictly matched to the input score. To preserve the note-level correspondence, we apply a hard pitch constraint: the model is free to sample expressive attributes but whenever a Pitch token is expected, we directly set it to the pitch from the input score instead of sampling. This enforces a one-to-one mapping between score notes and generated performance notes.  

![](images/42090fcf17c1665b978ee224e76b963df907b4000be7f211c93ace69898faafd.jpg)  
Figure 11: Overlapped Block-wise Generation Strategy. This figure illustrates how we generate long sequences using overlapping blocks. Block 1 is produced first. For the next block, we shift the window forward and reuse the stable overlapping region from the previous block as the decoder’s context. Block 2 then continues generation from this context.  

For pieces longer than 4096 tokens, we use an overlapped block-wise generation strategy illustrated in Figure 11. We first generate a 4096-token block. For the next block, we shift the window forward by 2048 tokens so that the new encoder input overlaps with the second half of the previous block. As decoder context, we reuse this overlapping region but drop a few unstable tokens at the tail before using it as the prompt. The model then continues generation from this context, and we append only the newly produced part. This procedure repeats until the piece is complete.  

# H. Ethics Statement  

The pre-training and fine-tuning of our model were conducted exclusively using publicly available datasets, as detailed in Appendix A. Our research did not involve the collection of new private data. For our subjective listening study, we recruited human participants. All participants were presented with an informed consent form prior to the study, which outlined the purpose of the research, the nature of the task, and how their data would be used. To protect participant privacy, all experimental responses were fully anonymized prior to analysis and were stored separately from any personal information required for compensation. We compensated each participant for their time and effort with a payment that exceeds the local minimum wage standard. We foresee no direct negative societal impacts resulting from this work, which is intended to advance research in computational music and creativity.  