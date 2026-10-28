# 实验运行与 GPU 调度约定

本仓库的方差分离实验遵守以下运行规则：

1. 一个 inference 任务只使用一张 GPU。不要把同一个 checkpoint 的一次 inference 拆到多张 GPU。
2. 同一张 GPU 可以同时运行多个相互独立的 inference 进程；在当前 3090 环境中，默认目标是每张 GPU 至少 8 个进程，以提高低 batch、窗口级 inference 的 GPU 利用率。
3. 不同进程必须使用独立的输出目录、sampling seed 和日志文件。禁止多个进程写同一个 prediction manifest 或同一个评估 JSON。
4. 多卡调度的基本单位是“任务到 GPU”的绑定，而不是 DDP。只有训练任务需要多卡 DDP 时才使用 `torchrun`；inference 和 PN/PP evaluator 使用单卡进程。
5. inference 和 evaluator 的 CPU worker 数默认设为 0 或 1。进程数增加时，不能再让每个进程创建大量 worker，否则 CPU oversubscription 会降低吞吐。
6. 启动前检查每张 GPU 的显存余量；若单进程显存约 1--2 GB，可逐步增加到 8 个进程。出现 OOM 时按 8、6、4、2 逐级降低，并记录实际并发度。
7. 每个任务必须写入 `runs.jsonl`，记录 GPU、checkpoint、training seed、sampling seed、配置、输出目录和状态。任务完成后以结果文件存在性作为断点续跑依据。
8. M0 的 evaluator 复现性实验可以在 CPU 上运行；M1/M2/M4 的每个 inference 任务固定单卡并高并发；M3 训练任务遵守显存限制，不能与高并发 inference 共用同一张卡。

推荐的进程布局是：先把每个独立 inference 任务分配给固定 GPU，再在该 GPU 上以 8 个并发槽运行任务；一个槽完成后立即补充同 GPU 队列中的下一个任务。调度器需要避免使用会随父 shell 退出而终止的后台子进程，长任务使用独立 session，并保留 launcher 日志。

# 论文质量指标评估约定

论文主表、rebuttal 表格以及任何跨模型质量比较中的 PP/PN Wasserstein 和 Human Relevance，必须使用 score-span-normalized 的时间协议。对每首曲目和每个生成样本，按 score onset span 对性能 timing 做一个全局缩放；human reference 也必须使用同一 score-span 协议。不能把未经缩放的绝对时间结果与 normalized 结果放在同一质量表中。

PianoFlow、ScorePerformer 或其他外部模型的结果在进入质量表前，必须先检查 prediction manifest、GT manifest 和 normalization report。只有明确完成 score-span normalization 的结果才可与 INR、PT、VirtuosoNet 和 DExter 的主表数值比较。若外部模型尚未完成该归一化，只报告效率或单独标注为 absolute-time audit，不报告其可比的质量结论。

评估记录中必须同时写明：时间缩放是否应用于 prediction 和 GT、tempo map 的来源、pedal support（二值或原始 CC64）、样本数、test score 数，以及 HR 使用的 human anchor 文件。任何因固定 120 BPM、缺少 CC64 或未应用 score tempo map 造成的协议差异，都必须在表格和 rebuttal 中明确标注。

musical51 模型的训练和 inference 必须读取相同布局的 tagged sidecar，并核对 source identity。不能将 JSON 中的 compact9 特征直接当成 musical51 的原始特征。score-span 评估使用与模型输入一致的 `score_raw` 时间基准；需要 MIDI 时在实验目录中显式序列化，不能依赖迁移前的 `data/refined` 路径。

逐 epoch 的 teacher-forced analytic/histogram HR 是训练诊断。若它使用 raw training timing targets，必须明确标注，不能作为 score-span-normalized AR 测试 HR 写入论文质量表。数据规模同时记录 metadata 演奏数和实际可用演奏数，并保留上游对齐失败的排除原因。
