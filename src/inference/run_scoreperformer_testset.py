import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import pandas as pd


SCORE_NAME_MAP = {
    'Beethoven,_Ludwig_van/Piano_Sonata_No.12_in_A_flat_major,_Op.26_("Funeral_March")/1._Andante_con_variazioni_(A_flat_major)/score_MS_refined.mid': "Beethoven/Piano_Sonatas/12-1/score",
    'Beethoven,_Ludwig_van/Piano_Sonata_No.24_in_F_sharp_major,_Op.78_("A_Thérèse")/1._Adagio_cantabile_-_Allegro_ma_non_troppo/score_ASAP_refined.mid': "Beethoven/Piano_Sonatas/24-1/score",
    'Beethoven,_Ludwig_van/Piano_Sonata_No.24_in_F_sharp_major,_Op.78_("A_Thérèse")/2._Allegro_vivace/score_ASAP_refined.mid': "Beethoven/Piano_Sonatas/24-2/score",
    "Beethoven,_Ludwig_van/Piano_Sonata_No.27_in_E_minor,_Op.90/1._Mit_Lebhaftigkeit_und_durchaus_mit_Empfindung_und_Ausdruck/score_ASAP_refined.mid": "Beethoven/Piano_Sonatas/27-1/score",
    "Beethoven,_Ludwig_van/Piano_Sonata_No.27_in_E_minor,_Op.90/2._Nicht_zu_geschwind_und_sehr_singbar_vorgetragen/score_ASAP_refined.mid": "Beethoven/Piano_Sonatas/27-2/score",
    "Beethoven,_Ludwig_van/Piano_Sonata_No.28_in_A_major,_Op.101/1._Etwas_lebhaft_und_mit_der_innigsten_Empfindung/score_ASAP_refined.mid": "Beethoven/Piano_Sonatas/28-1/score",
    "Beethoven,_Ludwig_van/Piano_Sonata_No.28_in_A_major,_Op.101/2._Lebhaft._Marschmassig/score_ASAP_refined.mid": "Beethoven/Piano_Sonatas/28-2/score",
    "Beethoven,_Ludwig_van/Piano_Sonata_No.32_in_C_minor,_Op.111/1._Maestoso_-_Allegro_con_brio_ed_appassionato/score_ASAP_refined.mid": "Beethoven/Piano_Sonatas/32-1/score",
    "Chopin,_Frédéric/Ballade_No.2_in_F_major,_Op.38/score_ASAP_refined.mid": "Chopin/Ballades/2/score",
    "Chopin,_Frédéric/Scherzo_No.2_in_B_flat_minor,_Op.31,_B.111/score_ASAP_refined.mid": "Chopin/Scherzos/31/score",
    "Debussy,_Claude/Pour_le_Piano/1._Prélude/score_ASAP_refined.mid": "Debussy/Pour_le_Piano/1/score",
    "Glinka,_Mikhail/A_Farewell_to_Saint_Petersburg/10._The_Lark/score_MS_refined.mid": "Glinka/The_Lark/score",
    "Haydn,_Joseph/Piano_Sonata_No.46,_Keyboard_Sonata_in_E_major,_Hob.XVI:31/1._Moderato_(E_major)/score_ASAP_refined.mid": "Haydn/Keyboard_Sonatas/31-1/score",
    "Haydn,_Joseph/Piano_Sonata_No.47,_Keyboard_Sonata_in_B_minor,_Hob.XVI:32/1._Allegro_moderato_(B_minor)/score_ASAP_refined.mid": "Haydn/Keyboard_Sonatas/32-1/score",
    "Haydn,_Joseph/Piano_Sonata_No.52,_Keyboard_Sonata_in_G_major,_Hob.XVI:39/1._Allegro_con_brio_(G_major)/score_ASAP_refined.mid": "Haydn/Keyboard_Sonatas/39-1/score",
    "Haydn,_Joseph/Piano_Sonata_No.52,_Keyboard_Sonata_in_G_major,_Hob.XVI:39/2._Adagio_(C_major)/score_ASAP_refined.mid": "Haydn/Keyboard_Sonatas/39-2/score",
    "Haydn,_Joseph/Piano_Sonata_No.52,_Keyboard_Sonata_in_G_major,_Hob.XVI:39/3._Prestissimo_(G_major)/score_ASAP_refined.mid": "Haydn/Keyboard_Sonatas/39-3/score",
    "Liszt,_Franz/Mephisto_Waltz_No.1,_S.514/score_ASAP_refined.mid": "Liszt/Mephisto_Waltz/score",
    "Mozart,_Wolfgang_Amadeus/Fantasia_in_C_minor,_K.475/score_ASAP_refined.mid": "Mozart/Fantasie_475/score",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--metadata", type=Path, default=Path("PianoCoRe/metadata.csv"))
    p.add_argument("--midi-root", type=Path, default=Path("PianoCoRe"))
    p.add_argument("--score-source-list", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--token-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--gpus", default="0,1,2")
    p.add_argument("--workers-per-gpu", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=2500)
    p.add_argument("--time-window", type=float, default=2.0)
    return p.parse_args()


def collect_items(args):
    allowed = [
        line.strip()
        for line in args.score_source_list.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    df = pd.read_csv(
        args.metadata,
        usecols=["tier_a", "split", "refined_score_midi_path", "refined_performance_midi_path", "performance_dataset"],
    )
    df = df[
        df["tier_a"].fillna(False).astype(bool)
        & df["split"].eq("test")
        & df["performance_dataset"].fillna("").astype(str).eq("ASAP")
        & df["refined_score_midi_path"].isin(allowed)
    ]
    items = []
    for score_source in sorted(allowed):
        rows = df[df["refined_score_midi_path"].eq(score_source)]
        if rows.empty:
            raise FileNotFoundError(f"Missing ASAP metadata row for {score_source}")
        if score_source not in SCORE_NAME_MAP:
            raise KeyError(f"No ScorePerformer token map for {score_source}")
        items.append(
            {
                "score_source": score_source,
                "scoreperformer_score": SCORE_NAME_MAP[score_source],
                "score_midi": str((args.midi_root / "refined" / score_source).resolve()),
                "ground_truth_paths": [
                    str((args.midi_root / "refined" / p).resolve())
                    for p in sorted(rows["refined_performance_midi_path"].dropna().unique())
                ],
            }
        )
    return items


def clamp_tokens(tokens, tokenizer):
    tokens = np.asarray(tokens, dtype=np.int64).copy()
    if isinstance(tokenizer.sizes, dict):
        sizes = [
            tokenizer.sizes[key]
            for key, _ in sorted(tokenizer.vocab_types_idx.items(), key=lambda item: item[1])
        ]
    else:
        sizes = list(tokenizer.sizes)
    for idx, size in enumerate(sizes):
        tokens[:, idx] = np.clip(tokens[:, idx], 0, int(size) - 1)
    return tokens


def build_runtime(args, device):
    import torch
    from omegaconf import OmegaConf
    from scoreperformer.data.collators import MixedLMScorePerformanceCollator
    from scoreperformer.data.datasets import LocalScorePerformanceDataset
    from scoreperformer.data.tokenizers import SPMuple2
    from scoreperformer.inference.generators import ScorePerformerGenerator
    from scoreperformer.inference.messengers import SPMuple2Messenger, SPMupleMessenger
    from scoreperformer.models.scoreperformer import ScorePerformer

    obj = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = json.loads(obj["experiment"]["config"])
    dcfg = cfg["data"]["dataset"]
    dcfg.update(
        root=str(args.token_root),
        split="all",
        performance_directions="external/scoreperformer/data/directions/direction_classes.json",
        score_directions_dict="external/checkpoints/scoreperformer/data/asap1.1_perf_directions.json",
        sample=False,
        preload=False,
        cache=True,
    )
    for key in ["_name_", "_splits_"]:
        dcfg.pop(key, None)
    dataset = LocalScorePerformanceDataset(**dcfg)
    ccfg = cfg["data"]["collator"]
    ccfg.pop("_name_", None)
    collator = MixedLMScorePerformanceCollator(**ccfg)
    mcfg = OmegaConf.create(cfg["model"])
    mcfg.pop("_name_", None)
    mcfg.pop("_version_", None)
    ScorePerformer.inject_data_config(mcfg, dataset)
    model = ScorePerformer.init(mcfg)
    model.load_state_dict(obj["model"]["state_dict"])
    model.to(device)
    model.eval()
    messenger = SPMuple2Messenger(dataset.tokenizer) if isinstance(dataset.tokenizer, SPMuple2) else SPMupleMessenger(dataset.tokenizer)
    generator = ScorePerformerGenerator(model, dataset, collator, messenger, device=device)
    return obj, dataset, model, generator


def generate_one(args, runtime, item, sample_idx, output_path):
    import torch

    obj, dataset, model, generator = runtime
    generator.reset()
    perfs = [p for p, (score, _) in dataset._performance_map.items() if score == item["scoreperformer_score"]]
    if not perfs:
        raise KeyError(f"Missing tokenized performance skeleton for {item['scoreperformer_score']}")
    perf_idx = dataset.performance_names.index(sorted(perfs)[0])
    start = time.perf_counter()
    score_emb, _, _ = generator.encode_embeddings(perf_idx, overlay_bars=0.5)
    if sample_idx == 0:
        style_emb = torch.zeros(score_emb.shape[0], model.perf_encoder.embedding_dim, device=score_emb.device)
        style_name = "zero"
    else:
        control = obj["control_embs"]["dynamic/mf"].to(score_emb.device)
        style_emb = control[None].repeat(score_emb.shape[0], 1)
        style_name = "dynamic/mf"
    generator.prepare_performance_notes(perf_idx, score_embeddings=score_emb, perf_embeddings=style_emb)
    current_time = 0.0
    steps = 0
    while not generator.perf_data.reached_eos and steps < args.max_steps:
        _, messages = generator.generate_performance_notes(
            start_time=current_time,
            time_window=args.time_window,
            time_window_overflow=0.5,
            filter_kwargs={"k": 8},
            disable_tqdm=True,
        )
        if messages is None or len(messages) == 0:
            current_time += args.time_window
        else:
            current_time = max(current_time + 0.5, float(messages[:, 0].max()))
        steps += 1
    tokens = generator.perf_data.gen_seq.detach().cpu().numpy()
    tokens = tokens[(tokens[:, 0] != generator.sos_token_id) & (tokens[:, 0] != generator.eos_token_id)]
    tokens = clamp_tokens(tokens, dataset.tokenizer)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    midi = dataset.tokenizer.performance_tokens_to_midi(tokens, output_path=None)
    midi.dump(str(output_path))
    return time.perf_counter() - start, steps, bool(generator.perf_data.reached_eos), style_name


def worker(worker_idx, gpu, args, jobs, results):
    device = f"cuda:{gpu}"
    runtime = build_runtime(args, device)
    while True:
        job = jobs.get()
        if job is None:
            return
        score_idx, sample_idx, item = job
        output_path = args.output_dir / "midis" / f"{score_idx:02d}__sample_{sample_idx:03d}.mid"
        elapsed, steps, reached_eos, style_name = generate_one(args, runtime, item, sample_idx, output_path)
        results.put(
            (
                score_idx,
                sample_idx,
                {
                    "path": str(output_path.resolve()),
                    "inference_seconds": elapsed,
                    "steps": steps,
                    "reached_eos": reached_eos,
                    "style": style_name,
                    "gpu": gpu,
                    "worker_idx": worker_idx,
                },
            )
        )


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    items = collect_items(args)
    gpus = [int(x) for x in args.gpus.split(",") if x.strip()]
    ctx = mp.get_context("spawn")
    jobs, results = ctx.Queue(), ctx.Queue()
    workers = []
    for gpu in gpus:
        for _ in range(args.workers_per_gpu):
            worker_idx = len(workers)
            proc = ctx.Process(target=worker, args=(worker_idx, gpu, args, jobs, results))
            proc.start()
            workers.append(proc)
    wall_start = time.perf_counter()
    num_samples = 2
    for score_idx, item in enumerate(items):
        for sample_idx in range(num_samples):
            jobs.put((score_idx, sample_idx, item))
    for _ in workers:
        jobs.put(None)
    per_score = {idx: {**item, "prediction_paths": [None] * num_samples, "inference_seconds": [None] * num_samples, "sample_details": [None] * num_samples} for idx, item in enumerate(items)}
    for done in range(len(items) * num_samples):
        score_idx, sample_idx, payload = results.get()
        per_score[score_idx]["prediction_paths"][sample_idx] = payload["path"]
        per_score[score_idx]["inference_seconds"][sample_idx] = payload["inference_seconds"]
        per_score[score_idx]["sample_details"][sample_idx] = payload
        print(f"completed {done + 1}/{len(items) * num_samples} score={score_idx:02d} sample={sample_idx}", flush=True)
    for proc in workers:
        proc.join()
        if proc.exitcode:
            raise RuntimeError(f"worker {proc.pid} exited with {proc.exitcode}")
    manifest = {
        "model": "ScorePerformer",
        "protocol": "scoreperformer_zero_mf",
        "checkpoint": str(args.checkpoint.resolve()),
        "token_root": str(args.token_root.resolve()),
        "score_source_list": str(args.score_source_list.resolve()),
        "split": "test",
        "gt_filter": "performance_dataset=ASAP",
        "num_samples": num_samples,
        "sample_styles": ["zero", "dynamic/mf"],
        "gpus": gpus,
        "workers_per_gpu": args.workers_per_gpu,
        "items": [per_score[i] for i in range(len(items))],
        "total_inference_seconds": sum(sum(x["inference_seconds"]) for x in per_score.values()),
        "wall_inference_seconds": time.perf_counter() - wall_start,
    }
    (args.output_dir / "prediction_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
