# Legacy PT/CPT Preprocessing

These scripts belong to the older PianistTransformer/tokenizer data pipeline:

- unpaired pretrain JSONL generation
- Arrow conversion and merge
- pair-level tokenizer SFT generation

They are not part of the current Hybrid Note pipeline. Current experiments use
PianoCoRe-A paired refined MIDI directly and do not require CPT data.

Use the parent directory entrypoints instead:

```bash
python src/data_process/generate_json_with_paired_midi.py --overwrite
python src/data_process/update_json_score_feature_with_xml.py
```
