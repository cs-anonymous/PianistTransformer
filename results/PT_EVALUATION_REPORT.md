# PT Evaluation Report

**Date**: 2026-06-13  
**Task**: Evaluate PT on ASAP subset only  
**Status**: ASAP corrected run completed

---

## 1. Result

The corrected ASAP evaluation has been completed and saved to:

```text
results/pt_official_subset_eval_corrected/pt_results_asap.json
```

### ASAP Subset (104 samples, 53,248 notes)

| Feature | JS ↓ | IA ↑ | MAE ↓ | RMSE ↓ | Pearson ↑ |
|---------|------|------|-------|--------|-----------|
| Velocity | 0.0180 | 0.8997 | 10.51 | 13.72 | 0.6662 |
| Duration | 0.0086 | 0.9118 | 101.23 | 199.64 | 0.8439 |
| IOI | 0.0031 | 0.9649 | 22.83 | 61.90 | 0.9600 |
| BPedal | 0.0059 | 0.9170 | 4.61 | 7.91 | 0.3732 |
| CPedal | 0.2952 | 0.5391 | 44.29 | 63.48 | 0.3967 |
| **Overall** | **0.0661** | **0.8465** | **36.69** | **69.33** | **0.6480** |

---

## 2. Notes

- This run uses the corrected PT evaluator with score-token recovery, forced token ranges, and decoder start token stripping.
- `Overall` is recomputed in the same style as `results/INR_EPR_3MODELS_SUBSET_EVALUATION_2026-06-12.md`: mean over `Velocity`, `Duration`, `IOI`, `BPedal`, and `CPedal`.
- The earlier `velocity JS = 1.0` / `duration JS = 0.6694` result should be treated as invalid.
- This ASAP result is now much closer to the paper numbers in `docs/Pianist_Transformer_2025.md`.
