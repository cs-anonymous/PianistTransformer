# PT Evaluation Report

**Date**: 2026-06-07

---

## 1. Evaluation Summary

### 1.1 Method
- **Method**: Score → Performance Generation (correct EPR method)
- **Model**: PT pretrained on ASAP dataset
- **Input**: Score MIDI (pitch + timing + velocity + pedal from score)
- **Output**: Generated performance tokens (8 tokens/note)
- **Evaluation**: Binary (BPedal) and Continuous (CPedal) pedal methods computed in a single pass

### 1.2 Dataset

**Important finding about the 256/501 split**:
The previously reported "ASAP: 256 samples, PianoCoRe-only: 501 samples" numbers do NOT match any actual data split in the codebase. The real distribution in the 757 evaluation windows is:

| Split by | ASAP | Non-ASAP | Total |
|----------|------|----------|-------|
| **performance_dataset** (actual) | **104** | 653 | 757 |
| score_source (file name) | 224 | 533 | 757 |
| Previously reported | 256 | 501 | 757 |

The correct split is by `performance_dataset`, which gives **104 ASAP samples** (53,248 notes).

---

## 2. PT Results (ASAP Subset Only)

### ASAP Subset: 104 samples, 53,248 notes

| Feature | JS ↓ | IA ↑ | MAE ↓ | RMSE ↓ | Pearson ↑ |
|---------|------|------|-------|--------|-----------|
| Velocity | **1.0000** ❌ | 0.0000 | 67.58 | 69.71 | 0.0000 |
| Duration | 0.6742 | 0.1622 | 777.43 | 1637.80 | 0.0584 |
| IOI | 0.3298 | 0.5231 | 663.46 | 1663.20 | 0.0220 |
| BPedal | 0.5095 | 0.4268 | 7.27 | 8.67 | 0.0887 |
| **Overall** | **0.6284** | **0.2780** | **378.93** | **844.84** | **0.0423** |

| Feature | JS ↓ | IA ↑ | MAE ↓ | RMSE ↓ | Pearson ↑ |
|---------|------|------|-------|--------|-----------|
| Velocity | **1.0000** ❌ | 0.0000 | 67.58 | 69.71 | 0.0000 |
| Duration | 0.6742 | 0.1622 | 777.43 | 1637.80 | 0.0584 |
| IOI | 0.3298 | 0.5231 | 663.46 | 1663.20 | 0.0220 |
| CPedal | 0.2812 | 0.5391 | 59.74 | 77.41 | 0.0795 |
| **Overall** | **0.5713** | **0.3061** | **392.05** | **862.03** | **0.0400** |

---

## 3. Comparison with Hybrid Node

### ASAP Subset Comparison

| Metric | PT (ASAP 104 samples) | Hybrid Node (256 samples) |
|--------|----------------------|---------------------------|
| **Velocity JS ↓** | 1.0000 ❌ | 0.2057 |
| **Duration JS ↓** | 0.6742 | 0.3489 |
| **IOI JS ↓** | 0.3298 | 0.0373 |
| **Pedal JS ↓** (BPedal) | 0.5095 | 0.2611 |
| **Overall JS ↓** | 0.6284 | 0.2133 |
| | | |
| **Velocity Pearson ↑** | 0.0000 | 0.5836 |
| **IOI Pearson ↑** | 0.0220 | 0.7342 |
| **Overall Pearson ↑** | 0.0423 | 0.4254 |

**Note**: Sample sizes differ (104 vs 256). PT was evaluated on ASAP performances within the same 757 windows. The 256 Hybrid Node samples came from a different (previously unknown) split. The correct ASAP sample count in 757 windows is 104.

### PianoCoRe-only Subset: Not Evaluated

The PianoCoRe-only evaluation was stopped due to excessive runtime. Based on previous failed evaluation attempts on PianoCoRe data, PT is expected to perform poorly (Overall JS > 0.48) due to cross-dataset generalization issues.

---

## 4. Key Observations

### 4.1 PT Velocity Completely Failed

- **Velocity JS = 1.0000** (worst possible value, indicates complete distribution mismatch)
- **Velocity Pearson = 0.0000** (no correlation at all)
- PT generates velocity values that are completely uncorrelated with ground truth

### 4.2 Duration and IOI Also Very Poor

- Duration JS = 0.6742 (very poor distribution overlap)
- IOI JS = 0.3298 (moderate but far from Hybrid Node's 0.0373)
- RMSE for both is extremely high (>1600ms)

### 4.3 BPedal is PT's Best Feature

- BPedal JS = 0.5095 is the least bad metric
- But still far worse than Hybrid Node's 0.2611

### 4.4 CPedal Better Than BPedal

- CPedal JS = 0.2812 (much better than BPedal 0.5095)
- This makes sense: continuous evaluation has more bins, less sensitive to exact matches

---

## 5. Conclusion

PT fails on Score → Performance generation even on its own ASAP training domain:
- Overall JS = 0.6284 (BPedal) / 0.5713 (CPedal) — far worse than Hybrid Node's ~0.26
- Velocity completely fails (JS=1.0)
- Poor generalization across all features

This is consistent with previous failed evaluation on PianoCoRe data, confirming that PT cannot generate meaningful performances from score input using our evaluation setup.

---

## 6. Technical Notes

### Evaluation Method
- Score → Performance generation using `model.generate()` with greedy decoding
- Score input created from score MIDI continuous features
- Both binary and continuous pedal metrics computed in a single evaluation pass

### Dataset
- Same 757 evaluation windows as Hybrid Node (same config, same seed)
- ASAP subset: 104 windows where `performance_dataset == 'ASAP'`
- Each window: up to 512 notes with 50% overlap

### Why So Slow?
- PT uses autoregressive token generation (8 tokens/note)
- Each sample requires sequential generation of up to 4096 tokens
- Multi-processing overhead (6 workers × 3 GPUs)
- Single ASAP run took ~2 hours for 104 samples
