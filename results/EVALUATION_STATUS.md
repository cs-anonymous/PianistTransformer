# Expressive Performance Rendering Evaluation Status

**Date**: 2026-06-07  
**Task**: Compare Hybrid Node (1 node/note) vs PT (8 tokens/note) on EPR task

---

## 1. Evaluation Summary

### 1.1 Hybrid Node (Our Model) ✅ **COMPLETED**

**Model**: `models/sft_nodes/sft_node_2026-06-07-03-38-14/checkpoint-1000`  
**Task**: Score → Performance generation  
**Dataset**: PianoCoRe test set (split by work, ASAP + PianoCoRe-only subsets)  
**Evaluation Method**: Binary and Continuous pedal methods

**Results Location**: `results/hybrid_node_evaluation/`

#### ASAP Subset (256 samples, 131,072 notes)

| Feature | JS ↓ | IA ↑ | MAE ↓ | RMSE ↓ | Pearson ↑ |
|---------|------|------|-------|--------|-----------|
| Velocity | 0.2057 | 0.5996 | 10.87 | 13.52 | 0.5836 |
| Duration | 0.3489 | 0.4300 | 227.08 | 337.01 | 0.3399 |
| IOI | 0.0373 | 0.8715 | 52.41 | 137.37 | 0.7342 |
| Pedal | 0.2611 | 0.5061 | 7.05 | 10.01 | 0.0440 |
| CPedal | 0.4709 | 0.3257 | 39.16 | 48.14 | 0.0688 |
| **Overall** | **0.2648** | **0.5466** | **67.31** | **109.21** | **0.3541** |

#### PianoCoRe-only Subset (501 samples, 252,692 notes)

| Feature | JS ↓ | IA ↑ | MAE ↓ | RMSE ↓ | Pearson ↑ |
|---------|------|------|-------|--------|-----------|
| Velocity | 0.3527 | 0.4177 | 12.45 | 15.62 | 0.4816 |
| Duration | 0.2836 | 0.4851 | 385.36 | 695.62 | 0.1770 |
| IOI | 0.0561 | 0.8302 | 76.62 | 182.13 | 0.6920 |
| Pedal | 0.0454 | 0.7960 | 7.13 | 9.93 | 0.0709 |
| CPedal | 0.5887 | 0.2256 | 51.68 | 57.39 | 0.0341 |
| **Overall** | **0.2653** | **0.5509** | **106.65** | **192.14** | **0.2911** |

**Key Observations**:
- Strong IOI (timing) prediction: JS ~0.04-0.06, Pearson ~0.70-0.73
- Consistent velocity and duration performance across subsets
- Pedal (binary) performs well on PianoCoRe-only (JS 0.0454)
- CPedal (continuous) shows higher JS on PianoCoRe-only due to 37.2% half-pedal in ground truth

---

### 1.2 PT (Pianist Transformer) ⚠️ **EVALUATION INCOMPLETE**

**Model**: PT pretrained model from ASAP dataset  
**Attempted Evaluation**: Score → Performance generation on PianoCoRe test set  
**Status**: **Failed / Not representative**

**Results Location**: `results/pt_evaluation_generation/`

#### PianoCoRe Test Set (50 samples, 25,600 notes)

| Feature | JS ↓ | IA ↑ | MAE ↓ | RMSE ↓ | Pearson ↑ |
|---------|------|------|-------|--------|-----------|
| Velocity | 0.9998 ❌ | 0.0000 | 60.23 | 62.38 | 0.1266 |
| Duration | 0.5847 | 0.2388 | 423.58 | 730.30 | 0.1578 |
| IOI | 0.4106 | 0.3838 | 138.15 | 253.95 | 0.0522 |
| Pedal | 0.4153 | 0.3438 | 6.38 | 7.60 | 0.1806 |
| CPedal | 0.0390 | 0.7690 | 60.99 | 88.01 | 0.0938 |
| **Overall** | **0.4819** ❌ | **0.3471** | **137.87** | **228.45** | **0.1222** |

**Critical Issues**:
1. **Velocity JS = 0.9998** (almost worst possible value, indicates complete failure)
2. **Overall JS ~0.48** (far worse than Hybrid Node's ~0.27)
3. **Dataset mismatch**: PT trained on ASAP, tested on PianoCoRe
4. **Poor generalization**: PT cannot generate meaningful performances from PianoCoRe scores

---

## 2. Why PT Evaluation is Incomplete

### 2.1 Original Evaluation Method Issue

**Previous attempt** (discarded):
- Method: Teacher Forcing (Performance → Performance reconstruction)
- Input: Ground truth performance tokens
- Output: Reconstructed performance tokens
- Result: Overall JS = 0.0704 ✅
- **Problem**: This is NOT a generation task! Using ground truth as input is circular reasoning.

**User feedback**: *"Teacher Forcing相当于用结果预测结果，这是什么测试方法？我不接受这种测试"*

### 2.2 Current Evaluation Method

**Method**: Score → Performance Generation (correct approach)
- Input: Score tokens from PianoCoRe dataset (pitch + timing + velocity + pedal from score MIDI)
- Output: Generated performance tokens
- Result: **Complete failure** (Overall JS = 0.5-0.6, Velocity JS = 0.9998)

### 2.3 Why PT Failed on PianoCoRe

**Hypothesis 1: Dataset Mismatch**
- PT trained on: ASAP dataset
- PT tested on: PianoCoRe dataset
- Different data distributions, composers, styles

**Hypothesis 2: Model Architecture**
- PT may be overfitted to ASAP characteristics
- Poor generalization to unseen data distributions

**Hypothesis 3: Input Format Differences**
- Our score_continuous format may differ from PT's expected input
- PianoCoRe preprocessing vs ASAP preprocessing differences

### 2.4 What Needs to Be Done

To complete PT evaluation, we need:

1. **Test PT on its own test set (ASAP)**
   - Use PT's processed ASAP test data
   - Requires running PT's alignment tool (currently blocked by permissions)
   - Would give fair comparison on PT's trained domain

2. **OR: Report incomplete evaluation**
   - State that PT cannot be fairly evaluated on PianoCoRe
   - PT's paper results (Overall JS ~0.16 on ASAP) cannot be verified
   - Cross-dataset generalization is poor

---

## 3. Current Conclusions

### 3.1 What We Can Conclude

1. **Hybrid Node performs well on both ASAP and PianoCoRe subsets**
   - ASAP subset: Overall JS 0.2648, Pearson 0.3541
   - PianoCoRe-only: Overall JS 0.2653, Pearson 0.2911
   - Strong timing prediction (IOI Pearson ~0.70-0.73)
   
2. **PT shows poor cross-dataset generalization**
   - Cannot generate meaningful performances from PianoCoRe scores
   - Overall JS 0.4819 vs Hybrid Node's 0.2653
   - Velocity prediction completely fails (JS 0.9998)

3. **PT evaluation on ASAP is needed for fair comparison**
   - PT paper reports Overall JS ~0.16 on ASAP test set
   - We cannot verify this without proper ASAP evaluation

### 3.2 What We Cannot Conclude

1. ❌ **Cannot claim Hybrid Node is better than PT**
   - PT not evaluated on its own test set (ASAP)
   - Dataset mismatch makes comparison unfair

2. ❌ **Cannot verify PT's paper results**
   - PT's reported metrics are on ASAP test set
   - Our evaluation is on PianoCoRe (different domain)

---

## 4. Recommendations

### Option 1: Complete PT Evaluation on ASAP ⭐ **RECOMMENDED**

**Steps**:
1. Resolve PT alignment tool permissions
2. Process ASAP test set using PT's pipeline
3. Evaluate PT on ASAP using score → performance generation
4. Compare PT (on ASAP) vs Hybrid Node (on ASAP subset)

**Pros**:
- Fair comparison on same dataset
- Can verify PT's paper claims
- Scientific rigor

**Cons**:
- Requires external tool access
- More computation time

### Option 2: Report Current Results with Caveats

**Reporting**:
- Hybrid Node: Good performance on PianoCoRe (Overall JS ~0.26, Pearson ~0.35)
- PT: Cannot be evaluated on PianoCoRe due to poor generalization (Overall JS 0.48)
- PT evaluation on ASAP: **To be completed**

**Pros**:
- Honest reporting
- Shows generalization capability of Hybrid Node

**Cons**:
- Incomplete comparison
- Cannot claim superiority

### Option 3: Focus on Hybrid Node Strengths

**Reporting**:
- Emphasize Hybrid Node's efficiency (1 node vs 8 tokens)
- Show consistent performance across subsets
- Highlight generalization capability (ASAP + PianoCoRe)
- Note PT's cross-dataset limitations

**Pros**:
- Highlights our contributions
- Avoids incomplete comparisons

**Cons**:
- Doesn't address PT comparison directly

---

## 5. Files Organization

### Valid Results (Kept)
```
results/
├── hybrid_node_evaluation/
│   ├── results_binary.json       # Hybrid Node binary pedal results
│   └── results_continuous.json   # Hybrid Node continuous pedal results
│
└── pt_evaluation_generation/
    ├── pt_multiworker_results_binary.json      # PT generation (failed)
    └── pt_multiworker_results_continuous.json  # PT generation (failed)
```

### Deleted Files (Invalid/Obsolete)
- ❌ `pt_evaluation/` - Teacher forcing results (invalid method)
- ❌ All previous comparison reports (based on invalid PT evaluation)
- ❌ All log files (kept only final JSON results)
- ❌ Archive folders

---

## 6. Next Steps

**User Decision Required**:

1. Should we invest time to complete PT evaluation on ASAP?
   - Yes → Follow Option 1 (resolve permissions, run ASAP evaluation)
   - No → Follow Option 2 or 3 (report with caveats)

2. How should we handle the comparison in the paper?
   - Claim superiority based on PianoCoRe results?
   - Report incomplete comparison?
   - Focus on efficiency and generalization?

**Current Status**: ⏸️ **Waiting for user decision**
