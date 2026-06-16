# INR Training Matrix

This matrix defines the planned INSPIRE / INR backbone comparison.

## Scope

The experiment grid is:

```text
5 backbones x 2 tasks = 10 training runs
```

Backbones:

- `t5_10_2`: T5 encoder-decoder, 10 encoder layers + 2 decoder layers
- `t5_2_10`: T5 encoder-decoder, 2 encoder layers + 10 decoder layers
- `t5_6_6`: T5 encoder-decoder, 6 encoder layers + 6 decoder layers
- `bert_17`: encoder-only, 17 layers
- `gpt_17`: decoder-only, 17 layers

Tasks:

- `EPR`: score to performance
- `CSR`: performance to canonical score

Shared scale:

```json
{
  "hidden_size": 1024,
  "intermediate_size": 4096,
  "block_notes": 512
}
```

## Planned Configs

| Task | Backbone | Config |
|------|----------|--------|
| `EPR` | `t5_10_2` | `configs/inr_experiments/inr_epr_t5_10_2_h1024_b512.json` |
| `EPR` | `t5_2_10` | `configs/inr_experiments/inr_epr_t5_2_10_h1024_b512.json` |
| `EPR` | `t5_6_6` | `configs/inr_experiments/inr_epr_t5_6_6_h1024_b512.json` |
| `EPR` | `bert_17` | `configs/inr_experiments/inr_epr_bert_17_h1024_b512.json` |
| `EPR` | `gpt_17` | `configs/inr_experiments/inr_epr_gpt_17_h1024_b512.json` |
| `CSR` | `t5_10_2` | `configs/inr_experiments/inr_csr_t5_10_2_h1024_b512.json` |
| `CSR` | `t5_2_10` | `configs/inr_experiments/inr_csr_t5_2_10_h1024_b512.json` |
| `CSR` | `t5_6_6` | `configs/inr_experiments/inr_csr_t5_6_6_h1024_b512.json` |
| `CSR` | `bert_17` | `configs/inr_experiments/inr_csr_bert_17_h1024_b512.json` |
| `CSR` | `gpt_17` | `configs/inr_experiments/inr_csr_gpt_17_h1024_b512.json` |

## Loss Plan

EPR:

```python
loss_epr = loss_ioi + loss_dur + loss_vel + 0.75 * loss_pedal
```

- `ioi`, `dur`: Laplace NLL
- `velocity`, `pedal`: continuous Huber regression
- `pedal`: linear output during training, clamp only at inference/export

CSR:

```python
loss_csr = (
    loss_mo
  + loss_md
  + loss_first
  + loss_ml
  + loss_staff
  + loss_trill
  + loss_grace
  + loss_staccato
)
```

- `mo`, `md`, `ml`: ordinal classification on the `1/24` score grid
- `first`, `staff`, `trill`, `grace`, `staccato`: BCE with logits
- `ml` uses `ml_mask = attention_mask * has_score_feature * first_target`
- `ml` coefficient stays `1.0` because `ml_mask` is sparse
- `staff` coefficient is `0.5`

## Implementation Status

These configs describe the intended 10-run matrix. The current `src/train/train_inr.py`
implementation still mainly supports the older EPR continuous-regression baseline.
Before launching this full matrix, the training code should be extended for:

- EPR timing Laplace NLL heads
- CSR input/target construction
- CSR ordinal heads for `mo/md/ml`
- CSR BCE heads for score annotation
