# Config Directory Guide

Current mainline INR/EPR configs live in `configs/inr_epr/`, copied from the
submission package. The canonical baseline is:

```text
configs/inr_epr/cinr__default_dlm_k1_bounded5.json
```

The August 7 baseline with ASAP pedal-group data and pedal loss masking is:

```text
configs/inr_epr/baseline0807_pedal_group_mask.json
```

Pianist Transformer and original training configs remain at the top level:

```text
pretrain_config*.json
sft_config*.json
pt_*.json
ds_config.json
```

Older exploratory INR, slot, FINE/PINE, and generated configs were moved to:

```text
backup/repo_cleanup_20260804/configs/
```
