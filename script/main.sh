CUDA_VISIBLE_DEVICES=0,1 \
CONFIG=results/train_ddp/configs/kp1_pedal2_ddp_bs64.json \
  bash script/run_inr_scoreperf_mask_train_asap_pipeline.sh

CUDA_VISIBLE_DEVICES=2,3 \
CONFIG=results/train_ddp/configs/kp05_pedal2_ddp_bs64.json \
  bash script/run_inr_scoreperf_mask_train_asap_pipeline.sh