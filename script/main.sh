CUDA_VISIBLE_DEVICES=0,1 \
CONFIG=configs/kp1_pedal2_ddp_bs64.json \
  bash script/run_inr_pipeline.sh

CUDA_VISIBLE_DEVICES=2,3 \
CONFIG=configs/kp05_pedal2_ddp_bs64.json \
  bash script/run_inr_pipeline.sh