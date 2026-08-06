#!/bin/bash

# ==== USER CONFIGURATION ====
OUTPUT_DIR="/data/tientoan/iclr_output/output_libero_object" # your checkpoint folder
LOG_NAME="iclr_libero_object_0" # your log name
DATASET_JSON="./config/dataset_config_libero_object_visual_trace.json"
MASTER_PORT=2450
GPUS="2,6"

# ==== TRAINING PARAMETERS ====
CMD_BASE="torchrun --nproc_per_node=2 --master_port=$MASTER_PORT scripts_libero/train_libero_visual_trace.py \
  --dataset-cfg.dataset-json $DATASET_JSON \
  --logging-cfg.output-dir $OUTPUT_DIR \
  --logging-cfg.log-name $LOG_NAME \
  --optimizer-cfg.warmup-epochs 1.25 \
  --trainer-cfg.epochs 125 \
  --model-cfg.vision-encoder-cfg.vision-encoder ./vision_encoder/cross-mae-rtx-vitb.pth \
  --model-cfg.policy-cfg.scratch-llama-config config/model_config/custom_transformer.json \
  --model-cfg.policy-cfg.no-prompt-loss \
  --dataset-cfg.non-overlapping 32 \
  --trainer-cfg.accum-iter 16 \
  --shared-cfg.batch-size 2 \
  --shared-cfg.save-every 1"

echo ">>> Starting training..."

CUDA_VISIBLE_DEVICES=$GPUS $CMD_BASE

echo ">>> Training finished."
