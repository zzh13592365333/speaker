#!/usr/bin/env bash
set -euo pipefail
python train.py \
  --config configs/iemocap.local.json \
  --dataset iemocap \
  --modalities text,audio,video \
  --gpu 0 \
  --epochs 30 \
  --lr 4e-5 \
  --dropout 0.45 \
  --aux_loss_weight 0.20 \
  --context_mode past
