#!/usr/bin/env bash
set -euo pipefail
python train.py \
  --config configs/meld.local.json \
  --dataset meld \
  --gpu 0 \
  --epochs 10 \
  --lr 8e-6 \
  --dropout 0.4 \
  --aux_loss_weight 0.15
