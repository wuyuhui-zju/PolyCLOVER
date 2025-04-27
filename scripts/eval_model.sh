#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python eval_model.py \
  --model_path ../models/pretrained/finetuned/base.pth \
  --dataset eval_antibacterial \
  --weight_decay 1e-6 \
  --dropout 0 \
  --lr 5e-4 \
  --seed 44 \
