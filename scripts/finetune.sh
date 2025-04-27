#!/bin/bash

for seed in $(seq 0 19)
do
  python finetune.py \
    --model_path ../models/pretrained/finetuned/base.pth \
    --dataset initial \
    --weight_decay 1e-6 \
    --dropout 0 \
    --lr 5e-4 \
    --save \
    --ensemble_idx $seed \
    --seed $seed

done
