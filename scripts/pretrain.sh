#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2 python -u -m torch.distributed.run --nproc_per_node=3 --nnodes=1 pretrain.py \
  --train_mode scratch \
  --save_path ../models/pretrained/finetuned/ \
  --n_threads 8 \
  --n_devices 3 \
  --config base \
  --n_steps 10000 \
  --data_path ../datasets/
