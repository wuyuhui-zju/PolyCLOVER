#!/bin/bash

python mobo.py \
  --config al \
  --model_path ../models/ensemble_models_initial \
  --dataset initial \
  --data_path ../datasets \
  --device cuda:0
