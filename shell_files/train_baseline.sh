#!/bin/bash

# This script demonstrates how to train baseline models with this repo
# We train models for three subtasks separately
# 1. knowledge-seeking turn detection
# 2. knowledge selection
# 3. response generation
# And we show how to generate responses for test dataset without labels.json at the end

# set path to dataset here
version="baseline"
dataroot="1_5_10_s713_m"
num_gpus=8
ver="1_5_10_s713_m"

# Knowledge-seeking turn detection
# distributed training, single-process multi-gpu training also supported
# use --params_file to specify the file containing training parameters
# use --exp_name to specify the name of this run, the checkpoints and logs will be stored in runs/{exp_name}
# use --eval_desc to specify the description of evaluation, which will be written in eval_results.txt

nohup python3 -u -m torch.distributed.launch --nproc_per_node ${num_gpus} baseline.py \
        --params_file baseline/configs/detection/params.json \
        --dataroot ${dataroot} \
        --exp_name ktd-${version} > logs/detection${ver}.txt &&

# Knowledge selection
# use --negative_sample_method to modify the setting in params.json for this training run
# the updated parameters will be saved to {checkpoint}/params.json
# note that the default negative_sample_method for testing is "oracle"
# which filters the candidates based on the ground truth entity, so the number of candidates
# is way less than the total number of snippets
nohup python3 -u -m torch.distributed.launch --nproc_per_node ${num_gpus} baseline.py \
    --negative_sample_method "all" \
    --params_file baseline/configs/selection/params.json \
    --dataroot ${dataroot} \
    --exp_name ks-all-${version} > logs/selection${ver}.txt &&

# Response generation
nohup python3 -u -m torch.distributed.launch --nproc_per_node ${num_gpus} baseline.py \
    --params_file baseline/configs/generation/params.json \
    --dataroot ${dataroot} \
    --exp_name rg-hml128-kml128-${version} > logs/generation${ver}.txt &