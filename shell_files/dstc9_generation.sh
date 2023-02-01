#!/bin/bash

export CUDA_VISIBLE_DEVICES=8,9,10,11,12,13,14,15

version="dstc9"
dataroot="data"
num_gpus=8
ver="dstc9"

nohup python3 -u -m torch.distributed.launch --nproc_per_node ${num_gpus} baseline.py \
    --params_file baseline/configs/generation/params9.json \
    --dataroot ${dataroot} \
    --exp_name rg-hml128-kml128-${version} > logs/generation${ver}.txt &