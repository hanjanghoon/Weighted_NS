#!/bin/bash


export CUDA_VISIBLE_DEVICES=0
version="baseline"
dataroot2="data_posttrain/aggregate/"
num_gpus=1
ver="post_1"

python3 baseline.py --generate runs/rg-hml128-kml128-post-train1/checkpoint-5783  \
   --generation_params_file baseline/configs/generation/generation_params.json \
   --eval_dataset val \
   --dataroot ${dataroot2} \
   --output_file dstc10_code/pt_${ver}.json &&

python3 dstc10_code/scripts/scores_gn.py --dataset val --dataroot ${dataroot2} --outfile dstc10_code/pt_${ver}.json \
   --scorefile dstc10_code/pt_${ver}.score.json &&


cat dstc10_code/pt_${ver}.score.json

