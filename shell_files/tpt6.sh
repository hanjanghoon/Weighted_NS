#!/bin/bash


export CUDA_VISIBLE_DEVICES=2
version="baseline"
dataroot2="data_posttrain/aggregate/"
num_gpus=1
ver="lr3_1"

python3 baseline.py --generate runs/rg-hml128-kml128-post-train3/checkpoint-5720  \
   --generation_params_file baseline/configs/generation/generation_params.json \
   --eval_dataset val \
   --dataroot ${dataroot2} \
   --output_file dstc10_code/pt_${ver}.json &&

python3 dstc10_code/scripts/scores_gn.py --dataset val --dataroot ${dataroot2} --outfile dstc10_code/pt_${ver}.json \
   --scorefile dstc10_code/pt_${ver}.score.json &&


cat dstc10_code/pt_${ver}.score.json

