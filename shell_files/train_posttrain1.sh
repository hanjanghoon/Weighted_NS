#!/bin/bash


export CUDA_VISIBLE_DEVICES=0
version="post-train1"
dataroot1="data_posttrain/aggregate/"
dataroot2="data_posttrain/aggregate/"
ver="pt_final"


# Response generation
python3 baseline.py \
    --params_file baseline/configs/post-training/params1.json \
    --dataroot ${dataroot1} \
    --exp_name rg-hml128-kml128-${version} &&

python3 baseline.py --generate runs/rg-hml128-kml128-${version} \
   --generation_params_file baseline/configs/generation/generation_params.json \
   --eval_dataset val \
   --dataroot ${dataroot2} \
   --output_file dstc10_code/pt_${ver}.json &&

python3 dstc10_code/scripts/scores_gn.py --dataset val --dataroot ${dataroot2} --outfile dstc10_code/pt_${ver}.json \
   --scorefile dstc10_code/pt_${ver}.score.json &&

cat dstc10_code/pt_${ver}.score.json

