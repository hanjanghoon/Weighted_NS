#!/bin/bash


export CUDA_VISIBLE_DEVICES=4
version="paper_base"
dataroot1="data_detection"
dataroot2="data_detection"
ver="gn_paper_base"


# Response generation
python3 baseline.py \
    --params_file baseline/configs/generation/params.json \
    --dataroot ${dataroot1} \
    --exp_name rg-hml128-kml128-${version} &&

python3 baseline.py --generate runs/rg-hml128-kml128-${version} \
   --generation_params_file baseline/configs/generation/generation_params.json \
   --eval_dataset val \
   --dataroot ${dataroot2} \
   --output_file dstc10_code/${ver}.json &&

python3 dstc10_code/scripts/scores_gn.py --dataset val --dataroot ${dataroot2} --outfile dstc10_code/${ver}.json \
   --scorefile dstc10_code/${ver}.score.json &&

cat dstc10_code/${ver}.score.json

