#!/bin/bash


export CUDA_VISIBLE_DEVICES=0
version="baseline"
dataroot1="data_final_mix"
dataroot2="data_final_mix"
ver="temp"


# Response generation
python3 baseline.py \
    --params_file baseline/configs/generation/params_fine.json \
    --dataroot ${dataroot1} \
    --exp_name rg-hml128-kml128-${version} &&

python3 baseline.py --generate runs/rg-hml128-kml128-baseline \
   --generation_params_file baseline/configs/generation/generation_params.json \
   --eval_dataset val \
   --dataroot ${dataroot2} \
   --output_file dstc10_code/gn_${ver}.json &&

python3 dstc10_code/scripts/scores.py --dataset val --dataroot ${dataroot2} --outfile dstc10_code/gn_${ver}.json \
   --scorefile dstc10_code/gn${ver}.score.json &&

cat dstc10_code/gn${ver}.score.json

