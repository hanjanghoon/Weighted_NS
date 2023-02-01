#!/bin/bash


export CUDA_VISIBLE_DEVICES=1
version="baseline"
dataroot2="data_final_mix"
ver="gen_mix"

python3 baseline.py --generate runs/rg-hml128-kml128-${dataroot2}/ \
   --generation_params_file baseline/configs/generation/generation_params.json \
   --eval_dataset val \
   --dataroot ${dataroot2} \
   --output_file dstc10_code/gn_${ver}.json &&

python3 dstc10_code/scripts/scores_gn.py --dataset val --dataroot ${dataroot2} --outfile dstc10_code/gn_${ver}.json \
   --scorefile dstc10_code/gn_${ver}.score.json &&

cat dstc10_code/gn_${ver}.score.json

