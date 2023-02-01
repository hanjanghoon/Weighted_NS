#!/bin/bash


export CUDA_VISIBLE_DEVICES=1
version="baseline"
dataroot2="data_final"
ver="gen_val"

python3 baseline.py --generate runs/rg-hml128-kml128-baseline \
   --generation_params_file baseline/configs/generation/generation_params.json \
   --eval_dataset val \
   --dataroot ${dataroot2} \
   --labels_file pred/val/baseline.ks.json \
   --output_file dstc10_code/gn_${ver}.json &&

python3 dstc10_code/scripts/scores.py --dataset val --dataroot dstc10_code/data/ --outfile dstc10_code/gn_${ver}.json \
   --scorefile dstc10_code/gn_${ver}.score.json &&

cat dstc10_code/gn_${ver}.score.json

