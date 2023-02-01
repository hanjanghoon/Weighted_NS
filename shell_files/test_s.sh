#!/bin/bash


export CUDA_VISIBLE_DEVICES=0
version="mlm"
dataroot1="data_selection"
dataroot2="data_selection"
num_gpus=8
ver="mlm"

python3 baseline.py --eval_only --checkpoint runs/ks-all-${version}/ \
   --eval_all_snippets \
   --dataroot ${dataroot2} \
   --eval_dataset val \
   --output_file pred/val/baseline.ks.json

python3 dstc10_code/scripts/scores_sl.py --dataset val --dataroot ${dataroot2} --outfile pred/val/baseline.ks.json \
   --scorefile dstc10_code/sl${ver}.score.json &&

cat dstc10_code/sl${ver}.score.json
