#!/bin/bash


export CUDA_VISIBLE_DEVICES=0
dataroot2="data_final/"
num_gpus=8

version="baseline"

python3 baseline.py --eval_only --checkpoint runs/ktd-${version}/ \
   --eval_dataset val \
   --dataroot ${dataroot2} \
   --no_labels \
   --output_file temp.json &&

python3 dstc10_code/scripts/scores_dt.py --dataset val --dataroot ${dataroot2} --outfile temp.json \
   --scorefile temp.score.json &&

cat temp.score.json

