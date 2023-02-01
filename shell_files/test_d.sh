#!/bin/bash


export CUDA_VISIBLE_DEVICES=8,9,10,11,12,13,14,15
version="baseline"
dataroot2="data_final_correction"
num_gpus=4
ver="test"

python3 baseline.py --eval_only --checkpoint runs/ktd-baseline/ \
   --eval_dataset val \
   --dataroot ${dataroot2} \
   --no_labels \
   --output_file pred/val/baseline.ktd.json &&

python3 dstc10_code/scripts/scores_dt.py --dataset val --dataroot dstc10_code/data/ --outfile pred/val/baseline.ktd.json \
   --scorefile dstc10_code/dt_${ver}.score.json &&

cat dstc10_code/dt_${ver}.score.json

