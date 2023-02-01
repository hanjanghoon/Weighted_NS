#!/bin/bash


export CUDA_VISIBLE_DEVICES=1
version="baseline"
dataroot2="dstc10_code/data"
num_gpus=1
ver="test_s"

python3 baseline.py --eval_only --checkpoint runs/ks-all-baseline/checkpoint-975 \
   --eval_all_snippets \
   --dataroot ${dataroot2} \
   --eval_dataset val \
   #--labels_file pred/val/baseline.ktd.json \
   --output_file pred/val/baseline.ks.json

python3 dstc10_code/scripts/scores_sl.py --dataset val --dataroot dstc10_code/data/ --outfile pred/val/baseline.ks.json \
   --scorefile dstc10_code/sl${ver}.score.json &&

cat dstc10_code/sl${ver}.score.json

