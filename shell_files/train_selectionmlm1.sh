#!/bin/bash


export CUDA_VISIBLE_DEVICES=10
version="mlm"
dataroot1="data_selection"
dataroot2="data_selection"
ver="mlm"

python3 baseline.py \
    --negative_sample_method "domain" \
    --params_file baseline/configs/selection/roberta_params.json \
    --dataroot ${dataroot1} \
    --exp_name ks-all-${version} > logs/sl_only${ver}.txt &&

python3 baseline.py --eval_only --checkpoint runs/ks-all-${version}/ \
   --eval_all_snippets \
   --dataroot ${dataroot2} \
   --eval_dataset val \
   --output_file pred/val/baseline.ks.json

python3 dstc10_code/scripts/scores_sl.py --dataset val --dataroot dstc10_code/data/ --outfile pred/val/baseline.ks.json \
   --scorefile dstc10_code/sl${ver}.score.json &&

cat dstc10_code/sl${ver}.score.json

