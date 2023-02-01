#!/bin/bash


export CUDA_VISIBLE_DEVICES=8,9,10,11,12,13,14,15
version="data_final_mix"
dataroot1="data_final_mix"
dataroot2="data_final_mix"
num_gpus=8
ver="data_final_mix"

python3 -u -m torch.distributed.launch --nproc_per_node ${num_gpus} baseline.py \
    --negative_sample_method "domain" \
    --params_file baseline/configs/selection/roberta_params.json \
    --dataroot ${dataroot1} \
    --exp_name ks-all-${version} > logs/sl_only${ver}.txt &&

python3 baseline.py --eval_only --checkpoint runs/ks-all-${version}/ \
   --eval_all_snippets \
   --dataroot ${dataroot2} \
   --eval_dataset val \
   --output_file pred/val/ks-${version}.json

python3 dstc10_code/scripts/scores_sl.py --dataset val --dataroot ${dataroot2} --outfile pred/val/ks-${version}.json \
   --scorefile dstc10_code/sl${ver}.score.json &&

cat dstc10_code/sl${ver}.score.json
