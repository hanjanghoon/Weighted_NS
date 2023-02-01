#!/bin/bash


export CUDA_VISIBLE_DEVICES=8,9,10,11,12,13,14,15
version="data_final_asr_same"
dataroot="data/data_final_asr"

num_gpus=8
ver="paper_ks_neg_same"
label_file="pred/val/ktd-data_final.json"


python3 -u -m torch.distributed.launch --nproc_per_node ${num_gpus} baseline.py \
    --negative_sample_method "weighted" \
    --params_file baseline/configs/selection/roberta_params.json \
    --dataroot ${dataroot} \
    --exp_name ks-all-${version} &&

python3 baseline.py --eval_only --checkpoint runs/ks-all-${version}/ \
   --eval_all_snippets \
   --dataroot ${dataroot} \
   --eval_dataset val \
   --labels_file ${label_file} \
   --output_file pred/val/${ver}.json &&

python3 dstc10_code/scripts/scores_sl.py --dataset val --dataroot ${dataroot} --outfile pred/val/${ver}.json \
   --scorefile dstc10_code/${ver}.score.json &&

cat dstc10_code/${ver}.score.json
