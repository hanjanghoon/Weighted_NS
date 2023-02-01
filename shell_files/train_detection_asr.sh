#!/bin/bash


export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
version="data_final_asr"
dataroot1="data_final_asr"
dataroot2="data_final_asr"
num_gpus=8
ver="data_final_asr"


python3 -u -m torch.distributed.launch --nproc_per_node ${num_gpus} baseline.py \
        --params_file baseline/configs/detection/params.json \
        --dataroot ${dataroot1} \
        --exp_name ktd-${version} > logs/dt_only${ver}.txt &&

python3 baseline.py --eval_only --checkpoint runs/ktd-${version}/ \
   --eval_dataset val \
   --dataroot ${dataroot2} \
   --no_labels \
   --output_file pred/val/ktd-${version}.json &&

python3 dstc10_code/scripts/scores_dt.py --dataset val --dataroot ${dataroot2} --outfile pred/val/ktd-${version}.json \
   --scorefile dstc10_code/dt${ver}.score.json &&

cat dstc10_code/dt${ver}.score.json

