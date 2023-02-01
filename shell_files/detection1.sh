#!/bin/bash


export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
version="baseline"
dataroot1="final"
dataroot2="dstc10_code/data"
num_gpus=8
ver="920"


python3 -u -m torch.distributed.launch --nproc_per_node ${num_gpus} baseline.py \
        --params_file baseline/configs/detection/params.json \
        --dataroot ${dataroot1} \
        --exp_name ktd-${version} > logs/dt_only${ver}.txt &&

python3 baseline.py --eval_only --checkpoint runs/ktd-baseline/ \
   --eval_dataset val \
   --dataroot ${dataroot2} \
   --no_labels \
   --output_file pred/val/baseline.ktd.json &&

python3 dstc10_code/scripts/scores_dt.py --dataset val --dataroot dstc10_code/data/ --outfile pred/val/baseline.ktd.json \
   --scorefile dstc10_code/dt${ver}.score.json &&

cat dstc10_code/dt${ver}.score.json

