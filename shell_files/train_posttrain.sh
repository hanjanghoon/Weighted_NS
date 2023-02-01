#!/bin/bash


export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
version="post-train5"
dataroot1="data_posttrain/aggregate/"
dataroot2="data_posttrain/aggregate/"
num_gpus=8
ver="lr2"


# Response generation
python3 -u -m torch.distributed.launch --nproc_per_node ${num_gpus} baseline.py \
    --params_file baseline/configs/post-training/params.json \
    --dataroot ${dataroot1} \
    --exp_name rg-hml128-kml128-${version} &&

python3 baseline.py --generate runs/rg-hml128-kml128-${version} \
   --generation_params_file baseline/configs/generation/generation_params.json \
   --eval_dataset val \
   --dataroot ${dataroot2} \
   --output_file dstc10_code/pt_${ver}.json &&

python3 dstc10_code/scripts/scores_gn.py --dataset val --dataroot ${dataroot2} --outfile dstc10_code/pt_${ver}.json \
   --scorefile dstc10_code/pt${ver}.score.json &&

cat dstc10_code/pt${ver}.score.json

