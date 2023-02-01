#!/bin/bash


export CUDA_VISIBLE_DEVICES=4,5,6,7
version="baseline"
dataroot1="gen_data"
dataroot2="dstc10_code/data"
num_gpus=4
ver="v4"


# Response generation
python3 -u -m torch.distributed.launch --nproc_per_node ${num_gpus} baseline.py \
    --params_file baseline/configs/generation/params.json \
    --dataroot ${dataroot1} \
    --exp_name rg-hml128-kml128-${version} &&

python3 baseline.py --generate runs/rg-hml128-kml128-baseline \
   --generation_params_file baseline/configs/generation/generation_params.json \
   --eval_dataset val \
   --dataroot ${dataroot2} \
   --labels_file pred/val/baseline.ks.json \
   --output_file dstc10_code/gn_${ver}.json &&

python3 dstc10_code/scripts/scores.py --dataset val --dataroot dstc10_code/data/ --outfile dstc10_code/gn_${ver}.json \
   --scorefile dstc10_code/gn${ver}.score.json &&

cat dstc10_code/gn${ver}.score.json

