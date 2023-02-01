#!/bin/bash


export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
version="post_train_paper"
dataroot2="dstc10_code/data"
ver="paper_val_gen"


python3 baseline.py --generate runs/rg-hml128-kml128-${version} \
   --generation_params_file baseline/configs/generation/generation_params.json \
   --eval_dataset val \
   --dataroot ${dataroot2} \
   --output_file dstc10_code/${ver}.json &&

python3 dstc10_code/scripts/scores_gn.py --dataset val --dataroot ${dataroot2} --outfile dstc10_code/${ver}.json \
   --scorefile dstc10_code/${ver}.score.json &&

cat dstc10_code/${ver}.score.json

