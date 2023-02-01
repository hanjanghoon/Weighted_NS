#!/bin/bash

dataroot2="dstc10_test"
export CUDA_VISIBLE_DEVICES=1

version="data_final_mix2"
source="ks-fam_single"



python3 baseline.py --generate runs/rg-hml128-kml128-${version} \
   --generation_params_file baseline/configs/generation/generation_params.json \
   --eval_dataset test \
   --dataroot ${dataroot2} \
   --labels_file pred/test/${source}.json \
   --output_file pred/test/final_${source}-${version}.json &&

python3 dstc10_code/scripts/check_results.py --dataset test --dataroot ${dataroot2} --outfile pred/test/final_${source}-${version}.json


