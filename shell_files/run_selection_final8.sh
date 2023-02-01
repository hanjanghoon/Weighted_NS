#!/bin/bash


export CUDA_VISIBLE_DEVICES=1
version="data_final"
dataroot2="dstc10_test"
ktd_source="ktd-data_final"

nohup python3 baseline.py --eval_only --checkpoint runs/ks-all-${version}/ \
   --eval_all_snippets \
   --dataroot ${dataroot2} \
   --eval_dataset test \
   --labels_file pred/test/${ktd_source}.json \
   --output_file pred/test/ks-${version}2_single.json > s8.txt &
