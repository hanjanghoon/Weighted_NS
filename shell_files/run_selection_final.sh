#!/bin/bash


export CUDA_VISIBLE_DEVICES=0
version="data_final"
dataroot2="dstc10_test"
ktd_source="assemble"
ver="data_final"

python3 baseline.py --eval_only --checkpoint runs/ks-all-${version}/checkpoint-1266 \
   --eval_all_snippets \
   --dataroot ${dataroot2} \
   --eval_dataset test \
   --labels_file pred/test/${ktd_source}.json \
   --output_file pred/test/ks-${version}.json
