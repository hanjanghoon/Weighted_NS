#!/bin/bash


export CUDA_VISIBLE_DEVICES=0
dataroot2="dstc10_test"
num_gpus=8

version="data_final"

python3 baseline.py --eval_only --checkpoint runs/ktd-${version}/ \
   --eval_dataset test \
   --dataroot ${dataroot2} \
   --no_labels \
   --output_file pred/test/ktd-${version}.json &&

version="data_final_asr"

python3 baseline.py --eval_only --checkpoint runs/ktd-${version}/ \
   --eval_dataset test \
   --dataroot ${dataroot2} \
   --no_labels \
   --output_file pred/test/ktd-${version}.json &&

version="data_final_mix"
python3 baseline.py --eval_only --checkpoint runs/ktd-${version}/ \
   --eval_dataset test \
   --dataroot ${dataroot2} \
   --no_labels \
   --output_file pred/test/ktd-${version}.json &&

version="data_final_asr_mix"
python3 baseline.py --eval_only --checkpoint runs/ktd-${version}/ \
   --eval_dataset test \
   --dataroot ${dataroot2} \
   --no_labels \
   --output_file pred/test/ktd-${version}.json


