#!/bin/bash


export CUDA_VISIBLE_DEVICES=10
version="ks-all-data_final_asr"
dataroot2="dstc10_code/data"
ver="paper_ktd-final_ks-final_asr"

label_file="pred/val/ktd-data_final.json"

python3 baseline.py --eval_only --checkpoint runs/${version}/ \
   --eval_all_snippets \
   --dataroot ${dataroot2} \
   --eval_dataset val \
   --labels_file ${label_file} \
   --output_file pred/val/${ver}.json

python3 scripts/scores_sl.py --dataset val --dataroot dstc10_code/data/ --outfile pred/val/${ver}.json \
   --scorefile dstc10_code/${ver}.score.json &&

cat ${ver}.score.json

