#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
finedata="data_train"
post_checkpoint="roberta-base"
fine_checkpoint="dstc10/rm_rr_1_64_temp"


nohup python3 -u baseline.py \
    --negative_sample_method "ranking" \
    --premodel "roberta" \
    --multi_task \
    --eval_all_snippets \
    --learning_rate 1e-5 \
    --model_name_or_path ${post_checkpoint} \
    --params_file baseline/configs/selection/roberta_params_fine.json \
    --dataroot ${finedata} \
    --exp_name ${fine_checkpoint} > logs_dstc10/rm_rr_1_64_temp.txt &



