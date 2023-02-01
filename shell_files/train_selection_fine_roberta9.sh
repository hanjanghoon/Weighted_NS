#!/bin/bash

export CUDA_VISIBLE_DEVICES=1
finedata="dstc9_data"
post_checkpoint="roberta-base"
fine_checkpoint="dstc9/rm_rr_512_c1"


nohup python3 -u baseline.py \
    --negative_sample_method "ranking" \
    --premodel "roberta" \
    --multi_task \
    --eval_all_snippets \
    --learning_rate 1e-5 \
    --ranking_step 2400 \
    --model_name_or_path ${post_checkpoint} \
    --params_file baseline/configs/selection/roberta_params_fine.json \
    --dataroot ${finedata} \
    --exp_name ${fine_checkpoint} > logs_dstc9/rm_rr_512_c1.txt &



