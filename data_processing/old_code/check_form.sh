#!/bin/bash

dataroot2="dstc10_test"
export CUDA_VISIBLE_DEVICES=5


python3 dstc10_code/scripts/check_results.py --dataset test --dataroot "dstc10_test" --outfile pred/test/post/single1.json


