#!/bin/bash
# ====================================================
#   Copyright (C) 2022  All rights reserved.
#
#   Author        : Xinyu Zhu
#   Email         : zhuxy21@mails.tsinghua.edu.cn
#   File Name     : inference.sh
#   Last Modified : 2022-03-20 00:05
#   Describe      : 
#
# ====================================================

export CUDA_VISIBLE_DEVICES=3
options=" \
        --seed 20020206 \
        --max_length 512 \
        --num_beams 1 \
        --top_k 100 \
        --top_p 0.9 \
        --model_type t5 \
        --ckpt_path /cognitive_comp/zhuxinyu/codes/t5_mrc_zxy/outputs/mengzi-t5-base-celebrity-cmrc-separated/epoch=482-avg_train_loss=0.0083.ckpt \
        --data_path /cognitive_comp/zhuxinyu/datasets/xiaobing/Celebrity_QA_100_ques.txt
        "

run_cmd="python -u t5_inference.py $@ ${options}"

sh -c "${run_cmd}"
       
