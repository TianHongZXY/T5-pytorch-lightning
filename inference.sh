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

export CUDA_VISIBLE_DEVICES=6
options=" \
        --seed 20020206 \
        --max_length 128 \
        --model_type mt5 \
        --model_name /cognitive_comp/zhuxinyu/pretrained_models/google/mt5-base \
        --ckpt_path /cognitive_comp/zhuxinyu/codes/t5_mrc_zxy/outputs/mt5-base-celebrity-cmrc-separated-04-06_23-26-lm-objective/last-epoch=199-avg_train_loss=0.0164.ckpt \
        --data_path /cognitive_comp/zhuxinyu/datasets/xiaobing/Celebrity_QA_100_ques.txt \
        --top_k 100 \
        --top_p 0.9 \
        --do_sample
        "
        # --ckpt_path /cognitive_comp/zhuxinyu/codes/t5_mrc_zxy/outputs/mengzi-t5-base-celebrity-cmrc-separated-03-26_16-01/epoch=194-avg_train_loss=0.0439.ckpt \
        # --num_beams 1 \

run_cmd="python -u t5_inference.py $@ ${options}"

sh -c "${run_cmd}"
       
