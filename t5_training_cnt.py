import pandas as pd
import numpy as np
import json
import os
import random
import torch
from t5_modeling import Seq2SeqTrainer, BaseSeq2SeqModel
from data_preprocess import (perform_span_corruption_seg,
                             perform_span_corruption_nonseg,
                             get_training_data,
                             get_training_data_lightning
                             )
import pytorch_lightning as pl
#os.environ["CUDA_VISIBLE_DEVICES"] = "6"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def train():
    torch.cuda.empty_cache()
    pl.seed_everything(42)

    # hyper parameters
    corruption_rate = 0.15
    source_max_token_len = 1024
    target_max_token_len = 512 # int(source_max_token_len * corruption_rate * 1.5)
    batch_size = 6
    precision = 16
    gpu_nums = 2
    outputdir = "/cognitive_comp/zhuxinyu/codes/t5_mrc_zxy/outputs"
    model_prefix = "mengzi-t5-base-celebrity-cmrc-separated"
    outputdir = os.path.join(outputdir, model_prefix)
    show_training_ex = 100
    max_epoch = 2 # 1000
    patience = 10

    # instantiate model
    # model_path = "google/mt5-large"
    model_path = "/cognitive_comp/zhuxinyu/pretrained_models/mengzi-t5-base"
    model_type = 't5'

    # get data
    #  data_dir = '/cognitive_comp/sunyuqing/datasets/cmrc2018-master/squad-style-data/'
    #  seg_data_dir = os.path.join(data_dir, 'cmrc2018_all_context_segs.json')
    #  nonseg_data_dir = os.path.join(data_dir, 'cmrc2018_all.json')
    data_dir = '/cognitive_comp/zhuxinyu/datasets/'
    data_path = os.path.join(data_dir, 'celebrity_rm_citation_index_for_lightning_cmrc_separated.txt')

    #  all_contexts_seg = get_training_data(True)
    #  all_contexts_nonseg = get_training_data(False)
    all_contexts_nonseg = get_training_data_lightning(data_path)

    # train on segmented corruption data
    #  seq2seq = BaseSeq2SeqModel(outputdir=outputdir)
    seq2seq = BaseSeq2SeqModel.from_pretrained(model_type=model_type, model_name=model_path)
    seq2seq_trainer = Seq2SeqTrainer(seq2seq, seq2seq.tokenizer)
    #  model.from_pretrained(model_type, model_path)
    seq2seq_trainer.tokenizer.save_pretrained(outputdir)
    seq2seq_trainer.train_span_corruption(
        train_context=all_contexts_nonseg,
        context_segmented=False,
        source_max_token_len=source_max_token_len, 
        target_max_token_len=target_max_token_len,
        batch_size=batch_size,
        max_epochs=max_epoch,
        gpu_nums=gpu_nums,
        precision=precision,
        early_stopping_patience_epochs=patience,  # 0 to disable early stopping feature
        outputdir=outputdir,
        show_training_ex=show_training_ex,
        corruption_rate=corruption_rate,
    )
    seq2seq_trainer.model.save_pretrained(outputdir)


if __name__ == "__main__":
    train()

