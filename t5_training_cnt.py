import pandas as pd
import os
import time
import json
import torch
import argparse
from t5_modeling import Seq2SeqTrainer, BaseSeq2SeqModel, LightningDataModule, T5LightningModel
from data_preprocess import get_training_data_lightning
import pytorch_lightning as pl
#os.environ["CUDA_VISIBLE_DEVICES"] = "6"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def train():
    parent_parser = argparse.ArgumentParser("T5")

    torch.cuda.empty_cache()

    # root output directory
    outputdir = "/cognitive_comp/zhuxinyu/codes/t5_mrc_zxy/outputs"
    # get data
    data_dir = '/cognitive_comp/zhuxinyu/datasets/'
    data_path = os.path.join(data_dir, 'celebrity_rm_citation_index_for_lightning_cmrc_separated.txt')
    all_contexts_nonseg = get_training_data_lightning(data_path)

    # hyper parameters
    parent_parser = T5LightningModel.add_model_specific_args(parent_parser)
    parent_parser = Seq2SeqTrainer.add_trainer_specific_args(parent_parser)
    parent_parser = LightningDataModule.add_data_specific_args(parent_parser)
    parent_parser = pl.Trainer.add_argparse_args(parent_parser=parent_parser)
    args = parent_parser.parse_args()

    # create checkpoint directory in root directory
    model_prefix = f"{os.path.split(args.model_name)[-1]}"
    data_prefix = "celebrity-cmrc-separated"
    timestamp = time.strftime("%m-%d_%H-%M", time.localtime(time.time()))
    outputdir = os.path.join(outputdir, model_prefix + '-' + data_prefix + '-'  + timestamp)
    args.outputdir = outputdir

    # train on segmented corruption data
    pl.seed_everything(args.seed)
    #  seq2seq = BaseSeq2SeqModel.from_pretrained(args=args)
    seq2seq = T5LightningModel.from_pretrained(args=args)
    seq2seq_trainer = Seq2SeqTrainer(seq2seq, seq2seq.tokenizer)
    seq2seq_trainer.tokenizer.save_pretrained(outputdir)

    # save and show args
    with open(os.path.join(outputdir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
    print('-' * 30 + 'Args' + '-' * 30)
    for k, v in vars(args).items():
        if v is not None:
            print("\t", k, ":", v)
    print('\n' + '-' * 64)

    # Training
    seq2seq_trainer.train_span_corruption(
        args=args,
        train_context=all_contexts_nonseg,
    )
    seq2seq_trainer.model.model.save_pretrained(outputdir)


if __name__ == "__main__":
    train()

