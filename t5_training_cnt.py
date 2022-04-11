import os
import argparse
import time
import json
import torch
from t5_modeling import Seq2SeqTrainer, BaseSeq2SeqModel, T5LightningModel
from data_model import LightningDataModel, T5Dataset
from data_preprocess import get_data_lightning
import pytorch_lightning as pl
#os.environ["CUDA_VISIBLE_DEVICES"] = "6"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def train():
    parent_parser = argparse.ArgumentParser("T5")
    torch.cuda.empty_cache()

    # hyper parameters
    parent_parser = T5LightningModel.add_model_specific_args(parent_parser)
    parent_parser = Seq2SeqTrainer.add_trainer_specific_args(parent_parser)
    parent_parser = LightningDataModel.add_data_specific_args(parent_parser)
    args = parent_parser.parse_args()

    # root save directory
    save_dir  = args.save_dir

    # data directory
    data_dir = args.data_dir
    train_data_path = os.path.join(data_dir, args.train_data)
    train_context = get_data_lightning(train_data_path)
    if args.valid_data:
        valid_data_path = os.path.join(data_dir, args.valid_data)
        valid_context = get_data_lightning(valid_data_path)
    if args.test_data:
        test_data_path = os.path.join(data_dir, args.test_data)
        test_context = get_data_lightning(test_data_path)


    # create checkpoint directory in root save directory and replace save_dir with it
    model_prefix = f"{os.path.split(args.model_name)[-1]}"
    data_prefix = "celebrity-cmrc-separated"
    timestamp = time.strftime("%m-%d_%H-%M", time.localtime(time.time()))
    save_dir = os.path.join(save_dir, model_prefix + '-' + data_prefix + '-'  + timestamp + '-' + args.training_objective)
    args.save_dir = save_dir

    # create model and trainer
    pl.seed_everything(args.seed)
    seq2seq = T5LightningModel.from_pretrained(args=args)
    tokenizer = seq2seq.tokenizer
    tokenizer.save_pretrained(save_dir)
    seq2seq_trainer = Seq2SeqTrainer(args, seq2seq)

    # save and show args
    with open(os.path.join(save_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
    print('-' * 30 + 'Args' + '-' * 30)
    for k, v in vars(args).items():
        if v is not None:
            print("\t", k, ":", v)
    print('\n' + '-' * 64)


    # Create T5Dataset and data model
    train_dataset = T5Dataset(
        raw_text=train_context,
        tokenizer=tokenizer,
        source_max_token_len=args.source_max_token_len,
        target_max_token_len=args.target_max_token_len,
        corruption_rate=args.corruption_rate,
        max_extra_id=args.max_extra_id,
        training_objective=args.training_objective,
    )
    if args.valid_data:
        valid_dataset = T5Dataset(
            raw_text=valid_context,
            tokenizer=tokenizer,
            source_max_token_len=args.source_max_token_len,
            target_max_token_len=args.target_max_token_len,
            corruption_rate=args.corruption_rate,
            max_extra_id=args.max_extra_id,
            training_objective=args.training_objective,
        )
    if args.test_data:
        test_dataset = T5Dataset(
            raw_text=test_context,
            tokenizer=tokenizer,
            source_max_token_len=args.source_max_token_len,
            target_max_token_len=args.target_max_token_len,
            corruption_rate=args.corruption_rate,
            max_extra_id=args.max_extra_id,
            training_objective=args.training_objective,
        )

    data_model = LightningDataModel(
            train_dataset=train_dataset,
            valid_dataset=valid_dataset if args.valid_data else None,
            test_dataset=test_dataset if args.test_data else None,
            micro_batch_size=args.micro_batch_size,
            valid_batch_size=args.valid_batch_size,
            num_workers=args.num_workers,
            )

    # Training
    seq2seq_trainer.train(data_model=data_model)
    seq2seq_trainer.model.model.save_pretrained(save_dir)


if __name__ == "__main__":
    train()

