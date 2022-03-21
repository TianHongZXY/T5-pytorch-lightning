import sys
import os
import json
import torch
import argparse
from t5_modeling import BaseSeq2SeqModel
import pytorch_lightning as pl
from data_preprocess import get_training_data_lightning
from tqdm import tqdm


def inference():
    torch.cuda.empty_cache()
    # get data
    source_texts = get_training_data_lightning(args.data_path)

    # train on segmented corruption data
    model = BaseSeq2SeqModel('t5', '/cognitive_comp/zhuxinyu/pretrained_models/mengzi-t5-base/')
    #  Stage 3
    #  state_dict = torch.load(os.path.join(args.ckpt_path, 'lightning_model.pt'))
    #  state_dict['state_dict']['model.encoder.embed_tokens.weight'] = state_dict['state_dict']['model.shared.weight']
    #  state_dict['state_dict']['model.decoder.embed_tokens.weight'] = state_dict['state_dict']['model.shared.weight']
    #  model.load_state_dict(state_dict=state_dict['state_dict'])

    #  Stage 2
    #  使用deepspeed stage2/3 训练的模型，只能使用lightning的脚本转checkpoint, 否则state_dict的键会多一个module.
    #  并且只能使用torch.load()和model.load_state_dict()的方式加载checkpoint
    #  除此以外, embed.token.weight会出现丢失，参见https://github.com/PyTorchLightning/pytorch-lightning/issues/10964
    state_dict = torch.load(os.path.join(args.ckpt_path, 'lightning_model.pt'))['state_dict']
    #  print(state_dict.keys())
    state_dict['model.encoder.embed_tokens.weight'] = state_dict['model.shared.weight']
    state_dict['model.decoder.embed_tokens.weight'] = state_dict['model.shared.weight']
    model.load_state_dict(state_dict=state_dict)

    if torch.cuda.is_available():
        model.cuda()
    predictions = model.inference(
        source_texts,
        max_length=args.max_length,
        num_return_sequences=1,
        num_beams=args.num_beams,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=1,
        no_repeat_ngram_size=0,
        length_penalty=1,
        early_stopping=True,
        skip_special_tokens=True,
    )

    return predictions

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20020206)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=100)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--model_type", type=str, default='t5')
    parser.add_argument("--ckpt_path", type=str)
    parser.add_argument("--data_path", type=str)
    args = parser.parse_args()

    pl.seed_everything(args.seed)

    predictions = inference()
    with open(os.path.join(args.ckpt_path, 'config_of_predictions.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, indent=4)

    with open(os.path.join(args.ckpt_path, 'predictions.txt'), 'w', encoding='utf-8') as f:
        for pred in tqdm(predictions):
            f.write(pred + '\n')
