import sys
import os
import json
import torch
import argparse
from t5_modeling import BaseSeq2SeqModel, T5LightningModel
import pytorch_lightning as pl
from data_preprocess import get_data_lightning
from tqdm import tqdm
from fengshen import T5Tokenizer as fengshenT5Tokenizer
from transformers import T5Tokenizer


def inference():
    torch.cuda.empty_cache()
    # get data
    source_texts = get_data_lightning(args.data_path)

    # train on segmented corruption data
    #  model = BaseSeq2SeqModel('t5', '/cognitive_comp/zhuxinyu/pretrained_models/mengzi-t5-base/')
    #  Stage 3
    #  state_dict = torch.load(os.path.join(args.ckpt_path, 'lightning_model.pt'))
    #  state_dict['state_dict']['model.encoder.embed_tokens.weight'] = state_dict['state_dict']['model.shared.weight']
    #  state_dict['state_dict']['model.decoder.embed_tokens.weight'] = state_dict['state_dict']['model.shared.weight']
    #  model.load_state_dict(state_dict=state_dict['state_dict'])

    #  Stage 2
    #  使用deepspeed stage2/3 训练的模型，只能使用lightning的脚本转checkpoint, 否则state_dict的键会多一个module.
    #  并且只能使用torch.load()和model.load_state_dict()的方式加载checkpoint
    #  除此以外, embed.token.weight会出现丢失，参见https://github.com/PyTorchLightning/pytorch-lightning/issues/10964
    #  state_dict = torch.load(args.ckpt_path)['state_dict']
    #  print(state_dict.keys())
    #  state_dict['model.encoder.embed_tokens.weight'] = state_dict['model.shared.weight']
    #  state_dict['model.decoder.embed_tokens.weight'] = state_dict['model.shared.weight']
    #  model.load_state_dict(state_dict=state_dict)

    model = T5LightningModel.from_pretrained(model_type=args.model_type, model_name=args.model_name)
    state_dict = torch.load(args.ckpt_path)['state_dict']
    #  model.model.load_state_dict(state_dict=state_dict)
    model.load_state_dict(state_dict=state_dict)
    model.tokenizer = T5Tokenizer.from_pretrained(args.model_name)
    #  model.model.save_pretrained(os.path.split(args.ckpt_path)[0])

    if torch.cuda.is_available():
        model.cuda()
    if args.model_type == "fengshent5":
        predictions = []
        input_ids = model.tokenizer(
            source_texts, 
            return_tensors="pt", 
            add_special_tokens=True,
            padding=True,
        ).input_ids
        input_ids = input_ids.to(model.device)
        for i in tqdm(range(input_ids.size(0))):
            #  print("inputs:", model.tokenizer.decode(input_ids[i], skip_special_tokens=True))
            generated_ids = model.model.generate(
                input_ids=input_ids[i],
                max_length=args.max_length,
                greedy=True,
            )
            print(generated_ids)
            predictions.append(model.tokenizer.decode(generated_ids, skip_special_tokens=True))
            print(predictions[-1])
        #  predictions = model.tokenizer.batch_decode(middle_results, skip_special_tokens=True)
    else:
        predictions = model.inference(
            source_texts,
            max_length=args.max_length,
            do_sample=args.do_sample,
            num_return_sequences=args.num_return_sequences,
            num_beams=args.num_beams,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            length_penalty=args.length_penalty,
            early_stopping=True,
            skip_special_tokens=True,
        )

    return predictions

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20020206)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--model_type", type=str, default='t5')
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--ckpt_path", type=str)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--do_sample", "-ds", action="store_true", default=False)
    parser.add_argument("--num_beams", "-nb", type=int, default=None)
    parser.add_argument("--no_repeat_ngram_size", "-nr", type=int, default=None)
    parser.add_argument("--num_return_sequences", "-ns", type=int, default=None)
    parser.add_argument("--length_penalty", "-lp", type=float, default=None)
    parser.add_argument("--repetition_penalty", "-rp", type=float, default=None)
    args = parser.parse_args()

    pl.seed_everything(args.seed)

    predictions = inference()
    with open(os.path.join(os.path.split(args.ckpt_path)[0], 'config_of_predictions.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, indent=4)

    with open(os.path.join(os.path.split(args.ckpt_path)[0], 'predictions.txt'), 'w', encoding='utf-8') as f:
        for pred in tqdm(predictions):
            f.write(pred + '\n')
