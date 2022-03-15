import pandas as pd
import numpy as np
import json
import os
import random
from t5_modeling import SimpleT5

# get data
def get_cqa():
    data_dir = '/cognitive_comp/sunyuqing/datasets/cmrc2018-master/squad-style-data/'
    with open(os.path.join(data_dir, 'cmrc2018_all.json')) as f:
        train_data = json.load(f)
    train_data = train_data['data']
    cqa = [d['paragraphs'][0] for d in train_data]
    return cqa

def get_all_c(cqa):
    return [t['context'] for t in cqa]

def get_qa(cqa, answer_max_char_len=256):
    qa = [t['qas'] for t in cqa]
    qa = [item for sublist in qa for item in sublist]
    if answer_max_char_len!=-1:
        qa_filtered = list(filter(lambda t:len(t['answers'][0]['text'])<answer_max_char_len+1, qa))
        return qa_filtered
    else:
        return qa

def get_cqa_flattened(cqa):
    result = [(t['context'],tq['question'],tq['answer']) for t in cqa for tq in t]
    return result

# process qa pair
def process_qa(qa):
    source_target = []
    for t in qa:
        q = '问题:'+t['question']+' 答案:'
        a = t['answers'][0]['text']
        source_target.append((q,a))
    qa_pd = pd.DataFrame.from_records(source_target, columns=['source_text','target_text'])
    return qa_pd

# process contexts
def process_contexts(contexts, context_prompt_len:int=256):
    source_target = []
    for c in contexts:
        source = '前文:'+c[:context_prompt_len]+' 后文:'
        target = c[context_prompt_len:]
        source_target.append((source, target))
    context_pd = pd.DataFrame.from_records(source_target, columns=['source_text','target_text'])
    return context_pd

# process qc pair
def process_qc(cqa_flattened):
    source_target = []
    for t in cqa_flattened:
        q = '问题:'+t[1]+' 该问题原文:'
        c = t[0]
        source_target.append((q,c))
    qc_pd = pd.DataFrame.from_records(source_target, columns=['source_text','target_text'])
    return qc_pd

# tuning
def finetune(
    ft_pd:pd.DataFrame, 
    model:SimpleT5, 
    outputdir:str,
    model_save_name:str,
    task_name:str,
    source_max_token_len:int=512,
    target_max_token_len:int=512, 
    batch_size:int=8, 
    gpu_nums:int=1,
):  
    model_dir = f"{outputdir}/t5-mengzi-mrc-cmrc2018"
    for i in range(3,4):
        model.load_model("t5",f"{model_dir}-{i+1}")
        model.train_supervised(
            ft_pd, 
            source_max_token_len=source_max_token_len,
            target_max_token_len=target_max_token_len,
            batch_size=batch_size,
            max_epochs=1,
            gpu_nums=gpu_nums,
            outputdir=outputdir,
            show_training_ex=10
        )
        model.model.save_pretrained(f"{outputdir}/{model_save_name}-{i+1}-{task_name}")

# inference
def inference(
    inf_pd:pd.DataFrame,
    model:SimpleT5,
    outputdir:str,
    model_save_name:str,
    task_name:str,
    result_save_name:str,
    target_max_token_len:int=256,
    num_return_sequences:int=1,
    num_beams:int=2,
    top_k:int=50,
    top_p:float=1, 
    repetition_penalty: float = 2.5,
    no_repeat_ngram_size:int = 3,
    length_penalty: float = 1.0,
    early_stopping: bool = True,
): 
    source_texts = inf_pd['source_text'].to_list()
    model_dir = f"{outputdir}/{model_save_name}"
    model_inf_results = []
    for i in range(4):
        print("inference on model {}".format(i))
        model.load_model("t5",f"{model_dir}-{i+1}-{task_name}", use_gpu=True)
        inf_results = []
        for j in range(100):
            inf_results.append(model.inference(
            source_texts[j],
            max_length = target_max_token_len,
            num_return_sequences=num_return_sequences,
            num_beams=num_beams,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            length_penalty=length_penalty,
            early_stopping=early_stopping
        ))
        model_inf_results.append(inf_results)
    label_texts = inf_pd['target_text'].to_list()
    result_pd = pd.DataFrame({
        'question':source_texts,
        'ground_truth_answer':label_texts,
        'model1':model_inf_results[0],
        'model2':model_inf_results[1],
        'model3':model_inf_results[2],
        'model4':model_inf_results[3]
    })
    result_pd.to_csv(f"{outputdir}/{result_save_name}")

# main() function
def main():
    model = SimpleT5()
    cqa = get_cqa()

    # Q->A
    qa_pair = get_qa(cqa, answer_max_char_len=256)
    #ft_qa = process_qa(qa_pair[:500])
    inf_qa = process_qa(qa_pair[500:600])
    """
    finetune(
        ft_qa, 
        model, 
        "/cognitive_comp/sunyuqing/models/t5/outputs",
        "t5-mengzi-mrc-cmrc2018",
        "ft-qa",
        source_max_token_len=128,
        target_max_token_len=256,
        batch_size=8,
        gpu_nums=1
    )"""
    inference(
        inf_qa,
        model,
        "/cognitive_comp/sunyuqing/models/t5/outputs",
        "t5-mengzi-mrc-cmrc2018",
        "ft-qa",
        "t5-mengzi-mrc-cmrc2018-ft-qa-inf-5.csv",
        target_max_token_len=256,
        num_return_sequences=1,
        num_beams=1,
        top_k=0,
        top_p=1,
        repetition_penalty=2.5,
        no_repeat_ngram_size=3,
        length_penalty=1.5,
        early_stopping=False
    )

    """
    # Context Continuation
    all_contexts = get_all_c(cqa)
    ft_contexts = process_contexts(all_contexts[:500],context_prompt_len=32)
    inf_contexts = process_contexts(all_contexts[500:600],context_prompt_len=32)
    
    finetune(
        ft_contexts, 
        model, 
        "/cognitive_comp/sunyuqing/models/t5/outputs",
        "t5-mengzi-mrc-cmrc2018",
        "ft-cc",
        source_max_token_len=32,
        target_max_token_len=1024,
        batch_size=1,
        gpu_nums=1
    )
    inference(
        inf_contexts,
        model,
        "/cognitive_comp/sunyuqing/models/t5/outputs",
        "t5-mengzi-mrc-cmrc2018",
        "ft-cc",
        "t5-mengzi-mrc-cmrc2018-ft-cc-inf.csv",
        target_max_token_len=1024,
        num_return_sequences=1,
        num_beams=1,
        top_k=0,
        top_p=1,
        repetition_penalty=2.5,
        no_repeat_ngram_size=3,
        length_penalty=2.5,
        early_stopping=False
    )
"""
"""
    # Q->C
    qc_pair = get_cqa_flattened(cqa)
    ft_qc = process_qc(qc_pair[:500])
    inf_qc = process_qc(qc_pair[500:])
    finetune(
        ft_qc, 
        model, 
        "/cognitive_comp/sunyuqing/models/t5/outputs",
        "t5-mengzi-mrc-cmrc2018",
        "ft-qc",
        source_max_token_len=128,
        target_max_token_len=1024,
        batch_size=8,
        gpu_nums=1
    )
    inference(
        inf_qc,
        model,
        "/cognitive_comp/sunyuqing/models/t5/outputs",
        "t5-mengzi-mrc-cmrc2018",
        "ft-qc",
        "t5-mengzi-mrc-cmrc2018-ft-qc-inf.csv",
        target_max_token_len=1024,
        num_return_sequences=1,
        num_beams=2,
        top_k=0,
        top_p=1,
        repetition_penalty=2.5,
        no_repeat_ngram_size=3,
        length_penalty=2.5,
        early_stopping=False
    )"""


if __name__ == "__main__":
    main()
