import pandas as pd
import numpy as np
import json
import os
import random
import torch
#os.environ["CUDA_VISIBLE_DEVICES"] = "6"
from t5_modeling import SimpleT5
from data_preprocess import perform_span_corruption_seg, perform_span_corruption_nonseg

# parameters
source_max_token_len = 1024
target_max_token_len = 512
batch_size = 1
gpu_nums = 1
outputdir = "/cognitive_comp/sunyuqing/models/t5/outputs"
show_training_ex = 100
corruption_cnt = 50

# instantiate model
model_path = "google/mt5-large"

# get data
data_dir = '/cognitive_comp/sunyuqing/datasets/cmrc2018-master/squad-style-data/'
seg_data_dir = os.path.join(data_dir, 'cmrc2018_all_context_segs.json')
nonseg_data_dir = os.path.join(data_dir, 'cmrc2018_all.json')

def get_training_data(segmented):
    if segmented:
        with open(seg_data_dir) as f:
            train_data = json.load(f)
        all_contexts = train_data['context_segs']
        return all_contexts
    else:
        with open(nonseg_data_dir) as f:
            train_data = json.load(f)
        train_data = train_data['data']
        cqa = [d['paragraphs'][0] for d in train_data]
        all_contexts = [t['context'] for t in cqa]
        return all_contexts

all_contexts_seg = get_training_data(True)
all_contexts_nonseg = get_training_data(False)

# train on non-segmented corruption data
"""
model = SimpleT5()
model.from_pretrained("t5", model_path)
model.train_span_corruption(
    all_contexts_nonseg,
    corruption_cnt = corruption_cnt,
    context_segmented = False,
    source_max_token_len = source_max_token_len,
    target_max_token_len = target_max_token_len,
    batch_size = batch_size,
    max_epochs = max_epochs,
    gpu_nums = gpu_nums,
    show_training_ex=show_training_ex
)
model.model.save_pretrained(f"{outputdir}/t5-mengzi-mrc-cmrc2018-1")
"""

# train on segmented corruption data
model = SimpleT5()
model.from_pretrained("mt5", model_path)
model.train_span_corruption(
    all_contexts_seg,
    context_segmented = True,
    source_max_token_len = source_max_token_len, 
    target_max_token_len = target_max_token_len,
    batch_size = batch_size,
    max_epochs = corruption_cnt,
    gpu_nums = gpu_nums,
    show_training_ex=show_training_ex
)
model.model.save_pretrained(f"{outputdir}/mt5-large-mrc-cmrc2018-2")


# train on half segmented data and half non-seg data
model = SimpleT5()
model.from_pretrained("mt5", model_path)
model.train_span_corruption(
    all_contexts_nonseg,
    context_segmented = False,
    source_max_token_len = source_max_token_len, 
    target_max_token_len = target_max_token_len,
    batch_size = batch_size,
    max_epochs = corruption_cnt,
    gpu_nums = gpu_nums,
    show_training_ex=show_training_ex
)
model.train_span_corruption(
    all_contexts_seg,
    context_segmented = True,
    source_max_token_len = source_max_token_len, 
    target_max_token_len = target_max_token_len,
    batch_size = batch_size,
    max_epochs = corruption_cnt,
    gpu_nums = gpu_nums,
    show_training_ex=show_training_ex
)
model.model.save_pretrained(f"{outputdir}/mt5-large-mrc-cmrc2018-3")

