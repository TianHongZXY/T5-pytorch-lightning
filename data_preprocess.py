import random
import os
import json
import pandas as pd
from LAC import LAC


def perform_span_corruption_nonseg(context, noise_prob=0.15, max_extra_id=100):
    """
    Args:
        context: the context to be corrupted
        noise_prob: the probability of tokens masked
        extra_id_max: the maximum id of extra id
    Returns:
        Source Context String
        Target Context String
    """
    context_len = len(context)
    corruption_idx = random.sample(range(context_len), int(context_len * noise_prob))
    corruption_idx = sorted(corruption_idx)
    target_seq = []
    source_seq = []

    prev_idx = 0
    pt = 0
    extra_id_cnt = 0
    while pt < len(corruption_idx):
        corrup_start = corruption_idx[pt]
        while pt < len(corruption_idx)-1 and corruption_idx[pt+1] == corruption_idx[pt]+1:
            pt += 1
        corrup_end = corruption_idx[pt]+1
        source_seq.append(context[prev_idx:corrup_start])
        source_seq.append('<extra_id_{}>'.format(extra_id_cnt))
        prev_idx = corrup_end
        target_seq.append('<extra_id_{}>'.format(extra_id_cnt)+context[corrup_start:corrup_end])
        extra_id_cnt += 1
        if extra_id_cnt == max_extra_id:
            #  source_seq.append(context[prev_idx:])
            break
        pt += 1
    source_seq.append(''.join(context_seg[prev_idx:]))
    target_seq.append('<extra_id_{}>'.format(extra_id_cnt))
    return ''.join(source_seq), ''.join(target_seq)


# perform corruption span on segmented contexts
def perform_span_corruption_seg(context_seg, noise_prob=0.05, max_extra_id=100):
    """
    Args:
        context_seg:the segmented context
        noise_prob:the probability of tokens masked
        extra_id_max: the maximum id of extra id
    Returns:
        Source Context String(concatenated from corrupted context segments)
        Target Context String
    """
    context_len = len(context_seg)
    corruption_idx = random.sample(range(context_len),int(context_len * noise_prob))
    corruption_idx = sorted(corruption_idx)
    target_seq = []
    source_seq = []
    
    prev_idx = 0
    pt = 0
    extra_id_cnt = 0
    while pt < len(corruption_idx):
        corrup_start = corruption_idx[pt]
        while pt < len(corruption_idx)-1 and corruption_idx[pt+1] == corruption_idx[pt]+1:
            pt += 1
        corrup_end = corruption_idx[pt]+1
        source_seq.append(''.join(context_seg[prev_idx:corrup_start]))
        source_seq.append('<extra_id_{}>'.format(extra_id_cnt))
        prev_idx = corrup_end
        target_seq.append('<extra_id_{}>'.format(extra_id_cnt)+''.join(context_seg[corrup_start:corrup_end]))
        extra_id_cnt += 1
        if extra_id_cnt == max_extra_id:
            #  source_seq.append(''.join(context_seg[prev_idx:]))
            break
        pt += 1
    source_seq.append(''.join(context_seg[prev_idx:]))
    target_seq.append('<extra_id_{}>'.format(extra_id_cnt))
    return ''.join(source_seq), ''.join(target_seq)


def get_training_data(data_path, segmented):
    if segmented:
        with open(data_path) as f:
            train_data = json.load(f)
        all_contexts = train_data['context_segs']
        return all_contexts
    else:
        with open(data_path) as f:
            train_data = json.load(f)
        train_data = train_data['data']
        cqa = [d['paragraphs'][0] for d in train_data]
        all_contexts = [t['context'] for t in cqa]
        return all_contexts


def get_training_data_lightning(data_path):
    with open(data_path) as f:
        train_data = f.readlines()
    all_contexts = list(train_data)
    return all_contexts

