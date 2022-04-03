import random
import os
import json
import pandas as pd
from LAC import LAC
from typing import List, Tuple, Dict
import numpy as np
import collections


# perform corruption span on segmented contexts
def perform_span_corruption_seg(context_seg,
                                noise_prob=0.05,
                                max_extra_id=100,
                                ) -> Tuple[List[str], List[str]]:
    """
    Args:
        context_seg:the segmented context
        noise_prob:the probability of tokens masked
        extra_id_max: the maximum id of extra id
    Returns:
        Source Context String(concatenated from corrupted context segments)
        Target Context String
    """
    # TODO 待解决问题，Whole Word Masking
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
        while pt < len(corruption_idx) - 1 and corruption_idx[pt + 1] == corruption_idx[pt] + 1:
            pt += 1
        corrup_end = corruption_idx[pt] + 1
        source_seq.extend(context_seg[prev_idx:corrup_start])
        source_seq.append('<extra_id_{}>'.format(extra_id_cnt))
        prev_idx = corrup_end
        target_seq.append('<extra_id_{}>'.format(extra_id_cnt))
        target_seq.extend(context_seg[corrup_start:corrup_end])
        extra_id_cnt += 1
        if extra_id_cnt == max_extra_id:
            break
        pt += 1
    source_seq.extend(context_seg[prev_idx:])
    target_seq.append('<extra_id_{}>'.format(extra_id_cnt))

    return source_seq, target_seq


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


# Code from NVIDIA Megatron
MaskedLmInstance = collections.namedtuple("MaskedLmInstance",
                                          ["index", "label"])


def is_start_piece(piece):
    """Check if the current word piece is the starting piece (BERT)."""
    # When a word has been split into
    # WordPieces, the first token does not have any marker and any subsequence
    # tokens are prefixed with ##. So whenever we see the ## token, we
    # append it to the previous set of word indexes.
    return not piece.startswith("##")


def create_masked_lm_predictions(
        tokens: List[int],
        vocab_id_list: List[int],
        vocab_id_to_token_dict: Dict[int, str],
        masked_lm_prob: float,
        cls_id: int,
        sep_id: int,
        mask_id: int,
        max_predictions_per_seq: int,
        np_rng: np.random.RandomState,
        max_ngrams: int = 3,
        do_whole_word_mask: bool = True,
        favor_longer_ngram: bool = False,
        do_permutation: bool = False,
        geometric_dist: bool = False,
        masking_style: str = "bert",
        ) -> Tuple[List[int], List[int], List[int], List[int], List[MaskedLmInstance]]:
    """Creates the predictions for the masked LM objective.
    Note: Tokens here are vocab ids and not text tokens."""

    cand_indexes = []
    # Note(mingdachen): We create a list for recording if the piece is
    # the starting piece of current token, where 1 means true, so that
    # on-the-fly whole word masking is possible.
    token_boundary = [0] * len(tokens)

    for (i, token) in enumerate(tokens):
        if token == cls_id or token == sep_id:
            token_boundary[i] = 1
            continue
        # Whole Word Masking means that if we mask all of the wordpieces
        # corresponding to an original word.
        #
        # Note that Whole Word Masking does *not* change the training code
        # at all -- we still predict each WordPiece independently, softmaxed
        # over the entire vocabulary.
        if (do_whole_word_mask and len(cand_indexes) >= 1 and
                not is_start_piece(vocab_id_to_token_dict[token])):
            cand_indexes[-1].append(i)
        else:
            cand_indexes.append([i])
            if is_start_piece(vocab_id_to_token_dict[token]):
                token_boundary[i] = 1

    output_tokens = list(tokens)

    masked_lm_positions = []
    masked_lm_labels = []

    if masked_lm_prob == 0:
        return (output_tokens, masked_lm_positions,
                masked_lm_labels, token_boundary)

    num_to_predict = min(max_predictions_per_seq,
                         max(1, int(round(len(tokens) * masked_lm_prob))))

    ngrams = np.arange(1, max_ngrams + 1, dtype=np.int64)
    if not geometric_dist:
        # Note(mingdachen):
        # By default, we set the probilities to favor shorter ngram sequences.
        pvals = 1. / np.arange(1, max_ngrams + 1)
        pvals /= pvals.sum(keepdims=True)
        if favor_longer_ngram:
            pvals = pvals[::-1]

    ngram_indexes = []
    for idx in range(len(cand_indexes)):
        ngram_index = []
        for n in ngrams:
            ngram_index.append(cand_indexes[idx:idx + n])
        ngram_indexes.append(ngram_index)

    np_rng.shuffle(ngram_indexes)

    (masked_lms, masked_spans) = ([], [])
    covered_indexes = set()
    for cand_index_set in ngram_indexes:
        if len(masked_lms) >= num_to_predict:
            break
        if not cand_index_set:
            continue
        # Note(mingdachen):
        # Skip current piece if they are covered in lm masking or previous ngrams.
        for index_set in cand_index_set[0]:
            for index in index_set:
                if index in covered_indexes:
                    continue

        if not geometric_dist:
            n = np_rng.choice(ngrams[:len(cand_index_set)],
                              p=pvals[:len(cand_index_set)] /
                              pvals[:len(cand_index_set)].sum(keepdims=True))
        else:
            # Sampling "n" from the geometric distribution and clipping it to
            # the max_ngrams. Using p=0.2 default from the SpanBERT paper
            # https://arxiv.org/pdf/1907.10529.pdf (Sec 3.1)
            n = min(np_rng.geometric(0.2), max_ngrams)

        index_set = sum(cand_index_set[n - 1], [])
        n -= 1
        # Note(mingdachen):
        # Repeatedly looking for a candidate that does not exceed the
        # maximum number of predictions by trying shorter ngrams.
        while len(masked_lms) + len(index_set) > num_to_predict:
            if n == 0:
                break
            index_set = sum(cand_index_set[n - 1], [])
            n -= 1
        # If adding a whole-word mask would exceed the maximum number of
        # predictions, then just skip this candidate.
        if len(masked_lms) + len(index_set) > num_to_predict:
            continue
        is_any_index_covered = False
        for index in index_set:
            if index in covered_indexes:
                is_any_index_covered = True
                break
        if is_any_index_covered:
            continue
        for index in index_set:
            covered_indexes.add(index)
            masked_token = None
            if masking_style == "bert":
                # 80% of the time, replace with [MASK]
                if np_rng.random() < 0.8:
                    masked_token = mask_id
                else:
                    # 10% of the time, keep original
                    if np_rng.random() < 0.5:
                        masked_token = tokens[index]
                    # 10% of the time, replace with random word
                    else:
                        masked_token = vocab_id_list[np_rng.randint(0, len(vocab_id_list))]
            elif masking_style == "t5":
                masked_token = mask_id
            else:
                raise ValueError("invalid value of masking style")

            output_tokens[index] = masked_token
            masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))

        masked_spans.append(MaskedLmInstance(
            index=index_set,
            label=[tokens[index] for index in index_set]))

    assert len(masked_lms) <= num_to_predict
    np_rng.shuffle(ngram_indexes)

    select_indexes = set()
    if do_permutation:
        for cand_index_set in ngram_indexes:
            if len(select_indexes) >= num_to_predict:
                break
            if not cand_index_set:
                continue
            # Note(mingdachen):
            # Skip current piece if they are covered in lm masking or previous ngrams.
            for index_set in cand_index_set[0]:
                for index in index_set:
                    if index in covered_indexes or index in select_indexes:
                        continue

            n = np.random.choice(ngrams[:len(cand_index_set)],
                                 p=pvals[:len(cand_index_set)] /
                                 pvals[:len(cand_index_set)].sum(keepdims=True))
            index_set = sum(cand_index_set[n - 1], [])
            n -= 1

            while len(select_indexes) + len(index_set) > num_to_predict:
                if n == 0:
                    break
                index_set = sum(cand_index_set[n - 1], [])
                n -= 1
            # If adding a whole-word mask would exceed the maximum number of
            # predictions, then just skip this candidate.
            if len(select_indexes) + len(index_set) > num_to_predict:
                continue
            is_any_index_covered = False
            for index in index_set:
                if index in covered_indexes or index in select_indexes:
                    is_any_index_covered = True
                    break
            if is_any_index_covered:
                continue
            for index in index_set:
                select_indexes.add(index)
        assert len(select_indexes) <= num_to_predict

        select_indexes = sorted(select_indexes)
        permute_indexes = list(select_indexes)
        np_rng.shuffle(permute_indexes)
        orig_token = list(output_tokens)

        for src_i, tgt_i in zip(select_indexes, permute_indexes):
            output_tokens[src_i] = orig_token[tgt_i]
            masked_lms.append(MaskedLmInstance(index=src_i, label=orig_token[src_i]))

    masked_lms = sorted(masked_lms, key=lambda x: x.index)
    # Sort the spans by the index of the first span
    masked_spans = sorted(masked_spans, key=lambda x: x.index[0])

    for p in masked_lms:
        masked_lm_positions.append(p.index)
        masked_lm_labels.append(p.label)
    return (output_tokens, masked_lm_positions, masked_lm_labels, token_boundary, masked_spans)


def convert_megatron_mask_tokens_to_extra_id(
        tokens,
        extra_token_ids,
        masked_spans,
        ) -> Tuple[List[int], List[int]]:
    extra_token_ids = collections.deque(extra_token_ids)
    t5_input = []
    labels = []
    (start_index, end_index) = (0, None)
    for span in masked_spans:
        flag = extra_token_ids.popleft()

        # Append the same tokens in decoder input and output
        labels.append(flag)
        labels.extend(span.label)

        end_index = span.index[0]
        t5_input.extend(tokens[start_index: end_index])
        t5_input.append(flag)

        # the next start index is the token after the last span token
        start_index = span.index[-1] + 1

    # Add the remaining tokens to the t5 input
    t5_input.extend(tokens[start_index:])

    return t5_input, labels


if __name__ == "__main__":
    from fengshen import T5Tokenizer
    text = "答案：易烊千玺生于清朝咸丰元年（1854年）生。年（1853年）生。籍贯汉城，族亦2008年汉手欠年婚年（1988年）生后生年"
    tokenizer = T5Tokenizer.from_pretrained("/cognitive_comp/zhuxinyu/pretrained_models/IDEA-CCNL/Randeng-770M/")
    tokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))
    target_seq_length = 256
    max_num_tokens = target_seq_length
    truncated = len(tokens) > max_num_tokens
    tokens = tokens[:max_num_tokens]
    masked_lm_prob = 0.15

    # Masking.
    max_predictions_per_seq = masked_lm_prob * max_num_tokens
    vocab_id_list = list(tokenizer.vocab.values()) + tokenizer.additional_special_tokens_ids
    vocab_id_to_token_dict = tokenizer.ids_to_tokens
    vocab_id_to_token_dict.update({k: v for k, v in zip(tokenizer.additional_special_tokens_ids, tokenizer.additional_special_tokens)})
    (tokens, masked_positions, masked_labels, _, masked_spans) = create_masked_lm_predictions(tokens,
                                           vocab_id_list=vocab_id_list,
                                           vocab_id_to_token_dict=vocab_id_to_token_dict,
                                           masked_lm_prob=0.15,
                                           cls_id=tokenizer.cls_token_id,
                                           sep_id=tokenizer.sep_token_id,
                                           mask_id=tokenizer.mask_token_id,
                                           max_predictions_per_seq=118,
                                           np_rng=np.random.RandomState(20020206),
                                           max_ngrams=3,
                                           do_whole_word_mask=True,
                                           favor_longer_ngram=False,
                                           do_permutation=False,
                                           geometric_dist=False,
                                           masking_style="t5",
                                           )
    print(tokenizer.decode(tokens))
    print(tokenizer.decode(masked_labels))
    tokens_enc, tokens_dec_in = convert_megatron_mask_tokens_to_extra_id(tokens, tokenizer.additional_special_tokens_ids[2:], masked_spans)
    print(tokenizer.decode(tokens_enc))
    print(tokenizer.decode(tokens_dec_in))

