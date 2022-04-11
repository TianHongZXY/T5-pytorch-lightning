import torch
import numpy as np
from transformers import PreTrainedTokenizer
from data_preprocess import (
    perform_span_corruption_seg,
    create_masked_lm_predictions,
    convert_megatron_mask_tokens_to_extra_id,
)
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from typing import List, Union, Tuple, Optional, Dict


class LightningDataModel(pl.LightningDataModule):
    """ PyTorch Lightning data class """
    @staticmethod
    def add_data_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('LightningDataModel')
        parser.add_argument('--data_dir',
                            default='./data',
                            type=str)
        parser.add_argument('--num_workers', default=8, type=int)
        parser.add_argument('--train_data', default=None, type=str)
        parser.add_argument('--valid_data', default=None, type=str)
        parser.add_argument('--test_data', default=None, type=str)
        parser.add_argument('--cached_train_data',
                            default='cached_train_data.pkl',
                            type=str)
        parser.add_argument('--cached_valid_data',
                            default='cached_valid_data.pkl',
                            type=str)
        parser.add_argument('--cached_test_data',
                            default='cached_test_data.pkl',
                            type=str)
        parser.add_argument('--micro_batch_size', default=4, type=int)
        parser.add_argument('--global_batch_size', default=8, type=int)
        parser.add_argument('--valid_batch_size', default=4, type=int)
        parser.add_argument('--corruption_rate', default=0.15, type=float)
        parser.add_argument('--max_extra_id', default=100, type=int)
        parser.add_argument('--source_max_token_len', default=512, type=int)
        parser.add_argument('--target_max_token_len', default=512, type=int) # int(source_max_token_len * corruption_rate * 1.5)
        #  parser.add_argument('--recreate_dataset', action='store_true', default=False)

        return parent_parser

    def __init__(
        self,
        train_dataset: torch.utils.data.Dataset,
        valid_dataset: torch.utils.data.Dataset = None,
        test_dataset: torch.utils.data.Dataset = None,
        micro_batch_size: int = 4,
        valid_batch_size: int = 4,
        num_workers: int = 8,
    ):
        """
        initiates a PyTorch Lightning Data Module
        Args:
            train_dataset (torch.utils.data.Dataset): train dataset.
            valid_dataset (torch.utils.data.Dataset, optional): valid dataset.
            test_dataset (torch.utils.data.Dataset, optional): test dataset.
            micro_batch_size (int, optional): batch size. Defaults to 4.
            valid_batch_size (int, optional): batch size. Defaults to 4.
            num_workers (int, optional): num workers for dataloader. Defaults to 8.
        """
        super().__init__()

        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset
        self.micro_batch_size = micro_batch_size
        self.valid_batch_size = valid_batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.micro_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        if self.valid_dataset is not None:
            return DataLoader(
                self.valid_dataset,
                batch_size=self.valid_batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
            )
        else:
            pass

    def test_dataloader(self):
        if self.test_dataset is not None:
            return DataLoader(
                self.test_dataset,
                batch_size=self.valid_batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
            )
        else:
            pass


class CommonsenseQADatasetForT5(Dataset):
    def __init__(
        self,
        raw_text: List[Dict],
        tokenizer: PreTrainedTokenizer,
        source_max_token_len: int = 512,
    ):
        """
        Dataset for loading CommonsenseQA and training T5
        Args:
            raw_context (list[dict]): raw context
            tokenizer (PreTrainedTokenizer): tokenizer of model
            source_max_token_len (int, optional): max length of source sequence
        """
        print("Number of raw samples: ", len(raw_text))
        assert isinstance(raw_text, (list, tuple)
            ), f"raw_text must be list or tuple, but type {type(raw_text)} is given"
        assert all(isinstance(t, dict) for t in raw_text)

        self.raw_text = raw_text
        self.tokenizer = tokenizer
        self.source_max_token_len = max_token_len

        self.examples = []
        self.labels = ['A', 'B', 'C', 'D', 'E']
        for line in lines:
          qid = line['id']
          question = "Q: " + line['question']['stem']
          label_index = self.labels.index(line.get('answerKey', 'A'))
          answers = [choice['text'] for choice in sorted(line['question']['choices'], key=lambda c: c['label'])]
          answers_choices = "Answer Choices: "
          answer_text = f"A: The answer is {answers[label_index]} ({self.labels[label_index]})."
          for index, answer_text in zip(self.labels, answers):
              answers_choices += f"({index}) {answer_text} "
          self.examples.append({"question": question, "answers_choices": answers_choice, "answer_text": answer_text})

    def __getitem__(self, index: int):
        """ returns dictionary of input tensors to feed into GPT model"""
        qa = self.examples[index]
        source_text = qa['question'] + qa['answers_choices']
        target_text = qa['answer_text']
        # tokenize separately 
        source_text_encoding = self.tokenizer(
            source_text,
            max_length=self.source_max_token_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        target_text_encoding = self.tokenizer(
            target_text,
            max_length=self.source_max_token_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        labels = target_text_encoding["input_ids"]
        labels[labels == self.tokenizer.pad_token_id] = -100

        return dict(
            source_text=source_text,
            target_text=target_text,
            source_text_input_ids=source_text_encoding["input_ids"].flatten(),
            source_text_attention_mask=source_text_encoding["attention_mask"].flatten(),
            labels=labels.flatten(),
            labels_attention_mask=target_text_encoding["attention_mask"].flatten(),
        )

    def __len__(self):
        return len(self.raw_text)


class T5Dataset(Dataset):
    def __init__(
        self,
        raw_text: List[str],
        tokenizer: PreTrainedTokenizer,
        source_max_token_len: int = 512,
        target_max_token_len: int = 512,
        corruption_rate: float = 0.15,
        max_extra_id: int = 100,
        training_objective: str = "span_corruption",
    ):
        """
        Dataset for training T5
        Args:
            raw_context (list[str]): raw context
            tokenizer (PreTrainedTokenizer): tokenizer of model
            source_max_token_len (int, optional): max length of source sequence
            target_max_token_len (int, optional: max length of target sequence
            corruption_rate (float, optional): rate of span corruption pre-train task
            max_extra_id (int, optional): max number of corrupted span per context
        """
        print("Initial number of raw context samples: ", len(raw_text))
        #  assert raw_text is a list containing text samples
        assert isinstance(raw_text, (list, tuple)
            ), f"raw_text must be list or tuple, but type {type(raw_text)} is given"
        assert all(isinstance(t, str) for t in raw_text)

        self.raw_text = raw_text
        self.training_objective = training_objective
        self.tokenized_text = []
        self.tokenized_ids = []

        for i in range(len(self.raw_text)):
            cur_context = tokenizer.tokenize(self.raw_text[i])
            while len(cur_context) > source_max_token_len:
                window_context = cur_context[:source_max_token_len]
                self.tokenized_text.append(window_context)
                cur_context = cur_context[source_max_token_len:]
            self.tokenized_text.append(cur_context)
        print("After tokenization and segmentation, number of context samples: ", len(self.tokenized_text))
        for t_text in self.tokenized_text:
            self.tokenized_ids.append(tokenizer.convert_tokens_to_ids(t_text))

        self.tokenizer = tokenizer
        #  self.vocab_id_list = list(tokenizer.vocab.values())  # + tokenizer.additional_special_tokens_ids, 不需要加上special tokens,
        #  self.vocab_id_to_token_dict = tokenizer.ids_to_tokens
        #  不需要加上special tokens
        #  self.vocab_id_to_token_dict.update({k: v for k, v in zip(tokenizer.additional_special_tokens_ids, tokenizer.additional_special_tokens)})
        self.source_max_token_len = source_max_token_len
        self.target_max_token_len = target_max_token_len
        self.corruption_rate = corruption_rate
        self.max_extra_id = max_extra_id
        
    def __len__(self):
        return len(self.tokenized_text)

    def qa_getitem(self, str_context: str):
        colon = str_context[2]
        str_context = str_context.split("答案" + colon)
        context_row = [str_context[0], "答案" + colon + str_context[1]]

        return context_row

    def corruption_getitem(self, index: int) -> Tuple[str]:
        (tokens, masked_positions, masked_labels, _, masked_spans) = create_masked_lm_predictions(
                tokens=self.tokenized_ids[index],
                vocab_size=self.tokenizer.vocab_size,
                convert_ids_to_tokens=self.tokenizer.convert_ids_to_tokens,
                #  vocab_id_list=self.vocab_id_list,
                #  vocab_id_to_token_dict=self.vocab_id_to_token_dict,
                masked_lm_prob=self.corruption_rate,
                cls_id=self.tokenizer.cls_token_id,
                sep_id=self.tokenizer.sep_token_id,
                mask_id=self.tokenizer.mask_token_id if self.tokenizer.mask_token_id is not None else self.tokenizer.unk_token_id,
                max_predictions_per_seq=self.corruption_rate * self.target_max_token_len,
                np_rng=np.random.RandomState(index + 20020206),
                max_ngrams=10,
                do_whole_word_mask=True,
                favor_longer_ngram=False,
                do_permutation=False,
                geometric_dist=True,
                masking_style="t5",
                )
        #  注意这里第二个参数是给的是包含所有extra_token_ids的列表
        extra_tokens = [f"<extra_id_{i}>" for i in range(self.max_extra_id)]
        extra_token_ids = self.tokenizer.convert_tokens_to_ids(extra_tokens)
        tokens_enc, labels = convert_megatron_mask_tokens_to_extra_id(tokens, extra_token_ids, masked_spans)
        #  context_row = perform_span_corruption_seg(context_row, noise_prob=self.corruption_rate, max_extra_id=self.max_extra_id)

        return (self.tokenizer.decode(tokens_enc), self.tokenizer.decode(labels))

    def prefix_lm_getitem(self, str_context: str):
        """You can implement supervised training using given source and target, prefix lm is just a simple example."""
        length = len(str_context)
        context_row = [str_context[:length // 5], str_context[length // 5:]]
        return context_row

    def __getitem__(self, index: int):
        """Returns dictionary of input tensors to feed into T5/MT5 model"""
        context_row = self.tokenized_text[index]
        str_context = self.tokenizer.convert_tokens_to_string(context_row).replace(" ", "")

        # qa data
        if str_context.startswith("问题:") or str_context.startswith("问题："):
            context_row = self.qa_getitem(str_context)
        # span corruption data
        elif self.training_objective == "span_corruption":
            context_row = self.corruption_getitem(index)
        # prefix lm data
        elif self.training_objective == "prefix_lm":
            context_row = self.prefix_lm_getitem(str_context)
        else:
            raise ValueError(f"Training objective `{self.training_objective}` is not acceptable!")

        # tokenize separately 
        source_text = context_row[0]
        target_text = context_row[1]

        source_text_encoding = self.tokenizer(
            source_text,
            max_length=self.source_max_token_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        target_text_encoding = self.tokenizer(
            target_text,
            max_length=self.target_max_token_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        labels = target_text_encoding["input_ids"]
        labels[labels == self.tokenizer.pad_token_id] = -100

        return dict(
            source_text=source_text,
            target_text=target_text,
            source_text_input_ids=source_text_encoding["input_ids"].flatten(),
            source_text_attention_mask=source_text_encoding["attention_mask"].flatten(),
            labels=labels.flatten(),
            labels_attention_mask=target_text_encoding["attention_mask"].flatten(),
        )

