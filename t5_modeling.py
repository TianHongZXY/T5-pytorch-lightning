import os
import torch
import argparse
import numpy as np
import pandas as pd
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AdamW,
    Adafactor,
    T5ForConditionalGeneration,
    T5Tokenizer,
    MT5ForConditionalGeneration,
    MT5TokenizerFast as MT5Tokenizer,
    ByT5Tokenizer,
    PreTrainedTokenizer,
    PreTrainedModel,
    T5PreTrainedModel,
)
from transformers.optimization import get_linear_schedule_with_warmup, get_constant_schedule_with_warmup
from data_preprocess import (
    perform_span_corruption_seg,
    create_masked_lm_predictions,
    convert_megatron_mask_tokens_to_extra_id,
)
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning import loggers
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, TQDMProgressBar, LearningRateMonitor
from pytorch_lightning.plugins import DDPPlugin
from sklearn.model_selection import train_test_split
from torchsnooper import snoop
from typing import List, Union, Tuple, Optional
from fengshen import T5Config as fengshenT5Config
from fengshen import T5ForConditionalGeneration as fengshenT5ForConditionalGeneration
from fengshen import T5Tokenizer as fengshenT5Tokenizer



class LightningDataModule(pl.LightningDataModule):
    """ PyTorch Lightning data class """
    @staticmethod
    def add_data_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('DataModel')
        #  parser.add_argument('--data_dir',
        #                      default='./data',
        #                      type=str)
        #  parser.add_argument('--num_workers', default=8, type=int)
        #  parser.add_argument('--train_data', default='train.json', type=str)
        #  parser.add_argument('--valid_data', default='dev.json', type=str)
        #  parser.add_argument('--test_data', default='test.json', type=str)
        #  parser.add_argument('--cached_train_data',
        #                      default='cached_train_data.pkl',
        #                      type=str)
        #  parser.add_argument('--cached_valid_data',
        #                      default='cached_valid_data.pkl',
        #                      type=str)
        #  parser.add_argument('--cached_test_data',
        #                      default='cached_test_data.pkl',
        #                      type=str)
        #  parser.add_argument('--train_batchsize', default=16, type=int)
        #  parser.add_argument('--valid_batchsize', default=32, type=int)
        #  parser.add_argument('--recreate_dataset', action='store_true', default=False)
        parser.add_argument('--batch_size', default=6, type=int)

        return parent_parser

    def __init__(
        self,
        args,
        train_dataset,
        #  train_df: pd.DataFrame,
        #  test_df: pd.DataFrame,
        #  tokenizer: PreTrainedTokenizer,
        #  batch_size: int = 4,
        #  source_max_token_len: int = 512,
        #  target_max_token_len: int = 512,
        #  num_workers: int = 2,
    ):
        """
        initiates a PyTorch Lightning Data Module
        Args:
            train_df (pd.DataFrame): training dataframe. Dataframe must contain 2 columns --> "source_text" & "target_text"
            test_df (pd.DataFrame): validation dataframe. Dataframe must contain 2 columns --> "source_text" & "target_text"
            tokenizer (PreTrainedTokenizer): PreTrainedTokenizer (T5Tokenizer, MT5Tokenizer, or ByT5Tokenizer)
            batch_size (int, optional): batch size. Defaults to 4.
            source_max_token_len (int, optional): max token length of source text. Defaults to 512.
            target_max_token_len (int, optional): max token length of target text. Defaults to 512.
        """
        super().__init__()

        self.args = args
        self.train_dataset = train_dataset
        #  self.train_df = train_df
        #  self.test_df = test_df
        #  self.batch_size = batch_size
        #  self.tokenizer = tokenizer
        #  self.source_max_token_len = source_max_token_len
        #  self.target_max_token_len = target_max_token_len
        #  self.num_workers = num_workers

    def setup(self, stage=None):
        pass
        #  if stage == "fit"
        #  self.train_dataset = PyTorchDataModule(
        #      self.train_df,
        #      self.tokenizer,
        #      self.source_max_token_len,
        #      self.target_max_token_len,
        #  )
        #  self.test_dataset = PyTorchDataModule(
        #      self.test_df,
        #      self.tokenizer,
        #      self.source_max_token_len,
        #      self.target_max_token_len,
        #  )

    def train_dataloader(self):
        """ training dataloader """
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=16,
            pin_memory=True,
        )

    #  def test_dataloader(self):
    #      """ test dataloader """
    #      return DataLoader(
    #          self.test_dataset,
    #          batch_size=self.batch_size,
    #          shuffle=False,
    #          num_workers=self.num_workers,
    #      )
    #
    #  def val_dataloader(self):
    #      """ validation dataloader """
    #      return DataLoader(
    #          self.test_dataset,
    #          batch_size=self.batch_size,
    #          shuffle=False,
    #          num_workers=self.num_workers,
    #      )

#TODO 然后把data统一到一个datamodule里，然后探索一下span corruption
class ContextData(Dataset):
    def __init__(
        self,
        raw_text:List[str],
        tokenizer:PreTrainedTokenizer,
        source_max_token_len:int=512,
        target_max_token_len:int=512,
        corruption_rate:float=0.15,
        max_extra_id:int=100,
    ):
        #  if not isinstance(raw_text, pd.DataFrame):
        #      self.raw_text = pd.DataFrame({'raw_text':raw_text})
        #      print(self.raw_text.head())
        #  else:
        print("Initial number of raw context samples: ", len(raw_text))
        #  assert all raw_text is already tokenized, otherwise tokenize them using given tokenizer
        assert isinstance(raw_text, (list, tuple)
            ), f"raw_text must be list or tuple, but type {type(raw_text)} is given"
        assert all(isinstance(t, str) for t in raw_text)

        self.raw_text = raw_text
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
        self.vocab_id_list = list(tokenizer.vocab.values()) + tokenizer.additional_special_tokens_ids
        self.vocab_id_to_token_dict = tokenizer.ids_to_tokens
        self.vocab_id_to_token_dict.update({k: v for k, v in zip(tokenizer.additional_special_tokens_ids, tokenizer.additional_special_tokens)})
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
                vocab_id_list=self.vocab_id_list,
                vocab_id_to_token_dict=self.vocab_id_to_token_dict,
                masked_lm_prob=self.corruption_rate,
                cls_id=self.tokenizer.cls_token_id,
                sep_id=self.tokenizer.sep_token_id,
                mask_id=self.tokenizer.mask_token_id,
                max_predictions_per_seq=self.corruption_rate * self.target_max_token_len,
                np_rng=np.random.RandomState(index +  20020206),
                max_ngrams=10,
                do_whole_word_mask=True,
                favor_longer_ngram=False,
                do_permutation=False,
                geometric_dist=True,
                masking_style="t5",
                )
        #  注意这里第二个参数是给的是包含所有extra_token_ids的列表
        tokens_enc, labels = convert_megatron_mask_tokens_to_extra_id(tokens, self.tokenizer.additional_special_tokens_ids[2:], masked_spans)
        #  context_row = perform_span_corruption_seg(context_row, noise_prob=self.corruption_rate, max_extra_id=self.max_extra_id)

        return (self.tokenizer.decode(tokens_enc), self.tokenizer.decode(labels))

    def lm_getitem(self, str_context: str):
        context_row = ["阅读下面的文章：", str_context]
        return context_row

    def __getitem__(self, index: int):
        """ returns dictionary of input tensors to feed into T5/MT5 model"""
        context_row = self.tokenized_text[index]
        str_context = self.tokenizer.convert_tokens_to_string(context_row).replace(" ", "")

        # qa data
        if str_context.startswith("问题:") or str_context.startswith("问题："):
            context_row = self.qa_getitem(str_context)
        # span corruption data
        else:
            context_row = self.corruption_getitem(index)
        # lm data
        #  else:
        #      context_row = self.lm_getitem(str_context)
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


class BaseSeq2SeqModel(pl.LightningModule):
    """
    initiates a PyTorch Lightning Seq2Seq base model, defines training and evaluation steps
    """
    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        Add model specific args
        Args:
            show_training_ex(int, optional): print the training examples for the batch idx. Set to -1 to disable.
            model_type (str, optional): "t5" or "mt5" . Defaults to "t5".
            model_name (str, optional): exact model architecture name, "t5-base" or "t5-large". Defaults to "t5-base".
        Returns:
            parent_parser
        """
        parser = parent_parser.add_argument_group('BaseSeq2SeqModel')
        # * Args for model setting
        parser.add_argument('--seed', default=20020206, type=int)
        parser.add_argument('--lr', default=1e-5, type=float)
        parser.add_argument('--l2', default=0., type=float)
        parser.add_argument('--warmup', default=0.1, type=float)
        parser.add_argument('--show_training_ex', default=-1, type=int)
        parser.add_argument('--model_type', default=None, type=str)
        parser.add_argument('--model_name', default=None, type=str)
        #  parser.add_argument('--save_dir', default='./save', type=str)

        return parent_parser

    def __init__(
        self,
        args=None,
    ):
        """
        Args:
            tokenizer : T5/MT5/ByT5 tokenizer
            model : T5/MT5/ByT5 model
            outputdir (str, optional): output directory to save model checkpoints. Defaults to "outputs".
            save_only_last_epoch (bool, optional): If True, save just the last epoch else models are saved for every epoch
        """
        super().__init__()

        if isinstance(args, dict):
            args = argparse.Namespace(**args)
        self.args = args
        self.save_hyperparameters(args)
        self.average_training_loss = None
        self.average_validation_loss = None
        self._consumed_samples = 0
        self._consumed_tokens = 0

    def setup(self, stage) -> None:
        if stage == 'fit':
            train_loader = self.trainer.datamodule.train_dataloader()  # self.trainer.train_dataloader
            num_gpus = self.trainer.gpus if self.trainer.gpus is not None else 0
            self.total_step = int(self.trainer.max_epochs * len(train_loader) / \
                (max(1, num_gpus) * self.trainer.accumulate_grad_batches))
            print('Total training step:', self.total_step)

    def forward(self, input_ids, attention_mask, decoder_attention_mask, labels=None):
        """ forward step """
        output = self.model(
            input_ids,
            attention_mask=attention_mask,
            labels=labels,
            decoder_attention_mask=decoder_attention_mask,
        )

        return output.loss, output.logits

    def training_step(self, batch, batch_idx):
        """ training step """
        input_ids = batch["source_text_input_ids"]
        batch_size = input_ids.size(0)
        self._consumed_samples += batch_size * max(self.trainer.gpus, 1)  # batch size * data parallel size
        attention_mask = batch["source_text_attention_mask"]
        labels = batch["labels"]
        labels_attention_mask = batch["labels_attention_mask"]
        self._consumed_tokens += (len(input_ids.flatten()) + len(labels.flatten())) * max(self.trainer.gpus, 1)

        loss, logits = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=labels_attention_mask,
            labels=labels,
        )

        if self.hparams.show_training_ex > -1 and batch_idx % self.hparams.show_training_ex == 0:
            prediction = torch.argmax(logits, dim=2)
            #  input_tokens = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
            #  predicted_tokens = self.tokenizer.batch_decode(prediction, skip_special_tokens=True)
            #  labels_tokens = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
            input_tokens = self.tokenizer.decode(input_ids[0], skip_special_tokens=False)
            predicted_tokens = self.tokenizer.decode(prediction[0], skip_special_tokens=False)
            labels_tokens = self.tokenizer.decode(labels[0], skip_special_tokens=False)
            print('-' * 50)
            print('input_token:     ', input_tokens.replace('[PAD]', ''))
            print('-' * 50)
            print('predicted_tokens:', predicted_tokens.replace('[PAD]','').replace('[EOS]',''))
            print('-' * 50)
            print('labels_tokens:   ', labels_tokens.replace('[PAD]','').replace('[UNK]', ''))
            print('-' * 50)
            # prediction_label_pair = list(zip(predicted_tokens, labels_tokens))
            # print(prediction_label_pair[0])

        self.log("train_loss", loss, prog_bar=True, logger=True, on_step=True, batch_size=batch_size)
        ts_logger = self.logger.experiment
        ts_logger.add_scalar("train_loss_vs_samples", loss.item(), self._consumed_samples)
        ts_logger.add_scalar("train_loss_vs_tokens", loss.item(), self._consumed_tokens)

        #  current_step = self.trainer.lr_schedulers[0]['scheduler']._step_count

        return loss

    def validation_step(self, batch, batch_idx):
        """ validation step """
        input_ids = batch["source_text_input_ids"]
        batch_size = input_ids.size(0)
        attention_mask = batch["source_text_attention_mask"]
        labels = batch["labels"]
        labels_attention_mask = batch["labels_attention_mask"]

        loss, logits = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=labels_attention_mask,
            labels=labels,
        )

        self.log("val_loss", loss, prog_bar=True, logger=True, on_step=True, batch_size=batch_size)
        return loss

    def test_step(self, batch, batch_idx):
        """ test step """
        input_ids = batch["source_text_input_ids"]
        batch_size = input_ids.size(0)
        attention_mask = batch["source_text_attention_mask"]
        labels = batch["labels"]
        labels_attention_mask = batch["labels_attention_mask"]

        loss, logits = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=labels_attention_mask,
            labels=labels,
        )

        self.log("test_loss", loss, prog_bar=True, logger=True, on_step=True, batch_size=batch_size)
        return loss

    def configure_optimizers(self):
        """ configure optimizers """
        paras = list(filter(lambda p: p.requires_grad, self.parameters()))
        #  no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        #  paras = [{
        #      'params':
        #      [p for n, p in paras if not any(nd in n for nd in no_decay)],
        #      'weight_decay': self.hparams.l2
        #  }, {
        #      'params': [p for n, p in paras if any(nd in n for nd in no_decay)],
        #      'weight_decay': 0.0
        #  }]
        optimizer = AdamW(paras, lr=self.hparams.lr)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, int(self.total_step * self.hparams.warmup),
            self.total_step)

        return [{
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            }
        }]

    def training_epoch_end(self, training_step_outputs):
        """ save tokenizer and model on epoch end """
        self.average_training_loss = np.round(
            torch.mean(torch.stack([x["loss"] for x in training_step_outputs])).item(),
            4,
        )
        self.log("avg_train_loss", self.average_training_loss, prog_bar=True, logger=True, on_epoch=True)
        #  path = f"{self.hparams.outputdir}/epoch-{self.current_epoch}-train-loss-{str(self.average_training_loss)}"
        #  self.tokenizer.save_pretrained(path)
        #  self.model.save_pretrained(path)

    def validation_epoch_end(self, validation_step_outputs):
        _loss = [x.cpu() for x in validation_step_outputs]
        self.average_validation_loss = np.round(
            torch.mean(torch.stack(_loss)).item(),
            4,
        )
        self.log("average_validation_loss", self.average_validation_loss, prog_bar=True, logger=True, on_epoch=True)

    #  TODO 需不需要保留一个通用的接口？它能做些什么通用的事情吗？学习一下huggingface的做法，虽然它好像没有给T5做通用的接口
    #  @classmethod
    #  def from_pretrained(cls, **kwargs) -> pl.LightningModule:
    #      return cls._from_pretrained(cls, **kwargs)
    #
    #  @classmethod
    #  def _from_pretrained(cls, **kwargs) -> pl.LightningModule:
    #      raise NotImplementedError

    def inference(
        self, 
        source_texts,
        max_length: int = 512,
        num_return_sequences: int = 1,
        num_beams: int = 1,
        top_k: int = 50,
        top_p: float = 0.9,
        do_sample: bool = False,
        repetition_penalty: float = None,
        no_repeat_ngram_size: int = None,
        length_penalty: float = None,
        early_stopping: bool = True,
        skip_special_tokens: bool = True,
    ):
        """
        generates prediction for T5/MT5 model
        Args:
            source_texts (list[str]): sequence of texts for generating predictions
            max_length (int, optional): max token length of prediction. Defaults to 512.
            num_return_sequences (int, optional): number of predictions to be returned. Defaults to 1.
            num_beams (int, optional): number of beams. Defaults to 2.
            top_k (int, optional): Defaults to 100.
            top_p (float, optional): Defaults to 0.95.
            do_sample (bool, optional): Defaults to True.
            repetition_penalty (float, optional): Defaults to 2.5.
            length_penalty (float, optional): Defaults to 1.0.
            early_stopping (bool, optional): Defaults to True.
            skip_special_tokens (bool, optional): Defaults to True.
            clean_up_tokenization_spaces (bool, optional): Defaults to True.

        Returns:
            list[str]: returns predictions
        """
        input_ids = self.tokenizer(
            source_texts, 
            return_tensors="pt", 
            add_special_tokens=True,
            padding=True,
        ).input_ids
        input_ids = input_ids.to(self.model.device)
        generated_ids = self.model.generate(
            input_ids=input_ids,
            num_beams=num_beams,
            max_length=max_length,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            early_stopping=early_stopping,
            top_p=top_p,
            top_k=top_k,
            num_return_sequences=num_return_sequences,
        )
        # if num_return_sequences>1, then batch_decode returns batch_size * num_return_sequences results
        preds = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=skip_special_tokens)

        return preds


class T5LightningModel(BaseSeq2SeqModel):
    def __init__(
        self,
        args=None,
    ):
        """
        Args:
            tokenizer : T5/MT5/ByT5 tokenizer
            model : T5/MT5/ByT5 model
            outputdir (str, optional): output directory to save model checkpoints. Defaults to "outputs".
            save_only_last_epoch (bool, optional): If True, save just the last epoch else models are saved for every epoch
        """
        super().__init__(args)

        #  TODO 一般用不到从头初始化一个模型，暂时把这部分注释了，现在的问题是没有办法直接load_from_checkpoint，
        #  因为load前cls.model还没有初始化，这块需要思考怎么让它和lightning的接口兼容，或者我自己写一个load_from_checkpoint？
        #  if model_type == 'fengshent5':
        #      self.tokenizer = fengshenT5Tokenizer.from_pretrained(model_name)
        #  if model_type == "t5" or model_type == 'byt5':
        #      self.config = AutoConfig.from_pretrained(model_name)
        #      self.model = T5ForConditionalGeneration(self.config)
        #      #  print(self.state_dict().keys())
        #  elif model_type == "mt5":
        #      self.config = AutoConfig.from_pretrained(model_name)
        #      self.model = MT5ForConditionalGeneration(self.config)
        #  elif model_type == "fengshent5":
        #      self.config = fengshenT5Config.from_pretrained(model_name)
        #      self.model = fengshenT5ForConditionalGeneration(self.config)
        #  else:
        #      raise ValueError(f"Given model type {model_type} is not in acceptable type list [t5, mt5, byt5, fengshent5]!")

    def configure_optimizers(self):
        """ configure optimizers """
        #  paras = list(filter(lambda p: p.requires_grad, self.parameters()))
        paras = self.parameters()
        #  T5一般用Adafactor而不是Adam
        optimizer = Adafactor(paras, lr=self.hparams.lr, scale_parameter=False, relative_step=False)
        #  fine-tune时T5用warmup + fixed lr一般来说效果比较好, see https://discuss.huggingface.co/t/t5-finetuning-tips/684/3
        scheduler = get_constant_schedule_with_warmup(
            optimizer, int(self.total_step * self.hparams.warmup),
            )

        return [{
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            }
        }]

    @classmethod
    def from_pretrained(cls, args=None, model_type=None, model_name=None) -> pl.LightningModule:
        """
        loads huggingface T5/MT5 Model as class.model for training/finetuning
        Args:
        """
        model_type = args.model_type if args is not None else model_type
        model_name = args.model_name if args is not None else model_name

        if model_type == "fengshent5":
            tokenizer = fengshenT5Tokenizer.from_pretrained(model_name)
            model = fengshenT5ForConditionalGeneration.from_pretrained(
                model_name, return_dict=True
            )
        elif model_type == "t5":
            tokenizer = T5Tokenizer.from_pretrained(model_name)
            model = T5ForConditionalGeneration.from_pretrained(
                model_name, return_dict=True
            )
        elif model_type == "mt5":
            tokenizer = MT5Tokenizer.from_pretrained(model_name)
            model = MT5ForConditionalGeneration.from_pretrained(
                model_name, return_dict=True
            )

        cls_instance = cls(args=args)
        cls_instance.tokenizer = tokenizer
        cls_instance.model = model

        return cls_instance



class Seq2SeqTrainer:
    @staticmethod
    def add_trainer_specific_args(parent_parser):
        """
        Add model specific args
        Args:
            show_training_ex(int, optional): print the training examples for the batch idx. Set to -1 to disable.
        Returns:
            parent_parser
        """
        parser = parent_parser.add_argument_group('Seq2SeqTrainer')
        # * Args for model setting
        parser.add_argument('--corruption_rate', default=0.15, type=float)
        parser.add_argument('--max_extra_id', default=100, type=int)
        parser.add_argument('--source_max_token_len', default=512, type=int)
        parser.add_argument('--target_max_token_len', default=512, type=int) # int(source_max_token_len * corruption_rate * 1.5)
        # TODO 把属于data的参数放到datamodule里面
        parser.add_argument('--patience', default=10, type=int)
        #  parser.add_argument('--save_dir', default='./save', type=str)

        return parent_parser

    def __init__(self, model:BaseSeq2SeqModel, tokenizer:Union[T5Tokenizer, MT5Tokenizer]) -> None:
        """initiates a Seq2SeqTrainer class, defines training procedures for seq2seq model like T5"""
        self.model = model
        self.tokenizer = tokenizer
    
    def train_span_corruption(
        self,
        args,
        train_context:List[str],
    ):
        #TODO 不要把所有参数都放到args去访问，不然都不知道传进来哪些参数，应该在函数声明中保留所有的参数，但外面都用args.xxx传进来
        """
        Train (Unsupervised pretraining) Seq2Seq Model with Span Corruption.
        Args:
            args: contain trainer and callback parameters
            train_context(list[str]): raw train context
            source_max_token_len(int, optional): max length of source sequence
            target_max_token_len(int, optional: max length of target sequence
            batch_size(int, optional): training batch size
            max_epoch(int, optional): max number of training epochs for each corruption on the dataset
            gpu_nums(int, optional): Number of gpus used for multi-gpu training. Set to 0 to disable gpus. 
            early_stopping_patience_epochs (int, optional): monitors val_loss on epoch end and stops training, if val_loss does not improve after the specied number of epochs. set 0 to disable early stopping. Defaults to 0 (disabled)
            precision(int, optional): Precision of float used in training. 16 or 32. 
        """
        callbacks = [TQDMProgressBar(refresh_rate=100)]
        lr_callback = LearningRateMonitor(logging_interval="step")
        callbacks.append(lr_callback)
        checkpoint = ModelCheckpoint(dirpath=args.outputdir,
                                     save_top_k=-1,
                                     save_last=True,
                                     monitor='avg_train_loss',
                                     mode='min',
                                     filename='{epoch:02d}-{avg_train_loss:.4f}',
                                     )
        checkpoint.CHECKPOINT_NAME_LAST = "last-{epoch:02d}-{avg_train_loss:.4f}"
        callbacks.append(checkpoint)

        if args.patience > 0:
            early_stop_callback = EarlyStopping(
                monitor="avg_train_loss",
                min_delta=0.00,
                patience=args.patience,
                verbose=True,
                mode="min",
                check_on_train_epoch_end=True,  # Check early stopping after every train epoch, ignore multi validation in one train epoch
            )
            callbacks.append(early_stop_callback)

        logger = loggers.TensorBoardLogger(save_dir=os.path.join(args.outputdir, 'logs/'), name="default")
        trainer = pl.Trainer.from_argparse_args(
            args,
            logger=logger,
            callbacks=callbacks,
            plugins=DDPPlugin(find_unused_parameters=False),
        )

        # process corruption within ContextData
        #  train_context, val_context = train_test_split(train_context, test_size=0.2)

        self.train_dataset = ContextData(
            train_context,
            self.tokenizer,
            source_max_token_len=args.source_max_token_len,
            target_max_token_len=args.target_max_token_len,
            corruption_rate=args.corruption_rate,
            max_extra_id=args.max_extra_id,
        )
        self.data_model = LightningDataModule(args, self.train_dataset)
        # DataLoader
        #  self.train_dataloader = DataLoader(
        #      self.train_dataset,
        #      batch_size=args.batch_size,
        #      shuffle=True,
        #      num_workers=16,
        #      pin_memory=True,
        #  )

        #  self.val_dataset = ContextData(
        #      val_context,
        #      self.tokenizer,
        #      source_max_token_len=source_max_token_len,
        #      target_max_token_len=target_max_token_len,
        #      corruption_rate=corruption_rate,
        #  )
        #  self.val_dataloader = DataLoader(
        #      self.val_dataset,
        #      batch_size=batch_size,
        #      shuffle=False,
        #      num_workers=4,
        #  )

        # Train
        trainer.fit(self.model, 
                    train_dataloaders=self.data_model,
                    #  train_dataloaders=self.train_dataloader,
                    #  val_dataloaders=self.val_dataloader,
                    )

    #  def train_supervised(
    #      self,
    #      train_data,
    #      source_max_token_len:int=512,
    #      target_max_token_len:int=512,
    #      batch_size:int=8,
    #      max_epochs:int=2,
    #      gpu_nums:int=1,
    #      precision:int=32,
    #      outputdir:str=None,
    #      show_training_ex: int=-1
    #  ):
    #      """
    #      Train (supervised finetuning) Seq2Seq Model.
    #      Args:
    #          train_data(list[tuple[str, str]] or pd.Dataframe): training dataset
    #          source_max_token_len(int, optional): max length of source sequence
    #          target_max_token_len(int, optional: max length of target sequence
    #          batch_size(int, optional): training batch size
    #          max_epoch(int, optional): max number of training epochs for each corruption on the dataset
    #          gpu_nums(int, optional): Number of gpus used for multi-gpu training. Set to 0 to disable gpus.
    #          precision(int, optional): Precision of float used in training. 16 or 32.
    #          outputdir(str, optional): output directory for saving trained model.
    #      """
    #      self.T5Model = BaseSeq2SeqModel(
    #          tokenizer=self.tokenizer,
    #          model=self.model,
    #          outputdir=outputdir,
    #          show_training_ex=show_training_ex
    #      )
    #      trainer = pl.Trainer(
    #          max_epochs=max_epochs,
    #          gpus=gpu_nums,
    #          progress_bar_refresh_rate=50,
    #          precision=precision,
    #          check_val_every_n_epoch=1
    #      )
    #      self.train_dataset = SupervisedData(
    #          train_data,
    #          self.tokenizer,
    #          source_max_token_len,
    #          target_max_token_len
    #      )
    #      self.train_dataloader = DataLoader(
    #          self.train_dataset,
    #          batch_size=batch_size,
    #          shuffle=True,
    #          num_workers=2
    #      )
    #      trainer.fit(self.T5Model, train_dataloaders=self.train_dataloader)

