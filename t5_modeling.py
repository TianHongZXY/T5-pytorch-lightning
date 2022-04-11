import os
import argparse
import torch
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
from typing import List, Union, Tuple, Optional, Dict
from fengshen import T5Config as fengshenT5Config
from fengshen import T5ForConditionalGeneration as fengshenT5ForConditionalGeneration
from fengshen import T5Tokenizer as fengshenT5Tokenizer


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
        parser.add_argument('--model_type', default=None, type=str, help="Acceptable model type: [t5, mt5, fengshent5]")
        parser.add_argument('--model_name', default=None, type=str)
        parser.add_argument('--training_objective', default="span_corruption", type=str, help="Acceptable objectives: [span_corruption, prefix_lm]")

        return parent_parser

    def __init__(
        self,
        args=None,
    ):
        """
        Args:
            tokenizer : T5/MT5/ByT5 tokenizer
            model : T5/MT5/ByT5 model
            save_dir (str, optional): output directory to save model checkpoints. Defaults to "outputs".
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
            train_loader = self.trainer.datamodule.train_dataloader()
            num_gpus = self.trainer.gpus if self.trainer.gpus is not None else 0
            self.total_step = int(self.trainer.max_epochs * len(train_loader) /
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
            #  Convert loss mask token into pad token
            labels = torch.where(labels != -100, labels, self.tokenizer.pad_token_id)
            labels_tokens = self.tokenizer.decode(labels[0], skip_special_tokens=False)
            print('-' * 50)
            print('input_token:     ', input_tokens.replace(self.tokenizer.pad_token, ''))
            print('-' * 50)
            print('predicted_tokens:', predicted_tokens.replace(self.tokenizer.pad_token,''))
            print('-' * 50)
            print('labels_tokens:   ', labels_tokens.replace(self.tokenizer.pad_token,''))
            print('-' * 50)

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
                'frequency': 1,
            }
        }]

    def training_epoch_end(self, training_step_outputs):
        """ save tokenizer and model on epoch end """
        self.average_training_loss = np.round(
            torch.mean(torch.stack([x["loss"] for x in training_step_outputs])).item(),
            4,
        )
        self.log("avg_train_loss", self.average_training_loss, prog_bar=True, logger=True, on_epoch=True)
        #  path = f"{self.hparams.save_dir}/epoch-{self.current_epoch}-train-loss-{str(self.average_training_loss)}"
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
        generates prediction for model
        Args:
            source_texts (list[str]): sequence of texts for generating predictions
            max_length (int, optional): max token length of prediction. Defaults to 512.
            num_return_sequences (int, optional): number of predictions to be returned. Defaults to 1.
            num_beams (int, optional): number of beams. Defaults to 1.
            top_k (int, optional): Defaults to 50.
            top_p (float, optional): Defaults to 0.9.
            do_sample (bool, optional): Defaults to True.
            repetition_penalty (float, optional): Defaults not used.
            no_repeat_ngram_size (int, optional): Defaults not used.
            length_penalty (float, optional): Defaults not used.
            early_stopping (bool, optional): Defaults to True.
            skip_special_tokens (bool, optional): Defaults to True.

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
        predictions = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=skip_special_tokens)

        return predictions


class T5LightningModel(BaseSeq2SeqModel):
    def __init__(
        self,
        args=None,
    ):
        """
        Args:
            tokenizer : T5/MT5/ByT5 tokenizer
            model : T5/MT5/ByT5 model
            save_dir (str, optional): output directory to save model checkpoints. Defaults to "outputs".
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
        paras = list(filter(lambda p: p.requires_grad, self.parameters()))
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
                'frequency': 1,
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
            tokenizer = T5Tokenizer.from_pretrained(model_name)
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
            patience(int, optional): monitors metric on epoch end and stops training, if metric does not improve after the specied number of epochs. set 0 to disable early stopping. Defaults to 3
            save_dir (str, optional): root save directory
            Additional args from pytorch-lightning Trainer:
                max_epoch(int, optional): max number of training epochs for each corruption on the dataset
                gpu_nums(int, optional): Number of gpus used for multi-gpu training. Set to 0 to disable gpus.
                precision(int, optional): Precision of float used in training. 16 or 32.
                strategy(str, optional): Supports different training strategies with aliases
                    as well custom training type plugins.
                etc.
        Returns:
            parent_parser
        """
        parser = parent_parser.add_argument_group('Seq2SeqTrainer')
        # * Args for trainer setting
        parser.add_argument('--patience', default=3, type=int)
        parser.add_argument('--save_dir', default='./outputs', type=str)
        parser.add_argument('--save_top_k', default=-1, type=int)
        parser.add_argument('--monitor', default='val_loss', type=str)
        parser.add_argument('--mode', default='min', type=str)

        parent_parser = pl.Trainer.add_argparse_args(parent_parser=parent_parser)

        return parent_parser

    def __init__(self, args, model:BaseSeq2SeqModel) -> None:
        """
        initiates a Seq2SeqTrainer class, defines training procedures for seq2seq model like T5
        Args:
            args: contain trainer and callback parameters
            model: seq2seq model to train
        """
        self.model = model
        callbacks = [TQDMProgressBar(refresh_rate=100)]
        lr_callback = LearningRateMonitor(logging_interval="step")
        callbacks.append(lr_callback)
        checkpoint = ModelCheckpoint(dirpath=args.save_dir,
                                     save_top_k=args.save_top_k,
                                     save_last=True,
                                     monitor=args.monitor,
                                     mode=args.mode,
                                     filename='{epoch:02d}-{avg_train_loss:.4f}',
                                     )
        checkpoint.CHECKPOINT_NAME_LAST = "last-{epoch:02d}-{avg_train_loss:.4f}"
        callbacks.append(checkpoint)

        if args.patience > 0:
            early_stop_callback = EarlyStopping(
                monitor=args.monitor,
                min_delta=0.00,
                patience=args.patience,
                verbose=True,
                mode=args.mode,
                check_on_train_epoch_end=True,  # Check early stopping after every train epoch, ignore multi validation in one train epoch
            )
            callbacks.append(early_stop_callback)

        logger = loggers.TensorBoardLogger(save_dir=os.path.join(args.save_dir, 'logs/'), name="default")
        accumulate_grad_batches = args.global_batch_size // (max(args.gpus, 1) * args.micro_batch_size)
        if not (args.global_batch_size % (max(args.gpus, 1) * args.micro_batch_size) == 0):
            #  Use larger global batch size instead smaller because small batch size with large learning rate may hurt performance.
            accumulate_grad_batches += 1
            print(f"Global batch size can not be divided by (number of gpus * mirco batch size), "
                  f"using global batch size = {(accumulate_grad_batches + 1) * (max(args.gpus, 1) * args.micro_batch_size)} instead.")
        self.trainer = pl.Trainer.from_argparse_args(
            args,
            logger=logger,
            callbacks=callbacks,
            plugins=DDPPlugin(find_unused_parameters=False),
            accumulate_grad_batches=accumulate_grad_batches,
        )

    def train(self, data_model: pl.LightningDataModule):
        """
        Train seq2seq model with given data model.
        Args:
            data_model: lightning data module
        """
        # Train
        self.trainer.fit(self.model, datamodule=data_model)

