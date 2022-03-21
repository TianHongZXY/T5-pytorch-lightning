import os
import torch
import numpy as np
import pandas as pd
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AdamW,
    T5ForConditionalGeneration,
    T5Tokenizer,
    MT5ForConditionalGeneration,
    MT5TokenizerFast as MT5Tokenizer,
    ByT5Tokenizer,
    PreTrainedTokenizer,
    PreTrainedModel,
    T5PreTrainedModel,
)
from data_preprocess import (
    perform_span_corruption_nonseg,
    perform_span_corruption_seg,
)
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning import loggers
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, TQDMProgressBar
from sklearn.model_selection import train_test_split
from torchsnooper import snoop
from typing import List, Union, Tuple, Optional


class LightningDataModule(pl.LightningDataModule):
    """ PyTorch Lightning data class """

    def __init__(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        tokenizer: PreTrainedTokenizer,
        batch_size: int = 4,
        source_max_token_len: int = 512,
        target_max_token_len: int = 512,
        num_workers: int = 2,
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

        self.train_df = train_df
        self.test_df = test_df
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.source_max_token_len = source_max_token_len
        self.target_max_token_len = target_max_token_len
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.train_dataset = PyTorchDataModule(
            self.train_df,
            self.tokenizer,
            self.source_max_token_len,
            self.target_max_token_len,
        )
        self.test_dataset = PyTorchDataModule(
            self.test_df,
            self.tokenizer,
            self.source_max_token_len,
            self.target_max_token_len,
        )

    def train_dataloader(self):
        """ training dataloader """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        """ test dataloader """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        """ validation dataloader """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

#TODO 然后把data统一到一个datamodule里，然后探索一下span corruption
class ContextData(Dataset):
    def __init__(
        self,
        context,
        tokenizer,
        context_segmented:bool=False,
        source_max_token_len:int=512,
        target_max_token_len:int=512,
        corruption_rate:float=0.15,
    ):
        #  if not isinstance(context, pd.DataFrame):
        #      self.context = pd.DataFrame({'context':context})
        #      print(self.context.head())
        #  else:
        self.context = []
        print("Initial number of context: ", len(context))
        for i in range(len(context)):
            cur_context = context[i]
            while len(cur_context) > source_max_token_len:
                window_context = cur_context[:source_max_token_len]
                self.context.append(window_context)
                cur_context = cur_context[source_max_token_len:]
            self.context.append(cur_context)
        print("After segmentation, number of context: ", len(self.context))

        self.tokenizer = tokenizer
        self.source_max_token_len = source_max_token_len
        self.target_max_token_len = target_max_token_len
        self.context_segmented = context_segmented
        self.corruption_rate = corruption_rate
        
    def __len__(self):
        return len(self.context)

    def __getitem__(self, index: int):
        """ returns dictionary of input tensors to feed into T5/MT5 model"""
        #  context_row = self.context.iloc[index]
        context_row = self.context[index]
        str_context = ''.join(context_row)

        # supervised data
        if str_context.startswith("问题:"):
            str_context = str_context.split("答案:")
            context_row = [str_context[0], "答案:" + str_context[1]]
        # span corruption data
        else:
            if self.context_segmented:
                context_row = perform_span_corruption_seg(context_row, noise_prob=self.corruption_rate, max_extra_id=100)
            else:
                context_row = perform_span_corruption_nonseg(context_row, noise_prob=self.corruption_rate, max_extra_id=100)
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
        #labels[labels==self.tokenizer.pad_token_id] = -100
        return dict(
            source_text=source_text,
            target_text=target_text,
            source_text_input_ids=source_text_encoding["input_ids"].flatten(),
            source_text_attention_mask=source_text_encoding["attention_mask"].flatten(),
            labels=labels.flatten(),
            labels_attention_mask=target_text_encoding["attention_mask"].flatten(),
        )

class SupervisedData(Dataset):
    def __init__(
        self,
        data,
        tokenizer:T5Tokenizer,
        source_max_token_len:int=512,
        target_max_token_len:int=512,
    ):
        if not isinstance(data, pd.DataFrame):
            self.data = pd.DataFrame.from_records(data, columns=['source_text','target_text'])
        else:
            self.data = data
        self.tokenizer = tokenizer
        self.source_max_token_len = source_max_token_len
        self.target_max_token_len = target_max_token_len
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index: int):
        """ returns dictionary of input tensors to feed into T5/MT5 model"""
        data_row = self.data.iloc[index]
        source_text = data_row['source_text']
        target_text = data_row['target_text']
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
        #labels[labels==self.tokenizer.pad_token_id] = -100
        return dict(
            source_text=source_text,
            target_text=target_text,
            source_text_input_ids=source_text_encoding["input_ids"].flatten(),
            source_text_attention_mask=source_text_encoding["attention_mask"].flatten(),
            labels=labels.flatten(),
            labels_attention_mask=target_text_encoding["attention_mask"].flatten(),
        )


class BaseSeq2SeqModel(pl.LightningModule):
    def __init__(
        self,
        model_type=None,
        model_name=None,
        show_training_ex = -1,
    ):
        """
        initiates a PyTorch Lightning Seq2Seq base model, defines training and evaluation steps
        Args:
            tokenizer : T5/MT5/ByT5 tokenizer
            model : T5/MT5/ByT5 model
            outputdir (str, optional): output directory to save model checkpoints. Defaults to "outputs".
            save_only_last_epoch (bool, optional): If True, save just the last epoch else models are saved for every epoch
        """
        super().__init__()
        self.average_training_loss = None
        self.average_validation_loss = None
        self.show_training_ex = show_training_ex

        if model_type is not None:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if model_type == "t5" or model_type == 'byt5':
            self.config = AutoConfig.from_pretrained(model_name)
            self.model = T5ForConditionalGeneration(self.config)
            #  print(self.state_dict().keys())
        elif model_type == "mt5":
            self.config = AutoConfig.from_pretrained(model_name)
            self.model = MT5ForConditionalGeneration(self.config)
        elif model_type is None:
            pass
        else:
            raise ValueError(f"Given model type {model_type} is not in acceptable type list [t5, mt5, byt5]!")

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
        attention_mask = batch["source_text_attention_mask"]
        labels = batch["labels"]
        labels_attention_mask = batch["labels_attention_mask"]

        loss, logits = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=labels_attention_mask,
            labels=labels,
        )

        if self.show_training_ex > -1 and batch_idx % self.show_training_ex == 0:
            prediction = torch.argmax(logits, dim=2)
            #  input_tokens = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
            #  predicted_tokens = self.tokenizer.batch_decode(prediction, skip_special_tokens=True)
            #  labels_tokens = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
            input_tokens = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
            predicted_tokens = self.tokenizer.decode(prediction[0], skip_special_tokens=True)
            labels_tokens = self.tokenizer.decode(labels[0], skip_special_tokens=True)
            print('-' * 50)
            print('input_token:     ', input_tokens)
            print('-' * 50)
            print('predicted_tokens:', predicted_tokens)
            print('-' * 50)
            print('labels_tokens:   ', labels_tokens)
            print('-' * 50)
            # prediction_label_pair = list(zip(predicted_tokens, labels_tokens))
            # print(prediction_label_pair[0])

        self.log("train_loss", loss, prog_bar=True, logger=True, on_step=True, batch_size=batch_size)
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
        return AdamW(self.parameters(), lr=0.0001)

    def training_epoch_end(self, training_step_outputs):
        """ save tokenizer and model on epoch end """
        self.average_training_loss = np.round(
            torch.mean(torch.stack([x["loss"] for x in training_step_outputs])).item(),
            4,
        )
        self.log("avg_train_loss", self.average_training_loss, prog_bar=True, logger=True, on_epoch=True)
        #  path = f"{self.outputdir}/simplet5-epoch-{self.current_epoch}-train-loss-{str(self.average_training_loss)}-val-loss-{str(self.average_validation_loss)}"
        #  self.tokenizer.save_pretrained(path)
        #  self.model.save_pretrained(path)

    def validation_epoch_end(self, validation_step_outputs):
        _loss = [x.cpu() for x in validation_step_outputs]
        self.average_validation_loss = np.round(
            torch.mean(torch.stack(_loss)).item(),
            4,
        )
        self.log("average_validation_loss", self.average_validation_loss, prog_bar=True, logger=True, on_epoch=True)

    @classmethod
    def from_pretrained(cls, model_type="t5", model_name="t5-base") -> pl.LightningModule:
        """
        loads huggingface T5/MT5 Model as class.model for training/finetuning
        Args:
            model_type (str, optional): "t5" or "mt5" . Defaults to "t5".
            model_name (str, optional): exact model architecture name, "t5-base" or "t5-large". Defaults to "t5-base".
        """
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        if model_type == "t5" or model_type == "byt5":
            model = T5ForConditionalGeneration.from_pretrained(
                model_name, return_dict=True
            )
        elif model_type == "mt5":
            model = MT5ForConditionalGeneration.from_pretrained(
                model_name, return_dict=True
            )
        
        cls_instance = cls()
        cls_instance.tokenizer = tokenizer
        cls_instance.model = model

        return cls_instance

    def inference(
        self, 
        source_texts,
        max_length: int = 512,
        num_return_sequences: int = 1,
        num_beams: int = 2,
        top_k: int = 100,
        top_p: float = 0.95,
        do_sample: bool = True,
        repetition_penalty: float = 2.5,
        no_repeat_ngram_size: int = 3,
        length_penalty: float = 1.0,
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


class Seq2SeqTrainer:
    def __init__(self, model:BaseSeq2SeqModel, tokenizer:Union[T5Tokenizer, MT5Tokenizer]) -> None:
        """initiates a Seq2SeqTrainer class, defines training procedures for seq2seq model like T5"""
        self.model = model
        self.tokenizer = tokenizer
    
    def train_span_corruption(
        self,
        train_context:Union[List[str], List[List[str]]],
        context_segmented:bool=False,
        source_max_token_len:int=512,
        target_max_token_len:int=512,
        batch_size:int=8,
        max_epochs:int=50,
        gpu_nums:int=1,
        precision:int=32,
        early_stopping_patience_epochs:int=0,  # 0 to disable early stopping feature
        outputdir:str=None,
        show_training_ex:int=-1,
        corruption_rate:float=0.15,
    ):
        """
        Train (Unsupervised pretraining) Seq2Seq Model with Span Corruption.
        Args:
            train_context(list[list[str]] if segmented, else list[str]): training dataset
            context_segmented(bool, optional): True if train_context is already segmented. 
            source_max_token_len(int, optional): max length of source sequence
            target_max_token_len(int, optional: max length of target sequence
            batch_size(int, optional): training batch size
            max_epoch(int, optional): max number of training epochs for each corruption on the dataset
            gpu_nums(int, optional): Number of gpus used for multi-gpu training. Set to 0 to disable gpus. 
            early_stopping_patience_epochs (int, optional): monitors val_loss on epoch end and stops training, if val_loss does not improve after the specied number of epochs. set 0 to disable early stopping. Defaults to 0 (disabled)
            precision(int, optional): Precision of float used in training. 16 or 32. 
            outputdir(str, optional): output directory for saving trained model. 
            show_training_ex(int, optional): print the training examples for the batch idx. Set to -1 to disable.
        """
        callbacks = [TQDMProgressBar(refresh_rate=100)]
        checkpoint = ModelCheckpoint(dirpath=outputdir,
                                     save_top_k=3,
                                     save_last=True,
                                     monitor='avg_train_loss',
                                     mode='min',
                                     filename='{epoch:02d}-{avg_train_loss:.4f}')
        checkpoint.CHECKPOINT_NAME_LAST = "last-{epoch:02d}-{avg_train_loss:.4f}"
        callbacks.append(checkpoint)

        if early_stopping_patience_epochs > 0:
            early_stop_callback = EarlyStopping(
                monitor="avg_train_loss",
                min_delta=0.00,
                patience=early_stopping_patience_epochs,
                verbose=True,
                mode="min",
                check_on_train_epoch_end=True,  # Check early stopping after every train epoch, ignore multi validation in one train epoch
            )
            callbacks.append(early_stop_callback)

        logger = loggers.TensorBoardLogger(save_dir=os.path.join(outputdir, 'logs/'), name="default")
        trainer = pl.Trainer(
            logger=logger,
            callbacks=callbacks,
            max_epochs=max_epochs,
            gpus=gpu_nums,
            precision=precision,
            check_val_every_n_epoch=1,
            strategy='deepspeed_stage_2',
        )
        # process corruption within ContextData
        #  train_context, val_context = train_test_split(train_context, test_size=0.2)

        if not context_segmented:
            for i in range(len(train_context)):
                train_context[i] = self.tokenizer.tokenize(train_context[i])
            context_segmented = True

        self.train_dataset = ContextData(
            train_context,
            self.tokenizer,
            context_segmented=context_segmented,
            source_max_token_len=source_max_token_len,
            target_max_token_len=target_max_token_len,
            corruption_rate=corruption_rate,
        )
        # DataLoader
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
        )
        #  self.val_dataset = ContextData(
        #      val_context,
        #      self.tokenizer,
        #      context_segmented=context_segmented,
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
                    train_dataloaders=self.train_dataloader, 
                    #  val_dataloaders=self.val_dataloader,
                    )

    def train_supervised(
        self,
        train_data,
        source_max_token_len:int=512,
        target_max_token_len:int=512,
        batch_size:int=8,
        max_epochs:int=2,
        gpu_nums:int=1,
        precision:int=32,
        outputdir:str=None,
        show_training_ex: int=-1
    ):
        """
        Train (supervised finetuning) Seq2Seq Model.
        Args:
            train_data(list[tuple[str, str]] or pd.Dataframe): training dataset
            source_max_token_len(int, optional): max length of source sequence
            target_max_token_len(int, optional: max length of target sequence
            batch_size(int, optional): training batch size
            max_epoch(int, optional): max number of training epochs for each corruption on the dataset
            gpu_nums(int, optional): Number of gpus used for multi-gpu training. Set to 0 to disable gpus. 
            precision(int, optional): Precision of float used in training. 16 or 32. 
            outputdir(str, optional): output directory for saving trained model. 
        """
        self.T5Model = BaseSeq2SeqModel(
            tokenizer=self.tokenizer, 
            model=self.model, 
            outputdir=outputdir,
            show_training_ex=show_training_ex
        )
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            gpus=gpu_nums,
            progress_bar_refresh_rate=50,
            precision=precision,
            check_val_every_n_epoch=1
        )
        self.train_dataset = SupervisedData(
            train_data,
            self.tokenizer,
            source_max_token_len,
            target_max_token_len
        )
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=2
        )
        trainer.fit(self.T5Model, train_dataloaders=self.train_dataloader)

