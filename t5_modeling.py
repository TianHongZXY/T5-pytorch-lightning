from matplotlib.pyplot import show
import torch
import numpy as np
import pandas as pd 
from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5Tokenizer,
    MT5ForConditionalGeneration,
    MT5TokenizerFast as MT5Tokenizer,
)
from data_preprocess import perform_span_corruption_nonseg, perform_span_corruption_seg
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

torch.cuda.empty_cache()
pl.seed_everything(42)
#path = "Langboat/mengzi-t5-base"
#tokenizer = T5Tokenizer.from_pretrained(path)
class ContextData(Dataset):
    def __init__(
        self,
        context,
        tokenizer:T5Tokenizer,
        context_segmented:bool=False,
        source_max_token_len:int=512,
        target_max_token_len:int=512,
    ):
        if not isinstance(context, pd.DataFrame):
            self.context = pd.DataFrame({'context':context})
            print(self.context)
        else:
            self.context = context
        self.tokenizer = tokenizer
        self.source_max_token_len = source_max_token_len
        self.target_max_token_len = target_max_token_len
        self.context_segmented = context_segmented
        
    def __len__(self):
        return len(self.context)

    def __getitem__(self, index: int):
        """ returns dictionary of input tensors to feed into T5/MT5 model"""
        context_row = self.context.iloc[index]
        # span corruption
        if self.context_segmented:
            context_row = perform_span_corruption_seg(context_row, noise_prob=0.15, max_extra_id=100)
        else:
            context_row = perform_span_corruption_nonseg(context_row, noise_prob=0.12, max_extra_id=100)
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


class LightningModel(pl.LightningModule):
    def __init__(self, tokenizer, model, dataset, outputdir: str = None, show_training_ex: int=-1):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        #self.tokenizer.padding_size= 'left'
        self.outputdir = outputdir
        self.show_training_ex = show_training_ex
    
    def forward(self, input_ids, attention_mask, decoder_attention_mask, labels=None):
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
        attention_mask = batch["source_text_attention_mask"]
        labels = batch["labels"]
        labels_attention_mask = batch["labels_attention_mask"]
        loss, logits = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=labels_attention_mask,
            labels=labels,
        )
        if self.show_training_ex > -1 and batch_idx == self.show_training_ex:
            prediction = torch.argmax(logits, dim=2)
            predicted_tokens = self.tokenizer.batch_decode(prediction, skip_special_tokens=False)
            labels_tokens = self.tokenizer.batch_decode(labels, skip_special_tokens=False)
            prediction_label_pair = list(zip(predicted_tokens, labels_tokens))
            print(prediction_label_pair[0])
        self.log("train_loss", loss, prog_bar=False, logger=True)
        return {'loss': loss}
    
        
    def configure_optimizers(self):
        """ configure optimizers """
        return AdamW(self.parameters(), lr=0.0001)
    
    def training_epoch_end(self, training_step_outputs):
        avg_training_loss = np.round(
            torch.mean(torch.stack([x["loss"] for x in training_step_outputs])).item(),
            5,
        )
        print("average training loss over one epoch:{:.4f}".format(avg_training_loss))
    

class SimpleT5():
    def __init__(self) -> None:
        """ initiates SimpleT5 class """
        print("simple t5 model instantiated")
        pass
    
    def from_pretrained(self, model_type="t5", model_name="t5-base") -> None:
        """
        loads T5/MT5 Model model for training/finetuning

        Args:
            model_type (str, optional): "t5" or "mt5" . Defaults to "t5".
            model_name (str, optional): exact model architecture name, "t5-base" or "t5-large". Defaults to "t5-base".
        """
        if model_type == "t5":
            self.tokenizer = T5Tokenizer.from_pretrained(f"{model_name}")
            self.model = T5ForConditionalGeneration.from_pretrained(
                f"{model_name}", return_dict=True
            )
        
        elif model_type == "mt5":
            self.tokenizer = MT5Tokenizer.from_pretrained(f"{model_name}")
            self.model = MT5ForConditionalGeneration.from_pretrained(
                f"{model_name}", return_dict=True
            )
        """
        elif model_type == "byt5":
            self.tokenizer = ByT5Tokenizer.from_pretrained(f"{model_name}")
            self.model = T5ForConditionalGeneration.from_pretrained(
                f"{model_name}", return_dict=True
            )
        """
    
    def load_model(
        self, model_type: str = "t5", model_dir: str = "outputs", use_gpu: bool = False
    ):
        """
        loads a checkpoint for inferencing/prediction"""
        if model_type == "t5":
            self.model = T5ForConditionalGeneration.from_pretrained(f"{model_dir}")
            self.tokenizer = T5Tokenizer.from_pretrained("google/mt5-large")
        
        elif model_type == "mt5":
            self.model = MT5ForConditionalGeneration.from_pretrained(f"{model_dir}")
            self.tokenizer = MT5Tokenizer.from_pretrained("google/mt5-large")
        """
        elif model_type == "byt5":
            self.model = T5ForConditionalGeneration.from_pretrained(f"{model_dir}")
            self.tokenizer = ByT5Tokenizer.from_pretrained(f"{model_dir}")
        """
        if use_gpu:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                raise "exception ---> no gpu found. set use_gpu=False, to use CPU"
        else:
            self.device = torch.device("cpu")
        self.model = self.model.to(self.device)
    
    def train_span_corruption(
        self,
        train_context,
        context_segmented:bool=False,
        source_max_token_len:int=512,
        target_max_token_len:int=512,
        batch_size:int=8,
        max_epochs:int=50,
        gpu_nums:int=1,
        precision:int=32,
        outputdir:str=None,
        show_training_ex: int=-1
    ):
        """
        Train (Unsupervised pretraining) T5 Model with Span Corruption.
        Args:
            train_context(list[str]): training dataset
            context_segmented(bool, optional): True if train_context is already segmented. 
            source_max_token_len(int, optional): max length of source sequence
            target_max_token_len(int, optional: max length of target sequence
            batch_size(int, optional): training batch size
            max_epoch(int, optional): max number of training epochs for each corruption on the dataset
            gpu_nums(int, optional): Number of gpus used for multi-gpu training. Set to 0 to disable gpus. 
            precision(int, optional): Precision of float used in training. 16 or 32. 
            outputdir(str, optional): output directory for saving trained model. 
            show_training_ex(int, optional): print the training examples for the batch idx. Set to -1 to disable.
        """
        self.T5Model = LightningModel(
            tokenizer=self.tokenizer, 
            model=self.model, 
            outputdir=outputdir,
            show_training_ex=show_training_ex
        )
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            gpus=gpu_nums,
            progress_bar_refresh_rate=100,
            precision=precision,
            check_val_every_n_epoch=1,
            strategy='deepspeed_stage_2',
        )
        # process corruption within ContextData
        self.train_dataset = ContextData(
            train_context,
            self.tokenizer,
            context_segmented=context_segmented,
            source_max_token_len=source_max_token_len,
            target_max_token_len=target_max_token_len
        )
        # DataLoader
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2
        )
        # Train
        trainer.fit(self.T5Model, train_dataloaders=self.train_dataloader)


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
        Train (supervised finetuning) T5 Model.
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
        self.T5Model = LightningModel(
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

    def inference(
        self, 
        source_texts,
        max_length: int = 512,
        num_return_sequences: int = 1,
        num_beams: int = 2,
        top_k: int = 0,
        top_p: float = 1,
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
            top_k (int, optional): Defaults to 50.
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
        input_ids = input_ids.to(self.device)
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

