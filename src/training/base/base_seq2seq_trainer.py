import os
from abc import abstractmethod
from typing import Optional, Dict, Any

from transformers import (
    Trainer,
    TrainingArguments,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)

from .base_trainer import BaseTrainer


class BaseSeq2SeqTrainer(BaseTrainer):
    def __init__(self, config_path: str):
        super().__init__(config_path)
        self.encoder_tokenizer = None
        self.decoder_tokenizer = None
        self.tokenized_data = None
    
    @abstractmethod
    def load_encoder_tokenizer(self):
        pass
    
    @abstractmethod
    def load_decoder_tokenizer(self):
        pass
    
    def load_tokenizer(self):
        self.load_encoder_tokenizer()
        self.load_decoder_tokenizer()
    
    @abstractmethod
    def load_encoder(self):
        pass
    
    @abstractmethod
    def load_decoder(self):
        pass
    
    @abstractmethod
    def build_encoder_decoder_model(self):
        pass
    
    def load_model(self):
        self.load_encoder()
        self.load_decoder()
        self.build_encoder_decoder_model()
    
    @abstractmethod
    def tokenize_function(self, examples) -> Dict[str, Any]:
        pass
    
    def tokenize_dataset(self, remove_columns: Optional[list] = None, **kwargs):
        self.logger.info("Tokenizing dataset for seq2seq training")
        
        map_kwargs = {
            "batched": kwargs.get("batched", True),
        }
        if remove_columns:
            map_kwargs["remove_columns"] = remove_columns
        if "num_proc" in kwargs:
            map_kwargs["num_proc"] = kwargs["num_proc"]
        
        self.tokenized_data = self.dataset.map(
            self.tokenize_function,
            **map_kwargs
        )
        self.logger.info("Tokenization finished")
    
    def create_data_collator(self):
        self.logger.info("Creating DataCollatorForSeq2Seq")
        return DataCollatorForSeq2Seq(
            tokenizer=self.decoder_tokenizer,
            model=self.model,
            padding=True,
            pad_to_multiple_of=8
        )
    
    def create_training_args(self) -> Seq2SeqTrainingArguments:
        args_cfg = self.cfg.training_args
        self.logger.info(f"Seq2SeqTrainingArguments configured: {args_cfg}")
        
        return Seq2SeqTrainingArguments(
            output_dir=self.work_dir,
            logging_dir=self.work_dir,
            predict_with_generate=True,
            **args_cfg
        )
    
    def setup_trainer(self):
        self.logger.info("Initializing Seq2SeqTrainer")
        
        data_collator = self.create_data_collator()
        training_args = self.create_training_args()
        
        self.trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=self.tokenized_data['train'],
            eval_dataset=self.tokenized_data.get('test'),
            tokenizer=self.decoder_tokenizer,
        )
    
    def train(self):
        self.logger.info("Starting Seq2Seq training")
        self.trainer.train()
    
    def save_model(self):
        final_model_dir = os.path.join(self.work_dir, "final_model")
        self.logger.info(f"Saving final model and tokenizers to {final_model_dir}")
        
        self.model.save_pretrained(final_model_dir)
        
        # Save encoder tokenizer
        encoder_tokenizer_dir = os.path.join(final_model_dir, "encoder_tokenizer")
        if hasattr(self.encoder_tokenizer, 'save_pretrained'):
            self.encoder_tokenizer.save_pretrained(encoder_tokenizer_dir)
        
        # Save decoder tokenizer
        decoder_tokenizer_dir = os.path.join(final_model_dir, "decoder_tokenizer")
        if hasattr(self.decoder_tokenizer, 'save_pretrained'):
            self.decoder_tokenizer.save_pretrained(decoder_tokenizer_dir)
        
        self.logger.info(f"Model and tokenizers saved to {final_model_dir}")
    
    def run(self, remove_columns: Optional[list] = None, **tokenize_kwargs):
        self.load_tokenizer()
        self.load_dataset()
        self.tokenize_dataset(remove_columns=remove_columns, **tokenize_kwargs)
        self.load_model()
        self.setup_trainer()
        self.train()
        self.save_model()

