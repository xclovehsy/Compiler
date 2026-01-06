from typing import Optional, Dict, Any
from abc import abstractmethod
import os

from transformers import AutoModelForMaskedLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_from_disk

from src.training.base.base_trainer import BaseTrainer


class BaseMLMTrainer(BaseTrainer):
    
    def __init__(self, config_path: str):
        super().__init__(config_path)
        self.tokenized_data = None
    
    def load_model(self):
        self.logger.info(f"Loading model from {self.cfg.model_id}")
        self.model = AutoModelForMaskedLM.from_pretrained(self.cfg.model_id)
    
    @abstractmethod
    def tokenize_function(self, examples) -> Dict[str, Any]:
        pass
    
    def tokenize_dataset(self, remove_columns: Optional[list] = None, **kwargs):
        self.logger.info(f"Tokenizing dataset with max_length={self.cfg.max_length}")
        
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
        self.logger.info(
            f"Creating DataCollatorForLanguageModeling with mlm_probability={self.cfg.mlm_probability}"
        )
        return DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=True,
            mlm_probability=self.cfg.mlm_probability,
            pad_to_multiple_of=8
        )
    
    def create_training_args(self) -> TrainingArguments:
        args_cfg = self.cfg.training_args
        self.logger.info(f"TrainingArguments configured: {args_cfg}")
        
        return TrainingArguments(
            output_dir=self.work_dir,
            logging_dir=self.work_dir,
            **args_cfg
        )
    
    def setup_trainer(self):
        self.logger.info("Initializing Trainer")
        
        data_collator = self.create_data_collator()
        training_args = self.create_training_args()
        
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=self.tokenized_data['train'],
            eval_dataset=self.tokenized_data.get('test')
        )
    
    def train(self):
        self.logger.info("Starting MLM training")
        self.trainer.train()
    
    def save_model(self):
        final_model_dir = os.path.join(self.work_dir, "final_model")
        self.logger.info(f"Saving final model and tokenizer to {final_model_dir}")
        
        self.model.save_pretrained(final_model_dir)
        if hasattr(self.tokenizer, 'save_pretrained'):
            self.tokenizer.save_pretrained(final_model_dir)
        
        self.logger.info(f"Model and tokenizer saved to {final_model_dir}")
    
    def run(self, remove_columns: Optional[list] = None, **tokenize_kwargs):
        self.load_tokenizer()
        self.load_dataset()
        self.tokenize_dataset(remove_columns=remove_columns, **tokenize_kwargs)
        self.load_model()
        self.setup_trainer()
        self.train()
        self.save_model()