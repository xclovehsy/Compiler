"""Base trainer module with common training logic."""
import os
import argparse
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any

from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from datasets import load_from_disk

from src.config import Config, load_config
from src.utils.utils import get_logger


class BaseMLMTrainer(ABC):
    """Base class for Masked Language Model training."""
    
    def __init__(self, config_path: str):
        self.cfg = Config(config_path)
        self.work_dir = self.cfg.create_work_dir()
        self.logger = get_logger(self.work_dir)
        
        self.model = None
        self.tokenizer = None
        self.dataset = None
        self.tokenized_data = None
        self.trainer = None
        
        self.logger.info(f"Work directory created at {self.work_dir}")
    
    @abstractmethod
    def load_tokenizer(self):
        """Load the tokenizer. Override in subclasses."""
        pass
    
    def load_model(self):
        """Load the model."""
        self.logger.info(f"Loading model from {self.cfg.model_id}")
        self.model = AutoModelForMaskedLM.from_pretrained(self.cfg.model_id)
    
    def load_dataset(self):
        """Load the dataset."""
        self.logger.info(f"Loading dataset from {self.cfg.data_dir}")
        self.dataset = load_from_disk(self.cfg.data_dir)
        self.logger.info(f"Dataset loaded with {len(self.dataset)} examples")
    
    @abstractmethod
    def tokenize_function(self, examples) -> Dict[str, Any]:
        """Tokenize examples. Override in subclasses."""
        pass
    
    def tokenize_dataset(self, remove_columns: Optional[list] = None, **kwargs):
        """Tokenize the dataset."""
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
        """Create the data collator for MLM."""
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
        """Create training arguments."""
        args_cfg = self.cfg.training_args
        self.logger.info(f"TrainingArguments configured: {args_cfg}")
        
        return TrainingArguments(
            output_dir=self.work_dir,
            logging_dir=self.work_dir,
            **args_cfg
        )
    
    def setup_trainer(self):
        """Setup the Trainer."""
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
        """Run training."""
        self.logger.info("Starting training")
        self.trainer.train()
    
    def save_model(self):
        """Save the final model and tokenizer."""
        final_model_dir = os.path.join(self.work_dir, "final_model")
        self.logger.info(f"Saving final model and tokenizer to {final_model_dir}")
        
        self.model.save_pretrained(final_model_dir)
        if hasattr(self.tokenizer, 'save_pretrained'):
            self.tokenizer.save_pretrained(final_model_dir)
        
        self.logger.info(f"Model and tokenizer saved to {final_model_dir}")
    
    def run(self, remove_columns: Optional[list] = None, **tokenize_kwargs):
        """Run the complete training pipeline."""
        self.load_tokenizer()
        self.load_dataset()
        self.tokenize_dataset(remove_columns=remove_columns, **tokenize_kwargs)
        self.load_model()
        self.setup_trainer()
        self.train()
        self.save_model()


class StandardMLMTrainer(BaseMLMTrainer):
    """Standard MLM trainer using HuggingFace tokenizer."""
    
    def load_tokenizer(self):
        """Load the HuggingFace tokenizer."""
        tokenizer_id = self.cfg.tokenizer_id or self.cfg.model_id
        self.logger.info(f"Loading tokenizer from {tokenizer_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
        self.logger.info(f"Tokenizer vocab size: {len(self.tokenizer)}")
    
    def tokenize_function(self, examples):
        """Standard tokenization function."""
        # 默认使用 'llvm' 字段，可以通过配置覆盖
        text_field = self.cfg.get("data.text_field", "llvm")
        return self.tokenizer(
            examples[text_field],
            padding=True,
            truncation=True,
            max_length=self.cfg.max_length,
            return_tensors="pt"
        )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to the YAML config file"
    )
    return parser.parse_args()

