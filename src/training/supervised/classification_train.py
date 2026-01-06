"""Classification training script.

Supervised learning for downstream classification tasks:
- Device mapping (CPU/GPU selection)
- Thread coarsening factor prediction
- Application classification (POJ-104)
"""
import os
from typing import Dict, Any, Optional

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from src.training.base import BaseTrainer
from src.training.base.base_trainer import parse_args
from src.model import Inst2VecTokenizer


class ClassificationTrainer(BaseTrainer):
    """Trainer for classification tasks.
    
    Supports various downstream tasks:
    - Device mapping: Predicting optimal device (CPU/GPU)
    - Thread coarsening: Predicting coarsening factor
    - Application classification: Classifying program functionality
    """
    
    def __init__(self, config_path: str):
        super().__init__(config_path)
        self.tokenized_data = None
        self.num_labels = None
    
    def load_tokenizer(self):
        """Load the tokenizer."""
        tokenizer_id = self.cfg.tokenizer_id or self.cfg.model_id
        tokenizer_type = self.cfg.get("model.tokenizer_type", "auto")
        
        self.logger.info(f"Loading tokenizer from {tokenizer_id}")
        
        if tokenizer_type == "inst2vec":
            self.tokenizer = Inst2VecTokenizer.from_pretrained(tokenizer_id)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
        
        self.logger.info(f"Tokenizer vocab size: {len(self.tokenizer)}")
    
    def load_model(self):
        """Load the classification model."""
        self.num_labels = self.cfg.get("model.num_labels", 2)
        
        self.logger.info(f"Loading model from {self.cfg.model_id} with {self.num_labels} labels")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.cfg.model_id,
            num_labels=self.num_labels
        )
    
    def tokenize_function(self, examples) -> Dict[str, Any]:
        """Tokenize examples for classification."""
        text_field = self.cfg.get("data.text_field", "llvm")
        label_field = self.cfg.get("data.label_field", "label")
        label_offset = self.cfg.get("data.label_offset", 0)  # For 1-indexed labels
        
        tokenized = self.tokenizer(
            examples[text_field],
            padding=True,
            truncation=True,
            max_length=self.cfg.max_length,
            return_tensors="pt"
        )
        
        # Handle label offset (e.g., POJ-104 uses 1-indexed labels)
        labels = int(examples[label_field]) - label_offset
        tokenized['labels'] = labels
        
        return tokenized
    
    def tokenize_dataset(self, remove_columns: Optional[list] = None, **kwargs):
        """Tokenize the dataset for classification."""
        self.logger.info(f"Tokenizing dataset with max_length={self.cfg.max_length}")
        
        self.tokenized_data = self.dataset.map(
            self.tokenize_function,
            batched=False,
            num_proc=kwargs.get("num_proc", 32),
            remove_columns=remove_columns or self.dataset['train'].column_names
        )
        self.logger.info("Tokenization finished")
    
    @staticmethod
    def compute_metrics(eval_pred):
        """Compute classification metrics."""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted'
        )
        accuracy = accuracy_score(labels, predictions)
        
        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
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
        """Setup the classification Trainer."""
        self.logger.info("Initializing Classification Trainer")
        
        training_args = self.create_training_args()
        
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.tokenized_data['train'],
            eval_dataset=self.tokenized_data.get('val') or self.tokenized_data.get('test'),
            compute_metrics=self.compute_metrics,
        )
    
    def train(self):
        """Run classification training."""
        self.logger.info("Starting classification training")
        self.trainer.train()
    
    def save_model(self):
        """Save the final model and tokenizer."""
        final_model_dir = os.path.join(self.work_dir, "final_model")
        self.logger.info(f"Saving final model and tokenizer to {final_model_dir}")
        
        self.trainer.save_model(final_model_dir)
        if hasattr(self.tokenizer, 'save_pretrained'):
            self.tokenizer.save_pretrained(final_model_dir)
        
        self.logger.info(f"Model and tokenizer saved to {final_model_dir}")
    
    def run(self, remove_columns: Optional[list] = None, **tokenize_kwargs):
        """Run the complete classification training pipeline."""
        self.load_tokenizer()
        self.load_dataset()
        self.tokenize_dataset(remove_columns=remove_columns, **tokenize_kwargs)
        self.load_model()
        self.setup_trainer()
        self.train()
        self.save_model()


def main():
    args = parse_args()
    trainer = ClassificationTrainer(args.config)
    trainer.run()


if __name__ == "__main__":
    main()

