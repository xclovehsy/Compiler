"""Inst2Vec + ModernBERT MLM training script."""
import argparse
from typing import Dict, Any

from transformers import AutoModelForMaskedLM

from src.training.base_trainer import BaseMLMTrainer, parse_args
from src.model import Inst2VecTokenizer


class Inst2VecModernBertMLMTrainer(BaseMLMTrainer):
    """Trainer for ModernBERT with Inst2Vec tokenizer."""
    
    def load_tokenizer(self):
        """Load the Inst2Vec tokenizer."""
        tokenizer_id = self.cfg.tokenizer_id
        self.logger.info(f"Loading Inst2Vec tokenizer from {tokenizer_id}")
        self.tokenizer = Inst2VecTokenizer.from_pretrained(tokenizer_id)
        self.logger.info(f"Tokenizer vocab size: {len(self.tokenizer)}")
    
    def tokenize_function(self, examples) -> Dict[str, Any]:
        """Tokenize using Inst2Vec tokenizer."""
        return self.tokenizer(
            examples['llvm'],
            max_length=self.cfg.max_length, 
            truncation=True,
            padding=True,
            return_tensors="pt"
        )
    
    def load_model(self):
        """Load and resize model for custom tokenizer."""
        super().load_model()
        
        # Resize token embeddings to match custom tokenizer
        old_vocab_size = self.model.get_input_embeddings().num_embeddings
        self.logger.info(f"Original model vocab size: {old_vocab_size}")
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.logger.info(f"Resized model vocab to {len(self.tokenizer)}")
        
        # Update model config
        self.model.config.vocab_size = len(self.tokenizer)
        self.logger.info(f"Updated model config vocab_size to {self.model.config.vocab_size}")
    
    def tokenize_dataset(self, remove_columns=None, **kwargs):
        """Tokenize dataset with Inst2Vec tokenizer (non-batched)."""
        self.logger.info(f"Tokenizing dataset with max_length={self.cfg.max_length}")
        
        self.tokenized_data = self.dataset.map(
            self.tokenize_function,
            batched=False,
            num_proc=kwargs.get("num_proc", 32),
            remove_columns=remove_columns or ['llvm', 'label']
        )
        self.logger.info("Tokenization finished")


def main():
    args = parse_args()
    trainer = Inst2VecModernBertMLMTrainer(args.config)
    trainer.run(remove_columns=['llvm', 'label'], num_proc=32)


if __name__ == "__main__":
    main()

