"""Standard BERT MLM training script.

Stage 1: Unsupervised representation learning with standard HuggingFace tokenizer.
"""
from typing import Dict, Any

from transformers import AutoTokenizer

from src.training.base import BaseMLMTrainer
from src.training.base.base_trainer import parse_args


class BertMLMTrainer(BaseMLMTrainer):
    """Trainer for BERT-style models with standard HuggingFace tokenizer."""
    
    def load_tokenizer(self):
        tokenizer_id = self.cfg.tokenizer_id or self.cfg.model_id
        self.logger.info(f"Loading tokenizer from {tokenizer_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
        self.logger.info(f"Tokenizer vocab size: {len(self.tokenizer)}")
    
    def tokenize_function(self, examples) -> Dict[str, Any]:
        text_field = self.cfg.get("data.text_field", "llvm")
        return self.tokenizer(
            examples[text_field],
            truncation=True,
            padding="max_length",
            max_length=self.cfg.max_length,
        )


def main():
    args = parse_args()
    
    # Determine columns to remove based on dataset
    remove_columns = ['llvm', 'label']
    
    trainer = BertMLMTrainer(args.config)
    trainer.run(remove_columns=remove_columns)


if __name__ == "__main__":
    main()

