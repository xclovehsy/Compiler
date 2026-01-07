import os
from typing import Optional, Dict, Any

from transformers import AutoModelForMaskedLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling

from src.training.base_trainer import BaseTrainer, parse_args
from src.model import Inst2VecTokenizer


class InstBertMLMTrainer(BaseTrainer):
    
    def __init__(self, config_path: str):
        super().__init__(config_path)
        self.tokenized_data = None
    
    def load_tokenizer(self):
        tokenizer_id = self.cfg.tokenizer_id
        self.logger.info(f"Loading Inst2Vec tokenizer from {tokenizer_id}")
        self.tokenizer = Inst2VecTokenizer.from_pretrained(tokenizer_id)
        self.logger.info(f"Tokenizer vocab size: {len(self.tokenizer)}")
    
    def load_model(self):
        """Load model and resize token embeddings."""
        self.logger.info(f"Loading model from {self.cfg.model_id}")
        self.model = AutoModelForMaskedLM.from_pretrained(self.cfg.model_id)
        
        old_vocab_size = self.model.get_input_embeddings().num_embeddings
        self.logger.info(f"Original model vocab size: {old_vocab_size}")
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.logger.info(f"Resized model vocab to {len(self.tokenizer)}")
        
        self.model.config.vocab_size = len(self.tokenizer)
        self.logger.info(f"Updated model config vocab_size to {self.model.config.vocab_size}")
    
    def tokenize_function(self, examples) -> Dict[str, Any]:
        return self.tokenizer(
            examples['llvm'],
            max_length=self.cfg.max_length, 
            truncation=True,
            padding=True,
            return_tensors="pt"
        )
    
    def tokenize_dataset(self, remove_columns: Optional[list] = None, **kwargs):
        self.logger.info(f"Tokenizing dataset with max_length={self.cfg.max_length}")
        
        self.tokenized_data = self.dataset.map(
            self.tokenize_function,
            batched=False,
            num_proc=kwargs.get("num_proc", 32),
            remove_columns=remove_columns or ['llvm', 'label']
        )
        self.logger.info("Tokenization finished")
    
    def setup_trainer(self):
        self.logger.info("Initializing Trainer")
        
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=True,
            mlm_probability=self.cfg.mlm_probability,
            pad_to_multiple_of=8
        )
        
        args_cfg = self.cfg.training_args
        training_args = TrainingArguments(
            output_dir=self.work_dir,
            logging_dir=self.work_dir,
            **args_cfg
        )
        
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


def main():
    args = parse_args()
    trainer = InstBertMLMTrainer(args.config)
    trainer.run(remove_columns=['llvm', 'label'], num_proc=32)


if __name__ == "__main__":
    main()
