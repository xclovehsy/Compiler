import os
from typing import Optional, Dict, Any

from transformers import (
    AutoTokenizer,
    AutoModel,
    EncoderDecoderModel,
    GPT2LMHeadModel,
    GPT2Config,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)

from src.training.base_trainer import BaseTrainer, parse_args
from src.model import Inst2VecTokenizer, OptiSeqTokenizer


class OptSeqGenTrainer(BaseTrainer):
    
    def __init__(self, config_path: str):
        super().__init__(config_path)
        self.encoder = None
        self.decoder = None
        self.encoder_tokenizer = None
        self.decoder_tokenizer = None
        self.tokenized_data = None
    
    def load_tokenizer(self):
        """Load encoder and decoder tokenizers."""
        # Encoder tokenizer
        encoder_tokenizer_id = self.cfg.get("model.encoder_tokenizer_id")
        self.logger.info(f"Loading encoder tokenizer from {encoder_tokenizer_id}")
        self.encoder_tokenizer = Inst2VecTokenizer.from_pretrained(encoder_tokenizer_id)
        self.logger.info(f"Encoder tokenizer vocab size: {len(self.encoder_tokenizer)}")
        
        # Decoder tokenizer
        decoder_tokenizer_id = self.cfg.get("model.decoder_tokenizer_id")
        self.logger.info(f"Loading decoder tokenizer from {decoder_tokenizer_id}")
        self.decoder_tokenizer = OptiSeqTokenizer.from_pretrained(decoder_tokenizer_id)
        self.logger.info(f"Decoder tokenizer vocab size: {len(self.decoder_tokenizer)}")
    
    def load_model(self):
        """Build the encoder-decoder model."""
        # Load encoder
        encoder_id = self.cfg.get("model.encoder_id")
        self.logger.info(f"Loading encoder from {encoder_id}")
        self.encoder = AutoModel.from_pretrained(encoder_id)
        
        # Create decoder
        gpt2_config = self.cfg.get("gpt2_config", {})
        gpt2_config["vocab_size"] = len(self.decoder_tokenizer)
        gpt2_config["add_cross_attention"] = True
        
        self.logger.info(f"Creating GPT-2 decoder with config: {gpt2_config}")
        config = GPT2Config(**gpt2_config)
        self.decoder = GPT2LMHeadModel(config)
        
        # Build encoder-decoder model
        self.logger.info("Building encoder-decoder model")
        self.model = EncoderDecoderModel(encoder=self.encoder, decoder=self.decoder)
        
        # Configure special tokens
        self.model.config.decoder_start_token_id = self.decoder_tokenizer.bos_token_id
        self.model.config.pad_token_id = self.decoder_tokenizer.pad_token_id
        self.model.config.eos_token_id = self.decoder_tokenizer.eos_token_id
        self.model.config.vocab_size = len(self.decoder_tokenizer)
        
        self.logger.info("Encoder-decoder model built successfully")
    
    def tokenize_function(self, example) -> Dict[str, Any]:
        """Tokenize a single example."""
        encoder_maxlen = self.cfg.get("data.encoder_maxlen", 512)
        decoder_maxlen = self.cfg.get("data.decoder_maxlen", 128)
        input_column = self.cfg.get("data.input_column", "LLVM_IR")
        target_column = self.cfg.get("data.target_column", "Commandline")
        
        # Tokenize encoder input (单条数据)
        encoder_outputs = self.encoder_tokenizer(
            example[input_column],
            truncation=True,
            padding="max_length",
            max_length=encoder_maxlen,
            return_tensors='pt'
        )
        
        # Tokenize decoder target (单条数据)
        decoder_outputs = self.decoder_tokenizer(
            example[target_column],
            truncation=True,
            padding="max_length",
            max_length=decoder_maxlen,
        )
        
        return {
            'input_ids': encoder_outputs['input_ids'].squeeze(0),
            'attention_mask': encoder_outputs['attention_mask'].squeeze(0),
            'labels': decoder_outputs['input_ids'].squeeze(0)
        }
    
    def tokenize_dataset(self, remove_columns: Optional[list] = None, **kwargs):
        """Tokenize the dataset."""
        self.logger.info("Tokenizing dataset")
        
        # 注意：由于 tokenizer 返回 PyTorch tensors，不能使用多进程
        # 使用 batched=False 逐条处理
        map_kwargs = {"batched": False}
        if remove_columns:
            map_kwargs["remove_columns"] = remove_columns
        
        self.tokenized_data = self.dataset.map(self.tokenize_function, **map_kwargs)
        self.logger.info("Tokenization finished")
    
    def setup_trainer(self):
        """Setup the Seq2SeqTrainer."""
        self.logger.info("Initializing Seq2SeqTrainer")
        
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.decoder_tokenizer,
            model=self.model,
            padding=True,
            pad_to_multiple_of=8
        )
        
        args_cfg = self.cfg.training_args
        training_args = Seq2SeqTrainingArguments(
            output_dir=self.work_dir,
            logging_dir=self.work_dir,
            predict_with_generate=True,
            **args_cfg
        )
        
        self.trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=self.tokenized_data['train'],
            eval_dataset=self.tokenized_data.get('test'),
            tokenizer=self.decoder_tokenizer,
        )
    
    def train(self):
        """Train the model."""
        self.logger.info("Starting optimization sequence generation training")
        self.trainer.train()
    
    def save_model(self):
        """Save the model and tokenizers."""
        final_model_dir = os.path.join(self.work_dir, "final_model")
        self.logger.info(f"Saving final model and tokenizers to {final_model_dir}")
        
        self.model.save_pretrained(final_model_dir)
        
        encoder_tokenizer_dir = os.path.join(final_model_dir, "encoder_tokenizer")
        if hasattr(self.encoder_tokenizer, 'save_pretrained'):
            self.encoder_tokenizer.save_pretrained(encoder_tokenizer_dir)
        
        decoder_tokenizer_dir = os.path.join(final_model_dir, "decoder_tokenizer")
        if hasattr(self.decoder_tokenizer, 'save_pretrained'):
            self.decoder_tokenizer.save_pretrained(decoder_tokenizer_dir)
        
        self.logger.info(f"Model and tokenizers saved to {final_model_dir}")
    
    def run(self, remove_columns: Optional[list] = None, **tokenize_kwargs):
        """Run the full training pipeline."""
        self.load_tokenizer()
        self.load_dataset()
        self.tokenize_dataset(remove_columns=remove_columns, **tokenize_kwargs)
        self.load_model()
        self.setup_trainer()
        self.train()
        self.save_model()


def main():
    args = parse_args()
    trainer = OptSeqGenTrainer(args.config)
    remove_columns = trainer.cfg.get("data.remove_columns", None)
    trainer.run(remove_columns=remove_columns)


if __name__ == "__main__":
    main()
