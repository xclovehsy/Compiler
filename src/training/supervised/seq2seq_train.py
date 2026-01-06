"""Optimization Sequence Generation training script.

Stage 2: Heuristic search-based sequence generation pretraining.
Uses Encoder-Decoder architecture:
- Encoder: InstBERT (pretrained in Stage 1)
- Decoder: GPT-2 for autoregressive sequence generation
"""
import os
from typing import Dict, Any

from transformers import (
    AutoTokenizer,
    EncoderDecoderModel,
    ModernBertModel,
    GPT2LMHeadModel,
    GPT2Config,
)

from src.training.base import BaseSeq2SeqTrainer
from src.training.base.base_trainer import parse_args
from src.model import Inst2VecTokenizer, OptiSeqTokenizer


class OptSeqTrainer(BaseSeq2SeqTrainer):
    """Trainer for LLVM Optimization Sequence Generation.
    
    This trainer is used for Stage 2 of the research:
    Heuristic search-based sequence generation pretraining.
    
    Architecture:
    - Encoder: InstBERT (ModernBERT with Inst2Vec tokenizer)
    - Decoder: GPT-2 with optimization sequence vocabulary
    """
    
    def __init__(self, config_path: str):
        super().__init__(config_path)
        self.encoder = None
        self.decoder = None
    
    def load_encoder_tokenizer(self):
        """Load the encoder tokenizer (Inst2Vec or standard)."""
        encoder_tokenizer_id = self.cfg.get("model.encoder_tokenizer_id")
        tokenizer_type = self.cfg.get("model.encoder_tokenizer_type", "inst2vec")
        
        self.logger.info(f"Loading encoder tokenizer from {encoder_tokenizer_id}")
        
        if tokenizer_type == "inst2vec":
            self.encoder_tokenizer = Inst2VecTokenizer.from_pretrained(encoder_tokenizer_id)
        else:
            self.encoder_tokenizer = AutoTokenizer.from_pretrained(encoder_tokenizer_id)
        
        self.logger.info(f"Encoder tokenizer vocab size: {len(self.encoder_tokenizer)}")
    
    def load_decoder_tokenizer(self):
        """Load the decoder tokenizer (OptiSeq for optimization passes)."""
        decoder_tokenizer_id = self.cfg.get("model.decoder_tokenizer_id")
        
        self.logger.info(f"Loading decoder tokenizer from {decoder_tokenizer_id}")
        self.decoder_tokenizer = OptiSeqTokenizer.from_pretrained(decoder_tokenizer_id)
        self.logger.info(f"Decoder tokenizer vocab size: {len(self.decoder_tokenizer)}")
    
    def load_encoder(self):
        """Load the encoder model (InstBERT/ModernBERT)."""
        encoder_id = self.cfg.get("model.encoder_id")
        self.logger.info(f"Loading encoder from {encoder_id}")
        self.encoder = ModernBertModel.from_pretrained(encoder_id)
    
    def load_decoder(self):
        """Load the decoder model (GPT-2)."""
        gpt2_config = self.cfg.get("model.gpt2_config", {})
        
        # Update vocab size to match decoder tokenizer
        gpt2_config["vocab_size"] = len(self.decoder_tokenizer)
        
        self.logger.info(f"Creating GPT-2 decoder with config: {gpt2_config}")
        config = GPT2Config(**gpt2_config)
        self.decoder = GPT2LMHeadModel(config)
    
    def build_encoder_decoder_model(self):
        """Build the encoder-decoder model."""
        self.logger.info("Building encoder-decoder model")
        
        self.model = EncoderDecoderModel(encoder=self.encoder, decoder=self.decoder)
        
        # Configure special tokens
        self.model.config.decoder_start_token_id = self.decoder_tokenizer.bos_token_id
        self.model.config.pad_token_id = self.decoder_tokenizer.pad_token_id
        self.model.config.eos_token_id = self.decoder_tokenizer.eos_token_id
        self.model.config.vocab_size = len(self.decoder_tokenizer)
        
        self.logger.info("Encoder-decoder model built successfully")
    
    def tokenize_function(self, examples) -> Dict[str, Any]:
        """Tokenize examples for seq2seq training."""
        encoder_maxlen = self.cfg.get("data.encoder_maxlen", 512)
        decoder_maxlen = self.cfg.get("data.decoder_maxlen", 128)
        
        # Tokenize LLVM IR (encoder input)
        llvm_ir = examples['LLVM_IR']
        input_ids = self.encoder_tokenizer(
            llvm_ir,
            truncation=True,
            padding="max_length",
            max_length=encoder_maxlen,
            return_tensors='pt'
        ).input_ids
        
        # Tokenize optimization sequence (decoder target)
        commandline = examples['Commandline']
        labels = self.decoder_tokenizer(
            commandline,
            truncation=True,
            padding="max_length",
            max_length=decoder_maxlen,
        )['input_ids']
        
        return {
            'input_ids': input_ids,
            'labels': labels
        }


def main():
    args = parse_args()
    
    remove_columns = [
        'Benchmark', 'CpuInfo', 'IrInstructionCountO0',
        'IrInstructionCountO3', 'IrInstructionCountOz',
        'InstCount', 'Autophase', 'Reward', 'LLVM_IR', 'Commandline'
    ]
    
    trainer = OptSeqTrainer(args.config)
    trainer.run(remove_columns=remove_columns)


if __name__ == "__main__":
    main()

