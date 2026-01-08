"""LLVM Optimization Sequence Generation training script."""
import os
import argparse
from datetime import datetime

from transformers import (
    Trainer,
    TrainingArguments,
    EncoderDecoderModel,
    AutoModel,
    AutoConfig,
    GPT2LMHeadModel,
    GPT2Config
)
from datasets import load_from_disk, DatasetDict

from src.config import load_config
from src.utils.utils import get_logger
from src.model import Inst2VecTokenizer, OptiSeqTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to the YAML config file"
    )
    return parser.parse_args()


def get_model(cfg, logger):
    """Build the encoder-decoder model."""
    instbert_id = cfg["model"]["instbert_id"]
    inst2vec_tokenizer_id = cfg["model"]["inst2vec_tokenizer_id"]
    opti_seq_tokenizer_id = cfg["model"]["opti_seq_tokenizer_id"]
    gpt2_cfg = cfg['gpt2_config']
    
    # 加载 tokenizer
    logger.info(f"Loading Inst2Vec tokenizer from {inst2vec_tokenizer_id}")
    inst2vec_tokenizer = Inst2VecTokenizer.from_pretrained(inst2vec_tokenizer_id)
    
    # Encoder
    logger.info(f"Loading encoder from {instbert_id}")
    encoder = AutoModel.from_pretrained(instbert_id)
    
    # Decoder
    logger.info(f"Loading OptiSeq tokenizer from {opti_seq_tokenizer_id}")
    opti_seq_tokenizer = OptiSeqTokenizer.from_pretrained(opti_seq_tokenizer_id)
    gpt2_config = GPT2Config(**gpt2_cfg)
    gpt2 = GPT2LMHeadModel(gpt2_config)
    
    # Encoder-decoder model
    logger.info("Building encoder-decoder model")
    model = EncoderDecoderModel(encoder=encoder, decoder=gpt2)
    model.config.decoder_start_token_id = opti_seq_tokenizer.eos_token_id
    model.config.pad_token_id = opti_seq_tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    
    return model, inst2vec_tokenizer, opti_seq_tokenizer


def main():
    args = parse_args()
    cfg = load_config(args.config)
    
    # Config values
    data_dir = cfg["data"]["data_dir"]
    encoder_maxlen = cfg["data"]["encoder_maxlen"]
    decoder_maxlen = cfg["data"]["decoder_maxlen"]
    base_work_dir = cfg["output"]["base_work_dir"]
    args_cfg = cfg["training_args"]
    
    # Create work directory
    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    work_dir = os.path.join(base_work_dir, time_str)
    os.makedirs(work_dir, exist_ok=True)
    
    # Setup logging
    logger = get_logger(work_dir)
    logger.info(f"Work directory created at {work_dir}")
    
    # Load model and tokenizers
    model, inst2vec_tokenizer, opti_seq_tokenizer = get_model(cfg, logger)
    
    # Load dataset
    logger.info(f"Loading dataset from {data_dir}")
    dataset = load_from_disk(data_dir)
    
    # 只取前100条用于测试
    logger.info("Limiting dataset to first 100 samples for testing")
    if isinstance(dataset, DatasetDict):
        dataset = DatasetDict({split: ds.select(range(min(100, len(ds)))) for split, ds in dataset.items()})
    else:
        dataset = dataset.select(range(min(100, len(dataset))))
    
    def tokenize_fn(example):
        llvm_ir = example['LLVM_IR']
        encoder_outputs = inst2vec_tokenizer(
            llvm_ir, 
            truncation=True,
            padding="max_length",
            max_length=encoder_maxlen,
            return_tensors='pt'
        )

        commandline = example['Commandline']
        decoder_outputs = opti_seq_tokenizer(
            commandline,
            truncation=True,
            padding=True,
            max_length=decoder_maxlen,
        )

        return {
            'input_ids': encoder_outputs['input_ids'].squeeze(0),
            'attention_mask': encoder_outputs['attention_mask'].squeeze(0),
            'labels': decoder_outputs['input_ids'].squeeze(0)
        }
    
    logger.info("Tokenizing dataset")
    remove_columns = [
        'Benchmark', 'CpuInfo', 'IrInstructionCountO0', 
        'IrInstructionCountO3', 'IrInstructionCountOz', 
        'InstCount', 'Autophase', 'Reward', 'LLVM_IR', 'Commandline'
    ]
    tokenized_data = dataset.map(
        tokenize_fn, 
        batched=False,
        remove_columns=remove_columns
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=work_dir,
        logging_dir=work_dir,
        **args_cfg
    )
    
    # Trainer
    logger.info("Initializing Trainer")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data['train'],
        eval_dataset=tokenized_data['test']
    )
    
    # Train
    logger.info("Starting training")
    trainer.train()
    
    # Save model
    final_model_dir = os.path.join(work_dir, "final_model")
    logger.info(f"Saving model to {final_model_dir}")
    model.save_pretrained(final_model_dir)
    inst2vec_tokenizer.save_pretrained(os.path.join(final_model_dir, 'encoder_tokenizer'))
    opti_seq_tokenizer.save_pretrained(os.path.join(final_model_dir, 'decoder_tokenizer'))
    logger.info("Training complete")


if __name__ == "__main__":
    main()

