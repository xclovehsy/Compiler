import os
import logging
from datetime import datetime
import argparse
import yaml
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    logging as hf_logging
)
from datasets import load_from_disk
from src.utils.utils import get_logger, compute_metrics
from src.model.tokenizer import Inst2VecTokenizer


# 注意这里没有改模型配置bos eos。。。

# === 1. 读取配置 YAML ===
parser = argparse.ArgumentParser()
parser.add_argument(
    "--config", type=str, required=True,
    help="Path to the YAML config file"
)
args = parser.parse_args()
with open(args.config, "r") as f:
    cfg = yaml.safe_load(f)

model_id = cfg["model"]["model_id"]
tokenizer_id = cfg["model"]["tokenizer_id"]
data_dir = cfg["data"]["data_dir"]
max_length = cfg["data"]["max_length"]
base_work_dir = cfg["output"]["base_work_dir"]
mlm_prob = cfg["mlm"]["mlm_probability"]
args_cfg = cfg["training_args"]

# === 2. 创建时间戳输出目录 ===
time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
work_dir = os.path.join(base_work_dir, time_str)
os.makedirs(work_dir, exist_ok=True)

# === 3. 配置日志 ===
logger = get_logger(work_dir)
logger.info(f"Work directory created at {work_dir}")

# === 4. 加载分词器 ===
logger.info(f"Loading tokenizer from {tokenizer_id}")
tokenizer = Inst2VecTokenizer.from_pretrained(tokenizer_id)
logger.info(f"Tokenizer vocab size: {len(tokenizer)}")

# === 5. 加载和预处理数据 ===
logger.info(f"Loading dataset from {data_dir}")
dataset = load_from_disk(data_dir)
logger.info(f"Dataset loaded with {len(dataset)} examples")

def tokenize_fn(examples):
    return tokenizer(
        examples['llvm'],
        max_length=max_length, 
        truncation=True,
        padding=True,
        return_tensors="pt"
    )

logger.info(f"Tokenizing dataset with max_length={max_length}")
tokenized_data = dataset.map(tokenize_fn,
                             batched=False,
                             num_proc=32,
                             remove_columns=[
                                 'llvm', 'label'
                             ])
logger.info("Tokenization finished")

# === 6. 加载模型 ===
logger.info(f"Loading model from {model_id}")
model = AutoModelForMaskedLM.from_pretrained(model_id)

# === 6.1. 调整 embedding & 输出层，使 vocab 与自定义 tokenizer 对齐 ===
old_vocab_size = model.get_input_embeddings().num_embeddings
logger.info(f"Original model vocab size: {old_vocab_size}")
model.resize_token_embeddings(len(tokenizer))
logger.info(f"Resized model vocab to {len(tokenizer)} (pad to multiple of 8)")

# 修改模型配置
model.config.vocab_size = len(tokenizer)
logger.info(f"Updated model config vocab_size to {model.config.vocab_size}")

# === 7. Huggingface 数据整理器 ===
logger.info(f"Creating DataCollatorForLanguageModeling with mlm_probability={mlm_prob}")
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=mlm_prob,
    pad_to_multiple_of=8
)

# # === 8. TrainingArguments  ===
args_cfg['learning_rate'] = float(args_cfg['learning_rate'])
training_args = TrainingArguments(
    output_dir=work_dir,
    logging_dir=work_dir,
    **args_cfg
)
logger.info(f"TrainingArguments configured: {args_cfg}")

# === 9. Trainer ===
logger.info("Initializing Trainer")
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_data['train'],
    eval_dataset=tokenized_data['test']
    # compute_metrics=compute_metrics
)

logger.info("Starting training")
trainer.train()

# === 10. 保存最终模型 ===
final_model_dir = os.path.join(work_dir, "final_model")
logger.info(f"Saving final model and tokenizer to {final_model_dir}")
model.save_pretrained(final_model_dir)
tokenizer.save_pretrained(final_model_dir)
logger.info(f"Model and tokenizer saved to {final_model_dir}")