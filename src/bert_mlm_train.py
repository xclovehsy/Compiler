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
# tokenizer_id = cfg["model"]["tokenizer_id"]
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

# === 4. 加载模型和分词器 ===
model = AutoModelForMaskedLM.from_pretrained(model_id)
# tokenizer = AutoTokenizer.from_pretrained(model_id)

# === 4.1 加载 Inst2VecTokenizer ===
# tokenizer = Inst2VecTokenizer.from_pretrained(tokenizer_id)

# === 5. 加载和预处理数据 ===
dataset = load_from_disk(data_dir)

def tokenize_fn(examples):
    return tokenizer(
        examples['content'],
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )

tokenized_data = dataset.map(tokenize_fn,
                             batched=True,
                             remove_columns=[
                                 'content', 'license_expression', 'license_source',
                                 'license_files', 'package_source', 'language'
                             ])
# tokenized_data = load_from_disk(data_dir)

# === 6. 数据整理器 ===
data_collator = DataCollatorForLanguageModeling(
    # tokenizer=tokenizer,
    mlm=True,
    mlm_probability=mlm_prob
)

# === 7. 构建 TrainingArguments 对象 ===
args_cfg['learning_rate'] = float(args_cfg['learning_rate'])
training_args = TrainingArguments(
    output_dir=work_dir,
    logging_dir=work_dir,
    **args_cfg
)

# === 9. Trainer 和训练过程 ===
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_data['train'],
    eval_dataset=tokenized_data['test']
    # compute_metrics=compute_metrics
)

trainer.train()

# === 10. 保存最终模型 ===
final_model_dir = os.path.join(work_dir, "final_model")
model.save_pretrained(final_model_dir)
tokenizer.save_pretrained(final_model_dir)
