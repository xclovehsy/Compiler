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
    EncoderDecoderModel, AutoTokenizer, ModernBertModel, GPT2LMHeadModel, GPT2Config
)
from datasets import load_from_disk
from src.utils.utils import get_logger, compute_metrics, convert_to_float
from src.model.tokenizer import OptiSeqTokenizer

# === 1. 读取配置 YAML ===
parser = argparse.ArgumentParser()
parser.add_argument(
    "--config", type=str, required=True,
    help="Path to the YAML config file"
)
args = parser.parse_args()
with open(args.config, "r") as f:
    cfg = yaml.safe_load(f)
    cfg = convert_to_float(cfg)

modern_bert_id = cfg["model"]["modern_bert_id"]
opti_seq_tokenizer_id = cfg["model"]["opti_seq_tokenizer_id"]
data_dir = cfg["data"]["data_dir"]
encoder_maxlen = cfg["data"]["encoder_maxlen"]
decoder_maxlen = cfg["data"]["decoder_maxlen"]
base_work_dir = cfg["output"]["base_work_dir"]
gpt2_cfg = cfg['gpt2_config']
args_cfg = cfg["training_args"]

# === 2. 创建时间戳输出目录 ===
time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
work_dir = os.path.join(base_work_dir, time_str)
os.makedirs(work_dir, exist_ok=True)

# === 3. 配置日志 ===
logger = get_logger(work_dir)

# === 4. 加载模型和分词器 ===
def get_model():
    # encoder
    modern_bert = ModernBertModel.from_pretrained(modern_bert_id)
    modern_bert_tokenizer = AutoTokenizer.from_pretrained(modern_bert_id)
    # decoder
    opti_seq_tokenizer = OptiSeqTokenizer.from_pretrained(opti_seq_tokenizer_id)
    gpt2_config = GPT2Config(**gpt2_cfg)
    gpt2 = GPT2LMHeadModel(gpt2_config)
    # encoder-decoder model
    model = EncoderDecoderModel(encoder=modern_bert, decoder=gpt2)
    model.config.decoder_start_token_id = opti_seq_tokenizer.eos_token_id
    model.config.pad_token_id = opti_seq_tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    return model, modern_bert_tokenizer, opti_seq_tokenizer
    
model, modern_bert_tokenizer, opti_seq_tokenizer = get_model()

# === 5. 加载和预处理数据 ===
dataset = load_from_disk(data_dir)
# dataset['train'] = dataset['train'].select(range(1000))
# dataset['test'] = dataset['test'].select(range(100))

def tokenize_fn(examples):
    llvm_ir = examples['LLVM_IR']
    input_ids = modern_bert_tokenizer(
        llvm_ir, 
        truncation=True,
        padding="max_length",
        max_length=encoder_maxlen,
        return_tensors='pt'
    ).input_ids

    commandline = examples['Commandline']
    labels = opti_seq_tokenizer(
        commandline,
        truncation=True,
        padding=True,
        max_length=decoder_maxlen,
    )['input_ids']

    # TODO attention mask

    return {
        'input_ids': input_ids,
        'labels': labels
    }

tokenized_data = dataset.map(
    tokenize_fn, 
    batched=True,
    remove_columns=['Benchmark', 'CpuInfo', 'IrInstructionCountO0', 'IrInstructionCountO3', 'IrInstructionCountOz', 'InstCount', 'Autophase', 'Reward', 'LLVM_IR', 'Commandline']
)

tokenized_data.save_to_disk('/home/xucong24/Compiler/datasets/llvm_opti_seq_tokenized')


# === 7. 构建 TrainingArguments 对象 ===
training_args = TrainingArguments(
    output_dir=work_dir,
    logging_dir=work_dir,
    **args_cfg
)

# === 9. Trainer 和训练过程 ===
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data['train'],
    eval_dataset=tokenized_data['test']
    # compute_metrics=compute_metrics
)

trainer.train()

# === 10. 保存最终模型 ===
final_model_dir = os.path.join(work_dir, "final_model")
model.save_pretrained(final_model_dir)
modern_bert_tokenizer.save_pretrained(os.path.join(final_model_dir, 'modern_bert_tokenizer'))
opti_seq_tokenizer.save_pretrained(os.path.join(final_model_dir, 'opti_seq_tokenizer'))
