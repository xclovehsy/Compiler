from transformers import EncoderDecoderModel, BertTokenizer, GPT2Tokenizer, GPT2LMHeadModel
from datasets import load_dataset
from transformers import Trainer, TrainingArguments

# 加载刚训练好的编码器（BERT）
encoder_name = "./bert-mlm-trained"
tokenizer_encoder = BertTokenizer.from_pretrained(encoder_name)
encoder = BertForMaskedLM.from_pretrained(encoder_name).bert  # 取bert编码器部分

# 加载解码器（这里以gpt2为例）
decoder_name = "gpt2"
tokenizer_decoder = GPT2Tokenizer.from_pretrained(decoder_name)
decoder = GPT2LMHeadModel.from_pretrained(decoder_name)

# 编码器-解码器模型
model = EncoderDecoderModel(encoder=encoder, decoder=decoder)

# 合并tokenizer，简单起见，使用同一个tokenizer（通常Seq2Seq模型用统一tokenizer较好）
tokenizer = tokenizer_encoder  # 或者你可以设计自定义tokenizer融合

# 加载自定义训练数据 (示例使用wikitext，改成你自己的数据)
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

def preprocess_function(examples):
    # 编码输入文本（encoder输入）
    inputs = tokenizer(examples["text"], truncation=True, max_length=128)
    # 编码目标文本（decoder输入）
    targets = tokenizer(examples["text"], truncation=True, max_length=128)
    model_inputs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "labels": targets["input_ids"],
    }
    return model_inputs

tokenized_datasets = dataset.map(preprocess_function, batched=True, remove_columns=["text"])

# --- 参考
tokenizer = BertTokenizer.from_pretrained(encoder_name)
tokenizer.pad_token = tokenizer.eos_token  # GPT2没有pad，设置成eos

def preprocess_function(examples):
    inputs = tokenizer(examples["text"], truncation=True, max_length=128)
    targets = tokenizer(examples["text"], truncation=True, max_length=128)
    model_inputs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "labels": targets["input_ids"],
    }
    # 将pad token替换为-100，跳过loss计算
    model_inputs["labels"] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label] 
        for label in model_inputs["labels"]
    ]
    return model_inputs

# 训练参数
training_args = TrainingArguments(
    output_dir="./encoder-decoder-clm",
    evaluation_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    save_total_limit=2,
    save_steps=500,
    logging_dir="./logs_enc_dec",
    logging_steps=100,
    learning_rate=5e-5,
    weight_decay=0.01,
    predict_with_generate=True,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
)

# 训练 CLM+Seq2Seq
trainer.train()

# 保存模型
model.save_pretrained("./encoder-decoder-clm-trained")
tokenizer.save_pretrained("./encoder-decoder-clm-trained")


# 使用不同的学习率 --------------------

from transformers import AdamW, Trainer, TrainingArguments
from transformers import EncoderDecoderModel

model = EncoderDecoderModel.from_encoder_decoder_pretrained('bert-base-uncased', 'gpt2')

# 查看参数名字，确认encoder和decoder的区分
for name, param in model.named_parameters():
    print(name)  # 方便确认哪个属于encoder，哪个属于decoder

# 给encoder和decoder分别组建参数列表
encoder_params = []
decoder_params = []

for name, param in model.named_parameters():
    if name.startswith("encoder"):
        encoder_params.append(param)
    elif name.startswith("decoder"):
        decoder_params.append(param)
    else:
        # 其他参数，如lm_head，放到decoder组或者分开也可以
        decoder_params.append(param)

# 定义参数组并指定不同LR
optimizer_grouped_parameters = [
    {"params": encoder_params, "lr": 5e-5},  # encoder较小学习率
    {"params": decoder_params, "lr": 1e-4},  # decoder较大学习率
]

optimizer = AdamW(optimizer_grouped_parameters)

# 定义训练参数
training_args = TrainingArguments(
    output_dir="./output",
    num_train_epochs=3,
    per_device_train_batch_size=8,
)

trainer = Trainer(
    model=model,
    args=training_args,
    optimizers=(optimizer, None),  # 手动传入optimizer，scheduler可以传None或自定义
    train_dataset=...,  # 你的数据集
    eval_dataset=...,   # 你的验证集
)

trainer.train()
