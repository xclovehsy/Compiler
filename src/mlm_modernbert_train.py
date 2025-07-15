from transformers import BertTokenizer, BertForMaskedLM, DataCollatorForLanguageModeling
from datasets import load_dataset
from transformers import Trainer, TrainingArguments

# 初始化BERT tokenizer和模型
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
encoder = BertForMaskedLM.from_pretrained(model_name)

# 准备数据集（这里以wikitext作为示例）
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

# 分词
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=128)

tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# 设置MLM数据collator，自动做mask
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

# 训练参数
training_args = TrainingArguments(
    output_dir="./bert-mlm",
    evaluation_strategy="epoch",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    save_total_limit=2,
    save_steps=500,
    logging_dir="./logs",
    logging_steps=100,
    learning_rate=5e-5,
    weight_decay=0.01,
)

trainer = Trainer(
    model=encoder,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
)

# 训练（MLM）
trainer.train()

# 保存训练好的编码器权重
encoder.save_pretrained("./bert-mlm-trained")
tokenizer.save_pretrained("./bert-mlm-trained")



# =========
from transformers import BertForMaskedLM, GPT2LMHeadModel

# 1. MLM训练编码器
encoder = BertForMaskedLM.from_pretrained('bert-base-uncased')
# 你在这里训练 encoder，只用MLM任务

# 2. 编码器编码文本获得 hidden_states
# freeze或者fine-tune encoder
encoder.eval()
with torch.no_grad():
    encoding = encoder.bert(input_ids).last_hidden_state  # shape (batch, seq_len, hidden)

# 3. 初始化解码器
decoder = GPT2LMHeadModel.from_pretrained('gpt2')

# 4. 利用 encoding 作为decoder cross-attention的memory
# GPT2默认是Decoder-only没有cross-attention，你需要改成Decoder带encoder-decoder Attention的结构，
# 或用transformer decoder实现，例如Huggingface的EncoderDecoderModel

# 5. 训练decoder用CLM任务生成文本

