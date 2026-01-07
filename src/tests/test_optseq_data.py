"""测试 OptSeqGen 数据集加载和 tokenizer。"""

import argparse
from datasets import load_from_disk

from src.model import Inst2VecTokenizer, OptiSeqTokenizer
from src.config import Config


def test_tokenizers(encoder_tokenizer_path: str, decoder_tokenizer_path: str):
    """测试 tokenizer 加载。"""
    print("=" * 50)
    print("测试 Tokenizer 加载")
    print("=" * 50)
    
    # Encoder tokenizer
    print(f"\n加载 encoder tokenizer: {encoder_tokenizer_path}")
    encoder_tokenizer = Inst2VecTokenizer.from_pretrained(encoder_tokenizer_path)
    print(f"  vocab size: {len(encoder_tokenizer)}")
    print(f"  pad_token_id: {encoder_tokenizer.pad_token_id}")
    print(f"  bos_token_id: {encoder_tokenizer.bos_token_id}")
    print(f"  eos_token_id: {encoder_tokenizer.eos_token_id}")
    
    # Decoder tokenizer
    print(f"\n加载 decoder tokenizer: {decoder_tokenizer_path}")
    decoder_tokenizer = OptiSeqTokenizer.from_pretrained(decoder_tokenizer_path)
    print(f"  vocab size: {len(decoder_tokenizer)}")
    print(f"  pad_token_id: {decoder_tokenizer.pad_token_id}")
    print(f"  bos_token_id: {decoder_tokenizer.bos_token_id}")
    print(f"  eos_token_id: {decoder_tokenizer.eos_token_id}")
    
    return encoder_tokenizer, decoder_tokenizer


def test_dataset(data_dir: str):
    """测试数据集加载。"""
    print("\n" + "=" * 50)
    print("测试数据集加载")
    print("=" * 50)
    
    print(f"\n加载数据集: {data_dir}")
    dataset = load_from_disk(data_dir)
    
    print(f"  splits: {list(dataset.keys())}")
    for split in dataset.keys():
        print(f"  {split}: {len(dataset[split])} 条数据")
    
    # 显示列名
    print(f"\n  列名: {dataset['train'].column_names}")
    
    # 显示第一条数据
    print("\n  第一条数据预览:")
    first = dataset['train'][0]
    for key, value in first.items():
        if isinstance(value, str) and len(value) > 200:
            print(f"    {key}: {value[:200]}... (长度: {len(value)})")
        else:
            print(f"    {key}: {value}")
    
    return dataset


def test_tokenize(encoder_tokenizer, decoder_tokenizer, dataset, 
                  input_column: str, target_column: str,
                  encoder_maxlen: int = 512, decoder_maxlen: int = 128):
    """测试 tokenize 单条数据。"""
    print("\n" + "=" * 50)
    print("测试 Tokenize")
    print("=" * 50)
    
    # 取第一条数据
    example = dataset['train'][0]
    
    # Encoder tokenize
    print(f"\n编码 encoder 输入 (max_length={encoder_maxlen}):")
    encoder_input = example[input_column]
    print(f"  原始长度: {len(encoder_input)} 字符")
    
    encoder_outputs = encoder_tokenizer(
        encoder_input,
        truncation=True,
        padding="max_length",
        max_length=encoder_maxlen,
        return_tensors='pt'
    )
    print(f"  input_ids shape: {encoder_outputs['input_ids'].shape}")
    print(f"  attention_mask shape: {encoder_outputs['attention_mask'].shape}")
    print(f"  前 10 个 token ids: {encoder_outputs['input_ids'][0, :10].tolist()}")
    
    # Decoder tokenize
    print(f"\n编码 decoder 目标 (max_length={decoder_maxlen}):")
    decoder_input = example[target_column]
    print(f"  原始内容: {decoder_input}")
    
    decoder_outputs = decoder_tokenizer(
        decoder_input,
        truncation=True,
        padding="max_length",
        max_length=decoder_maxlen,
    )
    print(f"  input_ids shape: {decoder_outputs['input_ids'].shape}")
    print(f"  token ids: {decoder_outputs['input_ids'][0, :20].tolist()}")
    
    # 解码验证
    print("\n解码验证:")
    decoded = decoder_tokenizer.decode(decoder_outputs['input_ids'][0])
    print(f"  解码结果: {decoded}")


def test_batch_tokenize(encoder_tokenizer, decoder_tokenizer, dataset,
                        input_column: str, target_column: str,
                        batch_size: int = 4):
    """测试批量 tokenize。"""
    print("\n" + "=" * 50)
    print(f"测试批量 Tokenize (batch_size={batch_size})")
    print("=" * 50)
    
    # 取前 batch_size 条数据
    batch = dataset['train'][:batch_size]
    
    print(f"\n批量编码 encoder 输入:")
    encoder_outputs = encoder_tokenizer(
        batch[input_column],
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors='pt'
    )
    print(f"  input_ids shape: {encoder_outputs['input_ids'].shape}")
    
    print(f"\n批量编码 decoder 目标:")
    decoder_outputs = decoder_tokenizer(
        batch[target_column],
        truncation=True,
        padding="max_length",
        max_length=128,
    )
    print(f"  input_ids shape: {decoder_outputs['input_ids'].shape}")


def main():
    parser = argparse.ArgumentParser(description="测试 OptSeqGen 数据和 tokenizer")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    parser.add_argument("--batch_size", type=int, default=4, help="批量测试大小")
    args = parser.parse_args()
    
    # 加载配置
    cfg = Config(args.config)
    
    encoder_tokenizer_path = cfg.get("model.encoder_tokenizer_id")
    decoder_tokenizer_path = cfg.get("model.decoder_tokenizer_id")
    data_dir = cfg.get("data.data_dir")
    input_column = cfg.get("data.input_column", "LLVM_IR")
    target_column = cfg.get("data.target_column", "Commandline")
    encoder_maxlen = cfg.get("data.encoder_maxlen", 512)
    decoder_maxlen = cfg.get("data.decoder_maxlen", 128)
    
    # 测试 tokenizer
    encoder_tokenizer, decoder_tokenizer = test_tokenizers(
        encoder_tokenizer_path, decoder_tokenizer_path
    )
    
    # 测试数据集
    dataset = test_dataset(data_dir)
    
    # 测试单条 tokenize
    test_tokenize(
        encoder_tokenizer, decoder_tokenizer, dataset,
        input_column, target_column,
        encoder_maxlen, decoder_maxlen
    )
    
    # 测试批量 tokenize
    test_batch_tokenize(
        encoder_tokenizer, decoder_tokenizer, dataset,
        input_column, target_column,
        args.batch_size
    )
    
    print("\n" + "=" * 50)
    print("✅ 所有测试通过!")
    print("=" * 50)


if __name__ == "__main__":
    main()

