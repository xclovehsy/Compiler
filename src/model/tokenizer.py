import os
import json
import torch
import csv
import numpy as np
from typing import List, Optional
import pickle
from .inst2vec_preprocess import *

# from transformers import PreTrainedTokenizerBase

class Inst2VecTokenizer:
    def __init__(self, vocab, 
                 unk_token="<unk>", 
                 pad_token="<pad>", 
                 bos_token="<bos>", 
                 eos_token="<eos>", 
                 mask_token="<mask>", 
                 **kwargs):
        self.vocab = vocab
        self.ids_to_tokens = {v: k for k, v in vocab.items()}

        # Special tokens
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.mask_token = mask_token

        # Special token ids
        self.unk_token_id = vocab.get(unk_token)
        self.pad_token_id = vocab.get(pad_token)
        self.bos_token_id = vocab.get(bos_token)
        self.eos_token_id = vocab.get(eos_token)
        self.mask_token_id = vocab.get(mask_token)

        # HuggingFace 兼容常用别名
        self.cls_token = self.bos_token
        self.sep_token = self.eos_token
        self.cls_token_id = self.bos_token_id
        self.sep_token_id = self.eos_token_id
        
    # -----------------------------
    # 必要属性
    # -----------------------------
    @property
    def vocab_size(self):
        return len(self.vocab)

    def get_vocab(self):
        return self.vocab
    
    def __len__(self):
        return self.vocab_size
    
    # -----------------------------
    # Token <-> ID
    # -----------------------------
    def _tokenize(self, ir: str) -> List[str]:
        """Produce a list of pre-processed statements from an IR."""
        if os.path.isfile(ir):
            with open(ir, "r") as f:
                ir = f.read()
            
        lines = [[x] for x in ir.split("\n")]
        try:
            structs = GetStructTypes(ir)
            for line in lines:
                for struct, definition in structs.items():
                    line[0] = line[0].replace(struct, definition)
        except ValueError:
            pass

        preprocessed_lines, _ = preprocess(lines)
        preprocessed_texts = [
            PreprocessStatement(x[0]) if len(x) else ""
            for x in preprocessed_lines
        ]
        return [x for x in preprocessed_texts if x]

    def _convert_token_to_id(self, token: str) -> int:
        return self.vocab.get(token, self.unk_token_id)

    def _convert_id_to_token(self, index: int) -> str:
        return self.ids_to_tokens.get(index, self.unk_token)


    # -----------------------------
    # Encode / Decode
    # -----------------------------
    def encode(self, llvm: str, max_length: Optional[int] = None, truncation=True, padding=True, return_tensors="pt"):
        tokens = self._tokenize(llvm)
        token_ids =  [self._convert_token_to_id(t) for t in tokens]

        token_ids = [self.bos_token_id] + token_ids + [self.eos_token_id]

        if truncation and max_length is not None:
            token_ids = token_ids[:max_length]

        attention_mask = [1]*len(token_ids)

        if padding and max_length is not None and len(token_ids) < max_length:
            pad_len = max_length - len(token_ids)
            token_ids += [self.pad_token_id]*pad_len
            attention_mask += [0]*pad_len
            
        if return_tensors == "pt":
            token_ids = torch.tensor(token_ids, dtype=torch.long)
            attention_mask = torch.tensor(attention_mask, dtype=torch.long)
            
        return {
            "input_ids": token_ids,
            "attention_mask": attention_mask,
        }


    def decode(self, token_ids: torch.Tensor, skip_special_tokens=True) -> str:
        assert token_ids.ndim == 1
        token_ids = token_ids.tolist()

        tokens = []
        special_tokens = {self.pad_token, self.bos_token, self.eos_token}
        for id_ in token_ids:
            token = self._convert_id_to_token(id_)
            if skip_special_tokens and token in special_tokens:
                continue
            tokens.append(token)
        return '\n'.join(tokens)
    
    def batch_decode(self, token_ids: torch.Tensor, **kwargs) -> List[str]:
        assert token_ids.ndim <= 2

        if token_ids.ndim == 1:
            return self.decode(token_ids, **kwargs)
        else:
            return [self.decode(ids, **kwargs) for ids in token_ids]
        
        
    def pad(self, encoded_inputs, max_length=None, padding=True, return_tensors="pt", **kwargs):
        # encoded_inputs: list[dict]  (batch)
        if isinstance(encoded_inputs, list):
            # 提取每个样本的 input_ids
            input_ids_list = [e["input_ids"] for e in encoded_inputs]
        else:
            input_ids_list = encoded_inputs["input_ids"]

        # 把所有 input_ids 转成 list[int]
        normed_ids_list = []
        for ids in input_ids_list:
            if isinstance(ids, list):
                normed_ids_list.append(ids)
            elif hasattr(ids, "tolist"):  # torch.Tensor or numpy
                normed_ids_list.append(ids.tolist())
            else:
                raise TypeError(f"Unsupported type for input_ids: {type(ids)}")

        # 找出最大长度
        if max_length is None:
            max_len = max(len(ids) for ids in normed_ids_list)
        else:
            max_len = max_length

        # padding
        padded_ids = []
        attention_masks = []
        for ids in normed_ids_list:
            pad_len = max_len - len(ids)
            padded_ids.append(ids + [self.pad_token_id] * pad_len)
            attention_masks.append([1] * len(ids) + [0] * pad_len)

        if return_tensors == "pt":
            import torch
            padded_ids = torch.tensor(padded_ids, dtype=torch.long)
            attention_masks = torch.tensor(attention_masks, dtype=torch.long)

        return {
            "input_ids": padded_ids,
            "attention_mask": attention_masks,
        }
        
    def get_special_tokens_mask(self, token_ids, already_has_special_tokens=False):
        """
        token_ids: list[int] / list[list[int]] / torch.Tensor
        返回: 同结构的 mask, 1 表示特殊 token, 0 表示普通 token
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()

        if len(token_ids) == 0:
            return []

        if isinstance(token_ids[0], list):
            return [
                [1 if t in {self.pad_token_id, self.bos_token_id, self.eos_token_id, self.mask_token_id} else 0
                for t in seq]
                for seq in token_ids
            ]
        else:
            return [1 if t in {self.pad_token_id, self.bos_token_id, self.eos_token_id, self.mask_token_id} else 0
                    for t in token_ids]
            
    def convert_tokens_to_ids(self, tokens):
        """
        tokens: str 或 list[str]
        返回: int 或 list[int]
        """
        if isinstance(tokens, str):
            return self._convert_token_to_id(tokens)
        elif isinstance(tokens, list):
            return [self._convert_token_to_id(t) for t in tokens]
        else:
            raise TypeError(f"Unsupported type for convert_tokens_to_ids: {type(tokens)}")

    def convert_ids_to_tokens(self, ids):
        if isinstance(ids, int):
            return self._convert_id_to_token(ids)
        elif isinstance(ids, list):
            return [self._convert_id_to_token(i) for i in ids]
        else:
            raise TypeError(f"Unsupported type for convert_ids_to_tokens: {type(ids)}")


    def save_vocabulary(self, save_directory):
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        vocab_file = os.path.join(save_directory, "dictionary.pickle")
        with open(vocab_file, "wb") as f:
            pickle.dump(self.vocab, f)
        return (vocab_file,)
    
    def save_pretrained(self, save_directory):
        files = self.save_vocabulary(save_directory)
        config = {
            "unk_token": self.unk_token,
            "pad_token": self.pad_token,
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
            "mask_token": self.mask_token,
        }
        config_file = os.path.join(save_directory, "tokenizer_config.json")
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        return files + (config_file,)

    @classmethod
    def from_pretrained(cls, pretrained_path, **kwargs):
        vocab_file = os.path.join(pretrained_path, "dictionary.pickle")
        config_path = os.path.join(pretrained_path, "tokenizer_config.json")

        assert os.path.exists(vocab_file) and os.path.exists(config_path)
        
        with open(vocab_file, "rb") as f:
            vocab = pickle.load(f)
        config = {}
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        config.update(kwargs)
        return cls(vocab, **config)
    
    def __call__(self, text, **kwargs):
        if isinstance(text, (list, tuple)):
            tmp = [self.encode(t, **kwargs) for t in text]
            input_ids = torch.stack([t['input_ids'] for t in tmp], dim=0)  # 使用stack而不是cat
            attention_mask = torch.stack([t['attention_mask'] for t in tmp], dim=0)
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask
            }
        else:
            return self.encode(text, **kwargs)

class OptiSeqTokenizer:
    def __init__(self, vocab, unk_token="<unk>", pad_token="<pad>", bos_token="<bos>", eos_token="<eos>", **kwargs):
        self.vocab = vocab
        self.ids_to_tokens = {v: k for k, v in vocab.items()}

        self.unk_token = unk_token
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token

        self.unk_token_id = vocab.get(unk_token)
        self.pad_token_id = vocab.get(pad_token)
        self.bos_token_id = vocab.get(bos_token)
        self.eos_token_id = vocab.get(eos_token)

    def get_vocab(self):
        return self.vocab
    
    def get_vocab_size(self):
        return len(self.vocab)

    def _tokenize(self, text: str) -> List[str]:
        return text.strip().split()

    def _convert_token_to_id(self, token: str) -> int:
        return self.vocab.get(token, self.unk_token_id)

    def _convert_id_to_token(self, index: int) -> str:
        return self.ids_to_tokens.get(index, self.unk_token)

    def encode(self, text: str, max_length: Optional[int] = None, truncation=True, padding=False):
        tokens = self._tokenize(text)
        token_ids = [self._convert_token_to_id(t) for t in tokens]

        token_ids = [self.bos_token_id] + token_ids + [self.eos_token_id]

        if truncation and max_length is not None:
            token_ids = token_ids[:max_length]

        attention_mask = [1]*len(token_ids)

        if padding and max_length is not None and len(token_ids) < max_length:
            pad_len = max_length - len(token_ids)
            token_ids += [self.pad_token_id]*pad_len
            attention_mask += [0]*pad_len

        return {
            "input_ids": torch.tensor([token_ids], dtype=torch.long),
            "attention_mask": torch.tensor([attention_mask], dtype=torch.long),
        }

    def decode(self, token_ids: torch.Tensor, skip_special_tokens=True) -> str:
        assert token_ids.ndim == 1
        token_ids = token_ids.tolist()

        tokens = []
        special_tokens = {self.pad_token, self.unk_token, self.bos_token, self.eos_token}
        for id_ in token_ids:
            token = self._convert_id_to_token(id_)
            if skip_special_tokens and token in special_tokens:
                continue
            tokens.append(token)
        return " ".join(tokens)
    
    def batch_decode(self, token_ids: torch.Tensor, **kwargs) -> List[str]:
        assert token_ids.ndim <= 2

        if token_ids.ndim == 1:
            return self.decode(token_ids, **kwargs)
        else:
            return [self.decode(ids, **kwargs) for ids in token_ids]

    def save_vocabulary(self, save_directory):
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        vocab_file = os.path.join(save_directory, "vocab.txt")
        with open(vocab_file, "w", encoding="utf-8") as f:
            writer = csv.writer(f)
            for token, idx in sorted(self.vocab.items(), key=lambda x:x[1]):
                writer.writerow([idx, token])
        return (vocab_file,)
    
    def save_pretrained(self, save_directory):
        files = self.save_vocabulary(save_directory)
        config = {
            "unk_token": self.unk_token,
            "pad_token": self.pad_token,
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
        }
        config_file = os.path.join(save_directory, "tokenizer_config.json")
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        return files + (config_file,)

    @classmethod
    def from_pretrained(cls, pretrained_path, **kwargs):
        vocab_file = os.path.join(pretrained_path, "vocab.txt")
        vocab = {}
        with open(vocab_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                vocab[row[1]] = int(row[0])
        config_path = os.path.join(pretrained_path, "tokenizer_config.json")
        config = {}
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
        config.update(kwargs)
        return cls(vocab, **config)
    
    def __call__(self, text, **kwargs):
        if isinstance(text, (list, tuple)):
            # return [self.encode(t, **kwargs) for t in text]
            tmp = [self.encode(t, **kwargs) for t in text]
            input_ids = torch.cat([t['input_ids'] for t in tmp], dim=0)
            attention_mask = torch.cat([t['attention_mask'] for t in tmp], dim=0)
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask
            }
        else:
            return self.encode(text, **kwargs)




if __name__ == '__main__':
    # test Inst2VecTokenizer
    tokenizer_path = '/home/xucong24/Compiler/checkpoints/Inst2VecTokenizer'
    tokenizer = Inst2VecTokenizer.from_pretrained(tokenizer_path)
    with open('/home/xucong24/Compiler/datasets/poj104/ir_test/1/24.ll', 'r') as f:
        llvm = f.read()
    llvms = [llvm, llvm, llvm]
    encoding = tokenizer(
        llvms, 
        max_length=512, 
        truncation=True,
        padding=True
    )
    print(encoding)
    print(tokenizer.decode(encoding['input_ids'][0]))
    print((encoding['input_ids'][0] == tokenizer.unk_token_id).sum().item())
    print((encoding['attention_mask'][0] == 1).sum().item())
    # test OptiSeqTokenizer
    # tokenizer_path = '/home/xucong24/Compiler/checkpoints/OptiSeqTokenizer'
    # tokenizer = OptiSeqTokenizer.from_pretrained(tokenizer_path)
    # opt = [
    #     '-adce -alignment-from-assumptions -loop-unroll',
    #     '-adce -alignment-from-assumptions -alignment-from-assumptions'
    # ]
    # encoding = tokenizer(
    #     opt,
    #     truncation=True,
    #     max_length=10,
    #     padding=True
    # )
    # print(encoding)
    # print(tokenizer.decode(encoding['input_ids'][0]))
