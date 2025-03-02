"""This module defines an API for processing LLVM-IR with inst2vec."""
import os
import pickle
from .inst2vec_preprocess import *
import numpy as np
from typing import List

vocab_path = "src/observation/inst2vec/pickle/dictionary.pickle"
embedding_path = "src/observation/inst2vec/pickle/embeddings.pickle"


class Inst2vecEncoder:
    """An LLVM encoder for inst2vec."""

    def __init__(self):
        # TODO(github.com/facebookresearch/CompilerGym/issues/122): Lazily
        # instantiate inst2vec encoder.
        with open(vocab_path, "rb") as f:
            self.vocab = pickle.load(f)

        with open(embedding_path, "rb") as f:
            self.embeddings = pickle.load(f)

        self.unknown_vocab_element = self.vocab["!UNK"]

    def preprocess(self, ir: str) -> List[str]:
        """Produce a list of pre-processed statements from an IR."""
        if os.path.isfile(ir): # 如果输入文件路径
            with open(ir, "r") as f:
                ir = f.read()
            
        lines = [[x] for x in ir.split("\n")]
        try:
            structs = inst2vec_preprocess.GetStructTypes(ir)
            for line in lines:
                for struct, definition in structs.items():
                    line[0] = line[0].replace(struct, definition)
        except ValueError:
            pass

        preprocessed_lines, _ = inst2vec_preprocess.preprocess(lines)
        preprocessed_texts = [
            inst2vec_preprocess.PreprocessStatement(x[0]) if len(x) else ""
            for x in preprocessed_lines
        ]
        return [x for x in preprocessed_texts if x]

    def encode(self, preprocessed: List[str]) -> List[int]:
        """Produce embedding indices for a list of pre-processed statements."""
        return [
            self.vocab.get(statement, self.unknown_vocab_element)
            for statement in preprocessed
        ]

    def embed(self, encoded: List[int]) -> np.ndarray:
        """Produce a matrix of embeddings from a list of encoded statements."""
        return np.vstack([self.embeddings[index] for index in encoded])


if __name__ == "__main__":

    llvm_ir = r"""
; ModuleID = 'example.bc'
source_filename = "example.c"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@.str = private unnamed_addr constant [14 x i8] c"Hello, World!\\00", align 1

define i32 @compare(i8* %0, i8* %1) #0 {
  %3 = alloca i8*, align 8

  ; Comparing the two pointers (simple equality check)
  %4 = icmp eq i8* %0, %1   ; compare if %0 == %1
  %5 = zext i1 %4 to i32    ; convert boolean to integer (0 or 1)
  ret i32 %5
}

declare i32 @printf(i8* nocapture readonly) #1
"""
    
    encoder = Inst2vecEncoder()
    text = encoder.preprocess(llvm_ir)
    encode_text = encoder.encode(text)
    embed_text = encoder.embed(encode_text)
    
    print(text, encode_text, embed_text)