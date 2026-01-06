# Model module - Encoders and Tokenizers for LLVM IR processing
#
# This module provides:
# - Inst2vecEncoder: LLVM IR encoder using inst2vec embeddings
# - Inst2VecTokenizer: Tokenizer for LLVM IR (HuggingFace compatible)
# - OptiSeqTokenizer: Tokenizer for optimization sequences

from .inst2vec import (
    Inst2vecEncoder,
    preprocess,
    PreprocessStatement,
    GetStructTypes,
    GetFunctionsDeclaredInFile,
)
from .tokenizer import Inst2VecTokenizer, OptiSeqTokenizer

__all__ = [
    # Encoders
    "Inst2vecEncoder",
    # Tokenizers
    "Inst2VecTokenizer",
    "OptiSeqTokenizer",
    # Preprocessing functions
    "preprocess",
    "PreprocessStatement",
    "GetStructTypes",
    "GetFunctionsDeclaredInFile",
]
