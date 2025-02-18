from observation.autophase import compute_autophase, AUTOPHASE_FEATURE_NAMES
from observation.instcount import compute_instcount, INST_COUNT_FEATURE_DIMENSIONALITY
from observation.inst2vec import Inst2vecEncoder

if __name__ == '__main__':
    # print(compute_autophase('/Users/xucong/Desktop/Compiler/optimized.ll'))
    # print(AUTOPHASE_FEATURE_NAMES)
    # print(compute_instcount('/Users/xucong/Desktop/Compiler/optimized.ll'))
    # print(INST_COUNT_FEATURE_DIMENSIONALITY)

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
    text = encoder.preprocess('/Users/xucong/Desktop/Compiler/optimized.ll')
    encode_text = encoder.encode(text)
    embed_text = encoder.embed(encode_text)
    
    print(text, encode_text, embed_text)