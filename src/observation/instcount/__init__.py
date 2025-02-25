from .instcount import *

INST_COUNT_FEATURE_NAMES = [
    "TotalInsts",
    "TotalBlocks",
    "TotalFuncs",
    "Ret",
    "Br",
    "Switch",
    "IndirectBr",
    "Invoke",
    "Resume",
    "Unreachable",
    "CleanupRet",
    "CatchRet",
    "CatchSwitch",
    "CallBr",
    "FNeg",
    "Add",
    "FAdd",
    "Sub",
    "FSub",
    "Mul",
    "FMul",
    "UDiv",
    "SDiv",
    "FDiv",
    "URem",
    "SRem",
    "FRem",
    "Shl",
    "LShr",
    "AShr",
    "And",
    "Or",
    "Xor",
    "Alloca",
    "Load",
    "Store",
    "GetElementPtr",
    "Fence",
    "AtomicCmpXchg",
    "AtomicRMW",
    "Trunc",
    "ZExt",
    "SExt",
    "FPToUI",
    "FPToSI",
    "UIToFP",
    "SIToFP",
    "FPTrunc",
    "FPExt",
    "PtrToInt",
    "IntToPtr",
    "BitCast",
    "AddrSpaceCast",
    "CleanupPad",
    "CatchPad",
    "ICmp",
    "FCmp",
    "PHI",
    "Call",
    "Select",
    "UserOp1",
    "UserOp2",
    "VAArg",
    "ExtractElement",
    "InsertElement",
    "ShuffleVector",
    "ExtractValue",
    "InsertValue",
    "LandingPad",
    "Freeze",
]

INST_COUNT_FEATURE_DIMENSIONALITY = len(INST_COUNT_FEATURE_NAMES)

INST_COUNT_FEATURE_SHAPE = (INST_COUNT_FEATURE_DIMENSIONALITY,)