import subprocess
import re
from collections import defaultdict

def get_inst_count(llvm_ir_file):
    # 运行 opt -stats
    result = subprocess.run(["opt", "-stats", "-disable-output", llvm_ir_file],
                            capture_output=True, text=True)

    inst_count = defaultdict(int)
    
    # 解析 opt 的 stderr 输出
    for line in result.stderr.split("\n"):
        match = re.match(r"(\d+)\s+(.*) Instruction", line)
        if match:
            count, inst = match.groups()
            inst_count[inst] = int(count)

    # 获取基本统计信息
    stats = {
        "TotalInsts": inst_count.get("TotalInsts", 0),
        "TotalBlocks": inst_count.get("TotalBlocks", 0),
        "TotalFuncs": inst_count.get("TotalFuncs", 0),
    }

    # 所有指令名称
    instruction_names = [
        "Ret", "Br", "Switch", "IndirectBr", "Invoke", "Resume", "Unreachable",
        "CleanupRet", "CatchRet", "CatchSwitch", "CallBr", "FNeg", "Add", "FAdd",
        "Sub", "FSub", "Mul", "FMul", "UDiv", "SDiv", "FDiv", "URem", "SRem",
        "FRem", "Shl", "LShr", "AShr", "And", "Or", "Xor", "Alloca", "Load",
        "Store", "GetElementPtr", "Fence", "AtomicCmpXchg", "AtomicRMW", "Trunc",
        "ZExt", "SExt", "FPToUI", "FPToSI", "UIToFP", "SIToFP", "FPTrunc", "FPExt",
        "PtrToInt", "IntToPtr", "BitCast", "AddrSpaceCast", "CleanupPad", "CatchPad",
        "ICmp", "FCmp", "PHI", "Call", "Select", "UserOp1", "UserOp2", "VAArg",
        "ExtractElement", "InsertElement", "ShuffleVector", "ExtractValue",
        "InsertValue", "LandingPad", "Freeze"
    ]

    # 统计每种指令的数量
    for inst in instruction_names:
        stats[inst] = inst_count.get(inst, 0)

    return stats

# 测试
llvm_ir_file = "/tmp/tmpha1rb81s/optimized.ll"  # 你的 LLVM IR 文件
counts = get_inst_count(llvm_ir_file)
print(counts)
