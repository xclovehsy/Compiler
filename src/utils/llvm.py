import os
import re
import tempfile
import subprocess
import llvmlite.binding as llvm

def count_llvm_ir_instructions(llvm_ir: str) -> int:
    """
    计算 LLVM-IR 代码中的指令数目。
    """

    llvm.initialize()
    llvm.initialize_native_target()
    llvm.initialize_native_asmprinter()

    # 解析字符串中的LLVM IR
    module = llvm.parse_assembly(llvm_ir)

    # 计算指令数量
    instruction_count = 0
    for function in module.functions:
        for block in function.blocks:
            for instruction in block.instructions:  # 使用 .instructions 获取块中的指令
                instruction_count += 1
                
    return instruction_count

def compile_c_to_llvm_ir(source, opt_flags=None):
    """
    编译 C 代码到 LLVM IR，并应用优化选项，使用 .bc 进行中间存储。

    参数:
    - source: C 源代码字符串或 C 文件路径。
    - opt_flags: 需要应用的 LLVM 优化选项列表，如 ["-adce", "-instcombine"]。
    - clang_path: Clang 编译器路径。
    - opt_path: opt 工具路径。
    - llvm_dis_path: llvm-dis 工具路径。

    返回:
    - 优化后的 LLVM IR 字符串
    """
    opt_flags = opt_flags or []

    # 创建临时目录
    temp_dir = tempfile.mkdtemp()
    if os.path.isfile(source):
        c_file = source
    else:
        c_file = os.path.join(temp_dir, "input.c")
        with open(c_file, "w") as f:
            f.write(source)

    bc_file = os.path.join(temp_dir, "output.bc")
    opt_bc_file = os.path.join(temp_dir, "optimized.bc")
    opt_ll_file = os.path.join(temp_dir, "optimized.ll")

    try:
        # 生成 LLVM Bitcode (.bc)
        subprocess.run([
            'clang', 
            "-S", 
            "-emit-llvm", 
            "-o", bc_file, 
            c_file, 
            "-O1", 
            "-Xclang", "-disable-llvm-passes", 
            "-Xclang", "-disable-llvm-optzns",
            "-isystem", "/usr/include/c++/11",
            "-isystem", "/usr/include/x86_64-linux-gnu/c++/11",
            "-isystem", "/usr/include/c++/11/backward",
            "-isystem", "/usr/lib/gcc/x86_64-linux-gnu/11/include",
            "-isystem", "/usr/local/include",
            "-isystem", "/usr/include/x86_64-linux-gnu",
            "-isystem", "/usr/include"
        ], check=True)

        # 应用优化
        subprocess.run(["opt", *opt_flags, bc_file, "-o", opt_bc_file], check=True)

        # 转换优化后的 Bitcode 为 LLVM IR (.ll) 反编译
        subprocess.run(["llvm-dis", opt_bc_file, "-o", opt_ll_file], check=True)

        with open(opt_ll_file, "r") as f:
            llvm_ir = f.read()

    except Exception as e:
        print(e)
        llvm_ir = ""

    finally:
        # os.remove(bc_file)
        # os.remove(opt_bc_file)
        # os.remove(opt_ll_file)
        if not os.path.isfile(source):
            os.remove(c_file)

    return llvm_ir


if __name__ == '__main__':
    opt_flags = ["-adce", "-instcombine", "-simplifycfg", "-mem2reg"]
    c_file_path = r'/root/Compiler/data/anghabench/qsort.c'
    llvm_ir = compile_c_to_llvm_ir(c_file_path)
    print(f"优化前IR指令数量: {count_llvm_ir_instructions(llvm_ir)}")

    llvm_ir = compile_c_to_llvm_ir(c_file_path, opt_flags=opt_flags)
    print(f"优化后IR指令数量: {count_llvm_ir_instructions(llvm_ir)}")