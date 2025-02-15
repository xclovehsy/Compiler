import os
import re
import tempfile
import subprocess


compiler_gym_clang = "/root/.local/share/compiler_gym/llvm-v0/bin/clang"

def count_llvm_ir_instructions(llvm_ir: str) -> int:
    """
    计算 LLVM-IR 代码中的指令数目。
    """
    # 匹配 LLVM IR 指令的正则表达式
    instruction_pattern = re.compile(r'^\s*(?:\%\d+\s*=\s*)?([a-zA-Z]+)\s', re.MULTILINE)
    
    instructions = [match.group(1) for match in instruction_pattern.finditer(llvm_ir)]
    return len(instructions)

def compile_c_to_llvm_ir(source, opt_flags=None, clang_path="/root/.local/share/compiler_gym/llvm-v0/bin/clang", opt_path="opt", llvm_dis_path="llvm-dis"):
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
            clang_path, 
            "-c", 
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
        subprocess.run([opt_path, *opt_flags, bc_file, "-o", opt_bc_file], check=True)

        # 转换优化后的 Bitcode 为 LLVM IR (.ll)
        subprocess.run([llvm_dis_path, opt_bc_file, "-o", opt_ll_file], check=True)

        with open(opt_ll_file, "r") as f:
            llvm_ir = f.read()

    except Exception as e:
        print(e)
        llvm_ir = ""

    finally:
        os.remove(bc_file)
        os.remove(opt_bc_file)
        os.remove(opt_ll_file)
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