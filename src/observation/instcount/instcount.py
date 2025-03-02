import os
import tempfile
from typing import List
from utils.system import run_executable

def compute_instcount(source: str) -> List[int]:
    # 创建临时目录
    temp_dir = tempfile.mkdtemp()
    if os.path.isfile(source):
        ir_file = source
    else:
        ir_file = os.path.join(temp_dir, "input.ll")
        with open(ir_file, "w") as f:
            f.write(source)

    executable_path = "src/observation/instcount/compute_ir_instruction_count"  # 替换为实际的可执行文件路径
    args = [ir_file]
    output = run_executable(executable_path, args)
    
    return [int(i) for i in output.split(' ')]


if __name__ == '__main__':
    print(compute_instcount('/Users/xucong/Desktop/Compiler/optimized.ll'))
