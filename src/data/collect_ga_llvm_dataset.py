import compiler_gym
import pandas as pd
import csv
from datasets import Dataset
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

input_path = Path("/home/xucong24/CompilerOptimizationByTransformer/data")
save_path = Path("/home/xucong24/Compiler/datasets/GA_Llvm_Opt_1")

# 全局映射，将在所有进程初始化时加载
action2idx = {}
idx2action = {}

with open(input_path / 'action2idx.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        action2idx[row[1]] = int(row[0])
        idx2action[int(row[0])] = row[1]

df = pd.read_csv(input_path / 'results.csv', index_col=0)
print(df.info())

# 进程内全局环境变量，必须声明为 global
_env = None

def init_worker():
    """每个进程启动时调用，初始化环境和映射"""
    global _env, action2idx

    _env = compiler_gym.make("llvm-v0")

    # 由于 action2idx 是模块全局变量，为了确保每个进程都有它，进行赋值
    # 一般锁在 fork 模型下不用全局重新加载，这里留空或根据情况调整
    # 也可以将 action2idx 用共享内存或文件加载来保证一致
    
def process_row(row_tuple):
    """每条数据处理函数，参数为df.iterrows()返回的(index, row)元组"""
    global _env, action2idx

    idx, row = row_tuple
    benchmark = row['benchmark']
    reward = row['reward']
    actions = row['commandline']
    # try:
    #     actions = [action2idx[action] for action in row['commandline'].split()]
    # except KeyError as e:
    #     print(f"Action not found in dictionary: {e}")
    #     return None

    try:
        _env.reset(benchmark=benchmark)
    except Exception as e:
        print(f"Error with benchmark {benchmark}: {e}")
        return None

    # 读取需要的观察量
    try:
        return {
            'Benchmark': benchmark,
            'CpuInfo': _env.observation['CpuInfo'],
            'IrInstructionCountO0': _env.observation['IrInstructionCountO0'],
            'IrInstructionCountO3': _env.observation['IrInstructionCountO3'],
            'IrInstructionCountOz': _env.observation['IrInstructionCountOz'],
            'InstCount': _env.observation['InstCount'],
            'Autophase': _env.observation['Autophase'],
            'Reward': reward,
            'Commandline': actions,
            'LLVM_IR': _env.observation['Ir']
        }
    except Exception as e:
        print(f"Error reading observations for {benchmark}: {e}")
        return None


def main():
    data_list = []
    # 使用ProcessPoolExecutor，初始化器函数在每个进程调用
    with ProcessPoolExecutor(max_workers=64, initializer=init_worker) as executor:
        # 提交任务，这是个生成器，每个元素是(index, row)
        futures = {executor.submit(process_row, item): item[0] for item in df.iterrows()}

        # 按原始索引顺序收集结果，而不是按完成顺序
        results_dict = {}
        for future in tqdm(as_completed(futures), total=len(futures)):
            idx = futures[future]
            result = future.result()
            if result is not None:
                results_dict[idx] = result
        
        # 按原始顺序排序
        for idx in sorted(results_dict.keys()):
            data_list.append(results_dict[idx])

    print(f"Processed {len(data_list)} rows")

    # 保存到磁盘
    Dataset.from_list(data_list).save_to_disk(save_path)


if __name__ == "__main__":
    main()
