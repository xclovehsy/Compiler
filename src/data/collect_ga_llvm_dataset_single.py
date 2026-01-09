"""单进程版本的数据集收集脚本"""
import compiler_gym
import pandas as pd
import csv
from datasets import Dataset
from tqdm import tqdm
from pathlib import Path

input_path = Path("/home/xucong24/CompilerOptimizationByTransformer/data")
save_path = Path("/home/xucong24/Compiler/datasets/llvm_opti_seq")

# 加载 action 映射
action2idx = {}
idx2action = {}

with open(input_path / 'action2idx.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        action2idx[row[1]] = int(row[0])
        idx2action[int(row[0])] = row[1]

# 加载数据
df = pd.read_csv(input_path / 'results.csv', index_col=0)
print(df.info())


def main():
    # 初始化环境
    env = compiler_gym.make("llvm-v0")
    
    data_list = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        benchmark = row['benchmark']
        reward = row['reward']
        actions = row['commandline']
        
        try:
            env.reset(benchmark=benchmark)
        except Exception as e:
            print(f"Error with benchmark {benchmark}: {e}")
            continue
        
        # 读取观察量
        try:
            data_list.append({
                'Benchmark': benchmark,
                'CpuInfo': env.observation['CpuInfo'],
                'IrInstructionCountO0': env.observation['IrInstructionCountO0'],
                'IrInstructionCountO3': env.observation['IrInstructionCountO3'],
                'IrInstructionCountOz': env.observation['IrInstructionCountOz'],
                'InstCount': env.observation['InstCount'],
                'Autophase': env.observation['Autophase'],
                'Reward': reward,
                'Commandline': actions,
                'LLVM_IR': env.observation['Ir']
            })
        except Exception as e:
            print(f"Error reading observations for {benchmark}: {e}")
            continue
    
    env.close()
    
    print(f"Processed {len(data_list)} rows")
    
    # 保存到磁盘
    dataset = Dataset.from_list(data_list)
    dataset.save_to_disk(save_path)
    print(f"Saved to {save_path}")


if __name__ == "__main__":
    main()

