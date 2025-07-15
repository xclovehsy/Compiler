import gym
import compiler_gym
from compiler_gym.envs.llvm import make_benchmark
from llvm import *

from compiler_gym.envs import LlvmEnv

    
if __name__ == '__main__':
    c_file_path = r'/root/Compiler/data/anghabench/qsort.c'

    with compiler_gym.make("llvm-v0") as env:
        env.reset()
        benchmark = make_benchmark(c_file_path)
        env = gym.make("llvm-autophase-ic-v0")
        env.observation
        env.reset()
        env.reset(benchmark=benchmark)
        print(env.benchmark)
        print(f"优化前IR指令数量: {env.observation['IrInstructionCount']}")
        
        for action in ["-adce", "-instcombine", "-simplifycfg", "-mem2reg"]:
            observation, reward, done, info = env.step(env.action_space.flags.index(action))
            # print(observation)
            # print(action, env.action_space.flags.index(action))
            # print(observation, reward, done, info)

        # print(env.observation['Ir'])

        print(f"优化后IR指令数量: {env.observation['IrInstructionCount']}")

        print(env.observation['InstCount'])

        print(env.commandline())
