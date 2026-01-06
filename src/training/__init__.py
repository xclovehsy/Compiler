from .base import BaseTrainer, BaseMLMTrainer, BaseSeq2SeqTrainer
from .pretrain import InstBertMLMTrainer, BertMLMTrainer

# # Stage 2: Supervised
# from .supervised import OptSeqTrainer, ClassificationTrainer

# # Stage 3: RL
# from .rl import PPOTrainer
# from .rl.env import CompilerEnv
# from .rl.reward import RewardFunction, IRReductionReward, ExecutionTimeReward

__all__ = [
    # Base
    "BaseTrainer",
    "BaseMLMTrainer",
    "BaseSeq2SeqTrainer",
    # # Pretrain (Stage 1)
    # "InstBertMLMTrainer",
    # "BertMLMTrainer",
    # # Supervised (Stage 2)
    # "OptSeqTrainer",
    # "ClassificationTrainer",
    # # RL (Stage 3)
    # "PPOTrainer",
    # "CompilerEnv",
    # "RewardFunction",
    # "IRReductionReward",
    # "ExecutionTimeReward",
]
