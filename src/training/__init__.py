"""Training module for LLVM compiler optimization research.

This module implements the three-stage training pipeline:

Stage 1 - Pretrain (pretrain/):
    Unsupervised program semantic representation learning using MLM.
    - InstBertMLMTrainer: InstBERT (Inst2Vec + ModernBERT) MLM pretraining
    - BertMLMTrainer: Standard BERT MLM pretraining

Stage 2 - Supervised (supervised/):
    Heuristic search-based sequence generation pretraining.
    - OptSeqTrainer: Encoder-Decoder for optimization sequence generation
    - ClassificationTrainer: Downstream classification tasks

Stage 3 - RL (rl/):
    MDP-based reinforcement learning fine-tuning.
    - PPOTrainer: PPO algorithm for sequence optimization
    - CompilerEnv: LLVM compiler optimization environment
    - Reward functions: IR reduction, execution time improvement

Usage:
    # Stage 1: MLM Pretraining
    from src.training.pretrain import InstBertMLMTrainer
    trainer = InstBertMLMTrainer("config.yaml")
    trainer.run()
    
    # Stage 2: Seq2Seq Training
    from src.training.supervised import OptSeqTrainer
    trainer = OptSeqTrainer("config.yaml")
    trainer.run()
    
    # Stage 3: RL Fine-tuning
    from src.training.rl import PPOTrainer
    trainer = PPOTrainer("config.yaml", encoder_model, encoder_tokenizer)
    trainer.train(ir_dataset)
"""

# Base trainers
from .base import BaseTrainer, BaseMLMTrainer, BaseSeq2SeqTrainer

# Stage 1: Pretrain
from .pretrain import InstBertMLMTrainer, BertMLMTrainer

# Stage 2: Supervised
from .supervised import OptSeqTrainer, ClassificationTrainer

# Stage 3: RL
from .rl import PPOTrainer
from .rl.env import CompilerEnv
from .rl.reward import RewardFunction, IRReductionReward, ExecutionTimeReward

__all__ = [
    # Base
    "BaseTrainer",
    "BaseMLMTrainer",
    "BaseSeq2SeqTrainer",
    # Pretrain (Stage 1)
    "InstBertMLMTrainer",
    "BertMLMTrainer",
    # Supervised (Stage 2)
    "OptSeqTrainer",
    "ClassificationTrainer",
    # RL (Stage 3)
    "PPOTrainer",
    "CompilerEnv",
    "RewardFunction",
    "IRReductionReward",
    "ExecutionTimeReward",
]
