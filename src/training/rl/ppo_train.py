"""PPO training script for compiler optimization.

Stage 3: MDP-based reinforcement learning fine-tuning using PPO.

This module implements PPO (Proximal Policy Optimization) for learning
optimal LLVM optimization sequences. The policy network is initialized
from the seq2seq model trained in Stage 2.
"""
import os
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from src.config import Config, load_config
from src.utils.utils import get_logger
from src.training.rl.env.compiler_env import CompilerEnv, EnvConfig
from src.training.rl.reward import RewardFunction, IRReductionReward


@dataclass
class PPOConfig:
    """Configuration for PPO training."""
    # PPO hyperparameters
    learning_rate: float = 3e-4
    gamma: float = 0.99  # Discount factor
    gae_lambda: float = 0.95  # GAE lambda
    clip_epsilon: float = 0.2  # PPO clip range
    value_coef: float = 0.5  # Value loss coefficient
    entropy_coef: float = 0.01  # Entropy bonus coefficient
    max_grad_norm: float = 0.5  # Gradient clipping
    
    # Training settings
    num_episodes: int = 1000
    steps_per_episode: int = 50
    batch_size: int = 64
    epochs_per_update: int = 4
    
    # Model settings
    hidden_size: int = 768
    num_actions: int = 50  # Number of optimization passes
    
    # Logging
    log_interval: int = 10
    save_interval: int = 100


class ActorCritic(nn.Module):
    """Actor-Critic network for PPO.
    
    The actor outputs action probabilities over optimization passes.
    The critic estimates the value function.
    """
    
    def __init__(self, input_size: int, hidden_size: int, num_actions: int):
        super().__init__()
        
        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        
        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, num_actions),
        )
        
        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            x: Input state tensor
            
        Returns:
            action_logits: Logits for action distribution
            value: Estimated state value
        """
        features = self.shared(x)
        action_logits = self.actor(features)
        value = self.critic(features)
        return action_logits, value
    
    def get_action(self, state: torch.Tensor) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """Sample action from policy.
        
        Args:
            state: Current state
            
        Returns:
            action: Sampled action index
            log_prob: Log probability of action
            value: Estimated state value
        """
        action_logits, value = self.forward(state)
        dist = Categorical(logits=action_logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob, value
    
    def evaluate_actions(
        self,
        states: torch.Tensor,
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate actions for PPO update.
        
        Args:
            states: Batch of states
            actions: Batch of actions taken
            
        Returns:
            log_probs: Log probabilities of actions
            values: Estimated state values
            entropy: Policy entropy
        """
        action_logits, values = self.forward(states)
        dist = Categorical(logits=action_logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, values.squeeze(-1), entropy


class RolloutBuffer:
    """Buffer for storing rollout data."""
    
    def __init__(self):
        self.states: List[np.ndarray] = []
        self.actions: List[int] = []
        self.rewards: List[float] = []
        self.log_probs: List[torch.Tensor] = []
        self.values: List[torch.Tensor] = []
        self.dones: List[bool] = []
    
    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        log_prob: torch.Tensor,
        value: torch.Tensor,
        done: bool
    ):
        """Add a transition to the buffer."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)
    
    def clear(self):
        """Clear the buffer."""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.log_probs.clear()
        self.values.clear()
        self.dones.clear()
    
    def __len__(self):
        return len(self.states)


class PPOTrainer:
    """PPO Trainer for compiler optimization.
    
    This trainer implements the PPO algorithm for Stage 3 of the research:
    MDP-based reinforcement learning fine-tuning.
    """
    
    def __init__(
        self,
        config_path: str,
        encoder_model=None,
        encoder_tokenizer=None
    ):
        """Initialize PPO trainer.
        
        Args:
            config_path: Path to configuration file
            encoder_model: Pretrained InstBERT encoder
            encoder_tokenizer: Inst2Vec tokenizer
        """
        self.cfg = load_config(config_path)
        self.ppo_cfg = PPOConfig(**self.cfg.get("ppo", {}))
        
        # Create work directory
        base_work_dir = self.cfg.get("output", {}).get("base_work_dir", "./work_dirs/ppo")
        time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.work_dir = os.path.join(base_work_dir, time_str)
        os.makedirs(self.work_dir, exist_ok=True)
        
        self.logger = get_logger(self.work_dir)
        self.logger.info(f"PPO Trainer initialized. Work dir: {self.work_dir}")
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")
        
        # Environment
        env_config = EnvConfig(**self.cfg.get("env", {}))
        self.env = CompilerEnv(
            config=env_config,
            encoder_model=encoder_model,
            encoder_tokenizer=encoder_tokenizer
        )
        
        # Reward function
        self.reward_fn = IRReductionReward()
        
        # Actor-Critic network
        input_size = self.cfg.get("model", {}).get("hidden_size", 768)
        self.policy = ActorCritic(
            input_size=input_size,
            hidden_size=self.ppo_cfg.hidden_size,
            num_actions=self.env.num_actions
        ).to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.policy.parameters(),
            lr=self.ppo_cfg.learning_rate
        )
        
        # Rollout buffer
        self.buffer = RolloutBuffer()
        
        # Statistics
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
    
    def compute_gae(
        self,
        rewards: List[float],
        values: List[torch.Tensor],
        dones: List[bool],
        next_value: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Generalized Advantage Estimation.
        
        Args:
            rewards: List of rewards
            values: List of value estimates
            dones: List of done flags
            next_value: Value estimate for final state
            
        Returns:
            advantages: GAE advantages
            returns: Discounted returns
        """
        gamma = self.ppo_cfg.gamma
        gae_lambda = self.ppo_cfg.gae_lambda
        
        advantages = []
        gae = 0
        
        values_tensor = torch.stack(values).squeeze()
        values_np = values_tensor.detach().cpu().numpy()
        next_val = next_value.detach().cpu().item()
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_v = next_val
            else:
                next_v = values_np[t + 1]
            
            delta = rewards[t] + gamma * next_v * (1 - dones[t]) - values_np[t]
            gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        
        advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns = advantages + values_tensor
        
        return advantages, returns
    
    def update(self) -> Dict[str, float]:
        """Perform PPO update.
        
        Returns:
            Dictionary of loss values
        """
        # Convert buffer to tensors
        states = torch.tensor(
            np.array(self.buffer.states),
            dtype=torch.float32,
            device=self.device
        )
        actions = torch.tensor(self.buffer.actions, device=self.device)
        old_log_probs = torch.stack(self.buffer.log_probs).detach()
        
        # Compute GAE
        with torch.no_grad():
            _, next_value = self.policy(states[-1:])
        advantages, returns = self.compute_gae(
            self.buffer.rewards,
            self.buffer.values,
            self.buffer.dones,
            next_value.squeeze()
        )
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO epochs
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        
        for _ in range(self.ppo_cfg.epochs_per_update):
            # Evaluate actions
            log_probs, values, entropy = self.policy.evaluate_actions(states, actions)
            
            # Policy loss (PPO clip)
            ratio = torch.exp(log_probs - old_log_probs.squeeze())
            surr1 = ratio * advantages
            surr2 = torch.clamp(
                ratio,
                1 - self.ppo_cfg.clip_epsilon,
                1 + self.ppo_cfg.clip_epsilon
            ) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = nn.functional.mse_loss(values, returns)
            
            # Entropy bonus
            entropy_loss = -entropy.mean()
            
            # Total loss
            loss = (
                policy_loss +
                self.ppo_cfg.value_coef * value_loss +
                self.ppo_cfg.entropy_coef * entropy_loss
            )
            
            # Backprop
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                self.policy.parameters(),
                self.ppo_cfg.max_grad_norm
            )
            self.optimizer.step()
            
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.mean().item()
        
        num_updates = self.ppo_cfg.epochs_per_update
        return {
            'policy_loss': total_policy_loss / num_updates,
            'value_loss': total_value_loss / num_updates,
            'entropy': total_entropy / num_updates
        }
    
    def collect_rollouts(self, ir_samples: List[str]) -> float:
        """Collect rollouts from environment.
        
        Args:
            ir_samples: List of LLVM IR samples
            
        Returns:
            Average episode reward
        """
        self.buffer.clear()
        total_reward = 0
        num_episodes = 0
        
        for ir_code in ir_samples:
            state = self.env.reset(ir_code)
            episode_reward = 0
            
            for step in range(self.ppo_cfg.steps_per_episode):
                state_tensor = torch.tensor(
                    state,
                    dtype=torch.float32,
                    device=self.device
                ).unsqueeze(0)
                
                action, log_prob, value = self.policy.get_action(state_tensor)
                
                next_state, reward, done, info = self.env.step(action)
                
                self.buffer.add(state, action, reward, log_prob, value, done)
                
                episode_reward += reward
                state = next_state
                
                if done:
                    break
            
            total_reward += episode_reward
            num_episodes += 1
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(step + 1)
        
        return total_reward / max(num_episodes, 1)
    
    def train(self, ir_dataset: List[str]):
        """Run PPO training.
        
        Args:
            ir_dataset: List of LLVM IR samples for training
        """
        self.logger.info(f"Starting PPO training with {len(ir_dataset)} samples")
        
        for episode in range(self.ppo_cfg.num_episodes):
            # Sample batch
            batch_size = min(self.ppo_cfg.batch_size, len(ir_dataset))
            indices = np.random.choice(len(ir_dataset), batch_size, replace=False)
            batch = [ir_dataset[i] for i in indices]
            
            # Collect rollouts
            avg_reward = self.collect_rollouts(batch)
            
            # Update policy
            losses = self.update()
            
            # Logging
            if (episode + 1) % self.ppo_cfg.log_interval == 0:
                self.logger.info(
                    f"Episode {episode + 1}/{self.ppo_cfg.num_episodes} | "
                    f"Avg Reward: {avg_reward:.4f} | "
                    f"Policy Loss: {losses['policy_loss']:.4f} | "
                    f"Value Loss: {losses['value_loss']:.4f} | "
                    f"Entropy: {losses['entropy']:.4f}"
                )
            
            # Save checkpoint
            if (episode + 1) % self.ppo_cfg.save_interval == 0:
                self.save_checkpoint(episode + 1)
        
        # Final save
        self.save_checkpoint('final')
        self.logger.info("PPO training completed")
    
    def save_checkpoint(self, tag: str):
        """Save model checkpoint.
        
        Args:
            tag: Checkpoint tag (episode number or 'final')
        """
        checkpoint_path = os.path.join(self.work_dir, f"checkpoint_{tag}.pt")
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
        }, checkpoint_path)
        self.logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.episode_rewards = checkpoint.get('episode_rewards', [])
        self.episode_lengths = checkpoint.get('episode_lengths', [])
        self.logger.info(f"Checkpoint loaded from {checkpoint_path}")


def main():
    """Main entry point for PPO training."""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    
    # Load pretrained encoder (from Stage 1)
    # encoder_model = ...
    # encoder_tokenizer = ...
    
    trainer = PPOTrainer(
        config_path=args.config,
        encoder_model=None,  # TODO: Load pretrained InstBERT
        encoder_tokenizer=None  # TODO: Load Inst2Vec tokenizer
    )
    
    # Load IR dataset
    # ir_dataset = ...
    
    # trainer.train(ir_dataset)
    print("PPO trainer initialized. Load encoder and dataset to start training.")


if __name__ == "__main__":
    main()

