"""Reward functions for reinforcement learning.

This module defines various reward functions for LLVM optimization:
- IR reduction reward: Based on code size reduction
- Execution time reward: Based on actual runtime improvement
- Composite reward: Combination of multiple objectives
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass
import subprocess
import tempfile
import os
import time


@dataclass
class RewardConfig:
    """Configuration for reward functions."""
    # IR reduction weights
    ir_reduction_weight: float = 1.0
    
    # Execution time weights
    execution_weight: float = 1.0
    timeout: float = 30.0  # seconds
    
    # Composite reward
    use_composite: bool = False
    ir_weight: float = 0.5
    exec_weight: float = 0.5
    
    # Penalty settings
    increase_penalty: float = 2.0  # Penalty multiplier for regression
    timeout_penalty: float = -1.0


class RewardFunction(ABC):
    """Abstract base class for reward functions."""
    
    def __init__(self, config: Optional[RewardConfig] = None):
        self.config = config or RewardConfig()
    
    @abstractmethod
    def calculate(
        self,
        old_state: Dict[str, Any],
        new_state: Dict[str, Any],
        action: int
    ) -> float:
        """Calculate reward for a transition.
        
        Args:
            old_state: State before action
            new_state: State after action
            action: Action taken
            
        Returns:
            Reward value
        """
        pass
    
    def normalize(self, reward: float, min_val: float = -1.0, max_val: float = 1.0) -> float:
        """Normalize reward to a range."""
        return max(min_val, min(max_val, reward))


class IRReductionReward(RewardFunction):
    """Reward based on IR instruction count reduction.
    
    This is the primary reward function for Stage 3:
    r_t = (count_{t-1} - count_t) / count_{t-1}
    
    Positive reward for reduction, negative for increase.
    """
    
    def calculate(
        self,
        old_state: Dict[str, Any],
        new_state: Dict[str, Any],
        action: int
    ) -> float:
        """Calculate IR reduction reward.
        
        Args:
            old_state: Must contain 'ir_count'
            new_state: Must contain 'ir_count'
            action: Action taken (unused)
            
        Returns:
            Reward value in [-1, 1]
        """
        old_count = old_state.get('ir_count', 0)
        new_count = new_state.get('ir_count', 0)
        
        if old_count == 0:
            return 0.0
        
        reduction = (old_count - new_count) / old_count
        
        # Apply penalty for code size increase
        if reduction < 0:
            reduction *= self.config.increase_penalty
        
        return self.normalize(reduction * self.config.ir_reduction_weight)


class ExecutionTimeReward(RewardFunction):
    """Reward based on execution time improvement.
    
    More accurate but more expensive to compute.
    Requires running the compiled program.
    """
    
    def __init__(
        self,
        config: Optional[RewardConfig] = None,
        llvm_bin_path: str = "/usr/bin",
        test_input: Optional[str] = None
    ):
        super().__init__(config)
        self.llvm_bin_path = llvm_bin_path
        self.test_input = test_input
    
    def calculate(
        self,
        old_state: Dict[str, Any],
        new_state: Dict[str, Any],
        action: int
    ) -> float:
        """Calculate execution time reward.
        
        Args:
            old_state: Must contain 'ir_code' or 'exec_time'
            new_state: Must contain 'ir_code' or 'exec_time'
            action: Action taken (unused)
            
        Returns:
            Reward value based on speedup
        """
        # If execution times are pre-computed
        if 'exec_time' in old_state and 'exec_time' in new_state:
            old_time = old_state['exec_time']
            new_time = new_state['exec_time']
        else:
            # Measure execution times
            old_time = self._measure_execution_time(old_state.get('ir_code', ''))
            new_time = self._measure_execution_time(new_state.get('ir_code', ''))
        
        if old_time <= 0:
            return 0.0
        
        # Speedup ratio
        speedup = (old_time - new_time) / old_time
        
        # Apply penalty for slowdown
        if speedup < 0:
            speedup *= self.config.increase_penalty
        
        return self.normalize(speedup * self.config.execution_weight)
    
    def _measure_execution_time(self, ir_code: str) -> float:
        """Measure execution time of compiled IR.
        
        Args:
            ir_code: LLVM IR code
            
        Returns:
            Execution time in seconds, or -1 on error
        """
        if not ir_code:
            return -1.0
        
        try:
            # Write IR to temp file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.ll', delete=False) as f:
                f.write(ir_code)
                ir_file = f.name
            
            # Compile to executable
            exe_file = ir_file.replace('.ll', '.exe')
            llc_path = os.path.join(self.llvm_bin_path, "llc")
            
            # Compile IR to object file
            obj_file = ir_file.replace('.ll', '.o')
            subprocess.run(
                [llc_path, '-filetype=obj', ir_file, '-o', obj_file],
                capture_output=True,
                timeout=self.config.timeout
            )
            
            # Link to executable (simplified, may need proper linking)
            subprocess.run(
                ['gcc', obj_file, '-o', exe_file],
                capture_output=True,
                timeout=self.config.timeout
            )
            
            # Measure execution time
            start_time = time.time()
            result = subprocess.run(
                [exe_file],
                input=self.test_input,
                capture_output=True,
                timeout=self.config.timeout,
                text=True
            )
            end_time = time.time()
            
            if result.returncode == 0:
                return end_time - start_time
            else:
                return -1.0
                
        except subprocess.TimeoutExpired:
            return self.config.timeout
        except Exception:
            return -1.0
        finally:
            # Cleanup
            for f in [ir_file, obj_file, exe_file]:
                if f and os.path.exists(f):
                    try:
                        os.remove(f)
                    except:
                        pass


class CompositeReward(RewardFunction):
    """Composite reward combining multiple objectives.
    
    Combines IR reduction and execution time rewards:
    r = w_ir * r_ir + w_exec * r_exec
    """
    
    def __init__(
        self,
        config: Optional[RewardConfig] = None,
        ir_reward: Optional[IRReductionReward] = None,
        exec_reward: Optional[ExecutionTimeReward] = None
    ):
        super().__init__(config)
        self.ir_reward = ir_reward or IRReductionReward(config)
        self.exec_reward = exec_reward
    
    def calculate(
        self,
        old_state: Dict[str, Any],
        new_state: Dict[str, Any],
        action: int
    ) -> float:
        """Calculate composite reward.
        
        Args:
            old_state: State before action
            new_state: State after action
            action: Action taken
            
        Returns:
            Weighted combination of rewards
        """
        # IR reduction component
        ir_r = self.ir_reward.calculate(old_state, new_state, action)
        reward = self.config.ir_weight * ir_r
        
        # Execution time component (if available)
        if self.exec_reward is not None:
            exec_r = self.exec_reward.calculate(old_state, new_state, action)
            reward += self.config.exec_weight * exec_r
        
        return self.normalize(reward)

