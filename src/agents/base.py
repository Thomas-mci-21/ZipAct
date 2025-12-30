from abc import ABC, abstractmethod
from typing import Tuple, Optional


class BaseAgent(ABC):
    """Base class for all agents with step limit support."""
    
    # Default maximum steps before giving up
    DEFAULT_MAX_STEPS = 50
    
    def __init__(self, max_steps: int = None):
        """
        Initialize base agent.
        
        Args:
            max_steps: Maximum steps allowed. None means use DEFAULT_MAX_STEPS.
        """
        self.max_steps = max_steps if max_steps is not None else self.DEFAULT_MAX_STEPS
        self.current_step = 0
        self.last_thought = ""
    
    @abstractmethod
    def reset(self, instruction: str):
        """Reset agent for a new task. Should reset current_step to 0."""
        pass

    @abstractmethod
    def step(self, observation: str) -> str:
        """
        Process observation and return action.
        Should increment current_step and check step limit.
        
        Returns:
            action: Action string to execute
        """
        pass
    
    def is_step_limit_reached(self) -> bool:
        """Check if step limit has been reached."""
        return self.current_step >= self.max_steps
    
    def get_step_count(self) -> int:
        """Get current step count."""
        return self.current_step
    
    def get_last_thought(self) -> str:
        """Return the last thought for logging."""
        return self.last_thought
