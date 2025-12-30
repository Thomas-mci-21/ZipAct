from abc import ABC, abstractmethod
from typing import Tuple, Any

class BaseEnv(ABC):
    @abstractmethod
    def reset(self) -> Tuple[str, dict]:
        """Resets the environment and returns the initial observation and info."""
        pass

    @abstractmethod
    def step(self, action: str) -> Tuple[str, float, bool, dict]:
        """Executes an action and returns (observation, reward, done, info)."""
        pass
