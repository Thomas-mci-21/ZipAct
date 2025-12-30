"""
SciWorld Environment Wrapper for ZipAct.
SciWorld is a text-based environment for scientific reasoning tasks.

Install: pip install scienceworld
"""

import re
from typing import Tuple, Dict, List, Optional
from .base import BaseEnv


class SciWorldEnv(BaseEnv):
    """
    Wrapper for SciWorld environment.
    SciWorld is a text-based environment for scientific reasoning tasks
    such as boiling water, melting ice, conducting electricity, etc.
    
    Note: This requires the scienceworld package to be installed.
    Install with: pip install scienceworld
    """
    
    # Available task types in SciWorld
    TASK_TYPES = [
        "boil",           # Boil water
        "melt",           # Melt ice/substances
        "freeze",         # Freeze water
        "change-the-state-of-matter-of",  # Generic phase change
        "use-thermometer",  # Temperature measurement
        "measure-melting-point-known-substance",
        "measure-melting-point-unknown-substance", 
        "power-component",  # Electrical circuits
        "power-component-renewable-vs-nonrenewable",
        "test-conductivity",
        "test-conductivity-of-unknown-substances",
        "find-living-thing",
        "find-non-living-thing",
        "find-plant",
        "find-animal",
        "grow-plant",
        "grow-fruit",
        "chemistry-mix",
        "chemistry-mix-paint-secondary-color",
        "chemistry-mix-paint-tertiary-color",
        "lifespan-longest-lived",
        "lifespan-shortest-lived",
        "identify-life-stages-1",
        "identify-life-stages-2",
        "inclined-plane-determine-friction",
        "inclined-plane-friction-named-surfaces",
        "mendelian-genetics-known-plant",
        "mendelian-genetics-unknown-plant",
    ]
    
    def __init__(
        self, 
        task_name: str = "boil", 
        variation_idx: int = 0,
        simplifications_preset: str = "easy",
        seed: int = 42
    ):
        """
        Initialize SciWorld environment.
        
        Args:
            task_name: Which SciWorld task to run (e.g., 'boil', 'melt', 'freeze')
            variation_idx: Task variation index (different scenarios for same task)
            simplifications_preset: Difficulty level ('easy', 'medium', 'hard')
            seed: Random seed
        """
        try:
            from scienceworld import ScienceWorldEnv
        except ImportError:
            raise ImportError(
                "SciWorld is not installed. Please install with:\n"
                "  pip install scienceworld\n"
                "For more info: https://github.com/allenai/ScienceWorld"
            )
        
        self.task_name = task_name
        self.variation_idx = variation_idx
        self.simplifications_preset = simplifications_preset
        self.seed = seed
        
        # Initialize SciWorld environment
        self.env = ScienceWorldEnv("")
        
        # Load the specific task
        self.env.load(task_name, variation_idx, simplifications_preset)
        
        self.current_task = ""
        self.task_description = ""
        self.max_score = 0
        self.current_score = 0
    
    def reset(self) -> Tuple[str, Dict]:
        """
        Reset environment and return initial observation.
        
        Returns:
            observation: Text description of the environment
            info: Additional information (task, max_score, valid_actions)
        """
        obs, info = self.env.reset()
        
        # Get task description
        self.task_description = self.env.taskdescription()
        self.current_task = self._extract_task(obs, self.task_description)
        self.max_score = self.env.getMaxScore()
        self.current_score = 0
        
        info = {
            'task': self.current_task,
            'task_description': self.task_description,
            'max_score': self.max_score,
            'variation': self.variation_idx,
        }
        
        return obs, info
    
    def step(self, action: str) -> Tuple[str, float, bool, Dict]:
        """
        Execute action and return result.
        
        Args:
            action: Action string to execute
            
        Returns:
            observation: New observation
            reward: Normalized reward (0-1)
            done: Whether episode is finished
            info: Additional information
        """
        obs, reward, done, info = self.env.step(action)
        
        # Update current score
        self.current_score = self.env.getScore()
        
        # Normalize reward to [0, 1]
        normalized_reward = self.current_score / self.max_score if self.max_score > 0 else 0
        
        # Add useful info
        info = info if isinstance(info, dict) else {}
        info['score'] = self.current_score
        info['max_score'] = self.max_score
        info['normalized_score'] = normalized_reward
        
        return obs, normalized_reward, done, info
    
    def _extract_task(self, observation: str, task_description: str) -> str:
        """Extract the task description."""
        # Use the task description from the environment
        if task_description:
            return task_description.strip()
        
        # Fallback: try to extract from observation
        task_match = re.search(r'Task:\s*(.+?)(?:\n|$)', observation, re.IGNORECASE)
        if task_match:
            return task_match.group(1).strip()
        
        # Fallback to task name
        return f"Complete the {self.task_name} task"
    
    def get_task(self) -> str:
        """Return the current task description."""
        return self.current_task if self.current_task else self.task_description
    
    def get_valid_actions(self) -> List[str]:
        """Return list of valid actions."""
        try:
            return self.env.getValidActionObjectCombinations()
        except:
            return []
    
    def get_valid_action_templates(self) -> List[str]:
        """Return list of valid action templates."""
        try:
            return self.env.getValidActionObjectCombinationsTemplates()
        except:
            return []
    
    def get_look(self) -> str:
        """Get description of current location."""
        try:
            return self.env.look()
        except:
            return ""
    
    def get_inventory(self) -> str:
        """Get current inventory."""
        try:
            return self.env.inventory()
        except:
            return ""
    
    def get_score(self) -> Tuple[int, int]:
        """Return (current_score, max_score)."""
        return self.current_score, self.max_score
    
    @classmethod
    def list_tasks(cls) -> List[str]:
        """Return list of available task types."""
        return cls.TASK_TYPES.copy()
    
    @classmethod
    def get_num_variations(cls, task_name: str) -> int:
        """Get number of variations for a task."""
        try:
            from scienceworld import ScienceWorldEnv
            env = ScienceWorldEnv("")
            return env.getNumVariations(task_name)
        except:
            return 0

