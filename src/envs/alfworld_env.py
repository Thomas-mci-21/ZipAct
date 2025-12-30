import os
import re
import yaml
import alfworld
import alfworld.agents.environment as environment
from typing import Tuple, Dict
from .base import BaseEnv

class ALFWorldEnv(BaseEnv):
    """
    Wrapper for ALFWorld environment.
    ALFWorld is a text-based embodied AI environment for household tasks.
    """
    
    def __init__(self, config_path: str = None, split: str = "eval_out_of_distribution", seed: int = 42):
        """
        Initialize ALFWorld environment.
        
        Args:
            config_path: Path to config yaml file. If None, use default config.
            split: Which split to use ('train', 'eval_out_of_distribution', 'eval_in_distribution')
            seed: Random seed
        """
        # Load configuration
        if config_path is None:
            # Use default config from alfworld package
            import alfworld.agents.environment as environment
            config_path = os.path.join(os.path.dirname(environment.__file__), 'configs', 'base_config.yaml')
            
            # If base config doesn't exist, create a minimal one
            if not os.path.exists(config_path):
                self.config = {
                    'env': {
                        'type': 'AlfredTWEnv',
                        'regen_game_files': False,
                        'domain_randomization': False,
                        'task_types': [1, 2, 3, 4, 5, 6]
                    },
                    'general': {
                        'random_seed': seed,
                        'train_eval': split
                    }
                }
            else:
                with open(config_path) as f:
                    self.config = yaml.safe_load(f)
                    self.config['general']['train_eval'] = split
                    self.config['general']['random_seed'] = seed
        else:
            with open(config_path) as f:
                self.config = yaml.safe_load(f)
                self.config['general']['train_eval'] = split
        
        # Initialize environment
        env_type = self.config['env']['type']
        self.env = getattr(environment, env_type)(self.config, train_eval=split)
        self.env = self.env.init_env(batch_size=1)
        
        self.current_task = None
        self.admissible_commands = []
    
    def reset(self) -> Tuple[str, Dict]:
        """
        Reset environment and return initial observation.
        
        Returns:
            observation: Text description of the environment
            info: Additional information (task description, admissible commands, etc.)
        """
        obs, infos = self.env.reset()
        obs = obs[0]  # Unwrap from batch
        info = infos if isinstance(infos, dict) else {}
        
        # Extract task instruction
        self.current_task = self._extract_task(obs)
        
        # Get admissible commands if available
        if 'admissible_commands' in info:
            self.admissible_commands = info['admissible_commands'][0] if isinstance(info['admissible_commands'], list) else []
        
        info['task'] = self.current_task
        
        return obs, info
    
    def step(self, action: str) -> Tuple[str, float, bool, Dict]:
        """
        Execute action and return result.
        
        Args:
            action: Action string to execute
            
        Returns:
            observation: New observation
            reward: Reward signal (1.0 for success, 0.0 otherwise)
            done: Whether episode is finished
            info: Additional information
        """
        obs, scores, dones, infos = self.env.step([action])
        
        obs = obs[0]
        score = scores[0]
        done = dones[0]
        info = infos if isinstance(infos, dict) else {}
        
        # Update admissible commands
        if 'admissible_commands' in info:
            self.admissible_commands = info['admissible_commands'][0] if isinstance(info['admissible_commands'], list) else []
        
        return obs, score, done, info
    
    def _extract_task(self, observation: str) -> str:
        """Extract the task description from the observation."""
        # ALFWorld observations typically contain the task in the format:
        # "You are in the middle of a room. Looking quickly around you, you see...
        #  Your task is to: put a clean apple in refrigerator."
        
        # Try to extract "Your task is to: ..." pattern
        task_match = re.search(r'Your task is to:\s*(.+?)(?:\n|$)', observation, re.IGNORECASE)
        if task_match:
            return task_match.group(1).strip()
        
        # Fallback: return last non-empty line
        lines = [line.strip() for line in observation.split('\n') if line.strip()]
        if lines:
            return lines[-1]
        
        return observation
    
    def get_task(self) -> str:
        """Return the current task description."""
        return self.current_task if self.current_task else ""
    
    def get_admissible_commands(self):
        """Return list of admissible commands (if available)."""
        return self.admissible_commands

