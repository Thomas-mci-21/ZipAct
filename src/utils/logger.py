import json
import os
from datetime import datetime
from typing import Dict, Any, List

class Logger:
    """Logger for tracking agent execution and evaluation metrics."""
    
    def __init__(self, log_dir: str = "logs", experiment_name: str = None):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        if experiment_name is None:
            experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.experiment_name = experiment_name
        self.log_file = os.path.join(log_dir, f"{experiment_name}.jsonl")
        self.summary_file = os.path.join(log_dir, f"{experiment_name}_summary.json")
        
        self.episodes = []
        self.current_episode = None
    
    def start_episode(self, task: str, episode_id: int = None):
        """Start logging a new episode."""
        self.current_episode = {
            "episode_id": episode_id,
            "task": task,
            "steps": [],
            "start_time": datetime.now().isoformat(),
            "success": False,
            "reward": 0,
            "num_steps": 0,
            "token_usage": {}
        }
    
    def log_step(self, step_num: int, observation: str, thought: str, action: str, state: Dict = None):
        """Log a single step within an episode."""
        if self.current_episode is None:
            return
        
        step_data = {
            "step": step_num,
            "observation": observation,
            "thought": thought,
            "action": action
        }
        
        if state:
            step_data["state"] = state
        
        self.current_episode["steps"].append(step_data)
    
    def end_episode(self, success: bool, reward: float, token_usage: Dict[str, int]):
        """End the current episode and save results."""
        if self.current_episode is None:
            return
        
        self.current_episode["success"] = success
        self.current_episode["reward"] = reward
        self.current_episode["num_steps"] = len(self.current_episode["steps"])
        self.current_episode["token_usage"] = token_usage
        self.current_episode["end_time"] = datetime.now().isoformat()
        
        # Save to JSONL
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(self.current_episode, ensure_ascii=False) + "\n")
        
        self.episodes.append(self.current_episode)
        self.current_episode = None
    
    def save_summary(self, agent_name: str, model_name: str, dataset_name: str):
        """Save summary statistics."""
        if not self.episodes:
            return
        
        total_episodes = len(self.episodes)
        successful_episodes = sum(1 for ep in self.episodes if ep["success"])
        success_rate = successful_episodes / total_episodes if total_episodes > 0 else 0
        
        avg_steps = sum(ep["num_steps"] for ep in self.episodes) / total_episodes
        avg_reward = sum(ep["reward"] for ep in self.episodes) / total_episodes
        
        total_input_tokens = sum(ep["token_usage"].get("input_tokens", 0) for ep in self.episodes)
        total_output_tokens = sum(ep["token_usage"].get("output_tokens", 0) for ep in self.episodes)
        total_tokens = total_input_tokens + total_output_tokens
        
        summary = {
            "agent": agent_name,
            "model": model_name,
            "dataset": dataset_name,
            "experiment_name": self.experiment_name,
            "total_episodes": total_episodes,
            "successful_episodes": successful_episodes,
            "success_rate": success_rate,
            "avg_steps": avg_steps,
            "avg_reward": avg_reward,
            "token_usage": {
                "total_input_tokens": total_input_tokens,
                "total_output_tokens": total_output_tokens,
                "total_tokens": total_tokens,
                "avg_tokens_per_episode": total_tokens / total_episodes if total_episodes > 0 else 0
            }
        }
        
        with open(self.summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        return summary
    
    def print_summary(self, summary: Dict = None):
        """Print summary statistics."""
        if summary is None and self.episodes:
            summary = self.save_summary("unknown", "unknown", "unknown")
        
        if summary:
            print("\n" + "="*50)
            print(f"EXPERIMENT SUMMARY: {summary.get('experiment_name', 'N/A')}")
            print("="*50)
            print(f"Agent: {summary.get('agent', 'N/A')}")
            print(f"Model: {summary.get('model', 'N/A')}")
            print(f"Dataset: {summary.get('dataset', 'N/A')}")
            print(f"Success Rate: {summary.get('success_rate', 0):.2%} ({summary.get('successful_episodes', 0)}/{summary.get('total_episodes', 0)})")
            print(f"Avg Steps: {summary.get('avg_steps', 0):.2f}")
            print(f"Avg Reward: {summary.get('avg_reward', 0):.2f}")
            print(f"Total Tokens: {summary.get('token_usage', {}).get('total_tokens', 0):,}")
            print(f"Avg Tokens/Episode: {summary.get('token_usage', {}).get('avg_tokens_per_episode', 0):.0f}")
            print("="*50 + "\n")
