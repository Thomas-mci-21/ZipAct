#!/usr/bin/env python
"""
ALFWorld-specific runner for ZipAct experiments.
This is a convenience wrapper around run.py for ALFWorld experiments.

For more options, use run.py directly:
    python run.py --env alfworld --agent zipact --episodes 10
"""

import argparse
import os
import sys
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.envs.alfworld_env import ALFWorldEnv
from src.llm.client import LLMClient
from src.agents import get_agent
from src.agents.reflexion import ReflexionAgent
from src.utils.logger import Logger


def run_episode(agent, env, max_steps=50, logger=None, episode_id=None, verbose=False):
    """Run a single episode."""
    obs, info = env.reset()
    task = env.get_task()
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"EPISODE {episode_id if episode_id is not None else '?'}")
        print(f"Task: {task}")
        print(f"{'='*60}")
    
    # Initialize agent
    agent.reset(task)
    
    # Start logging
    if logger:
        logger.start_episode(task, episode_id)
    
    success = False
    total_reward = 0
    step = 0
    
    for step in range(max_steps):
        if verbose:
            print(f"\n--- Step {step + 1} ---")
            print(f"Obs: {obs[:200]}...")  # Truncate long observations
        
        # Agent decides action
        action = agent.step(obs)
        
        # Get thought for logging
        thought = agent.get_last_thought() if hasattr(agent, 'get_last_thought') else ""
        
        # Log step
        if logger:
            state = agent.get_state() if hasattr(agent, 'get_state') else None
            logger.log_step(step + 1, obs, thought, action, state)
        
        if verbose:
            print(f"Action: {action}")
        
        # Execute action
        obs, reward, done, info = env.step(action)
        total_reward += reward
        
        if done:
            success = reward > 0
            if verbose:
                print(f"\n{'='*60}")
                print(f"Episode finished: {'SUCCESS' if success else 'FAILURE'}")
                print(f"Reward: {reward}")
                print(f"Steps: {step + 1}")
                print(f"{'='*60}")
            break
    
    # Handle Reflexion agent reflection
    if isinstance(agent, ReflexionAgent) and not success:
        agent.reflect(success, failure_reason="Max steps reached" if step == max_steps - 1 else "Task failed")
    
    return success, total_reward, step + 1


def main():
    parser = argparse.ArgumentParser(description="Run ZipAct experiments on ALFWorld")
    parser.add_argument("--agent", type=str, default="zipact", 
                       choices=["zipact", "react", "reflexion", "obs_mask", "summary"],
                       help="Agent type to use")
    parser.add_argument("--model", type=str, default="gpt-4o-mini",
                       help="LLM model name (e.g., gpt-4o, gpt-4o-mini, qwen2.5-7b-instruct)")
    parser.add_argument("--episodes", type=int, default=5,
                       help="Number of episodes to run")
    parser.add_argument("--max_steps", type=int, default=50,
                       help="Maximum steps per episode")
    parser.add_argument("--split", type=str, default="eval_out_of_distribution",
                       choices=["train", "eval_in_distribution", "eval_out_of_distribution"],
                       help="Dataset split to use")
    parser.add_argument("--log_dir", type=str, default="logs",
                       help="Directory to save logs")
    parser.add_argument("--verbose", action="store_true",
                       help="Print detailed execution logs")
    parser.add_argument("--verbose-tokens", action="store_true",
                       help="Print detailed token usage for each LLM call")
    parser.add_argument("--api_key", type=str, default=None,
                       help="OpenAI API key (or set OPENAI_API_KEY env variable)")
    parser.add_argument("--base_url", type=str, default=None,
                       help="OpenAI API base URL (for compatible endpoints)")
    
    args = parser.parse_args()
    
    # Check API key
    if args.api_key:
        os.environ["OPENAI_API_KEY"] = args.api_key
    
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not set. Please set it via --api_key or environment variable.")
        sys.exit(1)
    
    # Initialize LLM
    print(f"\nInitializing LLM: {args.model}")
    llm = LLMClient(model=args.model, base_url=args.base_url, verbose=args.verbose_tokens)
    
    # Initialize agent with environment type
    print(f"Initializing agent: {args.agent}")
    environment = "alfworld"
    agent = get_agent(args.agent, llm, environment=environment, verbose=args.verbose)
    
    # Initialize environment
    print(f"Initializing ALFWorld environment (split: {args.split})")
    try:
        env = ALFWorldEnv(split=args.split)
    except Exception as e:
        print(f"Failed to initialize ALFWorld: {e}")
        print("\nMake sure ALFWorld is properly installed:")
        print("  pip install alfworld")
        print("  export ALFWORLD_DATA=/path/to/alfworld/data")
        sys.exit(1)
    
    # Initialize logger
    experiment_name = f"{args.agent}_{args.model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger = Logger(log_dir=args.log_dir, experiment_name=experiment_name)
    
    # Run episodes
    print(f"\nRunning {args.episodes} episodes...")
    print(f"{'='*60}\n")
    
    successes = []
    
    for i in range(args.episodes):
        success, reward, steps = run_episode(
            agent, env, 
            max_steps=args.max_steps,
            logger=logger,
            episode_id=i + 1,
            verbose=args.verbose
        )
        
        successes.append(success)
        
        # Get token usage
        token_usage = llm.get_token_usage()
        logger.end_episode(success, reward, token_usage)
        
        # Reset token counter for next episode
        llm.reset_token_count()
        
        # Print progress
        if not args.verbose:
            print(f"Episode {i+1}/{args.episodes}: {'✓ SUCCESS' if success else '✗ FAILURE'} (Steps: {steps}, Tokens: {token_usage['total_tokens']:,})")
    
    # Print summary
    summary = logger.save_summary(
        agent_name=args.agent,
        model_name=args.model,
        dataset_name=f"alfworld_{args.split}"
    )
    logger.print_summary(summary)
    
    print(f"\nLogs saved to: {logger.log_file}")
    print(f"Summary saved to: {logger.summary_file}")


if __name__ == "__main__":
    main()

