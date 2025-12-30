"""
Universal experiment runner for ZipAct across multiple datasets.
Supports ALFWorld, SciWorld, and WebShop.
"""
import argparse
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.llm.client import LLMClient
from src.agents.zipact import ZipActAgent
from src.agents.react import ReActAgent
from src.agents.reflexion import ReflexionAgent
from src.agents.observation_masking import ObservationMaskingAgent
from src.agents.summary import SummaryAgent
from src.utils.logger import Logger


def get_environment(dataset: str, **kwargs):
    """Initialize environment based on dataset name."""
    if dataset == "alfworld":
        from src.envs.alfworld_env import ALFWorldEnv
        return ALFWorldEnv(split=kwargs.get('split', 'eval_out_of_distribution'))
    
    elif dataset == "sciworld":
        from src.envs.sciworld_env import SciWorldEnv
        return SciWorldEnv(
            task_name=kwargs.get('task_name', 'boil'),
            simplifications_preset=kwargs.get('difficulty', 'easy')
        )
    
    elif dataset == "webshop":
        from src.envs.webshop_env import WebShopEnv
        return WebShopEnv()
    
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def get_agent(agent_type: str, llm_client: LLMClient, environment: str = "alfworld", verbose: bool = False):
    """Initialize agent based on type.
    
    Args:
        agent_type: Agent type name
        llm_client: LLM client instance
        environment: Environment type ('alfworld', 'sciworld', 'webshop')
        verbose: Whether to print debug info
    """
    if agent_type == "zipact":
        return ZipActAgent(llm_client, environment=environment, verbose=verbose)
    elif agent_type == "react":
        return ReActAgent(llm_client, environment=environment, verbose=verbose)
    elif agent_type == "reflexion":
        return ReflexionAgent(llm_client, environment=environment, verbose=verbose)
    elif agent_type == "obs_mask":
        return ObservationMaskingAgent(llm_client, environment=environment, keep_recent=5, verbose=verbose)
    elif agent_type == "summary":
        return SummaryAgent(llm_client, environment=environment, summary_interval=10, verbose=verbose)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


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
    
    for step in range(max_steps):
        if verbose:
            print(f"\n--- Step {step + 1} ---")
            print(f"Obs: {obs[:200]}...")
        
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
    parser = argparse.ArgumentParser(description="Run ZipAct experiments on multiple datasets")
    
    # Agent and model
    parser.add_argument("--agent", type=str, default="zipact",
                       choices=["zipact", "react", "reflexion", "obs_mask", "summary"],
                       help="Agent type to use")
    parser.add_argument("--model", type=str, default="gpt-4o-mini",
                       help="LLM model name")
    
    # Dataset
    parser.add_argument("--dataset", type=str, default="alfworld",
                       choices=["alfworld", "sciworld", "webshop"],
                       help="Dataset to evaluate on")
    
    # Dataset-specific arguments
    parser.add_argument("--split", type=str, default="eval_out_of_distribution",
                       help="Dataset split (ALFWorld)")
    parser.add_argument("--task_name", type=str, default="task-1-boil",
                       help="Task name (SciWorld)")
    parser.add_argument("--difficulty", type=str, default="easy",
                       choices=["easy", "medium", "hard"],
                       help="Difficulty level (SciWorld)")
    
    # Experiment settings
    parser.add_argument("--episodes", type=int, default=5,
                       help="Number of episodes to run")
    parser.add_argument("--max_steps", type=int, default=50,
                       help="Maximum steps per episode")
    parser.add_argument("--log_dir", type=str, default="logs",
                       help="Directory to save logs")
    parser.add_argument("--verbose", action="store_true",
                       help="Print detailed execution logs")
    parser.add_argument("--verbose-tokens", action="store_true",
                       help="Print detailed token usage for each LLM call")
    
    # API settings
    parser.add_argument("--api_key", type=str, default=None,
                       help="OpenAI API key")
    parser.add_argument("--base_url", type=str, default=None,
                       help="OpenAI API base URL")
    
    args = parser.parse_args()
    
    # Set API key
    if args.api_key:
        os.environ["OPENAI_API_KEY"] = args.api_key
    
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not set.")
        sys.exit(1)
    
    # Initialize LLM
    print(f"\nInitializing LLM: {args.model}")
    llm = LLMClient(model=args.model, base_url=args.base_url, verbose=args.verbose_tokens)
    
    # Initialize agent with environment parameter
    print(f"Initializing agent: {args.agent}")
    agent = get_agent(args.agent, llm, environment=args.dataset, verbose=args.verbose)
    
    # Initialize environment
    print(f"Initializing {args.dataset} environment")
    try:
        env_kwargs = {
            'split': args.split,
            'task_name': args.task_name,
            'difficulty': args.difficulty
        }
        env = get_environment(args.dataset, **env_kwargs)
    except Exception as e:
        print(f"Failed to initialize {args.dataset}: {e}")
        sys.exit(1)
    
    # Initialize logger
    experiment_name = f"{args.agent}_{args.model}_{args.dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger = Logger(log_dir=args.log_dir, experiment_name=experiment_name)
    
    # Run episodes
    print(f"\nRunning {args.episodes} episodes on {args.dataset}...")
    print(f"{'='*60}\n")
    
    for i in range(args.episodes):
        success, reward, steps = run_episode(
            agent, env,
            max_steps=args.max_steps,
            logger=logger,
            episode_id=i + 1,
            verbose=args.verbose
        )
        
        # Get token usage
        token_usage = llm.get_token_usage()
        logger.end_episode(success, reward, token_usage)
        
        # Reset token counter
        llm.reset_token_count()
        
        # Print progress
        if not args.verbose:
            print(f"Episode {i+1}/{args.episodes}: {'✓ SUCCESS' if success else '✗ FAILURE'} "
                  f"(Steps: {steps}, Reward: {reward:.2f}, Tokens: {token_usage['total_tokens']:,})")
    
    # Print summary
    summary = logger.save_summary(
        agent_name=args.agent,
        model_name=args.model,
        dataset_name=f"{args.dataset}_{args.split if args.dataset == 'alfworld' else args.task_name}"
    )
    logger.print_summary(summary)
    
    print(f"\nLogs saved to: {logger.log_file}")
    print(f"Summary saved to: {logger.summary_file}")


if __name__ == "__main__":
    main()
