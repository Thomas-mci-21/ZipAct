import json
import os
import glob
from typing import List, Dict
import pandas as pd

def load_summary(file_path: str) -> Dict:
    """Load a summary JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def analyze_results(log_dir: str = "logs"):
    """Analyze all experiment results in log directory."""
    
    # Find all summary files
    summary_files = glob.glob(os.path.join(log_dir, "*_summary.json"))
    
    if not summary_files:
        print(f"No summary files found in {log_dir}")
        return
    
    # Load all summaries
    results = []
    for file_path in summary_files:
        try:
            summary = load_summary(file_path)
            results.append({
                "Agent": summary.get("agent", "unknown"),
                "Model": summary.get("model", "unknown"),
                "Dataset": summary.get("dataset", "unknown"),
                "Success Rate": f"{summary.get('success_rate', 0):.1%}",
                "Successes": summary.get("successful_episodes", 0),
                "Total": summary.get("total_episodes", 0),
                "Avg Steps": f"{summary.get('avg_steps', 0):.1f}",
                "Avg Tokens/Episode": f"{summary.get('token_usage', {}).get('avg_tokens_per_episode', 0):.0f}",
                "Total Tokens": f"{summary.get('token_usage', {}).get('total_tokens', 0):,}",
                "Experiment": summary.get("experiment_name", "")
            })
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    if not results:
        print("No valid results found")
        return
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Sort by agent and model
    df = df.sort_values(["Model", "Agent"])
    
    # Print results
    print("\n" + "="*100)
    print("EXPERIMENT RESULTS SUMMARY")
    print("="*100)
    print(df.to_string(index=False))
    print("="*100)
    
    # Group by agent for comparison
    print("\n" + "="*100)
    print("COMPARISON BY AGENT (Average across models)")
    print("="*100)
    
    agent_comparison = df.groupby("Agent").agg({
        "Success Rate": lambda x: x.iloc[0],  # Already formatted
        "Avg Steps": lambda x: x.iloc[0],
        "Avg Tokens/Episode": lambda x: x.iloc[0]
    })
    print(agent_comparison.to_string())
    print("="*100)
    
    # Save to CSV
    output_file = os.path.join(log_dir, "results_summary.csv")
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Analyze ZipAct experiment results")
    parser.add_argument("--log_dir", type=str, default="logs", help="Directory containing log files")
    args = parser.parse_args()
    
    analyze_results(args.log_dir)
