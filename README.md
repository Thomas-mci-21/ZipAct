# ZipAct: Zipping Interaction History into a Compact State for Efficient LLM Agents

## 🎯 Overview

**ZipAct** addresses the "Context Snowball" problem in LLM agents by shifting from **history-dependent** to **state-dependent** reasoning.

<div align="center">
<img src="assets/zipact_architecture.png" alt="ZipAct Architecture" width="800"/>
</div>

### System Workflow

<div align="center">
<img src="assets/zipact_workflow.png" alt="ZipAct System Workflow" width="900"/>
</div>

**ZipAct** operates through a cyclical interaction between the **Actor** and **State Updater**:

- **Actor (π)**: Takes actions based on the current compressed state $S_t$
- **State Updater (U)**: Updates the state table by processing observations and actions
- **State Table**: Maintains three compact components:
  - **Goal State ($G_t$)**: Tracks hierarchical task progress and current objectives
  - **World State ($W_t$)**: Abstracts environment into task-relevant variables  
  - **Constraint State ($C_t$)**: Anti-loop mechanism for avoiding repeated failures
- **Environment**: Provides observations based on agent actions across multiple domains (ALFWorld, ScienceWorld, WebShop)

### The Context Snowball Problem

<div align="center">
<img src="assets/context_snowball.png" alt="Context Snowball Problem" width="800"/>
</div>

**ReAct-based agents** suffer from quadratic complexity $O(T^2)$ as context grows unboundedly. **ZipAct** achieves linear complexity $O(T)$ through state compression, significantly reducing token usage while maintaining performance.

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/your-username/ZipAct.git
cd ZipAct
pip install -r requirements.txt
```

### Setup

```bash
# Set your OpenAI API key
export OPENAI_API_KEY="your-api-key-here"

# (Optional) For Qwen or other compatible APIs
export OPENAI_BASE_URL="https://your-api-endpoint"

# Install ALFWorld
pip install alfworld
```

### Run Your First Experiment

```bash
# Run ZipAct on ALFWorld
python run_alfworld.py --agent zipact --model gpt-4o-mini --episodes 5

# Run ReAct baseline for comparison
python run_alfworld.py --agent react --model gpt-4o-mini --episodes 5

# See detailed execution with verbose mode
python run_alfworld.py --agent zipact --episodes 1 --verbose
```

## 💡 Usage

### Available Agents

- **`zipact`**: Our method (state-dependent reasoning)
- **`react`**: ReAct baseline (history-dependent)
- **`reflexion`**: Reflexion (ReAct + self-reflection)
- **`obs_mask`**: Observation masking baseline
- **`summary`**: History summarization baseline

### Run Experiments

```bash
# Single experiment
python run_alfworld.py \
  --agent zipact \
  --model gpt-4o-mini \
  --episodes 10 \
  --max_steps 50

# Multi-dataset support
python run_experiment.py \
  --dataset alfworld \
  --agent zipact \
  --episodes 10
```

### Analyze Results

```bash
# Generate comparison tables and statistics
python analyze_results.py --log_dir logs
```

Results are saved in `logs/` as:
- `{experiment_name}.jsonl` - Detailed episode logs
- `{experiment_name}_summary.json` - Statistics summary

## 🏗️ Architecture

### State Structure

```json
{
  "goal_state": {
    "global_instruction": "put a clean apple in refrigerator",
    "sub_goal_queue": ["clean the apple", "find refrigerator"],
    "current_objective": "find an apple"
  },
  "world_state": {
    "location": "kitchen",
    "inventory": ["apple_1"],
    "entity_map": {"countertop_1": "visited", "apple_1": "dirty"},
    "discovered_objects": ["apple_1", "sink_1", "fridge_1"]
  },
  "constraint_state": {
    "negative_constraints": ["drawer_1 is locked"],
    "visited_locations": ["kitchen", "living room"],
    "attempted_actions": ["open drawer_1"]
  }
}
```

### Core Components

- **Actor (π)**: Memory-less decision engine that acts based on current state
- **State Updater (U)**: Semantic compressor that zips history into structured state
  - Goal Progression Analysis
  - World Patching
  - Failure Reflection

## 📂 Project Structure

```
ZipAct/
├── src/
│   ├── agents/          # Agent implementations
│   │   ├── zipact.py   # ZipAct (our method)
│   │   ├── react.py    # ReAct baseline
│   │   └── ...
│   ├── envs/           # Environment wrappers
│   ├── llm/            # LLM client with token tracking
│   └── utils/          # Logging and evaluation
├── run_alfworld.py     # Main evaluation script
├── run_experiment.py   # Multi-dataset runner
└── analyze_results.py  # Results analysis
```

## 🤝 Contributing

We welcome contributions! Please feel free to submit issues or pull requests.


