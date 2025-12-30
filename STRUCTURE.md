# Project Structure

## Directory Layout

```
ZipAct/
├── src/                          # Source code
│   ├── agents/                   # Agent implementations
│   │   ├── __init__.py          # Agent factory function
│   │   ├── base.py              # Abstract base class
│   │   ├── zipact.py            # ZipAct (our method)
│   │   ├── react.py             # ReAct baseline
│   │   ├── reflexion.py         # Reflexion baseline
│   │   ├── observation_masking.py  # Observation masking
│   │   └── summary.py           # History summarization
│   │
│   ├── envs/                     # Environment wrappers
│   │   ├── __init__.py          # Environment factory function
│   │   ├── base.py              # Abstract environment
│   │   ├── alfworld_env.py      # ALFWorld (household tasks)
│   │   ├── sciworld_env.py      # SciWorld (scientific reasoning)
│   │   └── webshop_env.py       # WebShop (e-commerce)
│   │
│   ├── llm/                      # LLM client
│   │   ├── __init__.py
│   │   └── client.py            # API wrapper with token tracking
│   │
│   ├── prompts/                  # Prompt templates (per environment)
│   │   ├── __init__.py          # Exports all prompts
│   │   ├── prompt_manager.py    # Unified prompt manager
│   │   ├── zipact_prompts.py    # ALFWorld ZipAct prompts
│   │   ├── react_prompts.py     # ALFWorld ReAct prompts
│   │   ├── sciworld_prompts.py  # SciWorld prompts
│   │   └── webshop_prompts.py   # WebShop prompts
│   │
│   └── utils/                    # Utilities
│       ├── __init__.py
│       └── logger.py            # Logging and evaluation
│
├── run.py                       # Universal runner (all envs)
├── run_alfworld.py              # ALFWorld-specific runner
├── run_experiment.py            # Multi-dataset runner
├── run_batch.ps1                # Batch experiments (Windows)
├── analyze_results.py           # Results analysis
├── test_setup.py                # Installation test
│
├── requirements.txt             # Dependencies
├── config.yaml                  # Configuration
├── README.md                    # Documentation
├── STRUCTURE.md                 # This file
└── LICENSE                      # MIT License
```

## Supported Environments

| Environment | Domain | Actions | Install |
|-------------|--------|---------|---------|
| **ALFWorld** | Household tasks | go to, take, put, open, close, clean, heat, cool | `pip install alfworld` |
| **SciWorld** | Scientific reasoning | pick up, put, activate, pour, mix, wait | `pip install scienceworld` |
| **WebShop** | E-commerce | search[], click[] | [GitHub](https://github.com/princeton-nlp/WebShop) |

## Core Components

### Agents (`src/agents/`)

**ZipActAgent** - State-dependent reasoning (Our Method)
- Maintains compact state $S_t = \langle G_t, W_t, C_t \rangle$
- State Updater: Compresses history into structured state
- Actor: Makes decisions from state only
- **Supports**: ALFWorld, SciWorld, WebShop

**ReActAgent** - History-dependent baseline
- Accumulates full interaction history
- O(T²) complexity
- **Supports**: ALFWorld, SciWorld, WebShop

**Other Baselines**
- Reflexion: Adds self-reflection
- ObservationMasking: Masks old observations
- Summary: Periodic history summarization

### Prompt System (`src/prompts/`)

Environment-specific prompts for optimal performance:

| Environment | ZipAct Prompts | ReAct Prompts |
|-------------|----------------|---------------|
| ALFWorld | `ALFWORLD_ZIPACT_*` | `ALFWORLD_REACT_*` |
| SciWorld | `SCIWORLD_ZIPACT_*` | `SCIWORLD_REACT_*` |
| WebShop | `WEBSHOP_ZIPACT_*` | `WEBSHOP_REACT_*` |

**PromptManager** provides unified access:
```python
from src.prompts import PromptManager

pm = PromptManager("sciworld")
updater = pm.get_zipact_updater_prompt()
actor = pm.get_zipact_actor_prompt()
```

### Environments (`src/envs/`)

All environments implement:
```python
def reset() -> Tuple[str, dict]
def step(action: str) -> Tuple[str, float, bool, dict]
def get_task() -> str
```

Factory function:
```python
from src.envs import get_env

env = get_env("alfworld", split="eval_out_of_distribution")
env = get_env("sciworld", task_name="boil")
env = get_env("webshop")
```

### LLM Client (`src/llm/`)

Features:
- OpenAI API support
- Compatible API endpoints (Qwen, etc.)
- Automatic token counting
- Input/output tracking

## Usage Examples

### Run on Different Environments

```bash
# ALFWorld
python run.py --env alfworld --agent zipact --episodes 5

# SciWorld
python run.py --env sciworld --agent zipact --task boil --episodes 5

# WebShop
python run.py --env webshop --agent zipact --episodes 5
```

### Compare Agents

```bash
# ZipAct vs ReAct on ALFWorld
python run.py --env alfworld --agent zipact --episodes 20
python run.py --env alfworld --agent react --episodes 20
python analyze_results.py
```

## Data Flow

```
Episode Execution:
1. env.reset() → initial observation
2. agent.reset(task) → initialize state/history
3. For each step:
   a. agent.step(obs) → action
   b. env.step(action) → new obs, reward, done
   c. logger.log_step(...)
4. logger.end_episode(success, reward, tokens)
```

## Extending the Project

### Add a New Environment

1. Create `src/envs/myenv_env.py` extending `BaseEnv`
2. Create `src/prompts/myenv_prompts.py` with environment-specific prompts
3. Update `PromptManager` in `prompt_manager.py`
4. Update `__init__.py` files

### Add a New Agent

```python
# src/agents/my_agent.py
from .base import BaseAgent

class MyAgent(BaseAgent):
    def __init__(self, llm_client, environment="alfworld", **kwargs):
        self.environment = environment
        # ...
    
    def reset(self, instruction: str):
        pass
    
    def step(self, observation: str) -> str:
        return action
```

## Key Files

- **`src/agents/zipact.py`** - Core ZipAct algorithm
- **`src/prompts/prompt_manager.py`** - Unified prompt management
- **`src/prompts/zipact_prompts.py`** - ALFWorld prompts
- **`src/prompts/sciworld_prompts.py`** - SciWorld prompts
- **`src/prompts/webshop_prompts.py`** - WebShop prompts
- **`run.py`** - Universal runner
- **`src/utils/logger.py`** - Evaluation system
