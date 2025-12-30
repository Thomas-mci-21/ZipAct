# Quick Start Guide

## Installation

```bash
git clone https://github.com/your-username/ZipAct.git
cd ZipAct
pip install -r requirements.txt
```

## Setup

```bash
# Set OpenAI API key
export OPENAI_API_KEY="your-api-key"

# Install ALFWorld
pip install alfworld
```

## Test Installation

```bash
python test_setup.py
```

## Run Your First Experiment

```bash
# ZipAct on ALFWorld
python run_alfworld.py --agent zipact --model gpt-4o-mini --episodes 5

# Compare with ReAct
python run_alfworld.py --agent react --model gpt-4o-mini --episodes 5
```

## Batch Evaluation

```powershell
# Run all agents (PowerShell)
.\run_batch.ps1 -Episodes 20

# Analyze results
python analyze_results.py
```

## Common Issues

### ALFWorld not found
```bash
pip install alfworld
```

### API key not set
```bash
export OPENAI_API_KEY="your-key"
# Or use --api_key argument
python run_alfworld.py --api_key "your-key" --agent zipact
```

## Next Steps

- See [README.md](README.md) for detailed documentation
- Check [STRUCTURE.md](STRUCTURE.md) for code architecture
- Explore `src/` for implementation details

