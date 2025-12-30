"""
Agent implementations for ZipAct.

Agents:
- ZipActAgent: State-dependent reasoning (our method)
- ReActAgent: History-dependent baseline
- ReflexionAgent: ReAct + self-reflection
- ObservationMaskingAgent: ReAct with masked old observations
- SummaryAgent: ReAct with periodic summarization
"""

from .base import BaseAgent
from .zipact import ZipActAgent
from .react import ReActAgent
from .reflexion import ReflexionAgent
from .observation_masking import ObservationMaskingAgent
from .summary import SummaryAgent

__all__ = [
    "BaseAgent",
    "ZipActAgent",
    "ReActAgent",
    "ReflexionAgent",
    "ObservationMaskingAgent",
    "SummaryAgent",
]


def get_agent(agent_name: str, llm_client, environment: str = "alfworld", **kwargs):
    """
    Factory function to get agent by name.
    
    Args:
        agent_name: Agent name ('zipact', 'react', 'reflexion', 'obs_mask', 'summary')
        llm_client: LLM client instance
        environment: Environment type ('alfworld', 'sciworld', 'webshop')
        **kwargs: Additional agent arguments (verbose, keep_recent, summary_interval, etc.)
        
    Returns:
        Agent instance
    """
    agent_map = {
        'zipact': ZipActAgent,
        'react': ReActAgent,
        'reflexion': ReflexionAgent,
        'obs_mask': ObservationMaskingAgent,
        'observation_masking': ObservationMaskingAgent,
        'summary': SummaryAgent,
    }
    
    agent_lower = agent_name.lower()
    if agent_lower not in agent_map:
        raise ValueError(
            f"Unknown agent: {agent_name}. "
            f"Supported: {list(agent_map.keys())}"
        )
    
    AgentClass = agent_map[agent_lower]
    
    # All agents now support environment parameter
    return AgentClass(llm_client, environment=environment, **kwargs)
