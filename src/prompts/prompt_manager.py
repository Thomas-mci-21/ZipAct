"""
Unified Prompt Manager for ZipAct.
Provides environment-specific prompts for different datasets.

Supported environments:
- ALFWorld: Household tasks (clean, heat, cool, put objects)
- SciWorld: Scientific reasoning (boil, melt, freeze, mix)
- WebShop: E-commerce navigation (search, browse, purchase)
"""

from typing import Dict, Tuple

# Import all environment-specific prompts
from .zipact_prompts import (
    ALFWORLD_ZIPACT_UPDATER_SYSTEM_PROMPT,
    ALFWORLD_ZIPACT_ACTOR_SYSTEM_PROMPT,
    ALFWORLD_ZIPACT_INIT_STATE_PROMPT,
)

from .react_prompts import (
    ALFWORLD_REACT_SYSTEM_PROMPT,
    ALFWORLD_REACT_INSTRUCTION_TEMPLATE,
)

from .sciworld_prompts import (
    SCIWORLD_ZIPACT_UPDATER_SYSTEM_PROMPT,
    SCIWORLD_ZIPACT_ACTOR_SYSTEM_PROMPT,
    SCIWORLD_ZIPACT_INIT_STATE_PROMPT,
    SCIWORLD_REACT_SYSTEM_PROMPT,
    SCIWORLD_REACT_INSTRUCTION_TEMPLATE,
)

from .webshop_prompts import (
    WEBSHOP_ZIPACT_UPDATER_SYSTEM_PROMPT,
    WEBSHOP_ZIPACT_ACTOR_SYSTEM_PROMPT,
    WEBSHOP_ZIPACT_INIT_STATE_PROMPT,
    WEBSHOP_REACT_SYSTEM_PROMPT,
    WEBSHOP_REACT_INSTRUCTION_TEMPLATE,
)


class PromptManager:
    """
    Manages prompts for different environments and agent types.
    
    Usage:
        pm = PromptManager("alfworld")
        updater_prompt = pm.get_zipact_updater_prompt()
        actor_prompt = pm.get_zipact_actor_prompt()
    """
    
    # Registry of prompts for each environment
    PROMPTS = {
        "alfworld": {
            "zipact": {
                "updater": ALFWORLD_ZIPACT_UPDATER_SYSTEM_PROMPT,
                "actor": ALFWORLD_ZIPACT_ACTOR_SYSTEM_PROMPT,
                "init": ALFWORLD_ZIPACT_INIT_STATE_PROMPT,
            },
            "react": {
                "system": ALFWORLD_REACT_SYSTEM_PROMPT,
                "template": ALFWORLD_REACT_INSTRUCTION_TEMPLATE,
            }
        },
        "sciworld": {
            "zipact": {
                "updater": SCIWORLD_ZIPACT_UPDATER_SYSTEM_PROMPT,
                "actor": SCIWORLD_ZIPACT_ACTOR_SYSTEM_PROMPT,
                "init": SCIWORLD_ZIPACT_INIT_STATE_PROMPT,
            },
            "react": {
                "system": SCIWORLD_REACT_SYSTEM_PROMPT,
                "template": SCIWORLD_REACT_INSTRUCTION_TEMPLATE,
            }
        },
        "webshop": {
            "zipact": {
                "updater": WEBSHOP_ZIPACT_UPDATER_SYSTEM_PROMPT,
                "actor": WEBSHOP_ZIPACT_ACTOR_SYSTEM_PROMPT,
                "init": WEBSHOP_ZIPACT_INIT_STATE_PROMPT,
            },
            "react": {
                "system": WEBSHOP_REACT_SYSTEM_PROMPT,
                "template": WEBSHOP_REACT_INSTRUCTION_TEMPLATE,
            }
        }
    }
    
    # Environment aliases
    ENV_ALIASES = {
        "alf": "alfworld",
        "alfworld": "alfworld",
        "sci": "sciworld",
        "sciworld": "sciworld",
        "scienceworld": "sciworld",
        "web": "webshop",
        "webshop": "webshop",
    }
    
    def __init__(self, environment: str = "alfworld"):
        """
        Initialize PromptManager for a specific environment.
        
        Args:
            environment: Environment name ('alfworld', 'sciworld', 'webshop')
        """
        env_lower = environment.lower()
        self.environment = self.ENV_ALIASES.get(env_lower, env_lower)
        
        if self.environment not in self.PROMPTS:
            raise ValueError(
                f"Unknown environment: {environment}. "
                f"Supported: {list(self.PROMPTS.keys())}"
            )
    
    def get_zipact_prompts(self) -> Dict[str, str]:
        """Get all ZipAct prompts for current environment."""
        return self.PROMPTS[self.environment]["zipact"].copy()
    
    def get_react_prompts(self) -> Dict[str, str]:
        """Get all ReAct prompts for current environment."""
        return self.PROMPTS[self.environment]["react"].copy()
    
    def get_zipact_updater_prompt(self) -> str:
        """Get ZipAct State Updater system prompt."""
        return self.PROMPTS[self.environment]["zipact"]["updater"]
    
    def get_zipact_actor_prompt(self) -> str:
        """Get ZipAct Actor system prompt."""
        return self.PROMPTS[self.environment]["zipact"]["actor"]
    
    def get_zipact_init_prompt(self) -> str:
        """Get ZipAct state initialization prompt template."""
        return self.PROMPTS[self.environment]["zipact"]["init"]
    
    def get_react_system_prompt(self) -> str:
        """Get ReAct system prompt."""
        return self.PROMPTS[self.environment]["react"]["system"]
    
    def get_react_template(self) -> str:
        """Get ReAct instruction template."""
        return self.PROMPTS[self.environment]["react"]["template"]
    
    @classmethod
    def list_environments(cls) -> list:
        """List all supported environments."""
        return list(cls.PROMPTS.keys())
    
    @classmethod
    def get_prompts_for_env(cls, environment: str, agent_type: str = "zipact") -> Dict[str, str]:
        """
        Static method to get prompts without instantiating.
        
        Args:
            environment: Environment name
            agent_type: 'zipact' or 'react'
            
        Returns:
            Dict of prompts for the specified agent type
        """
        env_lower = environment.lower()
        env = cls.ENV_ALIASES.get(env_lower, env_lower)
        
        if env not in cls.PROMPTS:
            raise ValueError(f"Unknown environment: {environment}")
        if agent_type not in cls.PROMPTS[env]:
            raise ValueError(f"Unknown agent type: {agent_type}")
        
        return cls.PROMPTS[env][agent_type].copy()


def get_zipact_prompts(environment: str = "alfworld") -> Tuple[str, str, str]:
    """
    Convenience function to get ZipAct prompts.
    
    Args:
        environment: Environment name
        
    Returns:
        Tuple of (updater_prompt, actor_prompt, init_prompt)
    """
    pm = PromptManager(environment)
    return (
        pm.get_zipact_updater_prompt(),
        pm.get_zipact_actor_prompt(),
        pm.get_zipact_init_prompt()
    )


def get_react_prompts(environment: str = "alfworld") -> Tuple[str, str]:
    """
    Convenience function to get ReAct prompts.
    
    Args:
        environment: Environment name
        
    Returns:
        Tuple of (system_prompt, template)
    """
    pm = PromptManager(environment)
    return (
        pm.get_react_system_prompt(),
        pm.get_react_template()
    )
