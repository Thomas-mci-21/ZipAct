"""
Environment wrappers for ZipAct.
Provides unified interface for different interactive environments.

Supported environments:
- ALFWorld: Household tasks
- SciWorld: Scientific reasoning
- WebShop: E-commerce navigation
"""

from .base import BaseEnv

# Lazy imports to avoid dependency issues
# These will be imported on-demand when get_env() is called
__all__ = [
    "BaseEnv",
    "ALFWorldEnv",
    "ALFWorldSimpleEnv",
    "SciWorldEnv",
    "WebShopEnv",
    "get_env",
]


def get_env(env_name: str, **kwargs) -> BaseEnv:
    """
    Factory function to get environment by name.
    
    Args:
        env_name: Environment name ('alfworld', 'sciworld', 'webshop', 'alfworld_simple')
        **kwargs: Environment-specific arguments
        
    Returns:
        Environment instance
    """
    env_lower = env_name.lower()
    
    if env_lower in ['alfworld', 'alf']:
        from .alfworld_env import ALFWorldEnv
        return ALFWorldEnv(**kwargs)
    elif env_lower == 'alfworld_simple':
        from .alfworld_simple import ALFWorldSimpleEnv
        return ALFWorldSimpleEnv(**kwargs)
    elif env_lower in ['sciworld', 'sci', 'scienceworld']:
        from .sciworld_env import SciWorldEnv
        return SciWorldEnv(**kwargs)
    elif env_lower in ['webshop', 'web']:
        from .webshop_env import WebShopEnv
        return WebShopEnv(**kwargs)
    else:
        raise ValueError(
            f"Unknown environment: {env_name}. "
            f"Supported: alfworld, alfworld_simple, sciworld, webshop"
        )
