"""
WebShop Environment Wrapper for ZipAct.
WebShop is a text-based web navigation environment for e-commerce tasks.

Install: Follow instructions at https://github.com/princeton-nlp/WebShop
"""

import re
from typing import Tuple, Dict, List, Optional
from .base import BaseEnv


class WebShopEnv(BaseEnv):
    """
    Wrapper for WebShop environment.
    WebShop is a text-based web navigation environment for e-commerce tasks
    where an agent needs to find and purchase products matching specific criteria.
    
    Note: This requires the webshop package to be installed.
    Install from: https://github.com/princeton-nlp/WebShop
    
    Setup:
        1. Clone: git clone https://github.com/princeton-nlp/WebShop
        2. Install: cd WebShop && pip install -e .
        3. Download data: bash setup.sh
    """
    
    def __init__(
        self, 
        observation_mode: str = "text",
        session: Optional[str] = None,
        server_url: Optional[str] = None,
        render: bool = False
    ):
        """
        Initialize WebShop environment.
        
        Args:
            observation_mode: 'text' for text observations, 'html' for raw HTML
            session: Session ID for specific task (None for random)
            server_url: WebShop server URL (if using server mode)
            render: Whether to render browser view
        """
        self.observation_mode = observation_mode
        self.session = session
        self.server_url = server_url
        self.render = render
        
        # Try to import WebShop
        try:
            if server_url:
                # Use HTTP client mode
                from web_agent_site.envs import WebAgentTextEnv
                self.env = WebAgentTextEnv(
                    observation_mode=observation_mode,
                    render=render,
                    server_url=server_url
                )
            else:
                # Use local environment
                from web_agent_site.envs import WebAgentTextEnv
                self.env = WebAgentTextEnv(
                    observation_mode=observation_mode,
                    render=render
                )
            self._webshop_available = True
        except ImportError:
            print(
                "Warning: WebShop is not installed. Environment will run in mock mode.\n"
                "To install WebShop:\n"
                "  git clone https://github.com/princeton-nlp/WebShop\n"
                "  cd WebShop && pip install -e .\n"
                "  bash setup.sh"
            )
            self._webshop_available = False
            self.env = None
        
        self.current_task = ""
        self.current_session = None
        self.current_page = "search"
    
    def reset(self, session: Optional[str] = None) -> Tuple[str, Dict]:
        """
        Reset environment and return initial observation.
        
        Args:
            session: Optional session ID to load specific task
            
        Returns:
            observation: Text description of the current page
            info: Additional information (task, session, etc.)
        """
        if not self._webshop_available:
            return self._mock_reset()
        
        # Reset with optional session
        obs = self.env.reset(session=session or self.session)
        
        # Extract task/goal
        self.current_task = self._extract_task(obs)
        self.current_session = self.env.session if hasattr(self.env, 'session') else None
        self.current_page = "search"
        
        info = {
            'task': self.current_task,
            'session': self.current_session,
            'page_type': self.current_page,
        }
        
        return obs, info
    
    def step(self, action: str) -> Tuple[str, float, bool, Dict]:
        """
        Execute action and return result.
        
        Args:
            action: Action string (e.g., "search[query]", "click[element]")
            
        Returns:
            observation: New page observation
            reward: Reward signal (product match score, 0-1)
            done: Whether episode is finished (purchase completed)
            info: Additional information
        """
        if not self._webshop_available:
            return self._mock_step(action)
        
        obs, reward, done, info = self.env.step(action)
        
        # Update page type based on observation
        self.current_page = self._detect_page_type(obs)
        
        # Add info
        info = info if isinstance(info, dict) else {}
        info['page_type'] = self.current_page
        info['action'] = action
        
        return obs, reward, done, info
    
    def _extract_task(self, observation: str) -> str:
        """Extract the task/goal from the observation."""
        # WebShop instructions usually in specific format
        patterns = [
            r'Instruction:\s*\[(.+?)\]',
            r'Instruction:\s*(.+?)(?:\n|$)',
            r'Goal:\s*(.+?)(?:\n|$)',
            r'I need\s+(.+?)(?:\n|$)',
            r'Find\s+(.+?)(?:\n|$)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, observation, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip()
        
        # Fallback: first line
        lines = [line.strip() for line in observation.split('\n') if line.strip()]
        if lines:
            return lines[0]
        
        return "Find and purchase the requested product"
    
    def _detect_page_type(self, observation: str) -> str:
        """Detect current page type from observation."""
        obs_lower = observation.lower()
        
        if 'search' in obs_lower and 'results' not in obs_lower:
            return 'search'
        elif 'results' in obs_lower or 'products' in obs_lower:
            return 'results'
        elif 'product' in obs_lower and ('details' in obs_lower or 'description' in obs_lower):
            return 'product_detail'
        elif 'cart' in obs_lower:
            return 'cart'
        elif 'checkout' in obs_lower or 'thank you' in obs_lower or 'order' in obs_lower:
            return 'checkout'
        else:
            return 'unknown'
    
    def get_task(self) -> str:
        """Return the current task description."""
        return self.current_task
    
    def get_available_actions(self) -> List[str]:
        """Return list of available actions on current page."""
        if not self._webshop_available:
            return self._mock_available_actions()
        
        if hasattr(self.env, 'get_available_actions'):
            return self.env.get_available_actions()
        return []
    
    def get_page_type(self) -> str:
        """Return current page type."""
        return self.current_page
    
    # =========================================================================
    # Mock methods for when WebShop is not installed
    # =========================================================================
    
    def _mock_reset(self) -> Tuple[str, Dict]:
        """Mock reset for testing without WebShop installed."""
        self.current_task = "Find a red cotton shirt in size medium, price under $30"
        self.current_page = "search"
        self._mock_state = {"step": 0}
        
        obs = """WebShop
Instruction: [Find a red cotton shirt in size medium, price under $30]

[Search]"""
        
        info = {
            'task': self.current_task,
            'session': 'mock_session',
            'page_type': self.current_page,
        }
        
        return obs, info
    
    def _mock_step(self, action: str) -> Tuple[str, float, bool, Dict]:
        """Mock step for testing without WebShop installed."""
        self._mock_state["step"] += 1
        done = False
        reward = 0.0
        
        if action.startswith("search["):
            self.current_page = "results"
            obs = """Results for "red cotton shirt":

[Product 1] Red Cotton T-Shirt - $25.99 ★★★★☆
[Product 2] Premium Red Shirt - $45.00 ★★★★★
[Product 3] Cotton Blend Red Top - $19.99 ★★★☆☆

[Back to Search]"""
        
        elif "click[Red Cotton T-Shirt" in action or "click[Product 1]" in action:
            self.current_page = "product_detail"
            obs = """Red Cotton T-Shirt - $25.99

Description: Comfortable 100% cotton t-shirt in vibrant red.
Material: Cotton
Available sizes: [small] [medium] [large] [xl]
Color: Red

[Buy Now] [Back to Search]"""
        
        elif action == "click[medium]":
            obs = """Red Cotton T-Shirt - $25.99

Description: Comfortable 100% cotton t-shirt in vibrant red.
Material: Cotton
Selected size: medium ✓
Color: Red

[Buy Now] [Back to Search]"""
        
        elif action == "click[Buy Now]":
            done = True
            reward = 0.85  # Simulated match score
            self.current_page = "checkout"
            obs = """Thank you for your purchase!

Order confirmed: Red Cotton T-Shirt (medium) - $25.99

Your reward score: 0.85"""
        
        else:
            obs = f"Action '{action}' executed. Current page: {self.current_page}"
        
        info = {
            'page_type': self.current_page,
            'action': action,
            'mock': True,
        }
        
        return obs, reward, done, info
    
    def _mock_available_actions(self) -> List[str]:
        """Return mock available actions."""
        if self.current_page == "search":
            return ["search[<query>]"]
        elif self.current_page == "results":
            return ["click[<product>]", "click[Back to Search]", "click[Next >]"]
        elif self.current_page == "product_detail":
            return ["click[<size>]", "click[<color>]", "click[Buy Now]", "click[Back to Search]"]
        else:
            return ["click[<element>]", "back"]

