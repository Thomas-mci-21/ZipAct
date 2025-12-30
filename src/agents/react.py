import re
from typing import Tuple, Optional
from .base import BaseAgent
from ..llm.client import LLMClient
from ..prompts.prompt_manager import PromptManager

class ReActAgent(BaseAgent):
    """
    ReAct Agent: History-dependent reasoning paradigm.
    
    Maintains complete interaction history and conditions decisions on it.
    This is the standard baseline that suffers from quadratic complexity O(T^2).
    
    Supports multiple environments:
    - ALFWorld: Household tasks
    - SciWorld: Scientific reasoning
    - WebShop: E-commerce navigation
    """
    
    def __init__(
        self, 
        llm_client: LLMClient, 
        environment: str = "alfworld",
        max_steps: int = 50,
        verbose: bool = False
    ):
        """
        Initialize ReAct agent.
        
        Args:
            llm_client: LLM client for API calls
            environment: Environment type ('alfworld', 'sciworld', 'webshop')
            max_steps: Maximum steps allowed before giving up (default: 50)
            verbose: Whether to print debug info
        """
        super().__init__(max_steps=max_steps)
        self.llm = llm_client
        self.verbose = verbose
        self.environment = environment.lower()
        
        # Load environment-specific prompts
        self.prompt_manager = PromptManager(environment)
        self.system_prompt = self.prompt_manager.get_react_system_prompt()
        self.instruction_template = self.prompt_manager.get_react_template()
        
        self.history = []
        self.instruction = ""
    
    def reset(self, instruction: str):
        """Reset for a new task."""
        self.instruction = instruction
        self.history = []
        self.last_thought = ""
        self.current_step = 0
        
        if self.verbose:
            print(f"\n[ReAct] Starting task: {instruction}")
            print(f"[ReAct] Environment: {self.environment}")
            print(f"[ReAct] Max steps: {self.max_steps}")
    
    def step(self, observation: str) -> str:
        """
        Process observation and return next action.
        
        Args:
            observation: Current environment observation
            
        Returns:
            action: Action string to execute
            
        Raises:
            StopIteration: If step limit reached
        """
        # Check step limit
        if self.is_step_limit_reached():
            raise StopIteration(f"Step limit ({self.max_steps}) reached")
        
        self.current_step += 1
        step_num = self.current_step
        
        # Add observation to history
        self.history.append(f"Observation {step_num}: {observation}")
        
        # Construct prompt with full history
        history_str = "\n".join(self.history)
        prompt = self.instruction_template.format(
            instruction=self.instruction,
            history=history_str
        )
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        response = self.llm.chat(messages, temperature=0.0, max_tokens=256)
        
        # Parse thought and action
        thought, action = self._parse_thought_action(response)
        
        # Add to history (same step number)
        self.history.append(f"Thought {step_num}: {thought}")
        self.history.append(f"Action {step_num}: {action}")
        
        self.last_thought = thought
        
        if self.verbose:
            print(f"\n[ReAct] Step {step_num}/{self.max_steps}")
            print(f"[ReAct] Thought: {thought}")
            print(f"[ReAct] Action: {action}")
        
        return action
    
    def get_history_length(self) -> int:
        """Return the length of history (for analysis)."""
        return len(self.history)
    
    def _parse_thought_action(self, text: str) -> Tuple[str, str]:
        """Parse thought and action from response."""
        thought = ""
        action = "look"  # Default action
        
        lines = text.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith("Thought:"):
                thought = line.replace("Thought:", "").strip()
            elif line.startswith("Action:"):
                action = line.replace("Action:", "").strip()
        
        # If no explicit thought/action found, try to extract them
        if not thought and not action:
            thought = text.strip()
            # Try to find action-like commands
            action_match = re.search(r'\b(go to|take|put|open|close|clean|heat|cool|toggle|use|look|inventory)\b.+', text, re.IGNORECASE)
            if action_match:
                action = action_match.group(0).strip()
        
        return thought, action
