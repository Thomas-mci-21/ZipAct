import re
from typing import Tuple
from .base import BaseAgent
from ..llm.client import LLMClient
from ..prompts.prompt_manager import PromptManager

class ObservationMaskingAgent(BaseAgent):
    """
    Observation Masking Agent: ReAct with masked old observations.
    
    Keeps full action history but masks observations older than K steps to reduce context.
    This is a heuristic compression method.
    
    Supports multiple environments:
    - ALFWorld: Household tasks
    - SciWorld: Scientific reasoning
    - WebShop: E-commerce navigation
    """
    
    def __init__(
        self, 
        llm_client: LLMClient, 
        environment: str = "alfworld",
        keep_recent: int = 5,
        max_steps: int = 50,
        verbose: bool = False
    ):
        """
        Args:
            llm_client: LLM client
            environment: Environment type ('alfworld', 'sciworld', 'webshop')
            keep_recent: Number of recent observations to keep unmasked
            max_steps: Maximum steps allowed before giving up (default: 50)
            verbose: Whether to print debug info
        """
        super().__init__(max_steps=max_steps)
        self.llm = llm_client
        self.verbose = verbose
        self.environment = environment.lower()
        self.keep_recent = keep_recent
        
        # Load environment-specific prompts
        self.prompt_manager = PromptManager(environment)
        self.system_prompt = self.prompt_manager.get_react_system_prompt()
        self.instruction_template = self.prompt_manager.get_react_template()
        
        self.observations = []
        self.thoughts = []
        self.actions = []
        self.instruction = ""
    
    def reset(self, instruction: str):
        """Reset for a new task."""
        self.instruction = instruction
        self.observations = []
        self.thoughts = []
        self.actions = []
        self.last_thought = ""
        self.current_step = 0
        
        if self.verbose:
            print(f"\n[ObsMask] Starting task: {instruction}")
            print(f"[ObsMask] Environment: {self.environment}")
            print(f"[ObsMask] Max steps: {self.max_steps}")
    
    def step(self, observation: str) -> str:
        """Process observation and return next action."""
        # Check step limit
        if self.is_step_limit_reached():
            raise StopIteration(f"Step limit ({self.max_steps}) reached")
        
        self.current_step += 1
        
        # Store observation
        self.observations.append(observation)
        
        # Build history with masked observations
        history_parts = []
        
        for i in range(len(self.observations)):
            step_num = i + 1
            
            # Mask old observations
            if i < len(self.observations) - self.keep_recent:
                obs_text = "[Observation masked]"
            else:
                obs_text = self.observations[i]
            
            history_parts.append(f"Observation {step_num}: {obs_text}")
            
            # Add thought and action if they exist
            if i < len(self.thoughts):
                history_parts.append(f"Thought {step_num}: {self.thoughts[i]}")
            if i < len(self.actions):
                history_parts.append(f"Action {step_num}: {self.actions[i]}")
        
        history_str = "\n".join(history_parts)
        
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
        
        # Store thought and action
        self.thoughts.append(thought)
        self.actions.append(action)
        self.last_thought = thought
        
        if self.verbose:
            print(f"\n[ObsMask] Step {self.current_step}/{self.max_steps}")
            print(f"[ObsMask] Thought: {thought}")
            print(f"[ObsMask] Action: {action}")
        
        return action
    
    def _parse_thought_action(self, text: str) -> Tuple[str, str]:
        """Parse thought and action from response."""
        thought = ""
        action = "look"
        
        lines = text.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith("Thought:"):
                thought = line.replace("Thought:", "").strip()
            elif line.startswith("Action:"):
                action = line.replace("Action:", "").strip()
        
        if not thought and not action:
            thought = text.strip()
            # Environment-specific action patterns
            if self.environment == "webshop":
                action_match = re.search(r'(search\[.+?\]|click\[.+?\]|back)', text, re.IGNORECASE)
            elif self.environment == "sciworld":
                action_match = re.search(r'\b(look around|go to|pick up|put down|put|pour|open|close|activate|deactivate|focus on|wait|mix|connect|use|read|examine)\b.+', text, re.IGNORECASE)
            else:  # alfworld
                action_match = re.search(r'\b(go to|take|put|open|close|clean|heat|cool|toggle|use|look|inventory)\b.+', text, re.IGNORECASE)
            
            if action_match:
                action = action_match.group(0).strip()
        
        return thought, action
