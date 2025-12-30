import re
from typing import Tuple
from .base import BaseAgent
from ..llm.client import LLMClient
from ..prompts.prompt_manager import PromptManager

SUMMARY_PROMPT = """Summarize the following interaction history into a concise summary (3-5 sentences) that captures:
1. What has been explored
2. What objects have been found
3. What actions have been attempted
4. Current progress toward the goal

History:
{history}

Summary:"""

class SummaryAgent(BaseAgent):
    """
    Summary Agent: ReAct with history summarization.
    
    Periodically summarizes history to compress context while retaining key information.
    
    Supports multiple environments:
    - ALFWorld: Household tasks
    - SciWorld: Scientific reasoning
    - WebShop: E-commerce navigation
    """
    
    def __init__(
        self, 
        llm_client: LLMClient, 
        environment: str = "alfworld",
        summary_interval: int = 10,
        max_steps: int = 50,
        verbose: bool = False
    ):
        """
        Args:
            llm_client: LLM client
            environment: Environment type ('alfworld', 'sciworld', 'webshop')
            summary_interval: Summarize every N steps
            max_steps: Maximum steps allowed before giving up (default: 50)
            verbose: Whether to print debug info
        """
        super().__init__(max_steps=max_steps)
        self.llm = llm_client
        self.verbose = verbose
        self.environment = environment.lower()
        self.summary_interval = summary_interval
        
        # Load environment-specific prompts
        self.prompt_manager = PromptManager(environment)
        self.system_prompt = self.prompt_manager.get_react_system_prompt()
        self.instruction_template = self.prompt_manager.get_react_template()
        
        self.history = []
        self.summary = ""
        self.instruction = ""
    
    def reset(self, instruction: str):
        """Reset for a new task."""
        self.instruction = instruction
        self.history = []
        self.summary = ""
        self.last_thought = ""
        self.current_step = 0
        
        if self.verbose:
            print(f"\n[Summary] Starting task: {instruction}")
            print(f"[Summary] Environment: {self.environment}")
            print(f"[Summary] Max steps: {self.max_steps}")
    
    def step(self, observation: str) -> str:
        """Process observation and return next action."""
        # Check step limit
        if self.is_step_limit_reached():
            raise StopIteration(f"Step limit ({self.max_steps}) reached")
        
        self.current_step += 1
        
        # Add observation to history
        self.history.append(f"Observation {self.current_step}: {observation}")
        
        # Check if we should summarize
        if len(self.history) >= self.summary_interval * 2 and len(self.history) % (self.summary_interval * 2) == 0:
            self._summarize_history()
        
        # Build prompt with summary + recent history
        if self.summary:
            context = f"Summary of earlier steps:\n{self.summary}\n\nRecent history:\n"
            # Keep last summary_interval*2 items
            recent_history = self.history[-(self.summary_interval * 2):]
            context += "\n".join(recent_history)
        else:
            context = "\n".join(self.history)
        
        prompt = self.instruction_template.format(
            instruction=self.instruction,
            history=context
        )
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        response = self.llm.chat(messages, temperature=0.0, max_tokens=256)
        
        # Parse thought and action
        thought, action = self._parse_thought_action(response)
        
        # Add to history
        self.history.append(f"Thought {self.current_step}: {thought}")
        self.history.append(f"Action {self.current_step}: {action}")
        
        self.last_thought = thought
        
        if self.verbose:
            print(f"\n[Summary] Step {self.current_step}/{self.max_steps}")
            print(f"[Summary] Thought: {thought}")
            print(f"[Summary] Action: {action}")
        
        return action
    
    def _summarize_history(self):
        """Summarize the history up to this point."""
        history_text = "\n".join(self.history)
        
        messages = [
            {"role": "user", "content": SUMMARY_PROMPT.format(history=history_text)}
        ]
        
        response = self.llm.chat(messages, temperature=0.0, max_tokens=300)
        self.summary = response.strip()
        
        if self.verbose:
            print(f"\n[Summary] Generated summary: {self.summary}")
        
        # Clear old history, keep only recent items
        self.history = self.history[-(self.summary_interval * 2):]
    
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
