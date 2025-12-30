import re
from typing import Tuple, List
from .base import BaseAgent
from ..llm.client import LLMClient
from ..prompts.prompt_manager import PromptManager

REFLEXION_PROMPT = """Based on the failed attempt, reflect on what went wrong and what you should do differently.

Previous attempt summary:
{history_summary}

Failure reason: {failure_reason}

Provide a brief reflection (2-3 sentences) on:
1. What went wrong?
2. What should you try differently?

Reflection:"""

class ReflexionAgent(BaseAgent):
    """
    Reflexion Agent: ReAct + Self-Reflection on failures.
    
    Maintains full history plus verbal reflections from past failures.
    Uses reflections to guide future attempts.
    
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
        Initialize Reflexion agent.
        
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
        self.reflections = []
        self.instruction = ""
        self.episode_count = 0
    
    def reset(self, instruction: str):
        """Reset for a new task (but keep reflections from previous episodes)."""
        self.instruction = instruction
        self.history = []
        self.last_thought = ""
        self.current_step = 0
        self.episode_count += 1
        
        if self.verbose:
            print(f"\n[Reflexion] Starting task (attempt #{self.episode_count}): {instruction}")
            print(f"[Reflexion] Environment: {self.environment}")
            print(f"[Reflexion] Max steps: {self.max_steps}")
            if self.reflections:
                print(f"[Reflexion] Using {len(self.reflections)} past reflections")
    
    def step(self, observation: str) -> str:
        """Process observation and return next action."""
        # Check step limit
        if self.is_step_limit_reached():
            raise StopIteration(f"Step limit ({self.max_steps}) reached")
        
        self.current_step += 1
        step_num = self.current_step
        
        # Add observation to history
        self.history.append(f"Observation {step_num}: {observation}")
        
        # Construct prompt with full history + reflections
        history_str = "\n".join(self.history)
        
        # Include reflections if available
        reflection_str = ""
        if self.reflections:
            reflection_str = "\n\nPast Reflections:\n" + "\n".join([f"- {r}" for r in self.reflections[-3:]])  # Last 3 reflections
        
        prompt = self.instruction_template.format(
            instruction=self.instruction,
            history=history_str + reflection_str
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
            print(f"\n[Reflexion] Step {self.current_step}/{self.max_steps}")
            print(f"[Reflexion] Thought: {thought}")
            print(f"[Reflexion] Action: {action}")
        
        return action
    
    def reflect(self, success: bool, failure_reason: str = "Task not completed"):
        """Generate reflection after episode ends."""
        if success:
            # No need to reflect on success
            return
        
        # Generate reflection based on history and failure
        history_summary = self._summarize_history()
        
        messages = [
            {"role": "user", "content": REFLEXION_PROMPT.format(
                history_summary=history_summary,
                failure_reason=failure_reason
            )}
        ]
        
        response = self.llm.chat(messages, temperature=0.3, max_tokens=200)
        reflection = response.strip()
        
        self.reflections.append(reflection)
        
        if self.verbose:
            print(f"\n[Reflexion] Generated reflection: {reflection}")
    
    def get_reflections(self) -> List[str]:
        """Return all reflections."""
        return self.reflections.copy()
    
    def _summarize_history(self) -> str:
        """Create a brief summary of the episode for reflection."""
        # Take first 500 and last 500 characters of history
        history_str = "\n".join(self.history)
        if len(history_str) > 1000:
            return history_str[:500] + "\n...\n" + history_str[-500:]
        return history_str
    
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

