import json
import re
from typing import Dict, Any, Tuple, Optional
from .base import BaseAgent
from ..llm.client import LLMClient
from ..prompts.prompt_manager import PromptManager

class ZipActAgent(BaseAgent):
    """
    ZipAct Agent: State-dependent reasoning paradigm.
    
    Maintains a compact structured state S_t = <G_t, W_t, C_t>:
    - G_t: Goal State (hierarchical task progress)
    - W_t: World State (environment abstraction)
    - C_t: Constraint State (anti-loop mechanism)
    
    Architecture:
    - Actor: Memory-less decision engine (π)
    - State Updater: Semantic compressor (U)
    
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
        Initialize ZipAct agent.
        
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
        self.updater_prompt = self.prompt_manager.get_zipact_updater_prompt()
        self.actor_prompt = self.prompt_manager.get_zipact_actor_prompt()
        self.init_prompt = self.prompt_manager.get_zipact_init_prompt()
        
        self.state = self._get_empty_state()
        self.last_action = None
    
    def reset(self, instruction: str):
        """Initialize state for a new task."""
        self.current_step = 0
        self.last_thought = ""
        
        if self.verbose:
            print(f"\n[ZipAct] Initializing state for task: {instruction}")
            print(f"[ZipAct] Environment: {self.environment}")
            print(f"[ZipAct] Max steps: {self.max_steps}")
        
        # Use LLM to initialize structured state
        messages = [
            {"role": "system", "content": "You are a helpful assistant that initializes agent state."},
            {"role": "user", "content": self.init_prompt.format(instruction=instruction)}
        ]
        
        response = self.llm.chat(messages, temperature=0.0, max_tokens=512)
        self.state = self._parse_json(response, default=self._get_empty_state())
        
        # Ensure global_instruction is set
        if not self.state.get("goal_state", {}).get("global_instruction"):
            self.state["goal_state"]["global_instruction"] = instruction
        
        self.last_action = None
        
        if self.verbose:
            print(f"[ZipAct] Initial state: {json.dumps(self.state, indent=2)}")
    
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
        
        # Step 1: Update state (if not first step)
        if self.last_action is not None:
            self._update_state(observation)
        
        # Step 2: Actor decides next action
        thought, action = self._act(observation)
        
        # Store for next iteration
        self.last_action = action
        self.last_thought = thought
        
        if self.verbose:
            print(f"\n[ZipAct] Step {self.current_step}/{self.max_steps}")
            print(f"[ZipAct] Thought: {thought}")
            print(f"[ZipAct] Action: {action}")
        
        return action
    
    def get_state(self) -> Dict[str, Any]:
        """Return current state for logging."""
        return self.state.copy()
    
    def _update_state(self, observation: str):
        """
        Update state based on last action and new observation.
        Implements U: S_t <- U(S_{t-1}, a_{t-1}, o_t)
        """
        # Ensure attempted_actions is tracked (code-level guarantee)
        if self.last_action:
            if "constraint_state" not in self.state:
                self.state["constraint_state"] = {"attempted_actions": [], "negative_constraints": [], "visited_locations": []}
            if "attempted_actions" not in self.state["constraint_state"]:
                self.state["constraint_state"]["attempted_actions"] = []
            if self.last_action not in self.state["constraint_state"]["attempted_actions"]:
                self.state["constraint_state"]["attempted_actions"].append(self.last_action)
        
        state_str = json.dumps(self.state, indent=2, ensure_ascii=False)
        
        prompt = f"""Previous State:
```json
{state_str}
```

Last Action: {self.last_action}

New Observation:
{observation}

Analyze the transition and output the updated state in JSON format."""
        
        messages = [
            {"role": "system", "content": self.updater_prompt},
            {"role": "user", "content": prompt}
        ]
        
        response = self.llm.chat(messages, temperature=0.0, max_tokens=800)
        new_state = self._parse_json(response, default=self.state)
        
        if new_state:
            # Preserve attempted_actions from code-level tracking
            if "constraint_state" in new_state:
                new_state["constraint_state"]["attempted_actions"] = self.state.get("constraint_state", {}).get("attempted_actions", [])
            self.state = new_state
            if self.verbose:
                print(f"\n[ZipAct] Updated state: {json.dumps(self.state, indent=2)}")
    
    def _act(self, observation: str) -> Tuple[str, str]:
        """
        Decide next action based on current state and observation.
        Implements π: a_t ~ π(·|S_t, o_t)
        
        Returns:
            (thought, action): Reasoning and action
        """
        state_str = json.dumps(self.state, indent=2, ensure_ascii=False)
        
        prompt = f"""Current State:
```json
{state_str}
```

Immediate Observation:
{observation}

Decide what to do next."""
        
        messages = [
            {"role": "system", "content": self.actor_prompt},
            {"role": "user", "content": prompt}
        ]
        
        response = self.llm.chat(messages, temperature=0.0, max_tokens=256)
        
        # Parse thought and action
        thought, action = self._parse_thought_action(response)
        
        return thought, action
    
    def _parse_json(self, text: str, default: Dict = None) -> Dict[str, Any]:
        """Parse JSON from LLM response."""
        if default is None:
            default = {}
        
        try:
            # Try to extract JSON block
            if "```json" in text:
                json_str = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                json_str = text.split("```")[1].split("```")[0].strip()
            else:
                json_str = text.strip()
            
            # Remove any trailing commas before closing braces/brackets
            json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
            
            return json.loads(json_str)
        except Exception as e:
            if self.verbose:
                print(f"[ZipAct] Failed to parse JSON: {e}")
                print(f"[ZipAct] Raw text: {text[:200]}...")
            return default
    
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
            # Maybe the format is different, use the whole text as thought
            thought = text.strip()
            # Try to find action-like commands
            action_match = re.search(r'\b(go to|take|put|open|close|clean|heat|cool|toggle|use|look|inventory)\b.+', text, re.IGNORECASE)
            if action_match:
                action = action_match.group(0).strip()
        
        return thought, action
    
    def _get_empty_state(self) -> Dict[str, Any]:
        """Return an empty state template."""
        return {
            "goal_state": {
                "global_instruction": "",
                "sub_goal_queue": [],
                "current_objective": ""
            },
            "world_state": {
                "location": "unknown",
                "inventory": [],
                "entity_map": {},
                "discovered_objects": []
            },
            "constraint_state": {
                "negative_constraints": [],
                "visited_locations": [],
                "attempted_actions": []
            }
        }

