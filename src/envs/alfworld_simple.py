"""
Simplified ALFWorld Environment for testing without full installation.
Uses a simulated household task interaction pattern similar to ALFWorld.
"""

import re
from typing import Tuple, Dict, List
from .base import BaseEnv


class ALFWorldSimpleEnv(BaseEnv):
    """
    Simplified ALFWorld environment that simulates household tasks.
    This version doesn't require the full alfworld installation.
    """
    
    # Predefined task for testing - similar to ALFWorld task 1
    TASK_DATA = {
        "task": "heat some mug and put it in cabinet.",
        "initial_obs": "You are in the middle of a room. Looking quickly around you, you see a cabinet 6, a cabinet 5, a cabinet 4, a cabinet 3, a cabinet 2, a cabinet 1, a coffeemachine 1, a countertop 3, a countertop 2, a countertop 1, a drawer 3, a drawer 2, a drawer 1, a fridge 1, a garbagecan 1, a microwave 1, a shelf 3, a shelf 2, a shelf 1, a sinkbasin 1, a stoveburner 4, a stoveburner 3, a stoveburner 2, a stoveburner 1, and a toaster 1.",
        "locations": {
            "countertop 1": ["creditcard 1", "potato 2"],
            "countertop 2": ["butterknife 1", "knife 1", "peppershaker 2", "spoon 1", "tomato 1", "vase 3", "vase 2"],
            "countertop 3": ["butterknife 3", "butterknife 2", "cellphone 2", "creditcard 2", "dishsponge 3", "glassbottle 3", "houseplant 1", "knife 3", "knife 2", "pan 2", "saltshaker 2", "soapbottle 3", "spoon 3", "statue 1", "tomato 2"],
            "cabinet 1": ["mug 3"],
            "cabinet 2": [],
            "cabinet 3": ["plate 1"],
            "cabinet 4": ["bowl 1"],
            "cabinet 5": [],
            "cabinet 6": ["soapbottle 1"],
            "drawer 1": ["fork 1"],
            "drawer 2": ["spoon 2"],
            "drawer 3": [],
            "fridge 1": ["apple 1", "lettuce 1", "mug 1", "tomato 3"],
            "microwave 1": [],
            "coffeemachine 1": ["mug 2"],
            "shelf 1": ["peppershaker 1"],
            "shelf 2": ["vase 1"],
            "shelf 3": ["statue 2"],
            "sinkbasin 1": [],
            "garbagecan 1": [],
            "stoveburner 1": [],
            "stoveburner 2": [],
            "stoveburner 3": [],
            "stoveburner 4": ["pot 1"],
            "toaster 1": [],
        },
        "needs_open": ["fridge 1", "microwave 1", "cabinet 1", "cabinet 2", "cabinet 3", "cabinet 4", "cabinet 5", "cabinet 6"],
    }
    
    def __init__(self, task_id: int = 0, max_steps: int = 30):
        """
        Initialize the simplified ALFWorld environment.
        
        Args:
            task_id: Task identifier (for now, only task 0 is available)
            max_steps: Maximum number of steps before timeout
        """
        self.task_id = task_id
        self.max_steps = max_steps
        self.reset()
    
    def reset(self) -> Tuple[str, Dict]:
        """Reset the environment to initial state."""
        self.current_location = None
        self.inventory = []
        self.heated_objects = set()
        self.cooled_objects = set()
        self.cleaned_objects = set()
        self.open_receptacles = set()
        self.step_count = 0
        self.done = False
        self.success = False
        
        # Deep copy locations to avoid modifying original
        self.locations = {k: list(v) for k, v in self.TASK_DATA["locations"].items()}
        
        task = self.TASK_DATA["task"]
        initial_obs = self.TASK_DATA["initial_obs"]
        
        full_obs = f"{initial_obs}\nYour task is to: {task}"
        
        info = {
            "task": task,
            "admissible_commands": self._get_admissible_commands(),
        }
        
        return full_obs, info
    
    def step(self, action: str) -> Tuple[str, float, bool, Dict]:
        """
        Execute an action and return the result.
        
        Args:
            action: Action string to execute
            
        Returns:
            observation: Text observation
            reward: 1.0 for success, 0.0 otherwise
            done: Whether episode is finished
            info: Additional info
        """
        self.step_count += 1
        action = action.strip().lower()
        
        # Check for timeout
        if self.step_count >= self.max_steps:
            self.done = True
            return "Episode timed out.", 0.0, True, {"success": False}
        
        # Parse and execute action
        obs = self._execute_action(action)
        
        # Check for success
        reward = 1.0 if self.success else 0.0
        
        info = {
            "success": self.success,
            "admissible_commands": self._get_admissible_commands(),
        }
        
        return obs, reward, self.done, info
    
    def _execute_action(self, action: str) -> str:
        """Execute an action and return observation."""
        
        # go to {recep}
        go_match = re.match(r"go to (.+)", action)
        if go_match:
            location = go_match.group(1).strip()
            return self._go_to(location)
        
        # take {obj} from {recep}
        take_match = re.match(r"take (.+) from (.+)", action)
        if take_match:
            obj = take_match.group(1).strip()
            recep = take_match.group(2).strip()
            return self._take(obj, recep)
        
        # put {obj} in/on {recep}
        put_match = re.match(r"put (.+) (?:in|on|in/on) (.+)", action)
        if put_match:
            obj = put_match.group(1).strip()
            recep = put_match.group(2).strip()
            return self._put(obj, recep)
        
        # open {recep}
        open_match = re.match(r"open (.+)", action)
        if open_match:
            recep = open_match.group(1).strip()
            return self._open(recep)
        
        # close {recep}
        close_match = re.match(r"close (.+)", action)
        if close_match:
            recep = close_match.group(1).strip()
            return self._close(recep)
        
        # heat {obj} with {recep}
        heat_match = re.match(r"heat (.+) with (.+)", action)
        if heat_match:
            obj = heat_match.group(1).strip()
            recep = heat_match.group(2).strip()
            return self._heat(obj, recep)
        
        # cool {obj} with {recep}
        cool_match = re.match(r"cool (.+) with (.+)", action)
        if cool_match:
            obj = cool_match.group(1).strip()
            recep = cool_match.group(2).strip()
            return self._cool(obj, recep)
        
        # clean {obj} with {recep}
        clean_match = re.match(r"clean (.+) with (.+)", action)
        if clean_match:
            obj = clean_match.group(1).strip()
            recep = clean_match.group(2).strip()
            return self._clean(obj, recep)
        
        # examine/look at
        examine_match = re.match(r"(?:examine|look at) (.+)", action)
        if examine_match:
            target = examine_match.group(1).strip()
            return self._examine(target)
        
        return "Nothing happened."
    
    def _go_to(self, location: str) -> str:
        """Go to a location."""
        if location not in self.locations:
            return "Nothing happened."
        
        self.current_location = location
        
        # Check if location needs to be opened
        if location in self.TASK_DATA["needs_open"] and location not in self.open_receptacles:
            return f"The {location} is closed."
        
        items = self.locations[location]
        if items:
            items_str = ", ".join([f"a {item}" for item in items])
            return f"On the {location}, you see {items_str}."
        else:
            return f"On the {location}, you see nothing."
    
    def _take(self, obj: str, recep: str) -> str:
        """Take an object from a receptacle."""
        if self.inventory:
            return "Nothing happened."  # Already holding something
        
        if recep not in self.locations:
            return "Nothing happened."
        
        if recep in self.TASK_DATA["needs_open"] and recep not in self.open_receptacles:
            return "Nothing happened."  # Need to open first
        
        if obj in self.locations[recep]:
            self.locations[recep].remove(obj)
            self.inventory.append(obj)
            return f"You pick up the {obj} from the {recep}."
        
        return "Nothing happened."
    
    def _put(self, obj: str, recep: str) -> str:
        """Put an object in/on a receptacle."""
        if obj not in self.inventory:
            return "Nothing happened."
        
        if recep not in self.locations:
            return "Nothing happened."
        
        if recep in self.TASK_DATA["needs_open"] and recep not in self.open_receptacles:
            return "Nothing happened."  # Need to open first
        
        self.inventory.remove(obj)
        self.locations[recep].append(obj)
        
        # Check win condition: heated mug in cabinet
        task = self.TASK_DATA["task"]
        if "heat" in task and "mug" in task and "cabinet" in task:
            if "mug" in obj and obj in self.heated_objects and "cabinet" in recep:
                self.success = True
                self.done = True
                return f"You put the {obj} in/on the {recep}."
        
        return f"You put the {obj} in/on the {recep}."
    
    def _open(self, recep: str) -> str:
        """Open a receptacle."""
        if recep not in self.locations:
            return "Nothing happened."
        
        if recep not in self.TASK_DATA["needs_open"]:
            return "Nothing happened."  # Can't open this
        
        if recep in self.open_receptacles:
            return f"The {recep} is already open."
        
        self.open_receptacles.add(recep)
        self.current_location = recep
        
        items = self.locations[recep]
        if items:
            items_str = ", ".join([f"a {item}" for item in items])
            return f"You open the {recep}. The {recep} is open. In it, you see {items_str}."
        else:
            return f"You open the {recep}. The {recep} is open. In it, you see nothing."
    
    def _close(self, recep: str) -> str:
        """Close a receptacle."""
        if recep not in self.TASK_DATA["needs_open"]:
            return "Nothing happened."
        
        if recep not in self.open_receptacles:
            return f"The {recep} is already closed."
        
        self.open_receptacles.discard(recep)
        return f"You close the {recep}."
    
    def _heat(self, obj: str, recep: str) -> str:
        """Heat an object with a receptacle (microwave)."""
        if obj not in self.inventory:
            return "Nothing happened."
        
        if "microwave" not in recep:
            return "Nothing happened."
        
        self.heated_objects.add(obj)
        return f"You heat the {obj} using the {recep}."
    
    def _cool(self, obj: str, recep: str) -> str:
        """Cool an object with a receptacle (fridge)."""
        if obj not in self.inventory:
            return "Nothing happened."
        
        if "fridge" not in recep:
            return "Nothing happened."
        
        self.cooled_objects.add(obj)
        return f"You cool the {obj} using the {recep}."
    
    def _clean(self, obj: str, recep: str) -> str:
        """Clean an object with a receptacle (sinkbasin)."""
        if obj not in self.inventory:
            return "Nothing happened."
        
        if "sinkbasin" not in recep:
            return "Nothing happened."
        
        self.cleaned_objects.add(obj)
        return f"You clean the {obj} using the {recep}."
    
    def _examine(self, target: str) -> str:
        """Examine an object or receptacle."""
        if target in self.inventory:
            status = []
            if target in self.heated_objects:
                status.append("heated")
            if target in self.cooled_objects:
                status.append("cooled")
            if target in self.cleaned_objects:
                status.append("cleaned")
            if status:
                return f"The {target} is {', '.join(status)}."
            return f"This is a {target}."
        
        return "Nothing happened."
    
    def _get_admissible_commands(self) -> List[str]:
        """Get list of valid commands (simplified)."""
        commands = []
        
        # Go to locations
        for loc in self.locations.keys():
            commands.append(f"go to {loc}")
        
        # Open/close receptacles
        for recep in self.TASK_DATA["needs_open"]:
            if recep in self.open_receptacles:
                commands.append(f"close {recep}")
            else:
                commands.append(f"open {recep}")
        
        # Take objects from current location
        if self.current_location and not self.inventory:
            for obj in self.locations.get(self.current_location, []):
                commands.append(f"take {obj} from {self.current_location}")
        
        # Put/heat/cool/clean with inventory
        if self.inventory:
            obj = self.inventory[0]
            for loc in self.locations.keys():
                commands.append(f"put {obj} in/on {loc}")
            commands.append(f"heat {obj} with microwave 1")
            commands.append(f"cool {obj} with fridge 1")
            commands.append(f"clean {obj} with sinkbasin 1")
        
        return commands
    
    def get_task_description(self) -> str:
        """Get the current task description."""
        return self.TASK_DATA["task"]
