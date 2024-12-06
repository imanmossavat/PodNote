import copy
import json
import os

class StateManager:
    """
    Manages the configuration state with support for undo and redo functionality.
    It allows updating the configuration, saving states, and reverting to previous states.
    The state history can be persisted to a file.

    Attributes:
        config (Config): The configuration instance being managed.
        history (list): A stack storing previous configuration states.
        redo_stack (list): A stack storing undone states for redo functionality.
        max_history_size (int): The maximum number of states to retain in history.
        state_file (str): The file path to save and load state history.
    """
    def __init__(self, config, max_history_size=10, state_file="state_history.json"):
        self.config = config  # Instance of Config class
        self.history = []
        self.redo_stack = []
        self.max_history_size = max_history_size
        self.state_file = state_file
        
        # Load history from file if it exists
        self.load_state_from_file()

    def update_config(self, model_name=None, chunk_size=None, report_interval=None):
        """Update configuration and save the state."""
        self.save_state()
        if model_name:
            self.config.model_config['model_name'] = model_name
        if chunk_size:
            self.config.model_config['chunk_size'] = chunk_size
        if report_interval:
            self.config.reporting_config['report_interval'] = report_interval
        
        # Optionally, save the state to a file
        self.save_state_to_file()

    def save_state(self):
        """Save the current state before making changes."""
        if len(self.history) >= self.max_history_size:
            self.history.pop(0)  # Limit history size
        
        self.history.append(copy.deepcopy(self.config))  # Save state of config
        self.redo_stack.clear()  # Clear redo stack

    def undo(self):
        """Undo the last change and revert to the previous state."""
        if self.history:
            self.redo_stack.append(copy.deepcopy(self.config))
            self.config = self.history.pop()
            self.save_state_to_file()
        else:
            print("No history to undo.")
        
    def redo(self):
        """Redo the last undone change."""
        if self.redo_stack:
            self.history.append(copy.deepcopy(self.config))
            self.config = self.redo_stack.pop()
            self.save_state_to_file()
        else:
            print("No history to redo.")

    def load_state_from_file(self):
        """Load state history from a file if it exists."""
        if os.path.exists(self.state_file):
            with open(self.state_file, 'r') as f:
                state_data = json.load(f)
                # Assuming that the state file contains a list of configurations
                self.history = [self.config] + state_data.get("history", [])
                self.redo_stack = state_data.get("redo_stack", [])

    def save_state_to_file(self):
        """Save the current history and redo stack to a file."""
        state_data = {
            "history": [self.config] + self.history,
            "redo_stack": self.redo_stack
        }
        with open(self.state_file, 'w') as f:
            json.dump(state_data, f, indent=4)
