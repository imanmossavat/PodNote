import copy
import json
import os

class StateManager:
    """
    Manages the configuration state, including history, undo/redo functionality, 
    and notifying subscribed components when the state changes.

    Attributes:
        config (Config): The current configuration being managed.
        history (list): A stack storing previous configuration states.
        redo_stack (list): A stack storing undone states for redo functionality.
        subscribers (list): List of subscribed components to notify on config updates.
    """
    def __init__(self, 
                 config, 
                 max_history_size=10, state_file="state_history.json"):
        self.config = config  # Instance of Config class
        self.history = []
        self.redo_stack = []
        self.max_history_size = max_history_size
        self.state_file = state_file
        self.subscribers = []  # List to manage subscribed components

        # Load history from file if it exists
        self.load_state_from_file()
        self.config.state_manager= self

    def update_config(self, config):
        """Update configuration with the provided new config object and notify subscribers."""
        # Save current state before applying the update
        self.save_state()

        # Replace the current config with the new one
        self.config = config

        # Notify all subscribers of the updated config
        self.notify_subscribers()

        # Optionally save the updated state to a file
        self.save_state_to_file()

    def subscribe(self, subscriber):
        """Add a subscriber to the list."""
        self.subscribers.append(subscriber)

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
            self.notify_subscribers()  # Notify subscribers of state change
            self.save_state_to_file()
        else:
            print("No history to undo.")

    def redo(self):
        """Redo the last undone change."""
        if self.redo_stack:
            self.history.append(copy.deepcopy(self.config))
            self.config = self.redo_stack.pop()
            self.notify_subscribers()  # Notify subscribers of state change
            self.save_state_to_file()
        else:
            print("No history to redo.")

    def notify_subscribers(self):
        """Notify all subscribers about the updated configuration."""
        for subscriber in self.subscribers:
            subscriber.update_config(self.config)

    def load_state_from_file(self):
        """Load state history from a file if it exists."""
        if os.path.exists(self.state_file):
            with open(self.state_file, 'r') as f:
                state_data = json.load(f)
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
