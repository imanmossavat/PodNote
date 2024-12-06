"""
The `system` module contains the core management components for the application. These components handle configuration management, state management, and user interface interactions.

Classes:
    - ConfigManager: Manages all configurations for the application, ensuring consistent settings for models, directories, and other configurations. It also handles logging and global settings.
    - StateManager: Manages the application state, allowing for saving, undoing, and redoing changes. It is essential for tracking and reverting to previous configurations or actions.
    - UIManager: Handles user interactions with the application, ensuring that the user interface is responsive to state changes and configuration updates. It also integrates with other managers to process and analyze audio files, generate reports, and more.

Each of these components is designed to work together, providing a centralized and modular approach for handling key aspects of the application's behavior, configuration, and user interaction.

Usage:
    - `ConfigManager`: Used to configure the application's settings, such as model names, chunk sizes, and logging.
    - `StateManager`: Used to manage the state of the application, track changes, and allow undo/redo functionality.
    - `UIManager`: Used to interact with the user, process inputs, manage workflows, and provide real-time feedback during application execution.
"""


from .config_manager import Config
from .state_manager import StateManager
from .ui_manager import UIManager 

__all__= ['Config','StateManager','UIManager']