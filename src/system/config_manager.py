"""

when setting config use: config.set_config_value

for example, when setting the audio_file name say: config.set_config_value('general.audio_file', audio_file)


Config Management System

This module defines the `Config` class, which is responsible for managing the configuration of the application.
The configuration is structured as a collection of nested dictionaries representing various aspects of the system,
such as model settings, reporting parameters, directory paths, and prompts.

The `Config` class provides the following features:

1. **Dynamic Attribute Management**:
   The configuration can be updated dynamically using attribute assignment, where nested fields in the configuration
   can be accessed and modified using dot-separated keys (e.g., `config.model_config.model_name = 'new_model'`).
   The `__setattr__` method overrides the default attribute assignment to route all updates through a central method
   (`set_config_value`), which handles nested dictionary structures.

2. **Automatic Notification of Configuration Changes**:
   When the configuration is updated, the `Config` class notifies a `StateManager` (if it exists) about the change via
   the `_notify_state_manager` method. This allows other components of the application that depend on the configuration
   to react to the changes (e.g., updating models, restarting processes).

3. **Handling Nested Dictionaries**:
   The configuration contains multiple nested dictionaries (e.g., `model_config`, `reporting_config`, `directories`, etc.).
   The `set_config_value` method allows for easy navigation and updating of deeply nested values using a dot-separated key.
   For example, setting a new chunk size can be done via `config.model_config.chunk_size = 1024`.

4. **Exemptions for Specific Attributes**:
   Some attributes, such as `state_manager`, are not intended to trigger notifications upon being set. These attributes
   are handled specially in the `__setattr__` method to ensure that assigning a new state manager or other similar attributes
   does not inadvertently notify all subscribers or update components unnecessarily.

5. **Flexible Configuration Initialization**:
   The `Config` class supports initialization with default values for various configuration options (e.g., model name,
   chunk size, report interval, etc.). These defaults can be customized during instantiation, and certain directories
   (e.g., logs, audio files, reports) are automatically created based on the provided or default paths.

6. **Logging Setup**:
   The `Config` class includes logging capabilities that capture significant events, errors, or configurations. The logger
   is set up to handle both console and file-based logging, ensuring that important information about the configuration
   and system state is available for troubleshooting.

7. **Directory Management**:
   Automatically manages the creation of required directories, ensuring that paths like `data_dir`, `audio_dir`, and
   `log_dir` exist when needed, avoiding errors when reading or writing to files.

### Rationale:

- **Dynamic Configuration**: The decision to use dot-separated keys for accessing nested configuration values allows for flexibility in managing and modifying configuration fields without tightly coupling the code to specific attributes. It simplifies updates and makes the configuration easier to extend.

- **State Management**: By integrating the state manager with configuration changes, the system ensures that updates to the configuration are consistently tracked, versioned, and reflected across all components that rely on it. This architecture makes it easier to maintain consistency and to revert to previous configurations if necessary.

- **Exemption of `state_manager`**: The `state_manager` attribute is an internal object responsible for managing the configuration's state and history. Since changing the `state_manager` does not affect the configuration directly, it is exempted from triggering notifications to prevent unnecessary updates or state tracking operations.

- **Passing Config by Reference**: The `Config` object is passed by reference throughout the application, meaning that any updates made to the configuration object will be reflected across all components that reference it. This is a deliberate design choice because it allows the configuration to be centrally managed and ensures that changes to the configuration state are immediately visible to all components without needing to duplicate or copy the configuration. The `Config` object is intended to be mutable and shared, making reference passing the most efficient and effective way to propagate changes across the system.

This design provides a robust, flexible, and extensible configuration management system that scales well as new features or settings are added to the application.
"""


import logging
import os
import sys
import time
import spacy

# Set root directory to 'src'
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(root_dir)

class Config:
    whisper_models = [
        "tiny.en", "tiny", "base.en", "base", "small.en", "small", 
        "medium.en", "medium", "large-v1", "large-v2", "large-v3", 
        "large", "large-v3-turbo", "turbo"
    ]

    def __init__(self, 
                 model_name="tiny.en", 
                 chunk_size=500, 
                 report_interval=10, 
                 root_dir=root_dir, 
                 data_dir=None,
                 job_name=None,
                 target_sample_rate= 16000,
                 timestamp_format="%Y%m%d-%H%M%S",
                ):
        
        object.__setattr__(self, '_initialized', False)  # We set it directly to avoid triggering __setattr__

        self.state_manager = None  # To be filled by state manager
        
        # Store timestamp and job_name in a dictionary (instead of directly as attributes)
        self.general = {
            'timestamp': time.strftime(timestamp_format),
            'job_name': job_name,
            'logger': None,
            'device': 'cpu',
            'audio_file': None,
            'target_sample_rate': target_sample_rate 
        }        
        if data_dir is None:
            data_dir = os.path.join(root_dir, 'data', f"{job_name}_{self.general['timestamp']}")
        
        
        self.directories = {
            'root': root_dir,
            'data_dir': data_dir,
            'report_dir': os.path.join(data_dir, "reports"),
            'log_dir': os.path.join(data_dir, "logs"),
            'pkl_dir': os.path.join(data_dir, "pkl")
        }

        self._create_directories()

        
        self.general['logger'] = self._create_default_logger()
        
        if model_name not in self.whisper_models:
            self.general['logger'].info(f"Model '{model_name}' not found. Using default 'tiny.en'.")
            model_name = "tiny.en"
        
        # Configuration dictionaries
        self.model_config = {
            'transcription_model_name': model_name,
            'chunk_size': chunk_size
        }
        self.reporting_config = {
            'report_interval': report_interval
        }
        
        self.prompt = {
            'filler': "I'm like, you know what I mean, kind of, um, ah, huh, and so, so um, uh, and um, like um, so like, like it's, it's like, i mean, yeah, ok so, uh so, so uh, yeah so, you know, it's uh, uh and, and uh, like, kind", 
            'domain': ""
        }
        

        self.nlp = {'spacy_model': spacy.load("en_core_web_sm"),
                    'user_highlight_keywords': [],
                    'filler_words_removed': ["um", "uh", "like"]
        }

                    
        

        object.__setattr__(self, '_initialized', True)  # We set it directly to avoid triggering __setattr__


    def get_full_prompt(self):
        return self.prompt['filler'] + self.prompt['domain']

    def __setattr__(self, name, value):
        """Override attribute assignment to route through set_config_value."""
        if hasattr(self, '_initialized') and self._initialized:
            self.general['logger'].info(f'setting config attributes for {name} to {value}')
        elif hasattr(self, 'general') and self.general['logger'] is not None:
            self.general['logger'].info(f'initializing config attributes for {name} to {value}')

        # Check if object has been fully initialized
        if not hasattr(self, '_initialized') or not self._initialized:
            # during initialization Skip custom logic during initialization
            super().__setattr__(name, value)
        else:
            # Custom handling after initialization
            if name in ['state_manager']:
                if isinstance(value, dict) and name in self.__dict__:
                    self.set_config_value(name, value)
                else:
                    super().__setattr__(name, value)
            elif name == "general.audio_file":
                self.update_audio_config(audio_file_path= value)
            else:
                self.set_config_value(name, value)

    def set_config_value(self, key, value):
        """Sets a value in the config using a dot-separated key to navigate nested dictionaries."""
        keys = key.split('.')  # Split the key into individual parts for nested dictionaries
        config_dict = self.__dict__

        # Navigate through the nested dictionaries based on the keys
        for part in keys[:-1]:  # Traverse all but the last key
            if part not in config_dict:
                raise KeyError(f"Key '{part}' not found in the configuration.")
            config_dict = config_dict[part]  # Go deeper into the nested dictionary

        # Set the final value in the nested dictionary
        final_key = keys[-1]
        if final_key not in config_dict:
            raise KeyError(f"Key '{final_key}' not found in the configuration.")
        config_dict[final_key] = value
        
        # Notify the state manager about the change
        if self.state_manager is not None: # state manager is off when initialization or on stand alone use of config, otherwise, it should be on
            self._notify_state_manager()

        

    def _notify_state_manager(self):
        """Notify the StateManager about changes to the config."""
        self.state_manager.update_config(self)

    def _create_directories(self):
        """Create the required directories if they don't exist."""
        for key, directory in self.directories.items():
            os.makedirs(directory, exist_ok=True)

    def _create_default_logger(self):
        logger = logging.getLogger(__name__)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        log_file = os.path.join(self.directories['log_dir'], "app.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        logger.setLevel(logging.INFO)
        return logger

    def update_audio_file(self, audio_file_path):
        """Update configuration based on the new audio file: audio_file."""
        if not audio_file_path:
            raise ValueError("Audio file path cannot be empty.")

        if not os.path.exists(audio_file_path):
            raise FileNotFoundError(f"Audio file not found: {audio_file_path}")

        # Extract details from the audio file
        audio_file_name = os.path.splitext(os.path.basename(audio_file_path))[0]
        self.general['audio_file'] = audio_file_path

        # Update directories based on the new audio file
        self.directories['report_dir'] = os.path.join(self.directories['report_dir'], f"audio_{audio_file_name}")


        self._create_directories()
        if self.state_manager is not None: # state manager is off when initialization or on stand alone use of config, otherwise, it should be on
            self._notify_state_manager(self)
        self.general['logger'].info(f"Configuration updated for audio file: {audio_file_path}")
