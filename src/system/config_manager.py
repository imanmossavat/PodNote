"""
Config Management System

The `Config` class manages the configuration for the application, providing:
- Structured, nested configuration fields for models, directories, reporting, and more.
- Methods for dynamic updates via `set_config_value` using dot-separated keys.
- Automatic directory management and logging setup.
- Integration with a `StateManager` to notify changes.

Key Features:
1. **Set Configuration Dynamically**:
   Use `config.set_config_value` to set values dynamically. For example:
config.set_config_value('general.audio_file', 'path/to/audio.mp3')


2. **Automatic Directory Management**:
Ensures required directories (e.g., data, logs, reports) are created if missing.

3. **Model Configuration**:
Default model settings with validation for supported Whisper models.

4. **Logging**:
Logs significant configuration changes and events, provided a logger is initialized.

5. **State Management**:
Automatically notifies the `StateManager` when configuration changes, ensuring consistent state.

6. **NLP Utilities**:
Loads a SpaCy model (`en_core_web_sm`) and allows customization of filler word processing.

Key Attributes:
- `general`: Contains general configuration such as job name, timestamp, logger, and audio file.
- `directories`: Manages paths for root, data, reports, logs, and pickles.
- `model_config`: Manages model settings like `transcription_model_name` and `chunk_size`.
- `reporting_config`: Defines reporting intervals.
- `prompt`: Stores default and custom filler/domain-specific prompts.
- `nlp`: Configuration for NLP processing.

Key Methods:
- `set_config_value(key, value)`: Sets configuration fields using dot-separated keys.
- `update_audio_file(audio_file_path)`: Updates configuration and paths based on the given audio file.

Notes:
- A logger is created during initialization. Additional logging points can be added to ensure traceability of updates.
- If `StateManager` is not set, changes will not be propagated outside the configuration.

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
                
        self.state_manager = None  
        
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
            if 'logger' in self.general and self.general['logger']:
                self.general['logger'].info(f"Model '{model_name}' not found. Using default 'tiny.en'.")
            model_name = "tiny.en"
        
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

    def set_config_value(self, key, value):
        keys = key.split('.')
        config_dict = self.__dict__

        for part in keys[:-1]:
            if part not in config_dict:
                raise KeyError(f"Key '{part}' not found in the configuration.")
            config_dict = config_dict[part]

        final_key = keys[-1]
        if final_key not in config_dict:
            raise KeyError(f"Key '{final_key}' not found in the configuration.")
        config_dict[final_key] = value
        
        if 'logger' in self.general and self.general['logger']:
            self.general['logger'].info(f"Updated configuration key '{key}' to value '{value}'")

        if self.state_manager:
            self._notify_state_manager()
            

    def _notify_state_manager(self):
        """Notify the StateManager about changes to the config."""
        self.state_manager.update_config(self)

    def _create_directories(self):
        """Create the required directories if they don't exist."""
        for key, directory in self.directories.items():
            os.makedirs(directory, exist_ok=True)
        
        if 'logger' in self.general and self.general['logger']:
            self.general['logger'].info(f"Ensuring directories exists")


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

        audio_file_name = os.path.splitext(os.path.basename(audio_file_path))[0]
        self.general['audio_file'] = audio_file_path

        self.directories['report_dir'] = os.path.join(self.directories['report_dir'], f"audio_{audio_file_name}")


        self._create_directories()
        if self.state_manager is not None: 
            self._notify_state_manager()
        self.general['logger'].info(f"Configuration updated for audio file: {audio_file_path}")
