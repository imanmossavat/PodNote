import logging
import os
import sys
import time

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
                 job_name='default_job',
                 target_sample_rate=16000,
                 timestamp_format="%Y%m%d-%H%M%S",
                ):
        
        self.general = {
            'timestamp': time.strftime(timestamp_format),
            'job_name': job_name,
            'logger': None,
            'device': 'cpu',
            'audio_file': None,
            'target_sample_rate': target_sample_rate,
            'open_report_after_save': False
        }        
        if data_dir is None:
            data_dir = os.path.join(root_dir, 
                                    'data', 
                                    job_name,
                                    self.general['timestamp'])
        
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
            'domain': "Iman Mossavat, Fontys, Eindhoven, Diederik roijers, Brainport, TUE (technical university of Eindhoven), legistavely, GDPR "
        }
        
        self.nlp = {'spacy_model':"en_core_web_sm",
                    'user_highlight_keywords': [],
                    'filler_words_removed': ["um", "uh", "like"]
        }

    def get_full_prompt(self):
        return self.prompt['filler'] + self.prompt['domain']

    def set_config_value(self, key, value,processing_service=None):

        """Set configuration fields using dot-separated keys"""
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
        
        if processing_service is not None:
            processing_service.update_config(self)






    def _create_directories(self):
        """Create the required directories if they don't exist."""
        for key, directory in self.directories.items():
            os.makedirs(directory, exist_ok=True)
        
        if 'logger' in self.general and self.general['logger']:
            self.general['logger'].info(f"Ensuring directories exist")

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



        self.general['logger'].info(f"Configuration updated for audio file: {audio_file_path}")

    def to_dict(self):
        """Convert the Config instance to a dictionary."""
        config_dict = self.__dict__.copy()
        
        config_dict.pop("logger", None) 

        return config_dict


