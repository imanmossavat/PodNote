import logging
import os
import sys
import time
import spacy

# Set root directory to 'src'
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(root_dir)

class Config:
    # Default model names
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
                 job_name="Job",
                 timestamp_format="%Y%m%d-%H%M%S",
                ):
        
        # Generate timestamp
        self.timestamp = time.strftime(timestamp_format)  # format: YYYYMMDD-HHMMSS

        # Set data_dir if not provided
        if data_dir is None:
            data_dir = os.path.join(root_dir, 'data', f"{job_name}_{self.timestamp}")
        
        # Initialize logger
        self.logger = self._create_default_logger()
                
        # Model initialization with logging in case of issues
        if model_name not in self.whisper_models:
            self.logger.info(f"Model '{model_name}' not found. Using default 'tiny.en'.")
            model_name = "tiny.en"
            
        # Model and reporting configuration
        self.model_config = {
            'model_name': model_name,
            'chunk_size': chunk_size
        }
        
        self.reporting_config = {
            'report_interval': report_interval  # in minutes
        }

        # Setting up the directories based on data_dir
        self.directories = {
            'root': root_dir,
            'data_dir': data_dir,
            'audio_dir': os.path.join(data_dir, "audio"),
            'report_dir': os.path.join(data_dir, "reports"),
            'log_dir': os.path.join(data_dir, "logs"),
            'pkl_dir': os.path.join(data_dir, "pkl")
        }

        self._create_directories()

        self.prompt = {
            'filler': "I'm like, you know what I mean, kind of, um, ah, huh, and so, so um, uh, and um, like um, so like, like it's, it's like, i mean, yeah, ok so, uh so, so uh, yeah so, you know, it's uh, uh and, and uh, like, kind", 
            'domain': ""
        }

        self.user_highlight_keywords= []
        self.filler_words_removed=["um", "uh", "like"]

        self.nlp= spacy.load("en_core_web_sm")

    def get_full_prompt(self):
        return self.prompt['filler']+self.prompt['domain']
    
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

        # Optional: Add file handler if required
        log_file = os.path.join(self.directories['log_dir'], "app.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        logger.setLevel(logging.INFO)
        return logger
