# src/config/config.py

class Config:
    # Default model name
    whisper_models = [
        "tiny.en", "tiny", "base.en", "base", "small.en", "small", 
        "medium.en", "medium", "large-v1", "large-v2", "large-v3", 
        "large", "large-v3-turbo", "turbo"
    ]
    
    def __init__(self, model_name="tiny.en", chunk_size=500, report_interval=10):
        # Initialize model configuration
        self.model_name = model_name if model_name in self.whisper_models else "tiny.en"
        
        # Reporting configuration
        self.reporting_config = {
            'report_interval': report_interval  # in minutes
        }

        # Model configuration
        self.model_config = {
            'model_name': self.model_name,
            'chunk_size': chunk_size
        }

    def get_model_config(self):
        return self.model_config

    def get_reporting_config(self):
        return self.reporting_config

    def set_model(self, model_name):
        if model_name in self.whisper_models:
            self.model_config['model_name'] = model_name
        else:
            raise ValueError(f"Invalid model name. Choose from: {', '.join(self.whisper_models)}")

    def set_chunk_size(self, chunk_size):
        self.model_config['chunk_size'] = chunk_size

    def set_report_interval(self, report_interval):
        self.reporting_config['report_interval'] = report_interval

    def get_all_models(self):
        return self.whisper_models
