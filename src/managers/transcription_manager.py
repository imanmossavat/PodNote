# src/managers/transcription_manager.py
import os
import sys

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
print(root_dir)
sys.path.append(root_dir)

import whisper

class TranscriptionManager:
    def __init__(self, config):
        self.config = config
        self.model = self.load_model(config.model_config['model_name'])
        self.transcription = None
        self.word_timestamps = None
        self.model = None

    def load_model(self, model_name):
        """Loads the Whisper model."""
        self.model = whisper.load_model(model_name)
    
    def transcribe_audio(self, audio):
        """Transcribes audio using the Whisper model."""

        domain_prompt= self.prompt['domain']
        

        prompt = self.config.get_full_prompt()

        
        result = self.model.transcribe(audio.squeeze().numpy(), word_timestamps=True, initial_prompt=prompt)
        self.transcription = result['text']
        self.word_timestamps = result['segments']
