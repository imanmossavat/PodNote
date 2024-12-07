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

        self.model= None
        self.config.general['logger'].info(f'transcription model init {self.model}')
        self.transcription = None
        self.word_timestamps = None
        
    def load_model(self, model_name, device="cpu"):
        """Loads the Whisper model."""
        self.model = whisper.load_model(model_name, device=device)
        if self.model is None:
            self.config.general['logger'].info(f'transcription model failed to load')
        else:
            self.config.general['logger'].info(f'transcription model {self.model} loaded in device: {device}')


    
    def transcribe_audio(self, audio, prompt= None):
        """Transcribes audio using the Whisper model."""
        
        if self.model is None:
            self.load_model(
                model_name= self.config.model_config['transcription_model_name'],device= self.config.general['device'])

        if prompt is None:
            prompt = self.config.get_full_prompt()

        
        result = self.model.transcribe(
            audio.squeeze().numpy(), 
            word_timestamps=True, 
            initial_prompt=prompt)
        
        self.transcription = result['text']
        self.word_timestamps = result['segments']
