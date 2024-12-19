# src/managers/transcription_manager.py
import os
import sys
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
print(root_dir)
sys.path.append(root_dir)

import whisper

class TranscriptionManager:
    def __init__(self, logger, transcription_model_name, device, prompt=None, report_dir=None, audio_file_name= None):
        self.logger = logger
        self.transcription_model_name= transcription_model_name
        self.device= device
        self.model= None
        self.logger.info(f'transcription model init {self.model}')
        self.transcription = None
        self.word_timestamps = None
        self.prompt= prompt
        self.report_dir= report_dir
        self.audio_file_name= audio_file_name
        
    def load_model(self):
        """Loads the Whisper model."""
        self.model = whisper.load_model(name= self.transcription_model_name,device= self.device)
        if self.model is None:
            self.logger.info(f'transcription model failed to load')
        else:
            self.logger.info(f'transcription model {self.model} loaded in device: {self.device}')


    
    def transcribe_audio(self, audio, prompt= None):
        """Transcribes audio using the Whisper model."""
        
        if self.model is None:
            self.load_model()

        if prompt is None:
            prompt = self.prompt

        
        result = self.model.transcribe(
            audio.squeeze().numpy(), 
            word_timestamps=True, 
            initial_prompt=prompt)
        
        self.transcription = result['text']
        self.word_timestamps = result['segments']

