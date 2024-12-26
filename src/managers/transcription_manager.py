# src/managers/transcription_manager.py
import os
import sys
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
print(root_dir)
sys.path.append(root_dir)
import json

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


    def save_state(self, state_file):
        """Save the state of the transcription to a file."""
        state = {
            "transcription": self.transcription,
            "word_timestamps": self.word_timestamps,
            "audio_file_name": self.audio_file_name,
            "prompt": self.prompt,
            "report_dir": self.report_dir,
        }
        with open(state_file, "w") as f:
            json.dump(state, f)

    def load_state(self, state_file):
        """Load the state of the transcription from a file."""
        with open(state_file, "r") as f:
            state = json.load(f)
        self.transcription = state["transcription"]
        self.word_timestamps = state["word_timestamps"]
        self.audio_file_name = state["audio_file_name"]
        self.prompt = state["prompt"]
        self.report_dir = state["report_dir"]
