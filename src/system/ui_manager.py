
import os
import sys

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
print(root_dir)
sys.path.append(root_dir)
from ..managers import ProcessingService

from .state_manager import StateManager

class UIManager:
    def __init__(self, config, state_manager):
        self.config = config
        self.state_manager = state_manager
        self.processing_service= ProcessingService(config)

        
        # Access logger from config
        self.logger = self.config.logger

    
    def process_audio(self, audio_file):
        try:
            self.processing_service.process_audio(audio_file)
            self.logger.info("Audio processed successfully.")
        except Exception as e:
            self.logger.error(f"Error processing audio: {e}")
            raise

    def transcribe(self, prompt):
        try:
            self.processing_service.transcribe(prompt)
            self.logger.info("Transcription processed successfully.")
        except Exception as e:
            self.logger.error(f"Error Transcription: {e}")
            raise


    def report(self):
        try:
            self.processing_service.report()
            self.logger.info("Report generate successfully.")
        except Exception as e:
            self.logger.error(f"Error generating report: {e}")
            raise