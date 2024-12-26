
import os
import sys

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
print(root_dir)
sys.path.append(root_dir)
from managers import ProcessingService


class UIManager:
    """
    Manages the user interface interactions and reacts to state updates.

    Attributes:
        config (Config): The current configuration instance.
        processing_service (ProcessingService): Handles processing tasks.
    """
    def __init__(self, config):
        self.config = config
        
        self.processing_service = ProcessingService(config)


        # Access logger from config
        self.logger = self.config.general['logger']

    def change_audio_file_name(self,audio_file):
        self.config.set_config_value(
            'general.audio_file', 
            audio_file, self.processing_service)

        

    def process_audio(self):
        audio_file= self.config.general['audio_file']

        try:
            self.processing_service.process_audio(audio_file)
            self.logger.info("Audio processed successfully.")
        except Exception as e:
            self.logger.error(f"Error processing audio: {e}")
            raise

    def transcribe(self, prompt=None):
        
        if prompt is not None:
            self.logger.info(f"Changing config prompt by ui-manager transcribe")


        try:
            self.processing_service.transcribe()
            self.logger.info("Transcription processed successfully.")
        except Exception as e:
            self.logger.error(f"Error transcribing: {e}")
            raise

    def report(self):
        try:
            self.processing_service.report()
#            self.logger.info("Report generated successfully.")
        except Exception as e:
            self.logger.error(f"Error generating report: {e}")
            raise


    def save_raw_transcription(self, text_filename=None, timestamp=None):
        self.processing_service.save_raw_transcription(
            text_filename=text_filename,
            timestamp=timestamp)
        

    def save_transcription_state(self, state_file):
        """Save the transcription state to a file."""
        self.processing_service.transcription_manager.save_state(state_file)
        self.logger.info(f'state saved to {state_file}')

    def load_transcription_state(self, state_file):
        """Load the transcription state from a file."""
        self.processing_service.transcription_manager.load_state(state_file)
        self.logger.info(f'state loaded from {state_file}')


