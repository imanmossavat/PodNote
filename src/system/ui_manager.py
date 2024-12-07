
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
        state_manager (StateManager): Manages application state and subscriptions.
        processing_service (ProcessingService): Handles processing tasks.
    """
    def __init__(self, config, state_manager):
        self.config = config
        self.state_manager = state_manager
        self.processing_service = ProcessingService(config)

        # Subscribe to state updates
        self.state_manager.subscribe(self)

        # Access logger from config
        self.logger = self.config.general['logger']

    def update_config(self, new_config):
        """React to updated configuration from the StateManager."""
        self.config = new_config
        self.logger = self.config.general['logger']
        self.logger.info("UI updated with new configuration.")

    def process_audio(self, audio_file):
        try:
            self.processing_service.process_audio(audio_file)
            self.logger.info("Audio processed successfully.")
        except Exception as e:
            self.logger.error(f"Error processing audio: {e}")
            raise

    def transcribe(self, prompt=None):
        
        if prompt is not None:
            self.state_manager.update_config(prompt=prompt)
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
            self.logger.info("Report generated successfully.")
        except Exception as e:
            self.logger.error(f"Error generating report: {e}")
            raise
