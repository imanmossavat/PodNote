import os
import sys

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
print(root_dir)
sys.path.append(root_dir)

from src import Config, UIManager, StateManager

import logging
import os



def main():
    # Step 1: Set up the configuration (Config) for the system
    config = Config()

    # Set up the logger (for logging information)
    logger = logging.getLogger('AudioProcessingApp')
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    config.logger = logger

    # Set directories in config for reports
    config.directories = {
        'report_dir': os.path.join(os.getcwd(), 'reports')
    }

    # Ensure the report directory exists
    if not os.path.exists(config.directories['report_dir']):
        os.makedirs(config.directories['report_dir'])

    # Step 2: Initialize the State Manager (tracking application state)
    state_manager = StateManager(config)

    # Step 3: Initialize the UI Manager (handling user requests and managing interactions)
    ui_manager = UIManager(config, state_manager)

    # Step 4: Simulate loading the audio file from the user's input
    audio_file = "path_to_audio_file.wav"  # Update this path as needed

    # Log the audio file processing start
    logger.info(f"Loading and processing the audio file: {audio_file}")

    try:
        # Step 5: Process the audio through the UI manager
        ui_manager.process_audio(audio_file)

        # Step 6: Log the successful processing and report generation
        logger.info(f"Audio file '{audio_file}' has been processed successfully.")
    
    except Exception as e:
        # Log any errors that occur during the process
        logger.error(f"An error occurred while processing the audio file: {e}")
        raise

if __name__ == "__main__":
    main()


