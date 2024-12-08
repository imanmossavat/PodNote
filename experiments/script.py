import os
import sys

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
print(root_dir)
sys.path.append(root_dir)

from src import Config, UIManager

import os



def main():
    # Step 1: Set up the configuration (Config) for the system
    config = Config()



    # Step 3: Initialize the UI Manager (handling user requests and managing interactions)
    ui_manager = UIManager(config)

    # Step 4: Simulate loading the audio file from the user's input
    audio_file = r"C:\Users\imanm\OneDrive\Documents\podcast\new_transcipt_code\MD_transcriptor\data\test\promo_.wav"
    ui_manager.change_audio_file_name(audio_file)

    # Log the audio file processing start

    try:
        # Step 5: Process the audio through the UI manager
        ui_manager.process_audio()
        ui_manager.transcribe()
        ui_manager.report()


        # Step 6: Log the successful processing and report generation
    
    except Exception as e:
        # Log any errors that occur during the process
        raise

if __name__ == "__main__":
    main()


