import os
import sys
import argparse

# Set up the root directory
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(root_dir)

from src import Config, UIManager

def main():
    # Step 1: Set up command-line argument parsing for the audio file
    parser = argparse.ArgumentParser(description="Transcribe a podcast audio file to markdown.")
    parser.add_argument('audio_file', type=str, help="Path to the audio file to transcribe.")
    args = parser.parse_args()



    # Step 2: Set up the configuration (Config) for the system
    config = Config()

    # Step 3: Initialize the UI Manager (handling user requests and managing interactions)
    ui_manager = UIManager(config)

    # Step 4: Set the audio file from the command-line argument
    audio_file = args.audio_file
    ui_manager.change_audio_file_name(audio_file)

    # Log the audio file processing start
    print(f"Processing audio file: {audio_file}")

    try:
        # Step 5: Process the audio through the UI manager
        ui_manager.process_audio()
        ui_manager.transcribe()
        ui_manager.report()
    
    except Exception as e:
        # Log any errors that occur during the process
        print(f"Error processing the audio file: {e}")
        raise

if __name__ == "__main__":
    main()
