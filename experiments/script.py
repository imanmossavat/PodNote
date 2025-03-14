

import os
import sys
import argparse
from pathlib import Path
import warnings

# Suppress FutureWarnings and UserWarnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# Set up the root directory
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(root_dir)

#sys.argv= ['script.py', r"YOURFILE"]

from src import Config, UIManager
def main():
    
    filename_transcription_state = None

    # Step 1: Set up command-line argument parsing for the audio file
    parser = argparse.ArgumentParser(description="Transcribe a podcast audio file to markdown.")
    parser.add_argument('audio_file', type=str, help="Path to the audio file to transcribe.")

    args = parser.parse_args()

    audio_file = args.audio_file
    data_dir= Path(audio_file).parent
    # Step 2: Set up the configuration (Config) for the system
    config = Config(data_dir=data_dir)
   

    # Add special names or domain-specific terms to assist transcription
    # Include:
    # - Your name or any variations
    # - Names of collaborators, companies, or institutions
    # - City names or other domain-specific keywords
    config.prompt['domain']='Iman Mossavat, \"Iman Mossavat\", Fontys, Eindhoven' 
    config.prompt['filler']='' # we turn off the filler detection by setting it to be empty

    # Step 3: Initialize the UI Manager (handling user requests and managing interactions)
    ui_manager = UIManager(config)

    # Step 4: Set the audio file from the command-line argument
    ui_manager.change_audio_file_name(audio_file)

    # Log the audio file processing start
    print(f"Processing audio file: {audio_file}")

    try:
        # Step 5: Process the audio through the UI manager
        ui_manager.process_audio()
        


        if filename_transcription_state is not None and os.path.exists(filename_transcription_state):
            # we skip transcription instead we load the file
            ui_manager.load_transcription_state(filename_transcription_state)
        else:
            ui_manager.transcribe()
            data_dir= config.directories['data_dir']
            filename_transcription_state = os.path.join(data_dir, f"transcription_state.json")
            ui_manager.save_transcription_state(filename_transcription_state)

            print(f'transcription data saved in {filename_transcription_state}')
            print('you can load the transcription file by ui_manager.load_transcription_state')
            ui_manager.save_raw_transcription() # if you need the information in a text file.
        
        ui_manager.report() # makes the final HTML report
    
    except Exception as e:
        # Log any errors that occur during the process
        print(f"Error processing the audio file: {e}")
        raise

if __name__ == "__main__":
    main()
