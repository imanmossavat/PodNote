import os
import sys

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
print(root_dir)
sys.path.append(root_dir)

import src 

def main():
    config = src.Config()
    ui_manager = src.UIManager(config)

    audio_file = "path_to_audio_file.wav"
    ui_manager.process_audio(audio_file)

if __name__ == "__main__":
    main()
