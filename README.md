# Podcast Audio to Markdown

This project provides a tool to convert podcast audio files into a structured markdown format. It uses speech recognition to transcribe the audio and then formats the output into a markdown file for easier reading and analysis.

## Features

- Converts podcast audio files (e.g., MP3, WAV) to text using **Whisper** (OpenAI’s speech-to-text model).
- Organizes the transcription into a markdown format for easy use and review.
- Includes functionality for logging, progress tracking, and error handling.

## Requirements

Before running the script, install the required Python libraries:

1. Clone the repository:
   ```bash
   git clone https://github.com/imanmossavat/podcast_audio2markdown.git
   ```

2. Install the required dependencies:
   ```bash
   pip install torchaudio torch spacy whisper
   ```
   or 
   ```bash
   pip install -r requirements.txt
   ```

## Usage

To convert an audio file (e.g., podcast episode) into a markdown file, run the following command:

```bash
python experiments/script.py path_to_audio_file
```

Replace `path_to_audio_file` with the path to your audio file.

For more information, visit [Deep Dive Podcast](https://deepdivepod.eu).


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

