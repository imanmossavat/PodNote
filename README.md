# PodNote

**PodNote** is a tool for transcribing podcast audio files, summarizing text with NLP, and generating HTML reports. It detects key quotes, filler words (but doesn’t remove them yet), and creates a merged summary with quotes and instructions. This tool is lightweight, runs well on laptops without a GPU, and offers optional GPU/LLM integration for advanced users.

## Use Case

- **Transcribes audio** and **summarizes text** with **NLP**.
- **Detects key quotes** from the transcription.
- **Creates a simple HTML report** to review both text and audio together.
- **Detects filler words** (but doesn’t remove them—yet).
- **Generates a merged summary** with quotes and instructions, which can be easily copied into GenAI tools for a coherent summary. This has been particularly useful for generating efficient summaries.
  
**Under construction:**
- **Performs semantic tiling**: For better text segmentation.
- **Local LLM Support**: For more advanced processing

The tool uses basic AI models that run effectively on a laptop without a GPU. My vision is to keep it accessible for everyone, while also adding optional GPU/LLM integration for those who need more power (Under Construction). In the future, users will be able to easily create operational prompts for external GenAI tools with just a few clicks.

## Features

- Converts podcast audio files (e.g., MP3, WAV) to text using **Whisper** 
- Organizes the transcription into a structured **HTML format** for easy use and review.
- Includes functionality for logging, progress tracking, and error handling.
- Simple and easy-to-use command-line interface.

## Requirements

Before running the script, install the required Python libraries:

1. Clone the repository:
   ```bash
   git clone https://github.com/imanmossavat/PodNote.git
   ```

2. Install the required dependencies:
   ```bash
   pip install torchaudio torch spacy whisper webbrowser sumy nltk transformers
   ```
   or 
   ```bash
   pip install -r requirements.txt
   ```

## Usage

To convert an audio file (e.g., podcast episode) into an HTML file, run the following command:

```bash
python experiments/script.py path_to_audio_file
```

Replace `path_to_audio_file` with the path to your audio file.

## Collaboration

We are actively looking for collaboration! If you’re interested in contributing to **PodNote**, whether it’s improving functionality, adding new features, or providing feedback, feel free to get in touch.

- **Open Issues**: You can open issues directly on this repository if you have suggestions, feature requests, or encounter any bugs.
- **Pull Requests**: If you want to contribute code, please fork the repository and submit a pull request.
For more information, visit [Deep Dive Podcast](https://deepdivepod.eu).

Looking forward to collaborating with you!

## License

This project is licensed under the **Creative Commons Attribution-NonCommercial 4.0 International License**. See the [LICENSE](LICENSE) file for details.
