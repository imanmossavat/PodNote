Here’s a suggested **README.md** for your project:

---

# Podcast Audio to Markdown Transcription

A Python tool for converting podcast audio files into well-formatted Markdown documents with time-stamped segments, keywords, and table of contents.

## Features
- Transcribe audio using OpenAI’s **Whisper** model.
- Segment audio and text into time-stamped chunks.
- Highlight keywords and remove filler words for cleaner transcripts.
- Automatically generate a **Table of Contents** for Markdown output.
- Extract and format **keywords** from transcriptions.
- Save output as a Markdown file with linked timestamps and chunked content.

## Installation

### Prerequisites
1. Python 3.8 or higher.
2. Install required libraries:
   ```bash
   pip install whisper torchaudio spacy
   ```
3. Download the spaCy English model:
   ```bash
   python -m spacy download en_core_web_sm
   ```

### Clone the Repository
```bash
git clone https://github.com/imanmossavat/podcast_audio2markdown.git
cd podcast_audio2markdown
```

## Usage

### 1. Running the Script
Run the script using (src folder):
```bash
python main_whisper.py
```

### 2. Input Parameters
- **Audio File**: Specify the path to the audio file for transcription.
- **Chunk Size**: Choose the maximum number of tokens per chunk (default: 500).

### 3. Output
- Transcription saved as a `.txt` file.
- Markdown document saved with time-stamped segments and a table of contents.

## How It Works
1. **Audio Preprocessing**: 
   - Resamples the audio to 16 kHz.
   - Converts stereo audio to mono.

2. **Transcription**:
   - Transcribes audio into text with word-level timestamps using OpenAI's Whisper model.
   
3. **Chunking**:
   - Splits text into chunks of specified size with corresponding time intervals.

4. **Keyword Extraction**:
   - Identifies top keywords using spaCy NLP.

5. **Markdown Generation**:
   - Creates a Markdown file with sections linked to timestamps.

## Example Output
Markdown files include:
- **Table of Contents**:
  - Linked headings for each chunk.
- **Time-Stamped Chunks**:
  - Segments with start and end times.

## Development

### Key Classes
- `AudioSegmentationManager`: Handles audio loading, transcription, segmentation, and markdown creation.
- **Main Methods**:
  - `transcribe_audio`: Performs audio transcription.
  - `split_text_into_chunks`: Segments transcriptions into smaller parts.
  - `save_markdown`: Generates and saves a Markdown document.

### Contributions
Feel free to open issues or submit pull requests to improve the project.

---

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

---

Let me know if you want additional sections, like examples or advanced options!