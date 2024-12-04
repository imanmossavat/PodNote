import whisper
import torchaudio
import os
import time
import spacy
from collections import Counter
import re
from datetime import timedelta


class AudioSegmentationManager:
    def __init__(self, audio_file, transcription_file=None, chunk_size=500, report_interval=10, model_name="whisper-large"):
        self.audio_file = audio_file
        
        if self.audio_file:
            self.audio, self.sample_rate = self.load_and_preprocess_audio(audio_file)
        else:
            self.audio, self.sample_rate = None, None

        if transcription_file:
            self.transcriptions = self.load_transcription(transcription_file)
        else:
            self.transcriptions = None
        
        self.transcription_file = transcription_file
        self.chunk_size = chunk_size
        self.report_interval = report_interval  # in minutes
        self.nlp = spacy.load("en_core_web_sm")  # Load spaCy model for sentence segmentation
        self.model = self.load_model(model_name)

    def load_and_preprocess_audio(self, file_path, target_sample_rate=16000):
        # Load audio using torchaudio
        waveform, sample_rate = torchaudio.load(file_path)
        
        # Resample audio if necessary
        if sample_rate != target_sample_rate:
            waveform = torchaudio.transforms.Resample(sample_rate, target_sample_rate)(waveform)
        
        # Convert to mono if it's stereo
        if waveform.size(0) > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        return waveform, target_sample_rate

    def load_transcription(self, file_path):
        # Read and parse the transcription (assumes SRT format)
        transcription_data = []
        with open(file_path, 'r') as f:
            content = f.read()
            segments = content.split("\n\n")  # Split by empty lines between sections
            for segment in segments:
                lines = segment.split("\n")
                if len(lines) >= 3:
                    time_range = lines[1]
                    text = lines[2]
                    start_time, end_time = self.parse_srt_time(time_range)
                    transcription_data.append((start_time, end_time, text))
        return transcription_data

    def parse_srt_time(self, time_range):
        # Parse timestamps in the format "00:00:05,000 --> 00:00:10,000"
        times = re.findall(r'\d{2}:\d{2}:\d{2},\d{3}', time_range)
        start_time = self.convert_to_seconds(times[0])
        end_time = self.convert_to_seconds(times[1])
        return start_time, end_time

    def convert_to_seconds(self, srt_time):
        # Convert SRT timestamp to seconds
        hours, minutes, seconds, milliseconds = map(float, re.split('[:,.]', srt_time))
        return hours * 3600 + minutes * 60 + seconds + milliseconds/1000.0

    def segment_audio(self):
        segments = []
        for start_time, end_time, _ in self.transcriptions:
            # Convert time to sample indices
            start_sample = int(start_time * self.sample_rate)
            end_sample = int(end_time * self.sample_rate)
            audio_segment = self.audio[:, start_sample:end_sample]
            segments.append(audio_segment)
        return segments

    def save_segment(self, segment, index):
        # Save the audio segment to a file
        torchaudio.save(f"segment_{index}.wav", segment, self.sample_rate)

    def report_time(self, current_time):
        # Report time every X minutes
        if current_time % (self.report_interval * 60) == 0:
            print(f"Time elapsed: {current_time / 60} minutes.")

    def format_text(self, text, filler_words=["um", "uh", "like"]):
        # Highlight filler words in *italics* (Markdown format)
        for word in filler_words:
            text = text.replace(f" {word} ", f" *{word}* ")
        return text

    def split_text_into_chunks(self, word_timestamps):
        # Split the text into chunks and associate timestamps with each chunk
        chunks = []
        current_chunk = ""
        current_start_time = word_timestamps[0]['start']  # Start time of the first segment
        current_end_time = None

        for i, segment in enumerate(word_timestamps):
            segment_text = segment['text']
            current_end_time = segment['end']  # End time of the current segment
            
            # Check if adding the current segment would exceed the chunk size
            if len(current_chunk.split()) + len(segment_text.split()) > self.chunk_size:
                # If chunk exceeds size, save current chunk with times and reset
                chunks.append((current_chunk.strip(), current_start_time, current_end_time))
                current_chunk = segment_text
                current_start_time = segment['start']  # Reset start time for new chunk
            else:
                # Else, keep adding the segment text to the current chunk
                current_chunk += " " + segment_text

        # Add the last chunk if it exists
        if current_chunk:
            chunks.append((current_chunk.strip(), current_start_time, current_end_time))

        return chunks


    def save_markdown(self, chunks, audio_file_name):
        folder = os.path.dirname(self.audio_file)
        timestamp = int(time.time())
        md_filename = os.path.join(folder, f"{os.path.splitext(audio_file_name)[0]}_{timestamp}.md")
        
        with open(md_filename, "w") as md_file:
            # Create Table of Contents at the start of the document
            md_file.write("# Table of Contents\n")
            for idx in range(len(chunks)):
                md_file.write(f"- [Chunk {idx + 1}](#chunk-{idx + 1})\n")
            md_file.write("\n")
            
            for idx, (chunk, start_time, end_time) in enumerate(chunks):
                md_file.write(f"### Chunk {idx + 1} (Start: {self.seconds_to_hms(start_time)}, End: {self.seconds_to_hms(end_time)}):\n")
                md_file.write(f"{chunk}\n\n")
        print(f"Markdown file saved: {md_filename}")

    def load_model(self, model_name):
        # Load the Whisper model
        model = whisper.load_model(model_name)
        return model

    def transcribe_audio(self, prompt= None):

        if prompt is None:
            prompt = "I'm like, you know what I mean, kind of, um, ah, huh, and so, so um, uh, and um, like um, so like, like it's, it's like, i mean, yeah, ok so, uh so, so uh, yeah so, you know, it's uh, uh and, and uh, like, kind. ASML, TUE, Fontys, Brainport, Eindhoven, Capegemini, YieldStar"

        # Transcribe the audio using Whisper and include timestamps
        result = self.model.transcribe(self.audio.squeeze().numpy(), word_timestamps=True, initial_prompt= prompt)
        transcription = result['text']
        word_timestamps = result['segments']  # Each segment contains start and end times
        return transcription, word_timestamps


    def save_transcription(self, transcription, filename):
        # Save the transcription to a file
        with open(filename, 'w') as f:
            f.write(transcription)

    def extract_keywords(self, transcript, top_m=10):
        # Process the transcript with spacy NLP
        doc = self.nlp(transcript)
        
        stop_words = set(["hello", "hi", "um", "uh", "like", "okay", "well", 'today', "thing", "things","kind"])
        
        # Extract nouns and proper nouns as potential keywords
        keywords = [token.text for token in doc if token.pos_ in ['NOUN', 'PROPN'] and token.text.lower() not in stop_words]
        
        
        # Get the top M most common keywords
        keyword_counts = Counter(keywords)
        top_keywords = keyword_counts.most_common(top_m)
        
        # Format and return the keywords
        formatted_keywords = [f"**{kw[0]}**: {kw[1]}" for kw in top_keywords]
        
        # Add the keywords at the top of the transcript
        updated_transcript = "\n".join(formatted_keywords) + "\n\n" + transcript
        
        return updated_transcript, [kw[0] for kw in top_keywords]

    def highlight_keywords(self, transcript, user_keywords):
        # Highlight user-defined keywords in bold
        for keyword in user_keywords:
            transcript = re.sub(f"({keyword})", r"**\1**", transcript, flags=re.IGNORECASE)
        
        return transcript

    def generate_table_of_contents(self, transcript):
        # Extract headings (lines starting with a '#' for markdown-style headings)
        lines = transcript.split("\n")
        toc = []
        for line in lines:
            if line.startswith("#"):
                toc.append(line.strip())
        
        toc_md = "\n".join([f"- [{heading}](#{heading.replace(' ', '-').lower()})" for heading in toc])
        return toc_md
    
    def seconds_to_hms(self, seconds):
            # Convert time from seconds to hh:mm:ss using timedelta
            return str(timedelta(seconds=seconds))[:8]  # This will return the format hh:mm:ss

def main():
    audio_file = input("Enter the audio file path (no r\"\" needed): ")
    transcription_filename = os.path.splitext(audio_file)[0] + "_transcription.txt"
    
    whisper_models = ["tiny.en", "tiny", "base.en", "base", "small.en", "small", "medium.en", "medium", "large-v1", "large-v2", "large-v3", "large", "large-v3-turbo", "turbo"]

    model_name = 'tiny.en'
    chunk_size = int(input("Enter the chunk size (number of tokens per chunk, default 500): ") or 500)

    segmentation_manager = AudioSegmentationManager(audio_file=audio_file,
                                                    chunk_size=chunk_size,
                                                    model_name=model_name)

    # Measure transcription time
    start_time = time.time()
    final_transcription, word_timestamps = segmentation_manager.transcribe_audio()  # Get both text and timestamps
    end_time = time.time()
    print(f"Transcription took {end_time - start_time:.2f} seconds")

    # Save the transcription
    segmentation_manager.save_transcription(final_transcription, transcription_filename)

    print(f"Transcription saved to {transcription_filename}")

    # Extract keywords
    updated_transcription, keywords = segmentation_manager.extract_keywords(final_transcription)
    print(f"Top keywords: {keywords}")

    # Highlight keywords
    highlighted_transcription = segmentation_manager.highlight_keywords(updated_transcription, keywords)

    # Generate Table of Contents
    toc = segmentation_manager.generate_table_of_contents(highlighted_transcription)
    print(f"Table of Contents:\n{toc}")

    # Format and chunk the text
    formatted_text = segmentation_manager.format_text(highlighted_transcription)

    # Now, split the text into chunks using word_timestamps
    chunks = segmentation_manager.split_text_into_chunks(word_timestamps)

    # Save the markdown file
    segmentation_manager.save_markdown(chunks, os.path.basename(audio_file))


if __name__ == "__main__":
    main()
