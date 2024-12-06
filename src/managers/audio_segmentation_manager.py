# src/managers/audio_segmentation_manager.py
import os
import sys

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
print(root_dir)
sys.path.append(root_dir)
from config import Config  # Import Config class



import whisper
import torchaudio
import torch
import spacy
import time
import re
from collections import Counter
from datetime import timedelta

class AudioSegmentationManager:
    def __init__(self, audio_file, transcription_file=None, config=None):
        # Use the provided config or fall back to default config
        self.config = config or Config()  # Default to Config if no config provided
        
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
        self.chunk_size = self.config.get_model_config()['chunk_size']
        self.report_interval = self.config.get_reporting_config()['report_interval']  # in minutes
        self.nlp = spacy.load("en_core_web_sm")  # Load spaCy model for sentence segmentation
        self.model = self.load_model(self.config.get_model_config()['model_name'])

    def load_and_preprocess_audio(self, file_path, target_sample_rate=16000):
        waveform, sample_rate = torchaudio.load(file_path)
        if sample_rate != target_sample_rate:
            waveform = torchaudio.transforms.Resample(sample_rate, target_sample_rate)(waveform)
        if waveform.size(0) > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        return waveform, target_sample_rate

    def load_transcription(self, file_path):
        transcription_data = []
        with open(file_path, 'r') as f:
            content = f.read()
            segments = content.split("\n\n")
            for segment in segments:
                lines = segment.split("\n")
                if len(lines) >= 3:
                    time_range = lines[1]
                    text = lines[2]
                    start_time, end_time = self.parse_srt_time(time_range)
                    transcription_data.append((start_time, end_time, text))
        return transcription_data

    def parse_srt_time(self, time_range):
        times = re.findall(r'\d{2}:\d{2}:\d{2},\d{3}', time_range)
        start_time = self.convert_to_seconds(times[0])
        end_time = self.convert_to_seconds(times[1])
        return start_time, end_time

    def convert_to_seconds(self, srt_time):
        hours, minutes, seconds, milliseconds = map(float, re.split('[:,.]', srt_time))
        return hours * 3600 + minutes * 60 + seconds + milliseconds/1000.0

    def segment_audio(self):
        segments = []
        for start_time, end_time, _ in self.transcriptions:
            start_sample = int(start_time * self.sample_rate)
            end_sample = int(end_time * self.sample_rate)
            audio_segment = self.audio[:, start_sample:end_sample]
            segments.append(audio_segment)
        return segments

    def save_segment(self, segment, index):
        torchaudio.save(f"segment_{index}.wav", segment, self.sample_rate)

    def report_time(self, current_time):
        if current_time % (self.report_interval * 60) == 0:
            print(f"Time elapsed: {current_time / 60} minutes.")

    def format_text(self, text, filler_words=["um", "uh", "like"]):
        for word in filler_words:
            text = text.replace(f" {word} ", f" *{word}* ")
        return text

    def split_text_into_chunks(self, word_timestamps):
        chunks = []
        current_chunk = ""
        current_start_time = word_timestamps[0]['start']
        current_end_time = None
        for i, segment in enumerate(word_timestamps):
            segment_text = segment['text']
            current_end_time = segment['end']
            if len(current_chunk.split()) + len(segment_text.split()) > self.chunk_size:
                chunks.append((current_chunk.strip(), current_start_time, current_end_time))
                current_chunk = segment_text
                current_start_time = segment['start']
            else:
                current_chunk += " " + segment_text
        if current_chunk:
            chunks.append((current_chunk.strip(), current_start_time, current_end_time))
        return chunks

    def save_markdown(self, chunks, audio_file_name):
        folder = os.path.dirname(self.audio_file)
        timestamp = int(time.time())
        md_filename = os.path.join(folder, f"{os.path.splitext(audio_file_name)[0]}_{timestamp}.md")
        with open(md_filename, "w") as md_file:
            md_file.write("# Table of Contents\n")
            for idx in range(len(chunks)):
                md_file.write(f"- [Chunk {idx + 1}](#chunk-{idx + 1})\n")
            md_file.write("\n")
            for idx, (chunk, start_time, end_time) in enumerate(chunks):
                md_file.write(f"### Chunk {idx + 1} (Start: {self.seconds_to_hms(start_time)}, End: {self.seconds_to_hms(end_time)}):\n")
                md_file.write(f"{chunk}\n\n")
        print(f"Markdown file saved: {md_filename}")

    def load_model(self, model_name):
        model = whisper.load_model(model_name)
        return model

    def transcribe_audio(self, prompt=None):
        if prompt is None:
            prompt = "I'm like, you know what I mean, kind of, um, ah, huh, and so, so um, uh, and um, like um, so like, like it's, it's like, i mean, yeah, ok so, uh so, so uh, yeah so, you know, it's uh, uh and, and uh, like, kind. ASML, TUE, Fontys, Brainport, Eindhoven, Capegemini, YieldStar"
        result = self.model.transcribe(self.audio.squeeze().numpy(), word_timestamps=True, initial_prompt=prompt)
        transcription = result['text']
        word_timestamps = result['segments']
        return transcription, word_timestamps

    def save_transcription(self, transcription, filename):
        with open(filename, 'w') as f:
            f.write(transcription)

    def extract_keywords(self, transcript, top_m=10):
        doc = self.nlp(transcript)
        keywords = Counter([token.text for token in doc if token.is_stop == False])
        return keywords.most_common(top_m)

    def seconds_to_hms(self, seconds):
        return str(timedelta(seconds=seconds))
