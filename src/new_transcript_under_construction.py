import torchaudio, torch
import re
import os
import markdown
import time
import spacy
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from collections import Counter

class AudioSegmentationManager:
    def __init__(self, audio_file, transcription_file=None, chunk_size=500, report_interval=10, model_name="facebook/wav2vec2-large-960h"):
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
        self.model, self.processor = self.load_model(model_name)

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
        # Highlight filler words in gray
        for word in filler_words:
            text = text.replace(f" {word} ", f" <span style='color:gray'>{word}</span> ")
        return text

    def split_text_into_chunks(self, text):
        # Split the text into sentences using spaCy for sentence segmentation
        doc = self.nlp(text)  # Apply POS tagging with spaCy
        sentences = list(doc.sents)  # Break text into sentences

        chunks = []
        current_chunk = ""

        for sentence in sentences:
            # Check if adding the sentence exceeds the chunk_size
            if len(current_chunk.split()) + len(sentence.text.split()) > self.chunk_size:
                # If chunk exceeds size, save current chunk and reset it
                chunks.append(current_chunk.strip())
                current_chunk = sentence.text
            else:
                # Else, keep adding the sentence to the current chunk
                current_chunk += " " + sentence.text

        # Add any remaining text as the last chunk
        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def save_markdown(self, chunks, audio_file_name):
        # Save the text into a markdown file with timestamp and the right filename
        folder = os.path.dirname(self.audio_file)
        timestamp = int(time.time())  # Current timestamp for unique filename
        md_filename = os.path.join(folder, f"{os.path.splitext(audio_file_name)[0]}_{timestamp}.md")
        
        with open(md_filename, "w") as md_file:
            for idx, chunk in enumerate(chunks):
                md_file.write(f"### Chunk {idx + 1}:\n")
                md_file.write(f"{chunk}\n\n")
        print(f"Markdown file saved: {md_filename}")

    def load_model(self, model_name):
        # Load the Wav2Vec2 model and processor
        processor = Wav2Vec2Processor.from_pretrained(model_name)
        model = Wav2Vec2ForCTC.from_pretrained(model_name)
        return model, processor

    def transcribe_audio(self):
        # Transcribe the audio using the model
        input_values = self.processor(self.audio.squeeze().numpy(), return_tensors="pt", sampling_rate=self.sample_rate).input_values
        with torch.no_grad():
            logits = self.model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.decode(predicted_ids[0])
        return transcription

    def save_transcription(self, transcription, filename):
        # Save the transcription to a file
        with open(filename, 'w') as f:
            f.write(transcription)

    def extract_keywords(self, transcript, top_m=10):
        # Process the transcript with spacy NLP
        doc = self.nlp(transcript)
        
        # Extract nouns and proper nouns as potential keywords
        keywords = [token.text for token in doc if token.pos_ in ['NOUN', 'PROPN']]
        
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


def main():
    audio_file = input("Enter the audio file path (no r\"\" needed): ")
    # transcription_file = input("Enter the transcription file path: ")
    transcription_filename = os.path.splitext(audio_file)[0] + "_transcription.txt"
    model_name = "facebook/wav2vec2-large-960h"  # Model to use
    chunk_size = int(input("Enter the chunk size (number of tokens per chunk, default 500): ") or 500)

    segmentation_manager = AudioSegmentationManager(audio_file=audio_file,
                                                    transcription_file= [],
                                                    chunk_size=chunk_size,
                                                    model_name=model_name)

    # Measure transcription time
    start_time = time.time()
    final_transcription = segmentation_manager.transcribe_audio()
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
    chunks = segmentation_manager.split_text_into_chunks(formatted_text)

    # Save the markdown file
    segmentation_manager.save_markdown(chunks, os.path.basename(audio_file))

if __name__ == "__main__":
    main()
