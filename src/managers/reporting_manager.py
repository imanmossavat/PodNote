# src/managers/audio_segmentation_manager.py
import os
from datetime import timedelta
import time
import re
from collections import Counter
#import spacy


class ReportingManager:
    def __init__(self, config):
        self.config = config
        self.nlp = self.config.nlp 
        self.logger = self.config.general['logger']  # Use the logger from config

    def report(self, transcription, word_timestamps, audio_file_name=None):
        # Log the report generation start
        self.logger.info("Generating report for audio file: %s", audio_file_name)

        # Extract keywords
        updated_transcription, keywords = self.extract_keywords(transcription)
        self.logger.info(f"Top keywords: {keywords}")

        # Highlight keywords
        highlighted_transcription = self.highlight_keywords(updated_transcription, keywords)

        # Generate Table of Contents
        toc = self.generate_table_of_contents(highlighted_transcription)
        self.logger.info(f"Table of Contents generated.")

        # Format and chunk the text
        formatted_text = self.format_filler_text(highlighted_transcription)

        # Now, split the text into chunks using word_timestamps
        chunks = self.split_text_into_chunks(word_timestamps)

        # Save the markdown file
        timestamp= self.config.general['timestamp']
        self.save_markdown(chunks, audio_file_name, timestamp= timestamp)

    def extract_keywords(self, transcript, top_m=10):
        doc = self.nlp(transcript)
        stop_words = set(["hello", "hi", "um", "uh", "like", "okay", "well", 'today', "thing", "things", "kind"])
        keywords = [token.text for token in doc if token.pos_ in ['NOUN', 'PROPN'] and token.text.lower() not in stop_words]
        keyword_counts = Counter(keywords)
        top_keywords = keyword_counts.most_common(top_m)
        formatted_keywords = [f"**{kw[0]}**: {kw[1]}" for kw in top_keywords]
        updated_transcript = "\n".join(formatted_keywords) + "\n\n" + transcript
        return updated_transcript, [kw[0] for kw in top_keywords]

    def highlight_keywords(self, transcript):
        user_keywords = self.config.nlp.user_highlight_keywords
        for keyword in user_keywords:
            transcript = re.sub(f"({keyword})", r"**\1**", transcript, flags=re.IGNORECASE)
        return transcript

    def generate_table_of_contents(self, transcript):
        lines = transcript.split("\n")
        toc = [line.strip() for line in lines if line.startswith("#")]
        toc_md = "\n".join([f"- [{heading}](#{heading.replace(' ', '-').lower()})" for heading in toc])
        return toc_md

    def seconds_to_hms(self, seconds):
        return str(timedelta(seconds=seconds))[:8]

    def format_filler_text(self, text):
        filler_words = self.config.nlp.filler_words_removed
        for word in filler_words:
            text = text.replace(f" {word} ", f" <span style='color: red;'>{word}</span> ")
        return text

    def save_markdown(self, chunks, audio_file_name, timestamp=None):
        # Define the report directory from the config
        report_dir = self.config.directories['report_dir']
        os.makedirs(report_dir, exist_ok=True)  # Ensure the directory exists

        if timestamp is None:
            timestamp = int(time.time())
        md_filename = os.path.join(report_dir, f"{os.path.splitext(audio_file_name)[0]}_{timestamp}.md")

        try:
            with open(md_filename, "w") as md_file:
                md_file.write("# Table of Contents\n")
                for idx in range(len(chunks)):
                    md_file.write(f"- [Chunk {idx + 1}](#chunk-{idx + 1})\n")
                md_file.write("\n")

                for idx, (chunk, start_time, end_time) in enumerate(chunks):
                    md_file.write(f"### Chunk {idx + 1} (Start: {self.seconds_to_hms(start_time)}, End: {self.seconds_to_hms(end_time)}):\n")
                    md_file.write(f"{chunk}\n\n")
            self.logger.info(f"Markdown file saved: {md_filename}")
        except Exception as e:
            self.logger.error(f"Failed to save markdown file: {e}")

    def split_text_into_chunks(self, word_timestamps):
        """
        Splits the transcription text into chunks based on word timestamps and sentence boundaries.
        Chunks are split at sentence boundaries to avoid cutting sentences in half, while also
        ensuring the chunk size does not exceed a specified limit.

        Parameters:
        ----------
        word_timestamps : list of dicts
            A list of word timestamps, where each entry contains the 'start' and 'end' times 
            and the word text.

        Returns:
        -------
        list of tuple
            A list of chunks, where each chunk is a tuple containing:
            - The chunked transcription text.
            - The start time of the chunk.
            - The end time of the chunk.
        """
        # Create a spacy document to perform sentence boundary detection
        doc = self.nlp(" ".join([segment['text'] for segment in word_timestamps]))
        chunks = []
        current_chunk = ""
        current_start_time = word_timestamps[0]['start']
        current_end_time = None
        sentence_start_time = word_timestamps[0]['start']

        for i, segment in enumerate(word_timestamps):
            segment_text = segment['text']
            current_end_time = segment['end']

            # Add the segment text to the current chunk
            current_chunk += " " + segment_text

            # Check if the current word is the end of a sentence (using spaCy's sentence boundary detection)
            if doc[sum([len([s['text'] for s in word_timestamps[:i]]) for i in range(i)])].is_sent_end:
                # Add the chunk and reset
                chunks.append((current_chunk.strip(), current_start_time, current_end_time))
                current_chunk = ""
                current_start_time = segment['start']

        # Add the last chunk if it exists
        if current_chunk:
            chunks.append((current_chunk.strip(), current_start_time, current_end_time))

        return chunks
