import os
from datetime import timedelta
import time
import re
from collections import Counter

class ReportingManager:
    def __init__(self, 
                 logger, 
                 spacy_model, user_highlight_keywords, 
                 filler_words_removed, report_dir= None,
                 audio_file_name=None,
                 open_report_after_save=False):

        self.spacy_model = spacy_model
        self.user_highlight_keywords = user_highlight_keywords
        self.filler_words_removed = filler_words_removed
        self.logger = logger  # Use the logger from config
        self.report_dir = report_dir
        self.audio_file_name = audio_file_name  # file name of the audio, to be used for playback

        print(f"audio file name is {audio_file_name}")
        self.open_report_after_save = open_report_after_save  # show the report after generation if True

    def report(self, transcription, word_timestamps):
        audio_file_name= self.audio_file_name
        # Log the report generation start
        self.logger.info("Generating report for audio file: %s", audio_file_name)

        # Extract keywords
        updated_transcription, keywords = self.extract_keywords(transcription)
        self.logger.info(f"Top keywords: {keywords}")

        # Highlight keywords
        highlighted_transcription = self.highlight_keywords(updated_transcription)

        # Generate Table of Contents
        toc = self.generate_table_of_contents(highlighted_transcription)
        self.logger.info(f"Table of Contents generated.")

        # Format and chunk the text
        formatted_text = self.format_filler_text(highlighted_transcription)

        # Now, split the text into chunks using word_timestamps
        chunks = self.split_text_into_chunks(word_timestamps)

        # Save the markdown file
        timestamp = int(time.time())  # every report has its own timestamp!
        self.save_markdown(chunks, timestamp=timestamp)

    def extract_keywords(self, transcript, top_m=10):
        doc = self.spacy_model(transcript)
        stop_words = set(["hello", "hi", "um", "uh", "like", "okay", "well", 'today', "thing", "things", "kind"])
        keywords = [token.text for token in doc if token.pos_ in ['NOUN', 'PROPN'] and token.text.lower() not in stop_words]
        keyword_counts = Counter(keywords)
        top_keywords = keyword_counts.most_common(top_m)
        formatted_keywords = [f"**{kw[0]}**: {kw[1]}" for kw in top_keywords]
        updated_transcript = "\n".join(formatted_keywords) + "\n\n" + transcript
        return updated_transcript, [kw[0] for kw in top_keywords]

    def highlight_keywords(self, transcript):
        user_keywords = self.user_highlight_keywords
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
        filler_words = self.filler_words_removed
        for word in filler_words:
            text = text.replace(f" {word} ", f" <span style='color: red;'>{word}</span> ")
        return text


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
        doc = self.spacy_model(" ".join([segment['text'] for segment in word_timestamps]))
        chunks = []
        current_chunk = ""
        current_start_time = word_timestamps[0]['start']
        current_end_time = None

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

    def generate_audio_player_html(self):
        """
        Generates the HTML for an audio player that can be used for embedding in markdown.
        The player allows playback control and seeks to the specified timestamps.
        """
        # Ensure the audio file is accessible via a URL (modify as necessary depending on your setup)
        audio_file_name= self.audio_file_name
        audio_url= audio_file_name

        #audio_url = f"/path/to/audio/files/{audio_file_name}"

        return f"""
        <audio id="audio_player" controls>
            <source src="{audio_url}" type="audio/mpeg">
            Your browser does not support the audio element.
        </audio>
        <script>
            function playAudioAtTime(time) {{
                var audio = document.getElementById('audio_player');
                audio.currentTime = time;
                audio.play();
            }}
        </script>
        """
    def save_markdown(self, chunks, timestamp=None):
        """
        Saves the segmented audio transcription into a Markdown file, formatted for Obsidian.
        """
        # Define the report directory from the config
        report_dir = self.report_dir
        os.makedirs(report_dir, exist_ok=True)  # Ensure the directory exists
        audio_file_name= self.audio_file_name

        if timestamp is None:
            timestamp = int(time.time())
        
        md_filename = os.path.join(report_dir, f"{os.path.splitext(os.path.basename(audio_file_name))[0]}_{timestamp}.md")


        try:
            with open(md_filename, "w") as md_file:
                # Add a link to the audio file
                md_file.write(f"[Audio File](file://{os.path.abspath(audio_file_name)})\n\n")
                
                # Add a horizontal line for structure
                md_file.write("---\n\n")

                # Table of Contents
                md_file.write("# Table of Contents\n")
                for idx, (chunk, start_time, end_time) in enumerate(chunks):
                    start_hms = self.seconds_to_hms(start_time)
                    end_hms = self.seconds_to_hms(end_time)
                    chunk_label = f"Chunk {idx + 1} ({start_hms}-{end_hms})"
                    chunk_link = f"[[#Chunk {idx + 1}|{chunk_label}]]"
                    md_file.write(f"- {chunk_link}\n")
                md_file.write("\n---\n\n")

                # Keywords Section (placeholder if needed)
                md_file.write("# Keywords\n")
                md_file.write("List any extracted or user-defined keywords here.\n\n")
                md_file.write("---\n\n")

                # Embed the audio player
                audio_player_html = self.generate_audio_player_html()
                md_file.write(f"{audio_player_html}\n\n")
                md_file.write("---\n\n")

                # Write Chunks
                for idx, (chunk, start_time, end_time) in enumerate(chunks):
                    start_hms = self.seconds_to_hms(start_time)
                    end_hms = self.seconds_to_hms(end_time)
                    md_file.write(f"### Chunk {idx + 1}\n")
                    md_file.write(f"**Start:** {start_hms}, **End:** {end_hms}\n\n")
                    md_file.write(f"{chunk}\n\n")
                    md_file.write("---\n\n")

            self.logger.info(f"Markdown file saved: {md_filename}")

            if self.open_report_after_save:
                # Open the markdown file in the default application
                os.system(f'open "{md_filename}"')

        except Exception as e:
            self.logger.error(f"Failed to save markdown file: {e}")
