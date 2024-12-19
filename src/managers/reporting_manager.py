import os
import time
from datetime import timedelta
import re
from collections import Counter
import urllib.parse  # This is for encoding the file paths properly

import logging


class ReportingManager:
    def __init__(self, logger, spacy_model, user_highlight_keywords,
                 filler_words_removed, chunk_size, report_dir=None, audio_file_name=None, open_report_after_save=False, report_format=None):
        self.logger = logger
        self.report_dir = report_dir
        self._audio_file_name = audio_file_name
        self.open_report_after_save = open_report_after_save
        self.report_format = report_format
        self.nlp_service = NLPService(spacy_model, user_highlight_keywords, filler_words_removed)
        self.audio_handler = AudioFileHandler(audio_file_name)
        self.chunk_formatter = ChunkFormatter(spacy_model, chunk_size)

        if report_format == 'html':
            self.report_saver = HTMLSaver(report_dir, audio_file_name, open_report_after_save)
        elif 'markdown':
            self.report_saver = MarkdownSaver(report_dir, audio_file_name, open_report_after_save)
        else:
            raise ValueError(f"Unidentified report format: {report_format}")



    @property
    def audio_file_name(self):
        return self._audio_file_name

    @audio_file_name.setter
    def audio_file_name(self, new_audio_file_name):
        self._audio_file_name = new_audio_file_name
        self.audio_handler.audio_file_name = new_audio_file_name
        self.report_saver.audio_file_name = new_audio_file_name
        self.logger.info(f"Audio file name updated to: {new_audio_file_name}")
  
    def report(self, transcription, word_timestamps):
        audio_file_name = self.audio_file_name
        self.logger.info("Generating report for audio file: %s", audio_file_name)

        # Process transcription and extract keywords
        updated_transcription, keywords = self.nlp_service.extract_keywords(transcription)
        self.logger.info(f"Top keywords: {keywords}")

        # Highlight keywords in the transcription
        highlighted_transcription = self.nlp_service.highlight_keywords(updated_transcription)

        # Format filler words
        formatted_text = self.nlp_service.format_filler_text(highlighted_transcription)

        # Split text into chunks while respecting sentence boundaries and token limits
        chunks = self.chunk_formatter.split_text_into_chunks(word_timestamps)
        formatted_chunks= self.chunk_formatter.format_chunks_as_html(chunks)

        # Generate Table of Contents
        toc_body = [
            f"<a href='#chunk_{idx + 1}'>Chunk {idx + 1} ({seconds_to_hms(start_time)}-{seconds_to_hms(end_time)})</a>"
            for idx, (_, start_time, end_time) in enumerate(chunks)
        ]

        # Define sections for the report
        sections = [
            {
                'type': 'text',
                'id': 'summary',
                'header': 'Summary',
                'body': 'This is an auto-generated summary of the audio file.'
            },
            {
                'type': 'audio',
                'id': 'audio_player',
                'header': 'Audio Player',
                'body': AudioFileHandler(self.audio_file_name).generate_audio_player_html()
            },
            {
                'type': 'toc',
                'id': 'toc',
                'header': 'Table of Contents',
                'body': toc_body  # Table of Contents with clickable links
            },
            {
                'type': 'chunks',
                'id': 'chunks',
                'header': 'Audio Chunks',
                'body': formatted_chunks  
            }
        ]

        timestamp = int(time.time())

        # Save the report in the chosen format (HTML or Markdown)
        if self.report_format == 'html':
            self.report_saver.create_and_save_html(timestamp=timestamp, sections=sections)
        else:
            self.report_saver.save_markdown(chunks, timestamp=timestamp)



class NLPService:
    def __init__(self, spacy_model, user_highlight_keywords, filler_words_removed):
        self.spacy_model = spacy_model
        self.user_highlight_keywords = user_highlight_keywords
        self.filler_words_removed = filler_words_removed

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
        for keyword in self.user_highlight_keywords:
            transcript = re.sub(f"({keyword})", r"**\1**", transcript, flags=re.IGNORECASE)
        return transcript

    def format_filler_text(self, text):
        for word in self.filler_words_removed:
            text = text.replace(f" {word} ", f" <span style='color: red;'>{word}</span> ")
        return text
    
    def generate_table_of_contents(self, transcript):
        lines = transcript.split("\n")
        toc = [line.strip() for line in lines if line.startswith("#")]
        toc_md = "\n".join([f"- [{heading}](#{heading.replace(' ', '-').lower()})" for heading in toc])
        return toc_md

class AudioFileHandler:
    def __init__(self, audio_file_name):
        self.audio_file_name = audio_file_name

    def generate_audio_player_html(self):
        audio_url = self.audio_file_name
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

class ChunkFormatter:
    def __init__(self, spacy_model, chunk_size):
        self.spacy_model = spacy_model
        self.chunk_size = chunk_size  # max token size for each chunk

    def split_text_into_chunks(self, word_timestamps):
        # Tokenize the entire transcription text
        doc = self.spacy_model(" ".join([segment['text'] for segment in word_timestamps]))
        chunks = []
        current_chunk = []
        current_start_time = word_timestamps[0]['start']
        current_end_time = None
        current_token_count = 0

        # Iterate over each word in the word_timestamps
        for i, segment in enumerate(word_timestamps):
            segment_text = segment['text']
            segment_start_time = segment['start']
            segment_end_time = segment['end']
            doc_segment = self.spacy_model(segment_text)  # Tokenize the segment text

            # Check if adding this segment exceeds the chunk size
            if current_token_count + len(doc_segment) > self.chunk_size:
                # Finalize the current chunk using the last segment's start time
                chunks.append((current_chunk, current_start_time, segment_start_time))
                current_chunk = [segment_text]  # Start a new chunk with this segment
                current_start_time = segment_start_time  # Update the start time
                current_token_count = len(doc_segment)  # Reset token count for the new chunk
            else:
                # Add the segment to the current chunk
                current_chunk.append(segment_text)
                current_token_count += len(doc_segment)
            
            # Update the current end time with the last segment's end time
            current_end_time = segment_end_time

        # Append the last chunk if there is any remaining text
        if current_chunk:
            chunks.append((current_chunk, current_start_time, current_end_time))

        return chunks

    
    def format_chunks_as_html(self, chunks):
        """Formats the chunks into an HTML string."""
        formatted_chunks = []
        for idx, (chunk, start_time, end_time) in enumerate(chunks):
            start_hms = seconds_to_hms(start_time)
            end_hms = seconds_to_hms(end_time)
            chunk_html = f"""
                <h3 id='chunk_{idx + 1}'>Chunk {idx + 1}</h3>
                <p><strong>Start:</strong> {start_hms}, <strong>End:</strong> {end_hms}</p>
                <p>{' '.join(chunk)}</p>
                <hr>
            """
            formatted_chunks.append(chunk_html)
        return formatted_chunks


    

class MarkdownSaver:
    def __init__(self, report_dir, audio_file_name, open_report_after_save=False):
        self.report_dir = report_dir
        self.audio_file_name = audio_file_name
        self.open_report_after_save = open_report_after_save

    def save_markdown(self, chunks, timestamp=None):
        report_dir = self.report_dir
        os.makedirs(report_dir, exist_ok=True)
        if timestamp is None:
            timestamp = int(time.time())

        md_filename = os.path.join(report_dir, f"{os.path.splitext(os.path.basename(self.audio_file_name))[0]}_{timestamp}.md")

        try:
            # Get absolute file path and replace backslashes with forward slashes
            abs_file_path = os.path.abspath(self.audio_file_name)
            abs_file_path = abs_file_path.replace("\\", "/")  # Convert backslashes to forward slashes

            # Encode the path
            encoded_audio_file_path = urllib.parse.quote(abs_file_path)

            # Write markdown content
            with open(md_filename, "w") as md_file:
                # Use the encoded path for the audio file link
                md_file.write(f"[Audio File](file://{encoded_audio_file_path})\n\n")
                md_file.write("---\n\n")
                md_file.write("# Table of Contents\n")

                # Write ToC with chunk start time links
                for idx, (chunk, start_time, end_time) in enumerate(chunks):
                    start_hms = seconds_to_hms(start_time)
                    end_hms = seconds_to_hms(end_time)
                    chunk_label = f"Chunk {idx + 1} ({start_hms}-{end_hms})"
                    chunk_link = f"[[#Chunk {idx + 1}|{chunk_label}]]"
                    md_file.write(f"- {chunk_link}\n")
                md_file.write("\n---\n\n")

                md_file.write("# Keywords\n")
                md_file.write("List any extracted or user-defined keywords here.\n\n")
                md_file.write("---\n\n")

                # Add audio player to markdown
                audio_player_html = AudioFileHandler(self.audio_file_name).generate_audio_player_html()
                md_file.write(f"{audio_player_html}\n\n")
                md_file.write("---\n\n")

                # Add the chunk details
                for idx, (chunk, start_time, end_time) in enumerate(chunks):
                    start_hms = seconds_to_hms(start_time)
                    end_hms = seconds_to_hms(end_time)
                    md_file.write(f"### Chunk {idx + 1}\n")
                    md_file.write(f"**Start:** {start_hms}, **End:** {end_hms}\n\n")
                    md_file.write(f"{' '.join(chunk)}\n\n")
                    md_file.write("---\n\n")

            if self.open_report_after_save:
                os.system(f'open "{md_filename}"')
            print(f"\nMarkdown file saved: {md_filename}\n")

        except Exception as e:
            print(f"Failed to save markdown file: {e}")

class HTMLSaver:
    def __init__(self, report_dir, audio_file_name, open_report_after_save=False, logger=None):
        self.report_dir = report_dir
        self.audio_file_name = audio_file_name
        self.open_report_after_save = open_report_after_save
        self.logger = logger or logging.getLogger(__name__)

    def create_and_save_html(self, timestamp=None, sections=None):
        """Main method to save the HTML report."""
        if sections is None:
            sections = []
        if timestamp is None:
            timestamp = int(time.time())

        html_filename = self._generate_html_filename(timestamp)
        self._create_directory(self.report_dir)

        try:
            html_content = self._generate_html_content(sections)
            self._write_to_file(html_filename, html_content)

            if self.open_report_after_save:
                self._open_html_file(html_filename)

        except Exception as e:
            self.logger.error(f"Failed to save HTML file: {e}")
            raise

        self.logger.info(f"HTML report saved to: {html_filename}")

    def _generate_html_filename(self, timestamp):
        """Generate the HTML file name based on the audio file and timestamp."""
        base_name = os.path.splitext(os.path.basename(self.audio_file_name))[0]
        return os.path.join(self.report_dir, f"{base_name}_{timestamp}.html")

    @staticmethod
    def _create_directory(directory):
        """Ensure the directory exists."""
        os.makedirs(directory, exist_ok=True)

    def _generate_html_content(self, sections):
        """Generate the full HTML content as a string."""
        self.logger.info(f"Creating HTML content for sections.")
        header = self._generate_html_header()
        body = self._generate_html_body(sections)
        return f"{header}{body}</html>"

    def _generate_html_header(self):
        """Generate the HTML header including JavaScript for copy functionality."""
        return """
        <html>
        <head>
            <title>Audio Report</title>
            <script>
                function copyToClipboard(elementId) {
                    const text = document.getElementById(elementId).innerText;
                    navigator.clipboard.writeText(text).then(() => {
                        alert('Copied to clipboard!');
                    }).catch(err => {
                        alert('Failed to copy: ' + err);
                    });
                }
            </script>
        </head>
        <body>
        """

    def _generate_html_body(self, sections):
        """Generate the HTML body by processing sections."""
        body = [self._generate_audio_file_section()]
        for section in sections:
            body.append(self._generate_section(section))
        return "\n".join(body)

    def _generate_audio_file_section(self):
        """Generate the audio file download section."""
        abs_file_path = os.path.abspath(self.audio_file_name).replace("\\", "/")
        encoded_audio_file_path = urllib.parse.quote(abs_file_path)
        return f"<h1>Audio File: <a href='file://{encoded_audio_file_path}'>Download</a></h1>"

    def _generate_section(self, section):
        """Generate an individual section based on its type."""
        section_id = section.get('id', '')
        section_header = section.get('header', '')
        section_type = section.get('type', '')
        section_body = section.get('body', '')

        self.logger.debug(f"Processing section: {section_id or section_header}")
        section_html = f"<section id='{section_id}'>"
        if section_header:
            section_html += f"<h2>{section_header}</h2>"

        if section_type == 'toc':
            section_html += self._generate_toc(section_body)
        elif section_type == 'text':
            section_html += self._generate_text(section_id, section_body)
        elif section_type == 'audio':
            section_html += section_body
        elif section_type == 'chunks':
            section_html += self._generate_chunks(section_body)
        else:
            section_html += f"<p>{section_body}</p>"

        section_html += "</section>"
        return section_html

    def _generate_toc(self, items):
        """Generate a Table of Contents section."""
        self.logger.debug(f"Rendering Table of Contents with {len(items)} items.")
        return "<ul>" + "".join(f"<li>{item}</li>" for item in items) + "</ul>"

    def _generate_text(self, section_id, text):
        """Generate a text section with a copy button."""
        element_id = f"text_{section_id}"
        return f"""
        <p id='{element_id}'>{text}</p>
        <button onclick="copyToClipboard('{element_id}')">Copy</button>
        """

    def _generate_chunks(self, chunks):
        """Generate HTML for chunks with copy buttons."""
        self.logger.debug(f"Rendering {len(chunks)} chunks.")
        html_chunks = []
        for idx, chunk in enumerate(chunks):
            chunk_id = f"chunk_{idx + 1}"
            html_chunks.append(f"""
            <div id='{chunk_id}'>{chunk}</div>
            <button onclick="copyToClipboard('{chunk_id}')">Copy</button>
            """)
        return "\n".join(html_chunks)

    @staticmethod
    def _write_to_file(filename, content):
        """Write the HTML content to a file."""
        with open(filename, "w") as file:
            file.write(content)

    @staticmethod
    def _open_html_file(filename):
        """Open the HTML file using the system's default browser."""
        os.system(f'open "{filename}"')

def seconds_to_hms(seconds):
    return str(timedelta(seconds=seconds))[:8]
