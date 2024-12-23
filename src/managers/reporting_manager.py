"""
This script processes an audio transcription and generates reports in either HTML or Markdown format. The report includes a summary, table of contents, audio player, and chunked content of the transcription with timestamps.

Classes:
1. **ReportingManager**: Manages the entire reporting process. It initializes necessary services and handles report creation, saving, and format handling.
2. **NLPService**: Responsible for processing the transcription. It extracts critical sentences, highlights them, and generates summaries. It also creates a table of contents for the transcription.
3. **AudioFileHandler**: Handles the generation of HTML audio player components and manages audio file paths.
4. **ChunkFormatter**: Splits the transcription into manageable chunks, ensuring no chunk exceeds the specified token size. It also formats these chunks into HTML for report generation.
5. **MarkdownSaver**: Saves the generated report in Markdown format. It handles file saving, link generation, and adds the audio player in the markdown.
6. **HTMLSaver**: Saves the generated report in HTML format. It manages file saving, generates the HTML content, and includes an audio player and sections such as summary, chunks, and table of contents.

Functions:
- **seconds_to_hms**: Converts seconds to a 'HH:MM:SS' format for displaying timestamps.

Architecture:
- **ReportingManager** orchestrates the entire process, using instances of `NLPService`, `AudioFileHandler`, `ChunkFormatter`, and either `HTMLSaver` or `MarkdownSaver` based on the user’s report format preference.
- The `NLPService` handles natural language processing tasks such as summarization and highlighting critical sentences in both the transcription and word timestamps.
- The `ChunkFormatter` splits the transcription into chunks to ensure they are within a manageable size for the report.
- The `AudioFileHandler` generates HTML audio player components for embedding in the report.
- Finally, the report is saved either in HTML or Markdown format depending on the selected report format.

This design enables flexibility in generating customized reports from audio transcriptions, with the ability to specify report format and chunk size.
"""

import os
import time
from datetime import timedelta
import urllib.parse  # This is for encoding the file paths properly
import logging
#
import re
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.nlp.tokenizers import Tokenizer
import nltk
nltk.download('punkt')
from transformers import T5Tokenizer, T5ForConditionalGeneration



class ReportingManager:
    def __init__(self, logger, spacy_model, user_highlight_keywords,
                 filler_words_removed, chunk_size, summary_ratio, report_dir=None, audio_file_name=None, open_report_after_save=False, report_format=None):
        self.logger = logger
        self.report_dir = report_dir
        self._audio_file_name = audio_file_name
        self.open_report_after_save = open_report_after_save
        self.report_format = report_format
        self.summary_ratio= summary_ratio
        self.nlp_service = NLPService(spacy_model, user_highlight_keywords, filler_words_removed, self.summary_ratio)
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

        # Process transcription using NLPService
        nlp_results = self.nlp_service.process_transcription(transcription, word_timestamps)

        # Extract results from NLPService processing
        critical_sentences = nlp_results['summary']
        highlighted_word_timestamps = nlp_results['highlighted_word_timestamps']

        # Split text into chunks while respecting sentence boundaries and token limits
        chunks = self.chunk_formatter.split_text_into_chunks(highlighted_word_timestamps)

        # Generate summaries for each chunk using T5-small
        chunk_summaries = self.nlp_service.summarize_chunks(chunks)

        # Aggregate chunk summaries into a section summary
        section_summary = " ".join(chunk_summaries)

        # Format chunks with highlighted sentences
        formatted_chunks = self.chunk_formatter.format_chunks_as_html(chunks, critical_sentences)

        # Generate Table of Contents for chunks
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
                'body': section_summary  # Aggregated summary of all chunks
            },
            {
                'type': 'audio',
                'id': 'audio_player',
                'header': 'Audio Player',
                'body': AudioFileHandler(audio_file_name).generate_audio_player_html()
            },
            {
                'type': 'toc',
                'id': 'toc',
                'header': 'Table of Contents',
                'body': toc_body  # Table of Contents with clickable links for chunks
            },
            {
                'type': 'chunks',
                'id': 'chunks',
                'header': 'Audio Chunks',
                'body': formatted_chunks  # Chunks formatted as HTML, with bolded critical sentences
            },
            {
                'type': 'text',
                'id': 'extractive_summary',
                'header': 'Extractive Summary',
                'body': "\n".join(critical_sentences)  # The critical sentences summary
            }
        ]

        timestamp = int(time.time())

        # Save the report in the chosen format (HTML or Markdown)
        if self.report_format == 'html':
            self.report_saver.create_and_save_html(timestamp=timestamp, sections=sections)
        else:
            self.report_saver.save_markdown(chunks, timestamp=timestamp)


 
class NLPService:
    def __init__(self, spacy_model, user_highlight_keywords, filler_words_removed, summary_ratio=None):
        self.spacy_model = spacy_model
        self.user_highlight_keywords = user_highlight_keywords
        self.filler_words_removed = filler_words_removed
        self.summary_ratio = summary_ratio  # Default ratio for summarization

    def _highlight_critical_sentences(self, transcription, critical_sentences):
        """Apply bold formatting to the critical sentences in the transcription."""
        for sentence in critical_sentences:
            transcription = transcription.replace(sentence, f"**{sentence}**")
        return transcription

    def _apply_highlighted_to_word_timestamps(self, word_timestamps, critical_sentences):
        """Apply bold formatting to critical sentences within word_timestamps."""
        for sentence in critical_sentences:
            words_in_sentence = sentence.split()  # Splitting sentence into words
            sentence_str = " ".join(words_in_sentence)  # Recreating sentence
            for word in word_timestamps:
                if word['text'] in sentence_str:
                    word['text'] = f"**{word['text']}**"
        return word_timestamps

    def _extract_summary(self, text):
        """Extract a summary using TextRank."""
        try:
            sentences = nltk.sent_tokenize(text)
            sentence_count = len(sentences)
            tokenizer = Tokenizer("english")
            parser = PlaintextParser.from_string(text, tokenizer)
            summarizer = TextRankSummarizer()
            summary_count = max(1, int(sentence_count * self.summary_ratio))
            summary = summarizer(parser.document, sentences_count=summary_count)
            summary = [str(sentence) for sentence in summary]
        except ValueError:
            summary = text.split()[:2]
        return summary

    def generate_table_of_contents(self, transcript):
        """Generate a Markdown Table of Contents from transcript headings."""
        lines = transcript.split("\n")
        toc = [line.strip() for line in lines if line.startswith("#")]
        toc_md = "\n".join([f"- [{heading}](#{heading.replace(' ', '-').lower()})" for heading in toc])
        return toc_md

    def process_transcription(self, transcription, word_timestamps):
        """Process the transcription to generate summaries, highlight sentences, and format chunks."""
        # Clean transcription for summarization
        cleaned_transcription = re.sub(r'\s+', ' ', transcription).strip()

        # Generate extractive summary from the cleaned transcription
        critical_sentences = self._extract_summary(cleaned_transcription)

        # Highlight critical sentences in the original transcription (not cleaned)
        highlighted_transcription = self._highlight_critical_sentences(transcription, critical_sentences)

        # Apply highlights to word timestamps
        highlighted_word_timestamps = self._apply_highlighted_to_word_timestamps(word_timestamps, critical_sentences)

        # Generate Table of Contents (from original transcription)
        toc = self.generate_table_of_contents(transcription)

        return {
            "summary": critical_sentences,
            "highlighted_transcription": highlighted_transcription,
            "highlighted_word_timestamps": highlighted_word_timestamps,
            "table_of_contents": toc
        }
    
    def summarize_chunks(self, chunks):
        """
        Summarize each chunk using T5-small and return a list of summaries.

        Args:
            chunks (list): List of transcription chunks, each with text and timestamps.

        Returns:
            list: List of summaries, one for each chunk.
        """

        # Load the T5-small model and tokenizer
        tokenizer = T5Tokenizer.from_pretrained("t5-small")
        model = T5ForConditionalGeneration.from_pretrained("t5-small")

        summaries = []
        for chunk_text, start_time, end_time in chunks:
            input_text = "summarize: " + " ".join(chunk_text)
            input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
            output_ids = model.generate(input_ids, max_length=150, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
            summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            summaries.append(summary)

        return summaries



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

    
    def format_chunks_as_html(self, chunks, critical_sentences):
        """Formats the chunks into an HTML string, with critical sentences highlighted."""
        formatted_chunks = []
        for idx, (chunk, start_time, end_time) in enumerate(chunks):
            start_hms = seconds_to_hms(start_time)
            end_hms = seconds_to_hms(end_time)
            
            # Highlight critical sentences within the chunk
            highlighted_chunk = " ".join(chunk)
            for sentence in critical_sentences:
                highlighted_chunk = highlighted_chunk.replace(sentence, f"<b>{sentence}</b>")
            
            # Format the chunk as HTML
            chunk_html = f"""
                <h3 id='chunk_{idx + 1}'>Chunk {idx + 1}</h3>
                <p><strong>Start:</strong> {start_hms}, <strong>End:</strong> {end_hms}</p>
                <p>{highlighted_chunk}</p>
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
        print(f"HTML report saved to: {html_filename}")

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
