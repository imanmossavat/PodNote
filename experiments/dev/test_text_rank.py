import os
import re
import pandas as pd
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.nlp.tokenizers import Tokenizer
import nltk
# Custom stop words list
custom_stop_words = {'don\'t', 'doesn\'t', "isn\'t", "wasn\'t", "weren\'t", 'is', 'are', 'the', 'a', 'an', 
                     'that', 'this', 'to', 'of', 'and', 'for', 'with', 'on', 'in', 'at', 'it', 'by', 'be', 
                     'uh', 'yeah', 'know', 'think', 'make', 'take', 'go', 'come', 'see', 'use', 'get', 'find', 
                     'give', 'basically'}

# Step 1: Read the text file and split it into chunks of 400 words
def read_and_chunk_text(file_path, chunk_size=400):
    with open(file_path, 'r') as file:
        text = file.read()
    
    # Split text into words
    words = text.split()
    
    # Split words into chunks of 'chunk_size' words
    chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

# Step 2: Preprocess the text (lowercase, remove stop words)
def preprocess_text(chunks):
    processed_chunks = []
    for chunk in chunks:
        chunk = chunk.lower()
        chunk = ' '.join([word for word in chunk.split() if word not in custom_stop_words and re.match(r'\w+', word)])
        processed_chunks.append(chunk)
    return processed_chunks

# Step 3: Extractive summarization using TextRank (Sumy)

def extract_summary_with_textrank(text, ratio=0.4):
    try:
        # Tokenize sentences using NLTK
        sentences = nltk.sent_tokenize(text)
        sentence_count = len(sentences)  # Number of sentences
        
        # Create a tokenizer and parse the text correctly
        tokenizer = Tokenizer("english")
        parser = PlaintextParser.from_string(text, tokenizer)
        
        summarizer = TextRankSummarizer()
        
        # Calculate summary length based on sentence count and ratio
        sentence_summary_count = max(1, int(sentence_count * ratio))  # Ensure at least one sentence
        summary = summarizer(parser.document, sentences_count=sentence_summary_count)
        
        summary = [str(sentence) for sentence in summary]
    except ValueError:  # If the text is too short to summarize
        summary = text.split()[:2]  # Take the first two sentences as a fallback
    return summary

# Step 4: Format the text with bolded summarized sentences
def format_chunks_with_bolded_sentences(chunks, summary):
    formatted_chunks = []
    for chunk in chunks:
        formatted_chunk = chunk
        for sentence in summary:
            escaped_sentence = re.escape(sentence)  # Escape special regex characters in the sentence
            formatted_chunk = re.sub(escaped_sentence, f'<b>{sentence}</b>', formatted_chunk)
        formatted_chunks.append(formatted_chunk)
    return formatted_chunks

# Main workflow
def main(file_path, chunk_size=400, summary_ratio=0.1):
    # Read and process the text file
    chunks = read_and_chunk_text(file_path, chunk_size)
    processed_chunks = preprocess_text(chunks)
    
    # Generate summary for the entire text
    entire_text = ' '.join(processed_chunks)  # Combine chunks into the full text
    full_summary = extract_summary_with_textrank(entire_text, ratio=summary_ratio)
    
    # Format the chunks with bolded sentences from the summary
    formatted_chunks = format_chunks_with_bolded_sentences(processed_chunks, full_summary)
    
    prompt = "Make a summary with a focus on topical contents. This is transcribed and extratively summarized, so do not be stuck on logical ordering of sentences and note there may be transcription errors: "

    full_summary_with_prompt = prompt + " ".join(full_summary)
    # Create a DataFrame for the formatted chunks
    df = pd.DataFrame({
        'chunk': formatted_chunks
    })
    
    # Generate HTML output
    html_output = df.to_html(escape=False)
    
    # Include the full summary in red at the end of the HTML
    full_summary_html = f'<p style="color:red; font-size:14px;">{ " ".join(full_summary_with_prompt) }</p>'
    
    # Return the HTML output with the full summary at the end
    html_output += full_summary_html
    return html_output, full_summary

# Example usage
file_path = r"C:\Users\imanm\OneDrive\Documents\podcast\new_transcipt_code\MD_transcriptor\data\default_job\20241221-162234\reports\processed_Robert_Iman Podcast-20241203_090154-Meeting Recording_transcription_1734794901.txt"

 # Path to your text file
html_output, full_summary = main(file_path)

# Save HTML output
input_folder = os.path.dirname(file_path)
output_file_path = os.path.join(input_folder, 'formatted_output.html')

with open(output_file_path, 'w') as f:
    f.write(html_output)

print(f"HTML output saved to {output_file_path}")
print("Full Summary:")
print(full_summary)

# Extract the folder of the input file
input_folder = os.path.dirname(file_path)

# Define the output file path
output_file_path = os.path.join(input_folder, '_sumy_formatted_output.html')

# Save the HTML content to a file
with open(output_file_path, 'w') as f:
    f.write(html_output)

print(f"HTML output saved to {output_file_path}")
