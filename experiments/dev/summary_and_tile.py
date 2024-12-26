#%%
import re
import numpy as np
import nltk
from sklearn.metrics.pairwise import cosine_similarity
from transformers import T5ForConditionalGeneration, T5Tokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from textsplit.tools import SimpleSentenceTokenizer
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from scipy.special import softmax
from sentence_transformers import SentenceTransformer
from gensim.models import KeyedVectors
import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from textsplit.tools import get_penalty, get_segments
from textsplit.algorithm import split_optimal
from collections import defaultdict
from nltk.tokenize import sent_tokenize
# Segment length for semantic tiling
segment_len = 30

# T5 model for summarization
model_name = 't5-small'
sentence_tokenizer = SimpleSentenceTokenizer()

# Custom stop words list
custom_stop_words = {'don\'t', 'doesn\'t', "isn\'t", "wasn\'t", "weren\'t", 'is', 'are', 'the', 'a', 'an', 
                     'that', 'this', 'to', 'of', 'and', 'for', 'with', 'on', 'in', 'at', 'it', 'by', 'be', 
                     'uh', 'yeah', 'know', 'think', 'make', 'take', 'go', 'come', 'see', 'use', 'get', 'find', 
                     'give', 'basically'}

# Step 1: Read and Chunk Text
def read_text(file_path):
    with open(file_path, 'r') as file:
        text = file.read()
    return text

def chunk_text(text, chunk_size=400):
    words = text.split()
    chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

# Step 2: Preprocess the text (lowercase, remove stop words)

# Function to preprocess and clean the text
def preprocess_text(text):
    # First, perform sentence tokenization to preserve sentence structure
    sentences = nltk.sent_tokenize(text)
    # print(f"Sentences before cleaning: {sentences}")  # Diagnostic print

    # Define filler words and repetitive phrases to remove
    filler_words = ["uh", "um", "yeah", "you know"]

    # Apply preprocessing steps to each sentence
    cleaned_sentences = []
    for sentence in sentences:
        cleaned_sentence = remove_filler_words(sentence, filler_words)
        cleaned_sentence = remove_repetitive_phrases(cleaned_sentence)
        cleaned_sentences.append(cleaned_sentence)

    # Rebuild the text from cleaned sentences
    cleaned_text = ' '.join(cleaned_sentences)
    
    # Normalize whitespace by splitting and joining words
    cleaned_text = ' '.join(cleaned_text.split())
    return cleaned_text

# Function to remove filler words manually (without regex)
def remove_filler_words(text, filler_words):
    # Normalize the text by removing punctuation before checking for filler words
    words = text.split()
    filtered_words = []
    for word in words:
        # Remove punctuation from the word for comparison
        clean_word = re.sub(r'[^\w\s]', '', word).lower()  # Remove non-alphanumeric characters
        # print(f"Checking word: '{word}' (cleaned: '{clean_word}')")  # Diagnostic print
        
        # Check if the word (after cleaning) is a filler word
        if clean_word not in filler_words:
            filtered_words.append(word)
        else:
            # print(f"Removing filler word: '{word}'")  # Diagnostic print
            pass
    
    return ' '.join(filtered_words)

# Function to remove repetitive phrases (without regex)
def remove_repetitive_phrases(text):
    words = text.split()  # Split sentence into words
    # print(f"Words in sentence before removing repetition: {words}")  # Diagnostic print
    
    unique_words = []
    prev_word = None
    for word in words:
        # Remove punctuation from the word for comparison
        clean_word = re.sub(r'[^\w\s]', '', word)  # Remove non-alphanumeric characters
        # print(f"Checking word: '{word}' (cleaned: '{clean_word}') with previous word: '{prev_word}'")  # Diagnostic print
        
        # Skip consecutive repetitions (e.g., "i i i" or "I, I, I")
        if prev_word and clean_word.lower() == prev_word.lower():
            print(f"Skipping word: '{word}' as it is a repetition")  # Diagnostic print
            continue
        
        unique_words.append(word)
        prev_word = clean_word.lower()  # Store cleaned version for comparison
    
    # Reconstruct sentence without repetitions
    cleaned_sentence = ' '.join(unique_words)
    # print(f"Cleaned sentence: '{cleaned_sentence}'")  # Diagnostic print
    return cleaned_sentence



# Step 4: Abstractive Summarization using T5
def summarize_with_t5(chunks):
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    
    t5_summary_chunks = []
    
    # Summarize each chunk individually
    for idx, chunk in enumerate(chunks):
        input_text = "summarize: " + chunk
        inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
        summary_ids = model.generate(inputs["input_ids"], max_length=150, num_beams=4, early_stopping=True)
        t5_summary_chunks.append(tokenizer.decode(summary_ids[0], skip_special_tokens=True))
        print(f"chunk {idx} processed")
    
    return t5_summary_chunks  # List of summarized chunks
# Step 5: Semantic Tiling of the Summary
def semantic_tiling(text, wrdvecs, segment_len=30):
    
    # Tokenize sentences from the text
    # sentence_tokenizer = SimpleSentenceTokenizer()
    # sentenced_text = sentence_tokenizer(text)

    # vecr = CountVectorizer(vocabulary=wrdvecs.index)

    # sentence_vectors = vecr.transform(sentenced_text).dot(wrdvecs)


    # Load the SentenceTransformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Replace with your preferred model
    # Tokenize the text into sentences
    sentenced_text = sent_tokenize(text)  # Tokenizes text into a list of sentences
    # Generate embeddings for each sentence
    sentence_vectors = model.encode(sentenced_text)  # Resulting shape: (num_sentences, embedding_dim)

    sentence_vectors = np.array(sentence_vectors)


    # Apply segmentation to the sentence vectors
    penalty = get_penalty([sentence_vectors], segment_len)
    print(f'Penalty: {penalty:.2f}')
    optimal_segmentation = split_optimal(sentence_vectors, penalty, seg_limit=250)
    segmented_text = get_segments(sentenced_text, optimal_segmentation)
    
    return segmented_text

def map_tiles_to_text(segmented_text, t5_summary):
    # Initialize a mapping dictionary where each chunk index maps to a set of tile indices
    chunk2tile_assignments = defaultdict(set)

    # Tokenize the summaries into sentences
    tokenized_summaries = [sent_tokenize(summary.lower()) for summary in t5_summary]

    # Iterate over each chunk summary
    for chunk_idx, chunk_sentences in enumerate(tokenized_summaries):
        # Iterate over each tile in the segmented text
        for tile_idx, tile_sentences in enumerate(segmented_text):
            # Check if any sentence in the chunk matches sentences in the tile
            for tile_sentence in tile_sentences:
                tile_sentence_cleaned = tile_sentence.lower().strip()
                if any(tile_sentence_cleaned in chunk_sentence for chunk_sentence in chunk_sentences):
                    # If there's a match, add the tile index to the chunk's assignment
                    chunk2tile_assignments[chunk_idx].add(tile_idx)

    # Convert defaultdict to a regular dictionary for ease of use
    return {chunk_idx: list(tile_indices) for chunk_idx, tile_indices in chunk2tile_assignments.items()}

# Step 7: Compute Sentence Probabilities and Plot
def compute_sentence_probabilities(similarity_scores):
    # Apply softmax to compute topic probabilities
    topic_similarities = np.array(similarity_scores)
    sentence_probabilities = softmax(topic_similarities, axis=1)
    return sentence_probabilities

def plot_sentence_probabilities(sentence_probabilities):
    topics = range(sentence_probabilities.shape[1])
    for topic in topics:
        plt.plot(sentence_probabilities[:, topic], label=f"Topic {topic}")
    plt.xlabel('Sentence Index')
    plt.ylabel('Sentence Probability')
    plt.title('Sentence Probabilities for Each Topic')
    plt.legend()
    plt.show()
def generate_html_report(chunk2tile_assignments, processed_chunks, t5_summary):
    # Start HTML structure
    html_content = """
    <html>
    <head>
        <title>Summary Report</title>
        <style>
            body { font-family: Arial, sans-serif; }
            .tile { margin-bottom: 20px; border: 1px solid #ccc; padding: 10px; border-radius: 5px; }
            .tile-header { font-size: 1.2em; font-weight: bold; }
            .tile-summary { margin-top: 10px; font-style: italic; }
            .tile-text { margin-top: 5px; white-space: pre-wrap; }
            .chunk-list { list-style-type: none; padding-left: 0; }
            .chunk-list li { margin: 5px 0; }
        </style>
    </head>
    <body>
        <h1>Summary Report</h1>
    """

    # Iterate through each chunk assignment to create the report
    for chunk_idx, tile_indices in chunk2tile_assignments.items():
        # Merged text for the chunk
        merged_text = " ".join([processed_chunks[i] for i in range(len(processed_chunks)) if chunk_idx in chunk2tile_assignments and i == chunk_idx])

        # Merged summary using T5
        merged_summary = t5_summary[chunk_idx] if chunk_idx < len(t5_summary) else ""

        # Create HTML for each tile
        for tile_idx in tile_indices:
            html_content += f"""
            <div class="tile">
                <div class="tile-header">Tile: {tile_idx + 1} Chunks: {', '.join(map(str, tile_indices))}</div>
                <div class="tile-summary"><strong>Summary:</strong> {merged_summary}</div>
                <div class="tile-text"><strong>Text:</strong> {merged_text}</div>
            </div>
            """
    
    # Closing the HTML tags
    html_content += """
    </body>
    </html>
    """
    return html_content


# Main Workflow
def main(file_path, chunk_size=400, summary_ratio=0.1):
    # Step 1: Chunk and preprocess the text
    text= read_text(file_path)
    processed_text = preprocess_text(text)
    processed_chunks = chunk_text(processed_text)
        
    # Step 2: Further Summarize using T5
    t5_summary = summarize_with_t5(processed_chunks)
    
    wrdvec_path = r'C:\Users\imanm\OneDrive\Documents\podcast\new_transcipt_code\MD_transcriptor\experiments\wrdvecs.bin'
    loaded_model = KeyedVectors.load_word2vec_format(wrdvec_path, binary=True)
    wrdvecs = pd.DataFrame(loaded_model.vectors, index=loaded_model.key_to_index)
    
    segmented_text = semantic_tiling(" ".join(t5_summary), wrdvecs, segment_len=5)  # Perform semantic tiling on the summary

    # Map the tiles to the chunks
    chunk2tile_assignments = map_tiles_to_text(segmented_text, t5_summary)
        
    return text, chunk2tile_assignments, processed_chunks, t5_summary

#%%
# Example usage
file_path = r"C:\Users\imanm\OneDrive\Documents\podcast\new_transcipt_code\MD_transcriptor\data\default_job\20241221-162234\reports\processed_Robert_Iman Podcast-20241203_090154-Meeting Recording_transcription_1734794901.txt"

text, chunk2tile_assignments, processed_chunks, t5_summary = main(file_path)
# Example of generating the HTML report
html_report = generate_html_report(chunk2tile_assignments, processed_chunks, t5_summary)

html_file_path= "summary_report.html"
# Saving the HTML report to a file
with open(html_file_path, "w") as file:
    file.write(html_report)

import os
print(f"The HTML report is saved at: {os.path.abspath(html_file_path)}")

# %%
