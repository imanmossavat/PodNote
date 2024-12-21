from .audio_manager import AudioManager
from .reporting_manager import ReportingManager
from .transcription_manager import TranscriptionManager
import os
import spacy
import time
from datetime import timedelta
import pickle

class ProcessingService:
    def __init__(self, config):

        
        self.target_sample_rate = config.general['target_sample_rate']
        self.logger= config.general['logger']
        self.device= config.general['device']
        self.audio_file_name = config.general['audio_file']
        self.report_format = config.general['report_format']

        self.report_dir = config.directories['report_dir']

        self.transcription_model_name= config.model_config['transcription_model_name']

        self.prompt =  config.get_full_prompt()
        self.spacy_model= spacy.load(config.nlp['spacy_model'])

        self.chunk_size= config.model_config['chunk_size']

        self.user_highlight_keywords= config.nlp['user_highlight_keywords']
        self.filler_words_removed=config.nlp['filler_words_removed']
        self.open_report_after_save = config.general['open_report_after_save']

        self.audio_manager = AudioManager(logger= self.logger, target_sample_rate= self.target_sample_rate)
        
        

        self.transcription_manager = TranscriptionManager(logger= self.logger,
                                                          transcription_model_name= self.transcription_model_name,
                                                          device= self.device,
                                                          prompt=  self.prompt
                                                          )

        self.reporting_manager = ReportingManager(logger= self.logger,
                                                  spacy_model= self.spacy_model,
                                                  user_highlight_keywords= self.user_highlight_keywords,
                                                  filler_words_removed= self.filler_words_removed,
                                                  chunk_size= self.chunk_size,
                                                  report_dir= self.report_dir,
                                                  audio_file_name= self.audio_file_name,
                                                  report_format= self.report_format)

    def update_config(self, config):
        self.report_dir = config.directories['report_dir']
        self.target_sample_rate = config.general['target_sample_rate']
        self.logger= config.general['logger']
        self.device= config.general['device']
        self.transcription_model_name= config.model_config['transcription_model_name']
        self.spacy_model= spacy.load(config.nlp['spacy_model'])
        self.user_highlight_keywords= config.nlp['user_highlight_keywords']
        self.filler_words_removed=config.nlp['filler_words_removed']
        self.prompt =  config.get_full_prompt()
        self.audio_file_name = config.general['audio_file']
        self.open_report_after_save = config.general['open_report_after_save']


        self.audio_manager.logger= self.logger
        self.audio_manager.target_sample_rate= self.target_sample_rate

        self.transcription_manager.logger= self.logger
        self.transcription_manager.prompt= self.prompt
        self.transcription_manager.transcription_model_name= self.transcription_model_name
        self.transcription_manager.device = self.device
        if self.transcription_manager.model is not None:
            self.load_model()

        self.reporting_manager.logger= self.logger
        self.reporting_manager.spacy_model= self.spacy_model
        self.reporting_manager.user_highlight_keywords= self.user_highlight_keywords
        self.reporting_manager.filler_words_removed= self.filler_words_removed
        self.reporting_manager.report_dir= self.report_dir
        self.reporting_manager.open_report_after_save= self.open_report_after_save
        self.reporting_manager.audio_file_name= self.audio_file_name
                                                

    def process_audio(self, audio_file):
        self.audio_manager.load_and_preprocess_audio(audio_file)

    
    def transcribe(self):

        if self.audio_manager.audio is None:
            self.config.general['logger'].info(f'audio is None cannot transcribe, first load audio')
        else: 
            self.transcription_manager.transcribe_audio(self.audio_manager.audio)
    
    def report(self):

        if self.transcription_manager.transcription is None:
            self.logger.info(f'no transcription available to report, first transcribe')
        elif  self.transcription_manager.word_timestamps is None:
            self.logger.info(f'no timestampes available to report, first transcribe')
        else:
            
            self.reporting_manager.report(
            transcription= self.transcription_manager.transcription,
            word_timestamps= self.transcription_manager.word_timestamps,
             )


    def save_raw_transcription(self, text_filename=None, timestamp=None):
        """Saves the transcription and timestamps to two separate text files."""
        if not hasattr(self.transcription_manager, 'transcription') or not hasattr(self.transcription_manager, 'word_timestamps'):
            self.logger.error("Transcription or word timestamps are missing. Run 'transcribe_audio' first.")

        # Ensure the report directory exists
        report_dir = self.report_dir
        os.makedirs(report_dir, exist_ok=True)

        # Generate a timestamp if not provided
        if timestamp is None:
            timestamp = int(time.time())

        # Create the filenames for both files
        if text_filename is None:
            text_filename_transcription = os.path.join(
                report_dir,
                f"{os.path.splitext(os.path.basename(self.audio_file_name))[0]}_transcription_{timestamp}.txt"
            )
            text_filename_timestamps = os.path.join(
                report_dir,
                f"{os.path.splitext(os.path.basename(self.audio_file_name))[0]}_timestamps_{timestamp}.txt"
            )
        else:
            # If a filename is provided, save to the same filename for both, appending the suffixes
            text_filename_transcription = text_filename
            text_filename_timestamps = os.path.splitext(text_filename)[0] + "_timestamps" + os.path.splitext(text_filename)[1]

        # Save the full transcription (without timestamps) to one file
        with open(text_filename_transcription, 'w', encoding='utf-8') as file:
            # Write the transcription text
            file.write("Transcription:\n")
            file.write(self.transcription_manager.transcription + "\n\n")
        
        print(f"Transcription saved to {text_filename_transcription}")

        # Save the segments with timestamps to another file
        with open(text_filename_timestamps, 'w', encoding='utf-8') as file:
            # Write the timestamps with corresponding text chunks
            file.write("Timestamps:\n")
            for segment in self.transcription_manager.word_timestamps:
                start_hms = str(timedelta(seconds=segment['start']))[:8]
                end_hms = str(timedelta(seconds=segment['end']))[:8]
                file.write(f"Start: {start_hms}, End: {end_hms}, Text: {segment['text']}\n")

        print(f"Timestamps and segments saved to {text_filename_timestamps}")




        
    def save_ui_manager_pickle(self, ui_manager_obj, pkl_filename=None, timestamp=None):
        """
        Pickles the UIManager object and saves it in the report directory with a timestamp.
        
        Arguments:
            ui_manager: The UIManager object to pickle.
            pkl_filename: Optional. The custom filename to use.
            timestamp: Optional. The timestamp to use for the filename. If None, it will be generated.
        """
        # Ensure timestamp is provided, otherwise use the current timestamp
        if timestamp is None:
            timestamp =  int(time.time())
  # Use the timestamp of the processing session
        
        # Use the provided filename or generate one with timestamp
        if pkl_filename is None:
            pkl_filename = f"ui_manager_{timestamp}.pkl"
        
        # Create the full path to save the pickle file in the report directory
        pkl_path = os.path.join(self.report_dir, pkl_filename)
        
        try:
            # Open the file and serialize (pickle) the UIManager
            with open(pkl_path, 'wb') as f:
                pickle.dump(ui_manager_obj, f)
            self.logger.info(f"UIManager pickled and saved to {pkl_path}")
        except Exception as e:
            self.logger.error(f"Error saving UIManager pickle: {e}")
            raise
