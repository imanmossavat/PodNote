from .audio_manager import AudioManager
from .reporting_manager import ReportingManager
from .transcription_manager import TranscriptionManager
import os
import spacy

class ProcessingService:
    def __init__(self, config):

        
        self.target_sample_rate = config.general['target_sample_rate']
        self.logger= config.general['logger']
        self.device= config.general['device']
        self.audio_file_name = config.general['audio_file']

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
                                                  audio_file_name= self.audio_file_name)

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

