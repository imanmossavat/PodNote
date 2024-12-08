from .audio_manager import AudioManager
from .reporting_manager import ReportingManager
from .transcription_manager import TranscriptionManager
import os
import spacy

class ProcessingService:
    def __init__(self, config):


        config.state_manager.subscribe(self)
        
        self.target_sample_rate = config.general['target_sample_rate']
        self.logger= config.general['logger']
        self.device= config.general['device']
        self.audio_file = config.general['audio_file']

        self.transcription_model_name= config.model_config['transcription_model_name']

        self.prompt =  config.get_full_prompt()
        self.spacy_model= spacy.load(config.nlp['spacy_model'])



        self.user_highlight_keywords= config.nlp['user_highlight_keywords']
        self.filler_words_removed=config.nlp['filler_words_removed']

        self.audio_manager = AudioManager(logger= self.logger, target_sample_rate= self.target_sample_rate)
        
        

        self.transcription_manager = TranscriptionManager(logger= self.logger,
                                                          transcription_model_name= self.transcription_model_name,
                                                          device= self.device,
                                                          prompt=  self.prompt
                                                          )

        self.reporting_manager = ReportingManager(logger= self.logger,
                                                  spacy_model= self.spacy_model,
                                                  user_highlight_keywords= self.user_highlight_keywords,
                                                  filler_words_removed= self.filler_words_removed)

    def update_config(self, config):
        self.target_sample_rate = config.general['target_sample_rate']
        self.logger= config.general['logger']
        self.device= config.general['device']
        self.transcription_model_name= config.model_config['transcription_model_name']
        self.spacy_model= spacy.load(config.nlp['spacy_model'])
        self.user_highlight_keywords= config.nlp['user_highlight_keywords']
        self.filler_words_removed=config.nlp['filler_words_removed']
        self.prompt =  config.get_full_prompt()
        self.audio_file = config.general['audio_file']


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
            
            audio_file_name = os.path.splitext(os.path.basename(self.audio_file))[0]

            self.reporting_manager.report(
            transcription= self.transcription_manager.transcription,
            word_timestamps= self.transcription_manager.word_timestamps,
            audio_file_name= audio_file_name
             )

