from .audio_manager import AudioManager
from .reporting_manager import ReportingManager
from .transcription_manager import TranscriptionManager
import os

class ProcessingService:
    def __init__(self, config):
        self.config= config
        
        target_sample_rate = config.general['target_sample_rate']
        logger= config.general['logger']
        device= config.general['device']
        transcription_model_name= config.model_config['transcription_model_name']


        spacy_model= config.nlp['spacy_model']
        user_highlight_keywords= config.nlp['user_highlight_keywords']
        filler_words_removed=config.nlp['filler_words_removed']

        self.audio_manager = AudioManager(logger= logger, target_sample_rate= target_sample_rate)
        
        

        self.transcription_manager = TranscriptionManager(logger= logger,
                                                          transcription_model_name= transcription_model_name,
                                                          device= device
                                                          )

        self.reporting_manager = ReportingManager(logger= logger,
                                                  spacy_model= spacy_model,
                                                  user_highlight_keywords= user_highlight_keywords,
                                                  filler_words_removed=filler_words_removed)

    def process_audio(self, audio_file):
        self.audio_manager.load_and_preprocess_audio(audio_file)

    
    def transcribe(self):

        if self.audio_manager.audio is None:
            self.config.general['logger'].info(f'audio is None cannot transcribe, first load audio')
        else: 
            self.transcription_manager.transcribe_audio(self.audio_manager.audio)
    
    def report(self):

        if self.transcription_manager.transcription is None:
            self.config.general['logger'].info(f'no transcription available to report, first transcribe')
        elif  self.transcription_manager.word_timestamps is None:
            self.config.general['logger'].info(f'no timestampes available to report, first transcribe')
        else:
            
            audio_file_path= self.config.general['audio_file']
            audio_file_name = os.path.splitext(os.path.basename(audio_file_path))[0]

            self.reporting_manager.report(
            transcription= self.transcription_manager.transcription,
            word_timestamps= self.transcription_manager.word_timestamps,
            audio_file_name= audio_file_name
             )

