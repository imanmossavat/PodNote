from .audio_manager import AudioManager
from .reporting_manager import ReportingManager
from .transcription_manager import TranscriptionManager
import os

class ProcessingService:
    def __init__(self, config):
        self.config= config
        self.audio_manager = AudioManager(config)
        self.transcription_manager = TranscriptionManager(config)
        self.reporting_manager = ReportingManager(config)

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

