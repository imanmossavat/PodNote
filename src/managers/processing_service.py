from .audio_manager import AudioManager
from .reporting_manager import ReportingManager
from .transcription_manager import TranscriptionManager

class ProcessingService:
    def __init__(self, config):
        self.audio_manager = AudioManager(config)
        self.transcription_manager = TranscriptionManager(config)
        self.reporting_manager = ReportingManager(config)

    def process_audio(self, audio_file):
        self.audio_manager.load_and_preprocess_audio(audio_file)

    
    def transcribe(self):

        self.transcription_manager.transcribe_audio(self.audio_manager.audio)
    
    def report(self):
        self.reporting_manager.report(
             self.transcription_manager.transcription,
             self.transcription_manager.word_timestamps,
             )

