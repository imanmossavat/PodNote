import os, sys
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
print(root_dir)
sys.path.append(root_dir)

import unittest
from src.new_transcript_under_construction import AudioSegmentationManager  # Replace with actual import path




class TestAudioSegmentationManager(unittest.TestCase):

    def test_load_audio(self):
        audio_file = os.path.join(root_dir, "data","test", "promo_.wav")
        manager = AudioSegmentationManager(audio_file, transcription_file=None)
        self.assertEqual(manager.audio.shape[0], 1)  # Should be mono after conversion
        self.assertEqual(manager.sample_rate, 16000)  # Expected sample rate after resampling

    def test_load_transcription(self):

        transcription_file = os.path.join(root_dir, "data", "test", "test_transcription.srt")

        manager = AudioSegmentationManager(audio_file=None, transcription_file=transcription_file)
        transcription_data = manager.load_transcription(transcription_file)
        self.assertGreater(len(transcription_data), 0)  # Ensure transcription data is loaded

    # Add other test cases here...

if __name__ == "__main__":
    unittest.main()
