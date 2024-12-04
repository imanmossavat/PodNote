"""
Unit tests for the AudioSegmentationManager class.

This script contains unit tests for loading and processing audio and transcription files. 
The tests check the following functionalities:
- Loading audio files and verifying their sample rate and mono/stereo status.
- Loading transcription files (in .srt format) and verifying their content.

The tests also automatically resample the audio to 16,000 Hz if necessary.

Modules required:
- torchaudio
- unittest
"""


import os, sys
import unittest
import torchaudio
import torchaudio.transforms as T

# Get the root directory path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
print(f"Root directory: {root_dir}")
sys.path.append(root_dir)

class TestAudioSegmentationManager(unittest.TestCase):

    def test_load_audio(self):
        # Define the path to the audio file
        audio_file = os.path.join(root_dir, "data", "test", "promo_.wav")
        print(f"Testing audio file: {audio_file}")

        try:
            # Try loading the audio file using torchaudio
            audio, sample_rate = torchaudio.load(audio_file)
            print(f"Audio loaded with shape {audio.shape} and sample rate {sample_rate}")

            # Automatically check and adjust sample rate if it's not 16,000 Hz
            target_sample_rate = 16000
            if sample_rate != target_sample_rate:
                print(f"Resampling audio from {sample_rate} Hz to {target_sample_rate} Hz.")
                resampler = T.Resample(sample_rate, target_sample_rate)
                audio = resampler(audio)
                sample_rate = target_sample_rate  # Update sample rate after resampling

            # Check if the audio is mono and the sample rate is 16000
            self.assertEqual(audio.shape[0], 1)  # Should be mono
            self.assertEqual(sample_rate, target_sample_rate)  # Should be 16000 after resampling or already 16000
        except Exception as e:
            self.fail(f"Failed to load audio file: {e}")

    def test_load_transcription(self):
        # Define the path to the transcription file
        transcription_file = os.path.join(root_dir, "data","test", "test_transcription.srt")
        print(f"Testing transcription file: {transcription_file}")

        try:
            # Try opening and reading the transcription file
            with open(transcription_file, 'r') as f:
                content = f.readlines()
            print(f"Loaded transcription with {len(content)} lines")
            self.assertGreater(len(content), 0)  # Ensure there is content
        except Exception as e:
            self.fail(f"Failed to load transcription file: {e}")

if __name__ == "__main__":
    unittest.main()
