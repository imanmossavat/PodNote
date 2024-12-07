# src/managers/audio_manager.py
import os
import sys

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
print(root_dir)
sys.path.append(root_dir)



import torchaudio
import os
import torch

class AudioManager:
    def __init__(self, config):
        self.config = config
        self.audio, self.sample_rate = None, None

    def load_and_preprocess_audio(self, file_path, target_sample_rate=16000):
        """Loads and preprocesses the audio file."""
        waveform, sample_rate = torchaudio.load(file_path)
        
        if sample_rate != target_sample_rate:
            waveform = torchaudio.transforms.Resample(sample_rate, target_sample_rate)(waveform)
        
        if waveform.size(0) > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        self.audio, self.sample_rate = waveform, target_sample_rate

    def segment_audio(self, audio, timestamps):
        """Segments the audio according to the timestamps."""
        segments = []
        for start_time, end_time, _ in timestamps:
            start_sample = int(start_time * self.sample_rate)
            end_sample = int(end_time * self.sample_rate)
            audio_segment = audio[:, start_sample:end_sample]
            segments.append(audio_segment)
        return segments

    def save_segment(self, segment, index, output_folder):
        # Save audio segment to file
        file_path = os.path.join(output_folder, f"segment_{index}.wav")
        torchaudio.save(file_path, segment, self.config.audio_sample_rate)
