import librosa
import numpy as np

file_path = "your_audio_file.wav"
try:
    y, sr = librosa.load(file_path, sr=22050)
    print("Audio Loaded Successfully!")
except Exception as e:
    print(f"Error loading audio: {e}")
