import librosa
import numpy as np
import torch
import soundfile as sf

# Load and process the audio (as you've done)
wave, sr = librosa.load("audiofiles/audio_0000.wav", sr=24000)
wave = wave / np.max(np.abs(wave))
wave = torch.from_numpy(wave).float()

# Convert the PyTorch tensor back to a numpy array
wave_numpy = wave.numpy()

# Save the processed audio to recov.wav
sf.write("recov.wav", wave_numpy, sr)

print("Processed audio saved as recov.wav")
