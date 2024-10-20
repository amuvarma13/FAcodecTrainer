import os
from datasets import DatasetDict
import soundfile as sf
from datasets import load_dataset
import numpy as np


dsn = "amuvarma/audio-lunatrejo-0"


ds = load_dataset(dsn)


def save_audio_to_wav(dataset: DatasetDict, output_folder: str):
    """
    Save non-empty audio elements from a dataset as WAV files in the specified folder.
    
    Args:
    dataset (DatasetDict): The dataset containing audio elements.
    output_folder (str): The path to the folder where WAV files will be saved.
    
    Returns:
    None
    """
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Access the 'train' split of the dataset
    train_dataset = dataset['train']
    
    # Counter for saved files
    saved_count = 0
    
    # Iterate through the dataset
    for i, example in enumerate(train_dataset):
        audio = example['audio']
        
        # Check if 'audio' is a dictionary (which is likely if it's from datasets library)
        if isinstance(audio, dict):
            audio_array = audio['array']
            sampling_rate = audio['sampling_rate']
        else:
            # If it's not a dictionary, assume it's already an array
            # You might need to adjust this part based on your actual data structure
            audio_array = audio
            sampling_rate = 16000  # You might need to change this default value
        
        # Check if the audio_array is not empty and contains actual samples
        if isinstance(audio_array, np.ndarray) and audio_array.size > 0 and not np.all(audio_array == 0):
            # Generate a filename
            filename = f"audio_{i:04d}.wav"
            filepath = os.path.join(output_folder, filename)
            
            # Save the audio as a WAV file
            sf.write(filepath, audio_array, sampling_rate)
            saved_count += 1
        else:
            print(f"Skipping empty or silent audio at index {i}")
    
    print(f"Saved {saved_count} non-empty audio files to {output_folder}")
    print(f"Skipped {len(train_dataset) - saved_count} empty or silent audio files")

save_audio_to_wav(ds, "audiofiles")