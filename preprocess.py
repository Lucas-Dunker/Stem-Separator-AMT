import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import os

def generate_cqt_spectrogram(wav_file, output_dir, track_id, spectrogram_id, sr=22050, hop_length=512, bins_per_octave=12, n_bins=84):
    """
    Generate a CQT spectrogram for a given WAV file and save it as a NumPy array.
    
    Parameters:
        wav_file (str): Path to the WAV file.
        output_dir (str): Directory to save the spectrogram.
        track_id (str): Identifier for the track (e.g., "Track00001").
        spectrogram_id (str): Identifier for the spectrogram (e.g., "S01").
        sr (int): Sampling rate.
        hop_length (int): Number of samples between successive frames.
        bins_per_octave (int): Number of bins per octave.
        n_bins (int): Total number of frequency bins.
    """
    # Load audio file
    y, sr = librosa.load(wav_file, sr=sr)

    # Generate CQT spectrogram
    cqt = librosa.cqt(y, sr=sr, hop_length=hop_length, bins_per_octave=bins_per_octave, n_bins=n_bins)
    cqt_db = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)

    # Create a unique file name
    spectrogram_file = os.path.join(output_dir, f"{track_id}_{spectrogram_id}.npy")
    np.save(spectrogram_file, cqt_db)
    print(f"Spectrogram saved to {spectrogram_file}")
    
    # Optional: Visualize and save the spectrogram image
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(cqt_db, sr=sr, hop_length=hop_length, bins_per_octave=bins_per_octave, x_axis='time', y_axis='cqt_note')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Constant-Q Power Spectrum')
    plt.tight_layout()
    plt.savefig(spectrogram_file.replace('.npy', '.png'))
    plt.close()

# Example usage in the loop
output_dir = "./spectrograms"
os.makedirs(output_dir, exist_ok=True)

for i in range(1, 21):
    track_num = str(i).zfill(2)  # Ensure two digits (e.g., "01", "02")
    track_id = f"Track000{track_num}"
    base_dir = f"Data/babyslakh_16k/babyslakh_16k/{track_id}/MIDI/"
    
    for j in range(16):
        wav_num = str(j).zfill(2)  # Ensure two digits (e.g., "00", "01")
        wav_file = os.path.join(base_dir, f"S{wav_num}.wav")

        if not os.path.isfile(wav_file):
            continue

        spectrogram_id = f"S{wav_num}"
        generate_cqt_spectrogram(wav_file, output_dir, track_id, spectrogram_id)




# wav_file = "Data/babyslakh_16k/babyslakh_16k/Track00001/MIDI/S00.wav"  # Replace with the path to your WAV file

# output_dir = "./spectrograms"
# os.makedirs(output_dir, exist_ok=True)
# generate_cqt_spectrogram(wav_file, output_dir)


# file = 'Data/babyslakh_16k/babyslakh_16k/Track00001/MIDI/S07.wav'
# print('HERE...')
# print(os.path.isfile(file))