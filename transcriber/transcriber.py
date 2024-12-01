import librosa
import numpy as np
import pretty_midi

class MusicTranscriber:
    def __init__(self, sr=22050, hop_length=512, n_bins=84, bins_per_octave=12):
        self.sr = sr
        self.hop_length = hop_length
        self.n_bins = n_bins
        self.bins_per_octave = bins_per_octave
        self.fs = sr / hop_length  # Frame rate for piano roll alignment
        # Calculate sequence_length based on 3 minutes duration
        self.standard_duration = 180  # 3 minutes in seconds
        self.sequence_length = int(np.ceil(self.standard_duration * self.fs))  # Number of frames

    def process_audio(self, audio_path):
        # Load audio file (up to 3 minutes)
        y, _ = librosa.load(audio_path, sr=self.sr, duration=self.standard_duration)
        # Compute CQT
        C = librosa.cqt(
            y,
            sr=self.sr,
            hop_length=self.hop_length,
            n_bins=self.n_bins,
            bins_per_octave=self.bins_per_octave
        )
        # Convert to magnitude and apply log scaling
        C_mag = librosa.amplitude_to_db(np.abs(C), ref=np.max)
        # Pad or truncate to fixed length
        if C_mag.shape[1] < self.sequence_length:
            pad_width = ((0, 0), (0, self.sequence_length - C_mag.shape[1]))
            C_mag = np.pad(C_mag, pad_width, mode='constant')
        else:
            C_mag = C_mag[:, :self.sequence_length]
        return C_mag

    def process_midi(self, midi_path):
        midi_data = pretty_midi.PrettyMIDI(midi_path)
        end_time = self.standard_duration  # Use standard duration
        # Generate time frames matching the spectrogram
        times = np.linspace(0, end_time, self.sequence_length)
        piano_roll = midi_data.get_piano_roll(fs=self.fs, times=times)
        # Restrict to MIDI note range 21–108 (88 keys)
        piano_roll = piano_roll[21:109, :]  # MIDI note numbers 21–108 inclusive
        # Ensure fixed length
        if piano_roll.shape[1] < self.sequence_length:
            pad_width = ((0, 0), (0, self.sequence_length - piano_roll.shape[1]))
            piano_roll = np.pad(piano_roll, pad_width, mode='constant')
        else:
            piano_roll = piano_roll[:, :self.sequence_length]
        # Normalize to binary values
        piano_roll = (piano_roll > 0).astype(np.float32)
        return piano_roll

    def prepare_dataset(self, audio_files, midi_files):
        spectrograms = []
        piano_rolls = []
        for audio_path, midi_path in zip(audio_files, midi_files):
            try:
                # Process audio
                cqt = self.process_audio(audio_path)
                # Process MIDI
                piano_roll = self.process_midi(midi_path)
                spectrograms.append(cqt)
                piano_rolls.append(piano_roll)
            except Exception as e:
                print(f"Error processing {audio_path}: {str(e)}")
                continue
        # Stack arrays
        X = np.stack(spectrograms)
        y = np.stack(piano_rolls)
        # Add channel dimension for CNN
        X = X[..., np.newaxis]  # Shape: (batch_size, n_bins, time_steps, 1)
        # Transpose y to have shape (batch_size, time_steps, n_pitches)
        y = y.transpose(0, 2, 1)
        return X, y

def segment_data(X, y, segment_length=512, hop_length=256):
    X_segments = []
    y_segments = []
    for i in range(X.shape[0]):  # Iterate over samples
        max_time = X.shape[2]  # Time dimension
        for start in range(0, max_time - segment_length + 1, hop_length):
            end = start + segment_length
            X_segments.append(X[i, :, start:end, :])
            y_segments.append(y[i, start:end, :])
    X_segments = np.array(X_segments)
    y_segments = np.array(y_segments)
    return X_segments, y_segments