import os
import subprocess
import time

# Path to your General MIDI SoundFont file (replace this with the path to your SoundFont file)
soundfont_path = "Data/FluidR3_GM/FluidR3_GM.sf2"  # Use a General MIDI SoundFont like FluidR3_GM.sf2

# Desired sample rate for output WAV files
sample_rate = 44100

# Function to convert MIDI file to WAV using FluidSynth
def midi_to_wav(midi_file_path, output_wav_path, soundfont_path, sample_rate):
    """
    Converts a MIDI file to WAV format using FluidSynth.
    """
    command = [
        "Data/fluidsynth-2.4.0-win10-x64/bin/fluidsynth.exe",
        "-ni",  # Non-interactive mode
        soundfont_path,
        midi_file_path,
        "-F", output_wav_path,
        "-r", str(sample_rate)
    ]
    try:
        subprocess.run(command, check=True)
        print(f"Converted {midi_file_path} to {output_wav_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error converting {midi_file_path} to WAV: {e}")

# Define the root folder where your track folders are located
root_folder = 'Data/babyslakh_16k/babyslakh_16k'

# Process each track and convert all MIDI files in the MIDI subfolder to WAV
def convert_all_midi_to_wav(root_folder, soundfont_path, sample_rate=44100):
    for track_folder in os.listdir(root_folder):
        full_track_path = os.path.join(root_folder, track_folder, "midi")
        
        if os.path.isdir(full_track_path):
            for file in os.listdir(full_track_path):
                if file.endswith('.mid') or file.endswith('.midi'):
                    midi_file_path = os.path.join(full_track_path, file)
                    output_wav_path = os.path.join(full_track_path, file.replace('.mid', '.wav').replace('.midi', '.wav'))
                    
                    # Convert each MIDI file to WAV
                    midi_to_wav(midi_file_path, output_wav_path, soundfont_path, sample_rate)
                    time.sleep(0.5)  # Brief pause to ensure smooth conversion

# Run the conversion
convert_all_midi_to_wav(root_folder, soundfont_path, sample_rate)
