# import pretty_midi
# import numpy as np
# import os

# def midi_to_label(midi_file, spectrogram_shape, hop_length, sr=22050, min_pitch=21, max_pitch=108, output_dir="./labels", track_id=None, midi_id=None):
#     """
#     Convert a MIDI file to a binary label matrix aligned with the spectrogram and save it to a file.

#     Parameters:
#         midi_file (str): Path to the MIDI file.
#         spectrogram_shape (tuple): Shape of the spectrogram (time_frames, n_bins).
#         hop_length (int): Hop length of the spectrogram.
#         sr (int): Sampling rate.
#         min_pitch (int): Minimum MIDI pitch to include in the label matrix.
#         max_pitch (int): Maximum MIDI pitch to include in the label matrix.
#         output_dir (str): Directory to save the labels.
#         track_id (str): Identifier for the track (e.g., "Track00001").
#         midi_id (str): Identifier for the MIDI file (e.g., "S00").

#     Returns:
#         str: Path to the saved label file.
#     """
#     # Ensure output directory exists
#     os.makedirs(output_dir, exist_ok=True)
    
#     # Process MIDI file
#     midi_data = pretty_midi.PrettyMIDI(midi_file)
#     time_frames, _ = spectrogram_shape
#     time_per_frame = hop_length / sr
#     n_notes = max_pitch - min_pitch + 1
#     labels = np.zeros((time_frames, n_notes), dtype=np.float32)  # Create dynamic label matrix

#     # Map MIDI notes to time frames
#     for instrument in midi_data.instruments:
#         for note in instrument.notes:
#             if min_pitch <= note.pitch <= max_pitch:  # Only include notes in the specified range
#                 start_frame = int(note.start / time_per_frame)
#                 end_frame = int(note.end / time_per_frame)
#                 if 0 <= start_frame < time_frames:
#                     labels[start_frame:end_frame, note.pitch - min_pitch] = 1  # Align to min_pitch
    
#     # Create a unique file name
#     if track_id and midi_id:
#         output_file = os.path.join(output_dir, f"{track_id}_{midi_id}.npy")
#     else:
#         output_file = os.path.join(output_dir, os.path.basename(midi_file).replace('.mid', '.npy'))

#     # Save labels as a .npy file
#     np.save(output_file, labels)
#     print(f"Labels saved to: {output_file}")
    
#     return output_file

# # Example usage in a loop
# output_dir = "./labels"
# os.makedirs(output_dir, exist_ok=True)

# spectrogram_shape = (100, 88)  # Adjust based on your spectrogram dimensions
# min_pitch = 36  # Example: C2
# max_pitch = 72  # Example: C6

# for i in range(1, 21):
#     track_num = str(i).zfill(2)  # Ensure two digits (e.g., "01", "02")
#     track_id = f"Track000{track_num}"
#     base_dir = f"Data/babyslakh_16k/babyslakh_16k/{track_id}/MIDI/"
    
#     for j in range(16):
#         midi_num = str(j).zfill(2)  # Ensure two digits (e.g., "00", "01")
#         midi_file = os.path.join(base_dir, f"S{midi_num}.mid")

#         if not os.path.isfile(midi_file):
#             continue

#         midi_id = f"S{midi_num}"
#         midi_to_label(
#             midi_file,
#             spectrogram_shape,
#             hop_length=512,
#             sr=22050,
#             min_pitch=min_pitch,
#             max_pitch=max_pitch,
#             output_dir=output_dir,
#             track_id=track_id,
#             midi_id=midi_id
#         )




# # import pretty_midi

# # midi_file = "Data/babyslakh_16k/babyslakh_16k/Track00001/MIDI/S04.mid"  # Replace with an actual file path
# # midi_data = pretty_midi.PrettyMIDI(midi_file)

# # # Print instrument and note information
# # for instrument in midi_data.instruments:
# #     print(f"Instrument: {instrument.name}")
# #     for note in instrument.notes:
# #         print(f"Note: Pitch={note.pitch}, Start={note.start}, End={note.end}")


# # import numpy as np

# # labels = np.load("labels/Track00020_S02.npy")
# # print(labels)



import numpy as np

labels = np.load("labels/Track00001_S00.npy")  # Replace with an actual file
print("Label Matrix Shape:", labels.shape)
print("Non-zero Entries:", np.count_nonzero(labels))
print(labels)


import matplotlib.pyplot as plt

plt.imshow(labels.T, aspect='auto', cmap='hot', origin='lower')
plt.xlabel("Time Frames")
plt.ylabel("Pitch (MIDI)")
plt.title("Generated Label Matrix")
plt.colorbar(label="Note Active (1=True, 0=False)")
plt.show()
