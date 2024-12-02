import pretty_midi
import numpy as np
import tensorflow as tf
from transcriber import MusicTranscriber
import os
from IPython.display import Audio, display

print("TensorFlow version:", tf.__version__)

transcriber = MusicTranscriber()

def segment_features(features, segment_length=512, hop_length=256):
    segments = []
    max_time = features.shape[1] 
    for start in range(0, max_time - segment_length + 1, hop_length):
        end = start + segment_length
        segments.append(features[:, start:end, :])  
    return np.array(segments)  

def piano_roll_to_midi(piano_roll, fs=43.06, program=0, velocity=100):
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=program)
    
    time_per_frame = 1.0 / fs

    for pitch in range(piano_roll.shape[1]):
        # Find all contiguous segments of 'on' notes
        active_times = np.where(piano_roll[:, pitch] > 0.5)[0]
        if len(active_times) == 0:
            continue
        
        # Group contiguous times
        note_start = active_times[0]
        for i in range(1, len(active_times)):
            if active_times[i] != active_times[i - 1] + 1:
                # Note off
                note_end = active_times[i - 1]
                start_time = note_start * time_per_frame
                end_time = (note_end + 1) * time_per_frame
                note = pretty_midi.Note(velocity=velocity, pitch=pitch + 21, start=start_time, end=end_time)
                instrument.notes.append(note)
                # Update note start
                note_start = active_times[i]
        
        # Add the final note
        note_end = active_times[-1]
        start_time = note_start * time_per_frame
        end_time = (note_end + 1) * time_per_frame
        note = pretty_midi.Note(velocity=velocity, pitch=pitch + 21, start=start_time, end=end_time)
        instrument.notes.append(note)
    
    midi.instruments.append(instrument)
    
    return midi

audio_file =  'Data/gdrive/Track002.wav'
cqt = transcriber.process_audio(audio_file)  

cqt = np.expand_dims(cqt, axis=-1) 

# Segment the input into slices of length 512
segments = segment_features(cqt, segment_length=512, hop_length=256) 

# Load the model
model_path = 'transcriber/models/piano_transcriberF1.keras'
model = tf.keras.models.load_model(model_path, compile=False)


predictions = model.predict(segments)  

piano_roll = np.concatenate(predictions, axis=0)  

midi = piano_roll_to_midi(piano_roll, fs=43.06)
midi.write('output.midi')


output_folder = "output_midi"
os.makedirs(output_folder, exist_ok=True)
output_path = os.path.join(output_folder, "output.midi")
midi.write(output_path)

print(f"MIDI file saved to {output_path}")

from IPython.display import Audio
print(f"Generated MIDI file: {output_path}")

display(Audio("output_midi/output.midi", autoplay=True))
