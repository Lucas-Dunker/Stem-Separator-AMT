import pretty_midi
import numpy as np
import tensorflow as tf
from transcriber import MusicTranscriber
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
    
    # Add the instrument to the PrettyMIDI object
    midi.instruments.append(instrument)
    
    return midi

cqt = transcriber.process_audio("Data/gdrive/Track002.wav")  # Shape: (84, time_steps)
cqt = np.expand_dims(cqt, axis=-1)  # Add channel dimension, Shape: (84, time_steps, 1)

# Segment the input into slices of length 512
segments = segment_features(cqt, segment_length=512, hop_length=256) 

# Load the model
model_path = 'transcriber\model\piano_transcriberF1.keras'
model = tf.keras.models.load_model(model_path, compile=False)

# Predict for each segment (batch input is expected)
predictions = model.predict(segments)  # Shape: (num_segm,ents, time_steps, n_pitches)

piano_roll = np.concatenate(predictions, axis=0)  # Combine time steps across all segments, shape: (full_time_steps, n_pitches)

midi = piano_roll_to_midi(piano_roll, fs=43.06)
midi.write('output.midi')

