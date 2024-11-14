import os
import music21

def extract_chords_from_midi(midi_file):
    """
    Extract chords from a MIDI file using music21.
    Returns a list of tuples with start time, end time, and chord name.
    """
    chords = []
    midi_data = music21.converter.parse(midi_file)

    for element in midi_data.flatten().notesAndRests.stream():
        if isinstance(element, music21.chord.Chord):
            start_time = element.offset
            duration = element.quarterLength
            chord_name = element.pitchedCommonName
            chords.append((start_time, start_time + duration, chord_name))
        elif isinstance(element, music21.note.Note):
            # Treat individual notes as chords (like C major for the note C)
            start_time = element.offset
            duration = element.quarterLength
            chord_name = element.name
            chords.append((start_time, start_time + duration, chord_name))

    return chords

def save_chords_to_lab_file(chords, output_file):
    """
    Save chords to a .lab file with start time, end time, and chord name.
    """
    with open(output_file, 'w') as f:
        for start_time, end_time, chord_name in chords:
            start_time = float(start_time)
            end_time = float(end_time)
            f.write(f"{start_time:.2f}\t{end_time:.2f}\t{chord_name}\n")

def process_individual_components(track_folder):
    """
    Process each MIDI file within a track's MIDI folder as individual components,
    saving separate .lab files for each component.
    """
    midi_folder = os.path.join(track_folder, "MIDI")
    if not os.path.isdir(midi_folder):
        print(f"No MIDI folder found in {track_folder}")
        return
    
    for midi_file in os.listdir(midi_folder):
        if midi_file.endswith('.mid') or midi_file.endswith('.midi'):
            midi_file_path = os.path.join(midi_folder, midi_file)
            chords = extract_chords_from_midi(midi_file_path)
            
            component_name = os.path.splitext(midi_file)[0]  # Get the name without extension
            output_file_path = os.path.join(midi_folder, f"{component_name}.lab")
            
            # Save the individual component's chord annotations
            save_chords_to_lab_file(chords, output_file_path)
            print(f"Processed {midi_file} in {track_folder} and saved to {output_file_path}")


def process_all_tracks(root_folder):
    """
    Process each track folder, generating individual .lab files for each MIDI component.
    """
    for track_folder in os.listdir(root_folder):
        full_track_path = os.path.join(root_folder, track_folder)
        if os.path.isdir(full_track_path):
            process_individual_components(full_track_path)

root_folder = 'Data/babyslakh_16k/babyslakh_16k'  # Folder containing Track00001, Track00002, etc.

# Process all tracks and generate individual .lab files for each component
process_all_tracks(root_folder)
