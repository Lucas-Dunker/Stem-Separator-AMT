import subprocess
import shutil
from pathlib import Path


class MidiToPdfConverter:
    def __init__(self, musescore_path="MuseScore4.exe"):
        self.musescore_path = musescore_path
        self._validate_musescore()

    def _validate_musescore(self):
        """Validate that MuseScore is accessible
        The user needs to have MuseScore installed and added to their PATH"""
        # we can change this to have it run in a docker container with musescore installed
        if not shutil.which(self.musescore_path):
            raise FileNotFoundError(
                f"'{self.musescore_path}' not found in PATH. Please ensure MuseScore is installed and added to PATH."
            )

    def convert(self, midi_path, output_path):
        """
        Convert MIDI file to PDF

        Args:
            midi_path: Path to input MIDI file (str or Path)
            output_path: Path for output PDF file (str or Path)
            create_dirs: Whether to create output directories if they don't exist

        Returns:
            bool: True if conversion was successful, False otherwise
        """
        midi_path = Path(midi_path)
        output_path = Path(output_path)

        if not midi_path.is_file():
            raise FileNotFoundError(f"MIDI file not found at '{midi_path}'")

        command = [self.musescore_path, str(midi_path), "-o", str(output_path)]

        try:
            subprocess.run(command, check=True, capture_output=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"Conversion error: {e}")
            return False
