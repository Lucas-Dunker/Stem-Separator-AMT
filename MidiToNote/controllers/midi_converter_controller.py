from pathlib import Path
from model.midi_to_note import MidiToPdfConverter


class MidiConverterController:
    def __init__(self, musescore_path: str = "MuseScore4.exe"):
        """
        Initialize the MIDI converter controller

        Args:
            musescore_path (str): Path to MuseScore executable
        """
        self._converter = None
        self.musescore_path = musescore_path

    def initialize_converter(self) -> bool:
        """
        Initialize the MIDI to PDF converter

        Returns:
            bool: True if initialization was successful, False otherwise
        """
        try:
            self._converter = MidiToPdfConverter(self.musescore_path)
            return True
        except FileNotFoundError as e:
            print(f"Failed to initialize converter: {e}")
            return False

    def convert_midi_to_pdf(
        self, input_path: str | Path, output_path: str | Path
    ) -> bool:
        """
        Convert a MIDI file to PDF

        Args:
            input_path: Path to input MIDI file
            output_path: Path for output PDF file
            create_dirs: Whether to create output directories if they don't exist

        Returns:
            bool: True if conversion was successful, False otherwise
        """
        if self._converter is None:
            if not self.initialize_converter():
                return False

        try:
            return self._converter.convert(input_path, output_path)
        except Exception as e:
            print(f"Error during conversion: {e}")
            return False
