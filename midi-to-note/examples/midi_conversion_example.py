from pathlib import Path
from controllers.midi_converter_controller import MidiConverterController


def single_file_example():
    """Example of converting a single MIDI file to PDF"""
    input_file = Path("data/input/o-christmas-tree.mid")
    output_file = Path("data/output/o-christmas-tree.pdf")

    controller = MidiConverterController()
    success = controller.convert_midi_to_pdf(input_file, output_file)

    if success:
        print(f"Converted MIDI to PDF notation: {input_file} to {output_file}")
    else:
        print("Conversion failed")


if __name__ == "__main__":
    print("Single file conversion example:")
    single_file_example()
