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


# def batch_conversion_example():
#     """Example of batch converting multiple MIDI files to PDF"""
#     input_files = []

#     output_dir = Path("data/output")

#     controller = MidiConverterController()
#     results = controller.batch_convert(input_files, output_dir)

#     for input_file, success in results.items():
#         print(f"Converting {input_file}: {'Success' if success else 'Failed'}")


if __name__ == "__main__":
    print("Single file conversion example:")
    single_file_example()

    # print("\nBatch conversion example:")
    # batch_conversion_example()
