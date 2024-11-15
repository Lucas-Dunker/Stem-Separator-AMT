import subprocess
import shutil
import os


def midi_to_pdf(midi_file, pdf_file):

    mscore_executable = "MuseScore4.exe"
    # we can change this implementation later to use an api
    # I'll just create a docker container with a simple flask route that takes a midi file and returns a pdf file
    # it'll also contain the MuseScore executable

    # check if MuseScore is accessible
    if not shutil.which(mscore_executable):
        raise FileNotFoundError(
            f"'{mscore_executable}' not found in PATH. Please ensure MuseScore is installed and added to PATH."
        )

    if not os.path.isfile(midi_file):
        raise FileNotFoundError(f"MIDI file not found at '{midi_file}'")

    command = [mscore_executable, midi_file, "-o", pdf_file]

    try:
        subprocess.run(command, check=True)
        print(f"Successfully converted '{midi_file}' to '{pdf_file}'")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred during conversion: {e}")


midi_file = os.path.abspath("../data/input/o-christmas-tree.mid")
output_dir = os.path.abspath("../data/output")
pdf_file = os.path.join(output_dir, "o-christmas-tree.pdf")
midi_to_pdf(midi_file, pdf_file)
