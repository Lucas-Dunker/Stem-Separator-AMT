
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

# Compute Constant-Q Transform
def compute_cqt(y, sr):
    # Compute the CQT (with default bins per octave = 12)
    C = librosa.cqt(y, sr=sr)
    # Convert the CQT to decibels for better visualization
    C_dB = librosa.amplitude_to_db(np.abs(C), ref=np.max)
    return C_dB


def get_harmonic_templates(cqt_frequencies, minimum_fequency, maximum_frequency, num_harmonics, bins_p_octave):
    # Define the range of fundamental frequencies based on musical notes within min_freq and max_freq
    num_notes = int(np.log2(maximum_frequency / minimum_fequency) * bins_p_octave)
    # Generates an array of frequencies (in Hz) evenly spaced in a logarithmic scale
    fundamental_frequencies = librosa.cqt_frequencies(n_bins=num_notes, fmin=minimum_fequency, bins_per_octave=bins_p_octave)

    #initialize the harmonic templates for all of the notes
    template_bank = np.zeros((len(fundamental_frequencies), len(cqt_frequencies)))

    # for each note's fundamental frequency
    for i, fundamental_freq in enumerate(fundamental_frequencies):
        harmonics = [fundamental_freq * (n + 1) for n in range(num_harmonics)]

        for harmonic in harmonics:
            # find the closest to the current harmonic
            nearest_bin = (np.abs(cqt_frequencies - harmonic)).argmin()
            template_bank[i, nearest_bin] = 1 

    return template_bank



def specmurt_analyis(cqt_frequencies, minimum_fequency, maximum_frequency, num_harmonics, bins_p_octave, C_dB):
    harmonic_templates = get_harmonic_templates(cqt_frequencies, minimum_fequency, maximum_frequency, num_harmonics, bins_p_octave)

    # initalize
    specmurt = np.zeros((harmonic_templates.shape[0], C_dB.shape[1]))

    for i, template in enumerate(harmonic_templates):
        # apply the convolution
        specmurt[i, :] = np.apply_along_axis(lambda x: np.correlate(x, template, mode='same'), axis=0, arr=C_dB)


    return specmurt





