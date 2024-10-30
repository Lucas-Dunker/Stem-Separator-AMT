
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

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
         specmurt[i, :] = np.correlate(C_dB[i, :], template, mode='same')


    return specmurt





