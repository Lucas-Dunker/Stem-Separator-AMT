# Imports and Setup
import torch
import numpy as np
import nussl
import scipy.io.wavfile

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu'


def process_audio(item, model):
    """
    Process audio data using the given model, returning audio of only the estimated filtered vocal data.

    Args:
    item: the audio data (in numpy array format, following our test data) to process
    model: the model to use for processing the audio data

    Returns:
    vocals_estimate_audio: The given model's estimated audio signal (raw data) for the vocals source.
    """ 
    # Convert mixture signal to tensor if needed
    if isinstance(item['mix_magnitude'], np.ndarray):
        item['mix_magnitude'] = torch.from_numpy(item['mix_magnitude']).float()
    
    # Transpose for model input
    item["mix_magnitude"] = item["mix_magnitude"].transpose(0, 1)
    
    # Get model output (estimate of vocals source)
    with torch.no_grad():
        output = model(item)

    # Process the vocals estimate
    vocals_estimate = output['vocals_estimate']
    if vocals_estimate.dim() == 5:  # Remove extra dimensions
        vocals_estimate = vocals_estimate.squeeze(0).squeeze(-1).squeeze(-1)
    vocals_estimate = vocals_estimate.cpu().data.numpy()

    # Get the original mixture phase
    mix_stft = item['mix'].stft()
    mix_phase = np.angle(mix_stft)

    # Make sure vocals_estimate matches the original mixture phase
    # We want shape to be (freq_bins, time_frames)
    vocals_estimate = vocals_estimate.transpose()
    
    # Match shapes for combining magnitude and phase
    if vocals_estimate.shape[-1] == 1:
        vocals_estimate = vocals_estimate.squeeze(-1)
    if mix_phase.shape[-1] == 1:
        mix_phase = mix_phase.squeeze(-1)

    # Verify shapes of magnitude and phase match exactly
    assert vocals_estimate.shape == mix_phase.shape, f"Shape mismatch: vocals_estimate {vocals_estimate.shape} vs mix_phase {mix_phase.shape}"
    
    # Reconstruct complex STFT
    vocals_estimate_stft = vocals_estimate * np.exp(1j * mix_phase)
    
    # Create new audio signal with the same parameters as the input audio
    vocals_estimate_audio = nussl.AudioSignal(
        stft=vocals_estimate_stft,
        sample_rate=item['mix'].sample_rate,
        stft_params=item['mix'].stft_params,
    )

    # Perform inverse STFT
    vocals_estimate_audio.istft()
    
    # Ensure the output length matches the input exactly
    target_length = len(item['mix'].audio_data[0])
    current_length = len(vocals_estimate_audio.audio_data[0])
    
    # Pad the output audio to match the target length, if necessary
    if current_length != target_length:
        if current_length < target_length:
            pad_length = target_length - current_length
            vocals_estimate_audio.audio_data = np.pad(
                vocals_estimate_audio.audio_data, 
                ((0, 0), (0, pad_length)), 
                mode="constant",
            )
        else:
            vocals_estimate_audio.audio_data = vocals_estimate_audio.audio_data[:, :target_length]
    
    return vocals_estimate_audio

def deployModel(model, test_data):
    """
    Utilize the given model to process audio data from the test data, saving the output audio files to disk.

    Args:
    model the model to use for processing and filtering audio data
    test_data: the data to process, separate, and save to disk
    """ 
    SAVE_WAV_FILE = True
    TEST_DATA_ITEM_INDEX = 4

    item = test_data[TEST_DATA_ITEM_INDEX]
    item['mix_magnitude'] = torch.from_numpy(item['mix_magnitude']).float()
    vocals_estimate_audio = process_audio(item, model)
    vocals_estimate_audio.embed_audio(display=False)

    if SAVE_WAV_FILE:
        vocals_estimate_audio.write_audio_to_file(f"./stem-separation/outputs/test-{TEST_DATA_ITEM_INDEX}-vocals-estimate.wav")
    
    print('saving true mixture...')
    if SAVE_WAV_FILE:
        scipy.io.wavfile.write(f"./stem-separation/outputs/test-{TEST_DATA_ITEM_INDEX}-mixture.wav", item['mix'].sample_rate, item['mix'].audio_data.T)

    for stem_label in item['sources'].keys():
        print(f"saving true {stem_label}...")
        if SAVE_WAV_FILE:
            scipy.io.wavfile.write(f"./stem-separation/outputs/test-{TEST_DATA_ITEM_INDEX}-{stem_label}.wav", item['sources'][stem_label].sample_rate, item['sources'][stem_label].audio_data.T)

    