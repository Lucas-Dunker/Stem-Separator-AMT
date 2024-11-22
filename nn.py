import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os
from sklearn.model_selection import train_test_split

# --- Utility: Pad or Truncate Data ---
def pad_or_truncate(data, target_length, axis=0):
    """
    Pads or truncates the data to the target length along the specified axis.
    """
    current_length = data.shape[axis]
    if current_length > target_length:
        slices = [slice(None)] * data.ndim
        slices[axis] = slice(0, target_length)
        return data[tuple(slices)]
    elif current_length < target_length:
        pad_width = [(0, 0)] * data.ndim
        pad_width[axis] = (0, target_length - current_length)
        return np.pad(data, pad_width, mode='constant')
    return data

# --- Load and Align Data ---
def load_data(spectrogram_dir, label_dir):
    """
    Load and align spectrogram and label data from directories.
    """
    spectrogram_files = sorted([f for f in os.listdir(spectrogram_dir) if f.endswith('.npy')])
    label_files = sorted([f for f in os.listdir(label_dir) if f.endswith('.npy')])

    spectrograms = []
    labels = []

    for s_file, l_file in zip(spectrogram_files, label_files):
        spectrogram = np.load(os.path.join(spectrogram_dir, s_file))
        label = np.load(os.path.join(label_dir, l_file))
        
        print(f"Original Spectrogram shape: {spectrogram.shape}, Label shape: {label.shape}, Files: {s_file}, {l_file}")
        
        # Align time frames
        target_time_frames = min(spectrogram.shape[1], label.shape[0])  # Match on time frames
        spectrogram = pad_or_truncate(spectrogram, target_time_frames, axis=1)  # Spectrogram: time frames on axis 1
        label = pad_or_truncate(label, target_time_frames, axis=0)  # Label: time frames on axis 0

        print(f"Aligned Spectrogram shape: {spectrogram.shape}, Aligned Label shape: {label.shape}")
        
        spectrograms.append(spectrogram)
        labels.append(label)

    return np.array(spectrograms), np.array(labels)

# --- Normalize Data ---
def normalize_data(X):
    """
    Normalize spectrogram data to range [0, 1].
    """
    return (X - X.min()) / (X.max() - X.min())

# --- Adjust Labels ---
def adjust_labels(labels, time_steps):
    """
    Adjust labels to match the specified number of time steps.
    Pads or truncates each label along the time dimension.

    Parameters:
        labels (numpy.ndarray): The labels to adjust.
        time_steps (int): The desired number of time steps.

    Returns:
        numpy.ndarray: Adjusted labels.
    """
    return np.array([pad_or_truncate(label, time_steps, axis=0) for label in labels])

# --- Build Model ---
def build_model(input_shape, output_shape):
    """
    Build a recurrent CNN model for audio-to-MIDI transcription.
    """
    model = models.Sequential()

    # Explicit Input layer
    model.add(layers.Input(shape=input_shape))

    # Convolutional layers with carefully adjusted pooling
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))  # First pooling
    print("After 1st pooling:", model.output_shape)

    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))  # Second pooling
    print("After 2nd pooling:", model.output_shape)

    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    print("After 3rd conv (no pooling):", model.output_shape)

    # Flatten CNN output
    model.add(layers.Flatten())
    flattened_size = model.output_shape[-1]
    print("Flattened size:", flattened_size)

    # Dynamically adjust time steps
    time_steps = output_shape[0]  # Desired number of time steps (100)
    if flattened_size % time_steps != 0:
        # Adjust time_steps to the largest factor of flattened_size less than or equal to output_shape[0]
        time_steps = max([i for i in range(1, flattened_size + 1) if flattened_size % i == 0 and i <= output_shape[0]])
        print(f"Adjusted time_steps to {time_steps} to match flattened size {flattened_size}")

    features_per_step = flattened_size // time_steps

    # Reshape for LSTM input
    model.add(layers.Reshape((time_steps, features_per_step)))

    # Recurrent layers
    model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True)))  # Keep sequences
    model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True)))  # Retain time dimension

    # Dense output layer
    model.add(layers.TimeDistributed(layers.Dense(output_shape[-1], activation='sigmoid')))

    return model

# --- Main Workflow ---
if __name__ == "__main__":
    # Directories
    spectrogram_dir = "./spectrograms"
    label_dir = "./labels"

    # Load and preprocess data
    X, Y = load_data(spectrogram_dir, label_dir)
    X = normalize_data(X)
    X = X[..., np.newaxis]  # Add channel dimension for CNN

    # Train/test split
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Dynamically adjust labels to match the model output time steps
    adjusted_time_steps = 96  # Match model's output time steps
    Y_train = adjust_labels(Y_train, adjusted_time_steps)
    Y_val = adjust_labels(Y_val, adjusted_time_steps)

    print(f"Adjusted Y_train shape: {Y_train.shape}, Adjusted Y_val shape: {Y_val.shape}")

    # Model input and output shapes
    input_shape = (X_train.shape[1], X_train.shape[2], 1)  # Time frames x frequency bins x 1 (channel)
    output_shape = (adjusted_time_steps, Y_train.shape[2])  # Time frames x note range

    # Build model
    model = build_model(input_shape, output_shape)

    # Compile model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        epochs=100,
        batch_size=16
    )

    # Save the trained model
    os.makedirs('saved_model', exist_ok=True)

# Save the model in .keras format
    model.save('saved_model/my_model.keras')
    print("Model saved!")

    # Evaluate the model
    loss, accuracy = model.evaluate(X_val, Y_val)
    print(f"Validation Loss: {loss:.4f}")
    print(f"Validation Accuracy: {accuracy:.4f}")
