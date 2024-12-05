# Imports and Setup
from common import data

from tqdm import tqdm
from pathlib import Path

import nussl
from nussl.datasets import transforms as nussl_tfm
from ignite.engine import Events

import numpy as np
import torch

from StemSeparationModel import StemSeparationModel

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu'

def prepareModel(train_folder, val_folder, test_folder):
    """
    Load in our audio dataset (music separated into 'bass', 'drums', 'vocals', and 'other' components), 
    set up our model, and prepare our training, validation, and test data.

    Args:
    train_folder: the file path containing the training data
    val_folder: the file path containing the validation data
    test_folder: the file path containing the test data

    Returns:
    model: a newly initialized model following the StemSeparationModel class
    train_data: the training data for the model
    train_dataloader: the training dataloader for the model
    val_dataloader: the validation dataloader for the model
    test_data: the test data for the model
    """ 
        
    data.prepare_musdb("./stem-separation/dataset")

    stft_params = nussl.STFTParams(window_length=512, hop_length=128, window_type='sqrt_hann')

    tfm = nussl_tfm.Compose([
    nussl_tfm.SumSources([['bass', 'drums', 'other']]),
    nussl_tfm.MagnitudeSpectrumApproximation(),
    nussl_tfm.IndexSources('source_magnitudes', 1),
    nussl_tfm.ToSeparationModel(),
    ])

    MAX_MIXTURES = int(1e8) # Set to some impossibly high number for on-the-fly mixing.

    train_data = data.on_the_fly(stft_params, transform=tfm, fg_path=train_folder, num_mixtures=MAX_MIXTURES, coherent_prob=1.0)
    train_dataloader = torch.utils.data.DataLoader(train_data, num_workers=1, batch_size=10)

    val_data = data.on_the_fly(stft_params, transform=tfm, fg_path=val_folder, num_mixtures=10, coherent_prob=1.0)
    val_dataloader = torch.utils.data.DataLoader(val_data, num_workers=1, batch_size=10)

    item = train_data[0]
    for key in item:
        print(key, type(item[key]), item[key].shape if isinstance(item[key], torch.Tensor) else "")

    test_tfm = nussl_tfm.Compose([
    nussl_tfm.SumSources( # Only outputting vocals for our model
        groupings=[['drums', 'bass', 'other']],
        group_names=['accompaniment'],
    ),
    nussl_tfm.MagnitudeSpectrumApproximation(),
])

    test_data = data.on_the_fly(stft_params, transform=test_tfm, fg_path=test_folder, num_mixtures=100)

    item = test_data[0]
    for key in item:
        print(key, type(item[key]), item[key].shape if isinstance(item[key], np.ndarray) else "")
    
    # Define Model
    num_features = stft_params.window_length // 2 + 1
    num_audio_channels = 1
    hidden_size = 50
    num_layers = 2
    bidirectional = True
    dropout = 0.3
    num_sources = 1
    activation = 'sigmoid'

    model = StemSeparationModel(
        num_features=num_features,
        num_audio_channels=num_audio_channels,
        hidden_size=hidden_size,
        num_layers=num_layers,
        bidirectional=bidirectional,
        dropout=dropout,
        num_sources=num_sources,
        activation=activation,
    )

    return model, train_data, train_dataloader, val_dataloader, test_data



def trainModel(model, NUM_EPOCHS, EPOCH_LENGTH, train_data, train_dataloader, val_dataloader):
    """
    Train the given model for a specified number of epochs and batch size with the given training and validation data.

    Args:
    model: the model to train
    NUM_EPOCHS: the number of epochs to train the model
    EPOCH_LENGTH: the number of batches per epoch
    train_data: the training data for the model
    train_dataloader: the training dataloader for the model
    val_dataloader: the validation dataloader for the model
    """ 

    # TODO - tweak our training step to improve performance and accuracy (epochs, learning rate, loss function, etc.)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nussl.ml.train.loss.L1Loss()

    def train_step(engine, batch):
        optimizer.zero_grad()
        output = model(batch) # forward pass
        loss = loss_fn(
            output['vocals_estimate'],
            batch['source_magnitudes']
        )
        
        loss.backward() # backwards + gradient step
        optimizer.step()
        
        loss_vals = {
            'L1Loss': loss.item(),
            'loss': loss.item()
        }

        progress_bar = engine.state.pbar
        progress_bar.set_description(f"Loss: {loss.item():.4f}")
        
        return loss_vals

    def val_step(engine, batch):
        with torch.no_grad():
            output = model(batch) # forward pass
        loss = loss_fn(
            output['vocals_estimate'],
            batch['source_magnitudes']
        )    
        loss_vals = {
            'L1Loss': loss.item(), 
            'loss': loss.item()
        }
        return loss_vals

    # Create the engines
    trainer, validator = nussl.ml.train.create_train_and_validation_engines(
        train_step, val_step, device=DEVICE
    )

    @trainer.on(Events.EPOCH_STARTED)
    def setup_pbar(engine):
        engine.state.pbar = tqdm(
            total=EPOCH_LENGTH,
            desc=f"Epoch {engine.state.epoch}/{NUM_EPOCHS}",
            leave=True,
        )

    @trainer.on(Events.ITERATION_COMPLETED)
    def update_pbar(engine):
        engine.state.pbar.update(1)

    @trainer.on(Events.EPOCH_COMPLETED)
    def close_pbar(engine):
        engine.state.pbar.close()

    # Save the output relative to this notebook
    output_folder = Path('.').absolute()

    # Adding handlers from nussl that print out details about model training
    nussl.ml.train.add_stdout_handler(trainer, validator)
    nussl.ml.train.add_validate_and_checkpoint(output_folder, model, optimizer, train_data, trainer, val_dataloader, validator)

    # Run the validation step and save the models
    trainer.run(
        train_dataloader,
        epoch_length=EPOCH_LENGTH,
        max_epochs=NUM_EPOCHS,
    )