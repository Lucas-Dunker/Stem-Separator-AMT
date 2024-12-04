# Imports and Setup
from nussl.ml.networks.modules import AmplitudeToDB, BatchNorm, RecurrentStack, Embedding
import numpy as np
import torch
from torch import nn

class StemSeparationModel(nn.Module):
    """
    This class defines the neural net model architecture for our stem separation model.

    The model includes the following components:
    - AmplitudeToDB: Converts the input mixture audio to log amplitude
    - BatchNorm: Normalizes the input mixture audio
    - RecurrentStack: A stack of LSTM layers
    - Embedding: A linear layer to generate the mask
    """ 
    def __init__(self, num_features, num_audio_channels, hidden_size,
                 num_layers, bidirectional, dropout, num_sources, 
                activation='sigmoid'):
        super().__init__()
        
        self.verbose = False

        self.amplitude_to_db = AmplitudeToDB()
        self.input_normalization = BatchNorm(num_features)
        self.recurrent_stack = RecurrentStack(
            num_features * num_audio_channels, hidden_size, 
            num_layers, bool(bidirectional), dropout
        )
        hidden_size = hidden_size * (int(bidirectional) + 1)
        self.embedding = Embedding(num_features, hidden_size, 
                                   num_sources, activation, 
                                   num_audio_channels)
        
        self.set_up_config(num_features, num_audio_channels, hidden_size,
                 num_layers, bidirectional, dropout, num_sources, 
                activation)

    def set_up_config(self, num_features, num_audio_channels, hidden_size,
                 num_layers, bidirectional, dropout, num_sources, 
                activation='sigmoid'):
        modules = {
            'model': {
                'class': 'StemSeparationModel',
                'args': {
                    'num_features': num_features,
                    'num_audio_channels': num_audio_channels,
                    'hidden_size': hidden_size,
                    'num_layers': num_layers,
                    'bidirectional': bidirectional,
                    'dropout': dropout,
                    'num_sources': num_sources,
                    'activation': activation,
                }
            }
        }

        connections = [
            ['model', ['mix_magnitude']]
        ]

        for key in ['mask', 'vocal_estimate']:
            modules[key] = {'class': 'Alias'}
            connections.append([key, [f'model:{key}']])
        
        output = ['vocal_estimate', 'mask',]
        self.config = {
            'name': 'StemSeparationModel',
            'modules': modules,
            'connections': connections,
            'output': output,
        }
        self.metadata = {
            'config': self.config,
            'nussl_version': '0.0.1',
        }

    def log(self, s):
        if self.verbose:
            print(s)

    def forward(self, item):
        # Get magnitude of mixture signal
        mixture_magnitude = item['mix_magnitude']
        if mixture_magnitude.dim() == 3:
            mixture_magnitude = mixture_magnitude.unsqueeze(0)  # Add a batch dimension to the mixture magnitude if needed
        self.log(f"Shape of mixture_magnitude: {mixture_magnitude.shape}")

        # Convert to log amplitude
        mixture_log_amplitude = self.amplitude_to_db(mixture_magnitude)
        self.log(f"Shape after amplitude to db: {mixture_log_amplitude.shape}")
        
        # Normalize the data
        normalized = self.input_normalization(mixture_log_amplitude)
        self.log(f"Shape after normalization: {normalized.shape}")

        # Pass through LSTM
        output = self.recurrent_stack(normalized)
        self.log(f"Shape after LSTM: {output.shape}")

        # Generate mask
        mask = self.embedding(output)
        self.log(f"Shape of mask: {mask.shape}")
    
        # Apply mask to get estimates - vocals source
        vocals_estimate = mixture_magnitude.unsqueeze(-1) * mask
        self.log(f"Shape of vocals estimate: {vocals_estimate.shape}")

        return {
            'mask': mask,
            'vocals_estimate': vocals_estimate,
        }

    def save(self, location, metadata=None, train_data=None, val_data=None, trainer=None):
        torch.save(self, location)
        return location
    
    def __repr__(self):
        output = super().__repr__()
        num_parameters = 0
        for p in self.parameters():
            if p.requires_grad:
                num_parameters += np.cumprod(p.size())[-1]
        output += '\nNumber of parameters: %d' % num_parameters
        return output

