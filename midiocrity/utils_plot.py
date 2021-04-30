from midiocrity_vae import MidiocrityVAE
import yaml
from rich import print as rprint
import torch
import numpy as np
import matplotlib.pyplot as plt; plt.rcParams['figure.dpi'] = 100
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_dummy_model():
    with open('../config/midiocrity_vae.yaml', 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    # rprint(config)

    # For reproducibility
    if 'seed' in config['train_params']:
        torch.manual_seed(config['train_params']['seed'])
        np.random.seed(config['train_params']['seed'])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    model = MidiocrityVAE(
        use_cuda = False,
        encoder_params={
            "n_cropped_notes": config['model_params']['n_cropped_notes'],
            "z_dim": config['model_params']['z_dim'],
            "phrase_size": config['model_params']['phrase_size'],
            "hidden_size": config['model_params']['encoder_params']['hidden_size'],
            "num_layers": config['model_params']['encoder_params']['num_layers'],
            "batch_size": config['data_params']['batch_size'],
            "n_tracks": config['model_params']['decoder_params']['n_tracks'],
        },
        decoder_params={
            "n_cropped_notes": config['model_params']['n_cropped_notes'],
            "z_dim": config['model_params']['z_dim'],
            "phrase_size": config['model_params']['phrase_size'],
            "hidden_size": config['model_params']['decoder_params']['hidden_size'],
            "num_layers": config['model_params']['decoder_params']['num_layers'],
            "batch_size": config['data_params']['batch_size'],
            "n_tracks": config['model_params']['decoder_params']['n_tracks'],
        }
    )
    return model


# Based on https://avandekleut.github.io/vae/
def plot_latent_nolabel(autoencoder, data, num_batches=100):
    """Plot data over a latent space of an autoencoder

    Parameters:
    autoencoder (nn.Module): The autoencoder. Must have the encode function.
    data ((x, y)): 
        -- x(batch size, seq_len, input_size, ntrack)
        -- y(artist, song)
    num_batches: max number of batches

    Returns:
    None

    """
    plt.figure(figsize=(5,5))
    for i, x in enumerate(data):
        # Encode enpoints
        mu, lv = autoencoder.encode(x)
        z = autoencoder.reparameterize(mu, lv)
        z = z.to('cpu').detach().numpy()
        plt.scatter(z[:, 0], z[:, 1], c='blue')
        if i > num_batches:
            plt.colorbar()
            break

def plot_latent(autoencoder, data, num_batches=100):
    """Plot data over a latent space of an autoencoder

    Parameters:
    autoencoder (nn.Module): The autoencoder. Must have the encode function.
    data ((x, y)): 
        -- x(batch size, seq_len, input_size, ntrack)
        -- y(artist, song)
    num_batches: max number of batches

    Returns:
    None

   """
    for i, (x, y) in enumerate(data):
        z = autoencoder.encode(x.to(device))
        z = z.to('cpu').detach().numpy()
        plt.scatter(z[:, 0], z[:, 1], c=y, cmap='tab10')
        if i > num_batches:
            plt.colorbar()
            break