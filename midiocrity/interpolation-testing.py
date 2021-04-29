import yaml
import argparse

# from midiocrity.data_unloader import unload_data
# from midiocrity import encoder, decoder, discriminator

from data_unloader import unload_data
import encoder
import decoder
import discriminator
from midiocrity_vae import MidiocrityVAE

import torch
import numpy as np

from dataset import MidiDataloader

from rich import print as rprint

def main():
    parser = argparse.ArgumentParser(description='Midiocrity VAE model training')
    parser.add_argument('--config', '-c',
                        dest="filename",
                        metavar='FILE',
                        help='path to the config file',
                        default='../config/midiocrity_vae.yaml')

    args = parser.parse_args()
    with open(args.filename, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    rprint(config)

    # For reproducibility
    if 'seed' in config['train_params']:
        torch.manual_seed(config['train_params']['seed'])
        np.random.seed(config['train_params']['seed'])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    model = MidiocrityVAE(
        encoder_params={
            "n_cropped_notes": config['model_params']['n_cropped_notes'],
            "z_dim": config['model_params']['z_dim'],
            "phrase_size": config['model_params']['phrase_size'],
            "hidden_size": config['model_params']['encoder_params']['hidden_size'],
            "num_layers": config['model_params']['encoder_params']['num_layers']
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

    # loader = MidiDataloader(
    #     config['data_params']['tensor_folder'],
    #     config['data_params']['batch_size'],
    #     config['data_params']['shuffle'],
    #     config['data_params']['num_workers'],
    #     batch_limit=config['data_params']['batch_limit']
    # )

    # # batch -> [X, y]
    # i = 0
    # for batch in loader:
    #     i += 1
    # print(i)

    # midiocrity.train()
    # INPUT -> (batch size, seq_len, input_size)  --  batch, phrase_size, notes, track
    # ENCODER Z OUTPUT -> (seq_len, batch, num_directions*hidden_size)
    # DECODER INPUT  -> (seq_len, batch, num_directions*hidden_size)
    # DECODER OUTPUT -> batch x phrase_size x notes x track 
    x_s = torch.rand(16, 256, 130)
    x_t = torch.rand(16, 256, 130)
    model.interpolate(x_s, x_t, length=14)




if __name__ == "__main__":
    # midiocrity = midiocrity()
    # midiocrity.train()
    batch = main()

