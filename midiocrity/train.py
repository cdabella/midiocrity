import yaml
import argparse

# from midiocrity.data_unloader import unload_data
# from midiocrity import encoder, decoder, discriminator

from data_unloader import unload_data
import encoder, decoder, discriminator

import torch
import numpy as np

from dataset import MidiDataloader

from rich import print as rprint

class midiocrity():
    def __init__(self, **kwargs):
        print("Initializing encoder...")
        self.encoder = None

        print("Initializing decoder...")
        self.decoder = decoder.build_decoder()

        print("Initializing discriminator...")
        self.z_discriminator = discriminator.build_gaussian_discriminator()

    def train(self):
        batch = unload_data()
        X = batch[0]
        X = X[:, :, :, 0]
        X = torch.squeeze(X)
        # single-track input X should now be of size (batch, time, pitch)
        self.encoder = encoder.Encoder(X.shape[2], 4)
        output_z_mean, output_z_logvar = self.encoder.forward(X)


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

    loader = MidiDataloader(
        config['data_params']['tensor_folder'],
        config['data_params']['batch_size'],
        config['data_params']['shuffle'],
        config['data_params']['num_workers'],
    )

    # batch -> [X, y]
    for batch in loader:
        pass



if __name__ == "__main__":
    # midiocrity = midiocrity()
    # midiocrity.train()
    batch = main()
