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
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

import numpy as np

from dataset import MidiDataloader

import math
import time
from datetime import timedelta

from rich import print as rprint
from rich.table import Table
from rich.progress import Progress

def main():
    #torch.backends.cudnn.enabled = False

    parser = argparse.ArgumentParser(description='Midiocrity VAE model training')
    parser.add_argument('--config', '-c',
                        dest="filename",
                        metavar='FILE',
                        help='path to the config file',
                        default='../config/midiocrity_vae.yaml')
    parser.add_argument('--model', '-m',
                        dest='model',
                        metavar='FILE',
                        help='Path to trained model',
                        default='../out/MidiocrityVAE.epoch-19.pt')

    args = parser.parse_args()
    with open(args.filename, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    rprint(args.model)
    rprint(config)

    # For reproducibility
    if 'seed' in config['train_params']:
        torch.manual_seed(config['train_params']['seed'])
        np.random.seed(config['train_params']['seed'])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if config["device"] == 'cuda' and torch.cuda.is_available():
        device = torch.device(config["device"])
        dtype = torch.float
    else:
        device = torch.device('cpu')
        dtype = torch.float

    mvae = MidiocrityVAE(
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
        },
        device=device,
    )

    mvae_state_dict = torch.load(args.model)
    mvae.load_state_dict(mvae_state_dict)

    loader = MidiDataloader(
        config['data_params']['tensor_folder'],
        config['data_params']['batch_size'],
        config['data_params']['shuffle'],
        config['data_params']['num_workers'],
        batch_limit=config['data_params']['batch_limit']
    )

    total_accuracy = 0.0
    test_accuracy = 0.0
    with torch.autograd.set_detect_anomaly(False):
        with Progress(auto_refresh=True) as progress:
            pprint = progress.console.print
            step_tot = 0
            step_test = 0
            tstart = time.time()
            tcycle = tstart
            total_batches = config['data_params']['batch_limit'] # * 0.9  # 0.7 train, 0.2 valid
            task = progress.add_task("Training...", total=total_batches)

            # Validation
            step_batch = 0
            mvae.eval()
            loader.set_phase('train')
            with torch.no_grad():
                for batch in loader:
                    step_tot += 1
                    X = batch[0]
                    X = X[:, :, :, 0:1]  # Only train the drum tracks
                    X = X.to(device=device, dtype=dtype)
                    mu, logvar, z, recon = mvae(X)
                    accuracy = mvae.accuracy(X, recon).detach().cpu().item()
                    total_accuracy += accuracy
                    # pprint(
                    #     f"Accuracy: {accuracy} "
                    #     f"Total Accuracy: {total_accuracy /step_tot}"
                    # )
                    progress.advance(task)

                loader.set_phase('valid')
                for batch in loader:
                    step_tot += 1
                    X = batch[0]
                    X = X[:, :, :, 0:1]  # Only train the drum tracks
                    X = X.to(device=device, dtype=dtype)
                    mu, logvar, z, recon = mvae(X)
                    accuracy = mvae.accuracy(X, recon).detach().cpu().item()
                    total_accuracy += accuracy
                    progress.advance(task)

                loader.set_phase('test')
                for batch in loader:
                    step_test += 1
                    X = batch[0]
                    X = X[:, :, :, 0:1]  # Only train the drum tracks
                    X = X.to(device=device, dtype=dtype)
                    mu, logvar, z, recon = mvae(X)
                    accuracy = mvae.accuracy(X, recon).detach().cpu().item()
                    test_accuracy += accuracy
                    progress.advance(task)

    print(
        f"Total accuracy ({total_batches} samples): {total_accuracy / step_tot}\n"
        f"Test accuracy ({step_test} samples): {test_accuracy / step_test}"
        f"Total batches: {total_batches} Total Steps: {step_tot}"
    )


if __name__ == "__main__":
    main()
