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
        self.encoder = encoder.Encoder(4, n_tracks=1)
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

    device = torch.device(config["device"])

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
        # kl_weight=config['model_params']['kl_weight']
    )


    rprint(mvae)

    table = Table(title="MVAE Trainable Parameters")
    table.add_column("Component")
    table.add_column("Module")
    table.add_column("Parameters", justify="right")
    params = {
        "encoder": 0,
        "decoder": 0
    }
    for name, param in mvae.named_parameters():
        if not param.requires_grad:
            continue
        component = name.split('.')[0]
        num_params = param.numel()
        table.add_row(component, name, str(num_params))

        params[component] += num_params

    rprint(table)
    rprint(params)
    rprint(f"Device: {mvae.current_device}")

    optimizer = optim.Adam(
        mvae.parameters(),
        lr=config['train_params']['LR'],
        weight_decay=config['train_params']['weight_decay']
    )
    scheduler = lr_scheduler.ExponentialLR(
        optimizer,
        gamma=config['train_params']['scheduler_gamma']
    )

    param_norm = lambda m: math.sqrt(sum([p.norm().item() ** 2 for p in m.parameters()]))
    grad_norm = lambda m: math.sqrt(sum([p.grad.norm().item() ** 2 for p in m.parameters() if p.grad is not None]))

    beta = config['train_params']['beta']

    with Progress() as progress:
        step = 0
        tstart = time.time()
        tcycle = tstart
        task = progress.add_task("Training...", total=config['train_params']['epochs'])
        metrics = np.zeros(3)
        for epoch in range(config['train_params']['epochs']):
            loader = MidiDataloader(
                config['data_params']['tensor_folder'],
                config['data_params']['batch_size'],
                config['data_params']['shuffle'],
                config['data_params']['num_workers'],
                batch_limit=config['data_params']['batch_limit']
            )

            # batch -> [X, y]
            for batch in loader:
                step += 1
                X = batch[0]
                X = X[:, :, :, 0]
                X[X == 255] = 0
                X = X.to(device=device, dtype=torch.float)
                # breakpoint()
                # X = X.to(config["device"])

                # X = torch.squeeze(X)
                mvae.zero_grad()
                mu, logvar, z, recon = mvae(X)
                # breakpoint()
                # progress.console.print(
                #     f"{'#'* 20} mu {'#'* 20}\n"
                #     f"{mu}\n"
                #     f"{'#'* 20} logvar {'#'* 20}\n"
                #     f"{logvar}\n"
                # )
                recon = torch.squeeze(recon)
                kl_loss, recon_loss, loss = mvae.loss(mu, logvar, X, recon, beta)
                # breakpoint()
                loss.backward()
                if config['train_params']['clip_norm'] is not None:
                    nn.utils.clip_grad_norm_(
                        mvae.parameters(),
                        config['train_params']['clip_norm']
                    )

                optimizer.step()
                # progress.console.print(
                #     f"Param Norm: {param_norm(mvae)}\n"
                #     f"Grad Norm: {grad_norm(mvae)}"
                # )
                # for name, param in mvae.named_parameters():
                #     # if param.requires_grad:
                #     print(name, torch.max(param.data))
                # breakpoint()
                metrics = metrics + np.array([kl_loss, recon_loss, loss])

                if step % config['output_params']['print_step'] == 0:
                    # Average metrics for this print cycle
                    metrics /= config['output_params']['print_step']
                    ttotal = time.time() - tstart
                    tcycle = time.time() - tcycle
                    progress.console.print(
                        f"[{step}] "
                        f"ttotal: {timedelta(seconds=ttotal)} "
                        f"tcycle: {timedelta(seconds=tcycle)} "
                        f"beta: {beta:.3f} "
                        f"KLDiv: {metrics[0]:.2f} "
                        f"ReconBCEL: {metrics[1]:.4f} "
                        f"Loss: {metrics[2]:.2f}"
                    )

                    metrics *= 0
                    tcycle = time.time()

                if step % config['output_params']['save_step'] == 0:
                    torch.save(
                        mvae.state_dict(), (
                            f"{config['output_params']['save_dir']}"
                            f"{config['output_params']['name']}"
                            f".iter-{step}"
                        )
                    )

                if step % config['train_params']['anneal_step'] == 0:
                    scheduler.step()
                    progress.console.print(f"Learning rate: {scheduler.get_lr():.6f}")

                # Increase KL weight (beta)
                if step % config['train_params']['beta_increase_step'] == 0:
                    beta = min(
                        config['train_params']['beta_max'],
                        beta + config['train_params']['beta_increase']
                    )

            progress.advance(task)




if __name__ == "__main__":
    # midiocrity = midiocrity()
    # midiocrity.train()
    batch = main()
