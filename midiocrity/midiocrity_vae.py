import torch
from torch import nn
from decoder import Decoder
from encoder import encoder

class MidiocrityVAE(nn.Module):

    def __init__(
            self,
            config_file=None,
            encoder_params=None,
            decoder_params=None,
            use_cuda=True
    ):
        super(MidiocrityVAE, self).__init__()

        self.z_dim = 32
        if config_file:
            pass
        else:
            if not encoder_params:
                self.encoder_params = {
                    "input_size": 4,
                    "seq_len": 256,
                    "hidden_size": 512,
                    "num_layers": 3
                }
            else:
                self.encoder_params = encoder_params
            if not decoder_params:
                self.decoder_params = {
                    "z_dim": 32,
                    "phrase_size": 256,
                    "batch_size": 16,
                    "hidden_size": 512,
                    "num_layers": 3,
                    "n_tracks": 4,
                    "n_cropped_notes": 130
                }
            else:
                self.decoder_params = decoder_params

        self.encoder = encoder(**self.encoder_params)
        self.decoder = Decoder(**self.decoder_params)

        self.use_cuda = use_cuda if torch.cuda.is_available() else False
        if self.use_cuda:
            self.cuda()
            self.current_device = torch.cuda.current_device()
        else:
            self.current_device = 'cpu'

    # Source: https://github.com/AntixK/PyTorch-VAE
    # Method from: https://arxiv.org/pdf/1312.6114v10.pdf
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def encode(self, x):
        mu, logvar = self.encoder(x)
        return mu, logvar

    def decode(self, z):
        out = self.decoder(z)
        return out

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        out = self.decode(z)
        return mu, logvar, z, out

    def sample(self, num_samples):
        z = torch.randn((num_samples, self.z_dim), device=self.current_device)
        out = self.decode(z)
        return out

