import torch
from torch import nn
from torch.nn import functional as F
from decoder import Decoder
from encoder import Encoder
from torch import lerp
from interpolator import slerp


class MidiocrityVAE(nn.Module):

    def __init__(
            self,
            config_file=None,
            encoder_params=None,
            decoder_params=None,
            use_cuda=True,
    ):
        super(MidiocrityVAE, self).__init__()

        self.z_dim = 32
        if config_file:
            pass
        else:
            if not encoder_params:
                self.encoder_params = {
                    "z_dim": 32,
                    "phrase_size": 256,
                    "batch_size": 16,
                    "hidden_size": 512,
                    "num_layers": 3,
                    "n_tracks": 4,
                    "n_cropped_notes": 130
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

        self.n_cropped_notes = self.encoder_params['n_cropped_notes']
        self.encoder = Encoder(**self.encoder_params).cuda()
        self.decoder = Decoder(**self.decoder_params).cuda()

        self.use_cuda = use_cuda if torch.cuda.is_available() else False
        if self.use_cuda:
            self.cuda()
            self.current_device = torch.cuda.current_device()
        else:
            self.current_device = 'cpu'

        self.cel = nn.CrossEntropyLoss()
        self.bcel = nn.BCELoss()

        # TODO:
        #  Implement beta scaling (https://openreview.net/forum?id=Sy2fzU9gl)
        # self.kl_norm =

    # Source: https://github.com/AntixK/PyTorch-VAE
    # Method from: https://arxiv.org/pdf/1312.6114v10.pdf
    # Discussion: https://www.jeremyjordan.me/variational-autoencoders/
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

    def loss(self, mu, logvar, X, recon, beta):
        kl_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1), dim=0)
        # reconstruction_loss = F.mse_loss(X, recon)
        reconstruction_loss = self.bcel(
            recon,
            X
        )
        return kl_loss, reconstruction_loss, kl_loss * beta + reconstruction_loss

    def sample(self, num_samples):
        z = torch.randn((num_samples, self.z_dim), device=self.current_device)
        out = self.decode(z)
        return out

    def interpolate(self, x_s, x_t, length = 30, method = 'lerp'):    
        interpolator = 0
        if method == 'lerp':
            interpolator = torch.lerp
        elif method == 'slerp':
            zi = slerp

        # Encode enpoints
        print('encoding ...')
        mu_s, lv_s = self.encode(x_s)
        mu_t, lv_t = self.encode(x_t)
        print('reparameterizing ...')
        z_s = self.reparameterize(mu_s, lv_s)
        z_t = self.reparameterize(mu_t, lv_t)

        print('encoded shape: ', z_s.shape)

        # Interpolation by length L
        print('generating interpolation ...')
        z = []
        for i in range(length):
            weight = i / length
            zi = interpolator(z_s, z_t, weight)
            z.append(zi)

        print('decoding interpolation ...')
        import time
        start = time.time()
        # Decode z
        d = []
        i=0
        for zi in z:
            di = self.decode(zi)
            print(di.shape)
            d.append(di)
        # d = list(map(lambda zi: self.decode(zi), z))
        # print(d[0].shape)
        print(time.time() - start)
        
        # d = [D(z0),...,D(zL)]
        return d