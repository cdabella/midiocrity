from data_unloader import unload_data
# import encoder, decoder, discriminator
import torch

from midiocrity_vae import MidiocrityVAE


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
        self.encoder = encoder.Encoder(X.shape[2])
        output_z_mean, output_z_logvar = self.encoder.forward(X)


if __name__ == "__main__":
    # midiocrity = midiocrity()
    midiocrity = MidiocrityVAE()
    # midiocrity.train()
    # INPUT -> (batch size, seq_len, input_size)
    # ENCODER Z OUTPUT -> (seq_len, batch, num_directions*hidden_size)
    # DECODER INPUT -> batch x phrase_size x notes x track 
    x_s = torch.rand(16, 256, 4)
    x_t = torch.rand(16, 256, 4)
    midiocrity.interpolate(x_s, x_t)
