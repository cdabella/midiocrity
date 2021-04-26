from midiocrity.data_unloader import unload_data
from midiocrity import encoder, decoder, discriminator
import torch


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
        X = batch[0, :, :, :, 0]
        X = torch.squeeze(X)
        # single-track input X should now be of size (batch, time, pitch)
        self.encoder = encoder.encoder(X.shape[2])
        output = self.encoder.forward(X)

        
if __name__ == "__main__":
    midiocrity = midiocrity()
    midiocrity.train()
