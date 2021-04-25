from midiocrity.data_unloader import unload_data
from midiocrity import encoder, decoder, discriminator


class midiocrity():
    def __init__(self, **kwargs):
        print("Initializing encoder...")
        self.encoder = encoder.build_encoder()

        print("Initializing decoder...")
        self.decoder = decoder.build_decoder()

        print("Initializing discriminator...")
        self.z_discriminator = discriminator.build_gaussian_discriminator()


    def train(self):
        batch = unload_data()




if __name__ == "__main__":
    midiocrity = midiocrity()
    midiocrity.train()
