import torch
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, z_dim, n_tracks, hidden_layers, output_dim):
        self.z_dim = z_dim
        self.n_tracks = n_tracks
        self.hidden_layers = hidden_layers
        self.output_dim


# flat version
def build_decoder():
    # Build simple decoder with pytorch based on musae decoders.py
    model = None
    return model