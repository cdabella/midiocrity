import torch
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, z_dim, phrase_size=256, batch_size=16, hidden_size=512, num_layers=3, n_tracks=4, n_cropped_notes=130):
        # input dims: batch x phrase_size x notes x track 
        super(Decoder, self).__init__()
        
        self.z_dim = z_dim
        self.n_tracks = n_tracks
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.phrase_size = phrase_size  # output_size
        self.batch_size = batch_size
        self.bidirectional = False
        self.num_directions = 2 if self.bidirectional else 1

        # MusAE uses the same initial states for each layer
        self.latent_to_h_0 = nn.Sequential(
            nn.Linear(z_dim, self.num_directions * self.hidden_size),
            nn.Tanh()
        )

        # self.latent_to_h_0 = nn.Sequential(
        #     nn.Linear(z_dim, self.num_layers * self.num_directions * self.hidden_size),
        #     nn.tanh()
        # )
        # self.latent_to_c_0 = nn.Sequential(
        #     nn.Linear(z_dim, self.num_directions * self.hidden_size),
        #     nn.Tanh()
        # )

        for track_idx in range(self.n_tracks):
            setattr(
                self, 
                f"track{track_idx}_lstm", 
                nn.LSTM(input_size=z_dim, hidden_size=hidden_size, num_layers=num_layers,
                        bidirectional=self.bidirectional, batch_first=True),
            )
            setattr(
                self,
                f"track{track_idx}_fc",
                nn.Sequential(
                    nn.Linear(hidden_size * self.num_directions, n_cropped_notes),
                    # nn.Softmax(dim=-1)
                )
            )

    def forward(self, x):
        #  batch x z_dim

        # # MusAE does not use c_0 and uses the same initial states for each LSTM layer
        # h_0 = self.latent_to_h_0(x).view(self.num_layers * self.num_directions, self.batch_size, self.hidden_size)
        # c_0 = self.latent_to_c_0(x).view(self.num_layers * self.num_directions, self.batch_size, self.hidden_size)
        
        h_0 = self.latent_to_h_0(x)
        h_0 = h_0.view(self.num_directions, self.batch_size, self.hidden_size)
        h_0 = h_0.expand(self.num_layers * self.num_directions, self.batch_size, self.hidden_size).contiguous()

        # c_0 = self.latent_to_c_0(x)
        # c_0 = c_0.view(self.num_directions, self.batch_size, self.hidden_size)
        # c_0 = c_0.expand(self.num_layers * self.num_directions, self.batch_size, self.hidden_size).contiguous()

        x = x.view(self.batch_size, 1, self.z_dim).expand(self.batch_size, self.phrase_size, self.z_dim)
        
        tracks = []
        for track_idx in range(self.n_tracks):
            track_x, _ = getattr(self, f"track{track_idx}_lstm")(x, (h_0, h_0))
            # track_x, _ = getattr(self, f"track{track_idx}_lstm")(x, (h_0, c_0))
            tracks.append(getattr(self, f"track{track_idx}_fc")(track_x))
        
        out = torch.stack(tracks, dim=-1)
        return out




# flat version
def build_decoder():
    # Build simple decoder with pytorch based on musae decoders.py
    model = None
    return model

def main():
    z_dim = 3
    batch_size = 4
    z = torch.rand(batch_size, z_dim)
    decoder = Decoder(3, phrase_size=12, batch_size=4, hidden_size=12, num_layers=3, n_tracks=4)
    # breakpoint()
    out = decoder(z)
    print(out.shape)

if __name__ == '__main__':
    main()