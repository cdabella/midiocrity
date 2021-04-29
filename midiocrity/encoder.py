import torch
import torch.nn as nn

class Encoder(nn.Module):
    '''
    Encoder consists of 3 stacked Bidirectional LSTM Layers followed by 4 Fully-Connected Layers.
    Input to LSTM is of size (batch size, seq_len, input_size) e.g. (128, 256, 1) for drums
    Output from LSTM is of size (seq_len, batch, num_directions*hidden_size)
    '''
    def __init__(self, z_dim, phrase_size=256, batch_size=16, hidden_size=32, num_layers=3, n_tracks=4, n_cropped_notes=130):
        super(Encoder, self).__init__()

        self.z_dim = z_dim
        self.phrase_size = phrase_size
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.n_tracks = n_tracks
        self.n_cropped_notes = n_cropped_notes

        for track_idx in range(self.n_tracks):
            setattr(
                self,
                f"track{track_idx}_lstm",
                torch.nn.LSTM(input_size=self.n_cropped_notes, hidden_size=self.hidden_size, num_layers=self.num_layers,
                              bidirectional=True, batch_first=True),
            )
            setattr(
                self,
                f"track{track_idx}_fc_z_mean",
                nn.Sequential(
                    nn.Linear(self.phrase_size * 2 * self.hidden_size, 128),
                    nn.ReLU(),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Linear(32, self.z_dim),
                    nn.ReLU()
                )
            )
            setattr(
                self,
                f"track{track_idx}_fc_logvar",
                nn.Sequential(
                    nn.Linear(self.phrase_size * 2 * self.hidden_size, 128),
                    nn.ReLU(),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Linear(32, self.z_dim),
                    nn.ReLU()
                )
            )

    '''
    Forward pass of encoder with pytorch based on musae encoders.py
    Input: tensor of size (batch, seq_len, sample)
    Output: tensor of size (batch, z_dim)
    '''
    def forward(self, input):
        track_z_means = []
        track_z_logvars = []
        for track_idx in range(self.n_tracks):
            track_input = input[track_idx]
            track_input = track_input.float()
            track_output, _ = getattr(self, f"track{track_idx}_lstm")(track_input)

            # Reshape, flattening each sample
            track_output = torch.flatten(track_output, start_dim=1)

            # 4 Fully Connected Layers for z_mean
            track_output_z_mean = getattr(self, f"track{track_idx}_fc_z_mean")(track_output)

            # 4 Fully Connected Layers for z_logvar
            track_output_z_logvar = getattr(self, f"track{track_idx}_fc_logvar")(track_output)

            track_z_means.append(track_output_z_mean)
            track_z_logvars.append(track_output_z_logvar)

        output_z_mean = torch.stack(track_z_means, dim=-1)
        output_z_logvar = torch.stack(track_z_logvars, dim=1)

        return output_z_mean, output_z_logvar
