import torch
import torch.nn as nn

class Encoder():
    '''
    Encoder consists of 3 stacked Bidirectional LSTM Layers followed by 4 Fully-Connected Layers.
    Input to LSTM is of size (batch size, seq_len, input_size) e.g. (128, 256, 1) for drums
    Output from LSTM is of size (seq_len, batch, num_directions*hidden_size)
    '''
    def __init__(self, input_size, seq_len=256, hidden_size=32, num_layers=3):
        self.lstm = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                                  bidirectional=True, batch_first=True)

        self.fc_z_mean = nn.Sequential(
            nn.Linear(seq_len * 2 * hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.ReLU()
        )

        self.fc_logvar = nn.Sequential(
            nn.Linear(seq_len * 2 * hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.ReLU()
        )

    '''
    Forward pass of encoder with pytorch based on musae encoders.py
    Input: tensor of size (batch, seq_len, sample)
    Output: tensor of size (batch, 1) -- track is currently 1 for just drums
    '''
    def forward(self, input):
        input = input.float()
        output, hidden = self.lstm(input)

        # Reshape, flattening each sample
        output = torch.flatten(output, start_dim=1)

        # 4 Fully Connected Layers for z_mean
        output_z_mean = self.fc_z_mean(output)

        # 4 Fully Connected Layers for z_logvar
        output_z_logvar = self.fc_logvar(output)

        return output_z_mean, output_z_logvar
