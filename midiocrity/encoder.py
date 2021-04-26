import torch


class encoder:
    '''
    Encoder consists of 3 stacked Bidirectional LSTM Layers followed by 4 Fully-Connected Layers.
    Input to LSTM is of size (batch size, seq_len, input_size) e.g. (128, 256, 1) for drums
    Output from LSTM is of size (seq_len, batch, num_directions*hidden_size)
    '''
    def __init__(self, input_size, seq_len=256, hidden_size=32, num_layers=3):
        self.lstm = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                                  bidirectional=True, batch_first=True)
        self.fc_1 = torch.nn.Linear(seq_len * 2 * hidden_size, 128)
        self.fc_2 = torch.nn.Linear(128, 64)
        self.fc_3 = torch.nn.Linear(64, 32)
        self.fc_4 = torch.nn.Linear(32, 4)

    '''
    Build simple decoder with pytorch based on musae decoders.py
    '''
    def forward(self, input):
        output, (h_n, c_n) = self.lstm(input)

        # Reshape, flattening each sample
        output = torch.transpose(output, dim0=0, dim1=1)
        output = torch.flatten(output, start_dim=1)

        # 4 Fully Connected Layers
        output = self.fc_1(output)
        output = self.fc_2(output)
        output = self.fc_3(output)
        output = self.fc_4(output)
        return output
