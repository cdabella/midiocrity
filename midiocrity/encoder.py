import torch

class encoder():
    def __init__(self, input_shape):
        self.lstm_1 = torch.nn.LSTM(input_shape[0] / 2, batch_first=True)
        self.lstm_2 = torch.nn.LSTM(128, batch_first=True)
        self.lstm_3 = torch.nn.LSTM(32, batch_first=True)
        self.fc_1 = torch.nn.Linear(1024, 128)
        self.fc_2 = torch.nn.Linear(128, 64)
        self.fc_3 = torch.nn.Linear(64, 32)
        self.fc_4 = torch.nn.Linear(32, 4)




    def forward(self, input):
        # Build simple decoder with pytorch based on musae decoders.py
        # Input is single channel X (dimensions time x pitch)

        # Reshape
        output = torch.reshape(input, (input.shape[0] / 2, input.shape[1] * 2))

        # 3 LSTM Layers
        output = self.lstm_1(output)
        output = self.lstm_2(output)
        output = self.lstm_3(output)

        # Reshape
        output = torch.flatten(output)

        # 3 Fully Connected Layers
        output = self.fc_1(output)
        output = self.fc_2(output)
        output = self.fc_3(output)
        return output