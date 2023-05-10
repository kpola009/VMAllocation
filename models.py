import torch
from torch import nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LSTMForecaster(nn.Module):

    def __init__(self, n_features, n_hidden, n_outputs, sequence_len, n_lstm_layers=1, n_deep_layers=10, dropout=0.2):

        '''

        n_features: number of input features
        n_hidden: number of neurons in each hidden layer
        n_outputs: number of outputs to predict for each training example
        n_lstm_layers: number of lstm layers
        n_deep_layers: number of hidden dense layers after the lstm layers
        sequence_len: number of steps to look back at for prediction

        '''

        super().__init__()

        self.n_lstm_layers = n_lstm_layers
        self.nhid = n_hidden

        self.lstm = nn.LSTM(n_features,
                            n_hidden,
                            num_layers=n_lstm_layers,
                            batch_first=True)

        self.fc1 = nn.Linear(n_hidden * sequence_len, n_hidden)

        self.dropout = nn.Dropout(p=dropout)

        dnn_layers = []
        for i in range(n_deep_layers):
            if i == n_deep_layers - 1:
                dnn_layers.append(nn.ReLU())
                dnn_layers.append(nn.Linear(n_hidden, n_outputs))

            else:
                dnn_layers.append(nn.ReLU())
                dnn_layers.append(nn.Linear(n_hidden, n_hidden))
                if dropout:
                    dnn_layers.append(nn.Dropout(p=dropout))

        self.dnn = nn.Sequential(*dnn_layers)

    def forward(self, x):

        hidden_state = torch.zeros(self.n_lstm_layers, x.shape[0], self.nhid).to(device)
        cell_state = torch.zeros(self.n_lstm_layers, x.shape[0], self.nhid).to(device)

        self.hidden = (hidden_state, cell_state)

        x, h = self.lstm(x, self.hidden)
        x = self.dropout(x.contiguous().view(x.shape[0], -1))
        x = self.fc1(x)
        x = self.dnn(x)

        return x