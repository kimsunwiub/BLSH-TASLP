import torch.nn as nn
import torch

def initialize_weights(network):
    """ Init weights with Xavier initialization """
    for name, param in network.named_parameters():
        if 'bias' in name:
            torch.nn.init.constant_(param, 0.0)
        elif 'weight' in name:
            torch.nn.init.xavier_normal_(param)

class RNN_model(nn.Module):
    def __init__(self, hidden_size, num_layers, stft_features):
        super(RNN_model, self).__init__()
        self.rnn = nn.LSTM(
            input_size=stft_features, hidden_size=hidden_size, 
            num_layers=num_layers, batch_first=True, bidirectional=True)
        self.dnn = nn.Linear(hidden_size*2, stft_features)
#         self.dnn = nn.Linear(hidden_size, stft_features)
        self.sigmoid = nn.Sigmoid()
        initialize_weights(self)

    def forward(self, x):
        (batch_size, seq_len, num_features) = x.shape
        rnn_out, hn = self.rnn(x) 
        _, _, hidden_size = rnn_out.shape
        # x: (seq_len, batch, hidden_size)
        # hn: (num_layers, batch, hidden_size)
        
        rnn_out = rnn_out.reshape(batch_size*seq_len, hidden_size)
        rnn_out = self.dnn(rnn_out)
        rnn_out = self.sigmoid(rnn_out)
        rnn_out = rnn_out.reshape(batch_size, seq_len, num_features)
        return rnn_out
    
class FC3_model(nn.Module):
    def __init__(self, hidden_size, stft_features):
        super(FC3_model, self).__init__()
        self.dnn1 = nn.Linear(stft_features, hidden_size)
        self.dnn2 = nn.Linear(hidden_size, hidden_size)
        self.dnn3 = nn.Linear(hidden_size, hidden_size)
        self.dnn4 = nn.Linear(hidden_size, stft_features)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        initialize_weights(self)

    def forward(self, x):
        out = self.relu(self.dnn1(x))
        out = self.relu(self.dnn2(out))
        out = self.relu(self.dnn3(out))
        out = self.sigmoid(self.dnn4(out))
        return out
    
class FC2_model(nn.Module):
    def __init__(self, hidden_size, stft_features):
        super(FC2_model, self).__init__()
        self.dnn1 = nn.Linear(stft_features, hidden_size)
        self.dnn2 = nn.Linear(hidden_size, hidden_size)
        self.dnn3 = nn.Linear(hidden_size, stft_features)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        initialize_weights(self)
        self.bn1 = nn.BatchNorm1d(num_features=hidden_size)
        self.bn2 = nn.BatchNorm1d(num_features=hidden_size)

    def forward(self, x):        
        out = self.relu(self.bn1(self.dnn1(x).transpose(1, 2)).transpose(1, 2))
        out = self.relu(self.bn2(self.dnn2(out).transpose(1, 2)).transpose(1, 2))
        out = self.sigmoid(self.dnn3(out))
        return out