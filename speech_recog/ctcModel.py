import torch
import torch.nn as nn
from cca import opt_layer

class CTCLstmProx(nn.Module):
    '''lstm based model for metaprox'''
    def __init__(self, config):
        super(CTCLstmProx, self).__init__()

        input_size_v1 = config['input_size_v1']
        input_size_v2 = config['input_size_v2']
        hidden_size = config['hidden_size']
        num_layers = config['num_layers']
        bidirection = config['bidirectional']
        self.proj_k = config['proj_k']
        num_classes = config['num_classes']
        self.device = config['device']
        self.avg = config['avg_logit']

        self.lstm1 = nn.LSTM(input_size=input_size_v1,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            bidirectional=bidirection)
    
        self.lstm2 = nn.LSTM(input_size=input_size_v2,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            bidirectional=bidirection)
    
        self.fc1 = nn.Linear(hidden_size, self.proj_k)
        self.fc2 = nn.Linear(hidden_size, self.proj_k)
        self.prox = opt_layer.apply

        self.fc = nn.Linear(self.proj_k, num_classes)
        self.fc_ = nn.Linear(self.proj_k, num_classes)

    def forward(self, v1, v2, alpha=.1):
        out_lstm1 = self.lstm1(v1)[0]
        out_lstm2 = self.lstm2(v2)[0]

        out_fc1 = self.fc1(out_lstm1)
        out_fc2 = self.fc2(out_lstm2)

        seq_len, batch_size, num_feat = out_fc1.shape

        if alpha == 0:
            out_v1, out_v2 = out_fc1, out_fc2
        else:
            in_prox = torch.cat((out_fc1.view(-1, num_feat), out_fc2.view(-1, num_feat)), dim=0)
            out_prox = self.prox(in_prox, alpha, self.proj_k, self.device)
            out_v1, out_v2 = torch.split(out_prox, out_prox.shape[0]//2, dim=0)
        
        prob1 = self.fc(out_v1.view(seq_len, batch_size, num_feat))
        prob2 = self.fc_(out_v2.view(seq_len, batch_size, num_feat))
        prob = (prob1 + prob2) if self.avg else prob1
        log_prob = nn.functional.log_softmax(prob, dim=2)
        return out_fc1, out_fc2, out_v1, out_v2, log_prob


