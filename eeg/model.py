import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import regularizer

class Chrysalis(nn.Module):
    def __init__(self, options):
        super(Chrysalis, self).__init__()
        self.n_layers = 1
        self.device = options.device
        self.input_size = options.num_params
        self.hidden_size = options.hidden_size
        self.batch_size = options.batch_size
        self.regtype = options.regtype
        self.decode_position = options.decode_position
        self.sequence_length = options.sequence_length
        self.reg = None
        
        # RNN structure
        self.encoder = nn.Linear(self.input_size, self.hidden_size)
        self.rnn = nn.RNN(self.input_size, self.hidden_size, self.n_layers, batch_first = True, 
                          nonlinearity = options.activation, bias=False)
        self.decoder = nn.Linear(self.hidden_size, 11, bias=False)
        
        if self.decode_position == 'parallel':
            self.classifier = nn.Linear(11*int(self.sequence_length), 11)
        elif self.decode_position == 'all':
            self.classifier = nn.Linear(int(self.hidden_size) * int(self.sequence_length), 11)
        elif self.decode_position == 'attention':
            self.attn = nn.Linear(self.hidden_size, 1)
        # functions to run at init
        self.get_regularizer(options)
        
    def forward(self, w, x):
        hidden = F.tanh(self.encoder(w)[None])
        out, hidden = self.rnn(x, hidden)
        out = out.contiguous().view(self.batch_size, -1, self.hidden_size)
        if self.decode_position == 'final':
            out = self.decoder(out)[:,-1,:]
        elif self.decode_position == 'parallel':
            out = self.classifier(self.decoder(out).view(self.batch_size, -1))
        elif self.decode_position == 'all':
            out = self.classifier(out.view(self.batch_size, -1))
        elif self.decode_position == 'attention':
            imp = F.softmax(self.attn(out), dim = 1)
            out = self.decoder(out)
            out = torch.sum(imp*out, dim=1)
        return out, hidden
    def init_hidden(self):
        # For LSTMs, hidden state is a tuple:
        #hidden = (weight.new(self.n_layers, self.batch_size, self.n_hidden).zero_().to(self.device),
        #          weight.new(self.n_layers, self.batch_size, self.n_hidden).zero_().to(self.device))
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, self.batch_size, self.hidden_size).zero_().to(self.device)
        return hidden
    def get_regularizer(self, options):
        reg = regularizer.regularizer(options)
        self.reg = reg.reg
    def regularizer(self):
        return torch.mean(self.reg*torch.abs(self.rnn.weight_hh_l0)**int(self.regtype))

class LSTM(nn.Module):
    def __init__(self, options):
        super(LSTM, self).__init__()
        self.n_layers = 1
        self.device = options.device
        self.input_size = options.num_params
        self.hidden_size = options.hidden_size
        self.batch_size = options.batch_size
        self.regtype = options.regtype
        self.decode_position = options.decode_position
        self.sequence_length = options.sequence_length
        self.reg = None
        
        # RNN structure
        self.encoder_h = nn.Linear(self.input_size, self.hidden_size)
        self.encoder_c = nn.Linear(self.input_size, self.hidden_size)
        self.rnn = nn.LSTM(self.input_size, self.hidden_size, self.n_layers, batch_first = True, 
                          bias=False)
        self.decoder = nn.Linear(self.hidden_size, 11, bias=False)
        self.drop = nn.Dropout(.5)
        
        if self.decode_position == 'parallel':
            self.classifier = nn.Linear(11*self.sequence_length, 11)
        elif self.decode_position == 'all':
            self.classifier = nn.Linear(self.hidden_size * self.sequence_length, 11)
        elif self.decode_position == 'attention':
            self.attn = nn.Linear(self.hidden_size, 1)
        # functions to run at init
        self.get_regularizer(options)
        # functions to run at init
        self.get_regularizer(options)
        
    def forward(self, w, x):
        hidden = self.encoder_h(w)[None], self.encoder_c(w)[None]
        out, hidden = self.rnn(x, hidden)
        out = out.contiguous().view(self.batch_size, -1, self.hidden_size)
        out = self.drop(out)
        #out = self.decoder(out)
        if self.decode_position == 'final':
            out = self.decoder(out)[:,-1,:]
        elif self.decode_position == 'parallel':
            out = self.classifier(self.decoder(out).view(self.batch_size, -1))
        elif self.decode_position == 'all':
            out = self.classifier(out.view(self.batch_size, -1))
        elif self.decode_position == 'attention':
            imp = self.drop(F.softmax(self.attn(out), dim = 1))
            out = self.drop(self.decoder(out))
            out = torch.sum(imp*out, dim=1)
        return out, hidden
    def init_hidden(self):
        # For LSTMs, hidden state is a tuple:
        #hidden = (weight.new(self.n_layers, self.batch_size, self.n_hidden).zero_().to(self.device),
        #          weight.new(self.n_layers, self.batch_size, self.n_hidden).zero_().to(self.device))
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, self.batch_size, self.hidden_size).zero_().to(self.device)
        return hidden
    def get_regularizer(self, options):
        reg = regularizer.regularizer(options)
        self.reg = torch.cat((reg.reg, reg.reg, reg.reg, reg.reg), dim=0)
    def regularizer(self):
        return torch.mean(self.reg*torch.abs(self.rnn.weight_hh_l0)**int(self.regtype))