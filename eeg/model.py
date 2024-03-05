import torch
import torch.nn as nn
import numpy as np

# This class isn't going to use lattitude/longitude data in any way, and leave it to the model to learn those
# traits. Check out Butterfly for model with lat/long included
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
        self.reg = None
        
        # RNN structure
        #self.encoder = nn.Linear(self.input_size, self.hidden_size)
        self.rnn = nn.RNN(self.input_size, self.hidden_size, self.n_layers, batch_first = True, 
                          nonlinearity = options.activation, bias=False)
        self.decoder = nn.Linear(self.hidden_size, 11, bias=False)
        
        # functions to run at init
        self.get_regularizer(options)
        
    def forward(self, x, hidden):
        out, hidden = self.rnn(x, hidden)
        out = out.contiguous().view(self.batch_size, -1, self.hidden_size)
        out = self.decoder(out)
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

        
