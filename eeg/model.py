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
        self.d = 2 if options.bidirectional else 1
        self.reg = None
        
        # RNN structure
        self.drop = nn.Dropout(0.5)
        self.encoder = nn.Linear(self.input_size, self.d*self.n_layers*self.hidden_size)
        self.rnn = nn.RNN(self.input_size, self.hidden_size, self.n_layers, batch_first = True, 
                          nonlinearity = options.activation, bias=False, bidirectional=options.bidirectional)
        self.decoder = nn.Linear(self.d*self.hidden_size, 11, bias=False)
        
        if self.decode_position == 'parallel':
            self.classifier = nn.Linear(11*int(self.sequence_length), 11)
        elif self.decode_position == 'all':
            self.classifier = nn.Linear(self.d*int(self.hidden_size) * int(self.sequence_length), 11)
        elif self.decode_position == 'attention':
            self.attn = nn.Linear(self.d*self.hidden_size, 1)
        elif self.decode_position == 'conv':
            self.conv = nn.Conv1d(self.d*self.hidden_size, 11, kernel_size = 7, padding='same')
            self.classifier = nn.Linear(11*int(self.sequence_length), 11)
        elif self.decode_position == 'conv_attn':
            self.conv = nn.Conv1d(self.d*self.hidden_size, 1, kernel_size = 7, padding = 'same')
        # functions to run at init
        self.get_regularizer(options)
        
    def forward(self, w, x):
        hidden = self.encoder_h(w).view(self.batch_size, self.hidden_size, self.d*self.n_layers).permute(2, 0, 1).contiguous()
        out, hidden = self.rnn(x, hidden)
        out = out.contiguous().view(self.batch_size, -1, self.d*self.hidden_size)
        out = self.drop(out)
        if self.decode_position == 'final':
            out = self.decoder(out)[:,-1,:]
        elif self.decode_position == 'parallel':
            out = self.classifier(F.elu(self.decoder(out).view(self.batch_size, -1)))
        elif self.decode_position == 'all':
            out = self.classifier(F.elu(out.view(self.batch_size, -1)))
        elif self.decode_position == 'attention':
            imp = F.softmax(self.attn(out), dim = 1)
            out = F.elu(self.decoder(out))
            out = torch.sum(imp*out, dim=1)
        elif self.decode_position == 'conv':
            out = out.permute(0, 2, 1)
            out = F.elu(self.conv(out))
            out = self.classifier(out.view(self.batch_size, -1))
        elif self.decode_position == 'conv_attn':
            imp = F.softmax(self.conv(out.permute(0, 2, 1)), dim = 1).permute(0, 2, 1)
            out = F.elu(self.decoder(out))
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
        self.n_layers = 2
        self.device = options.device
        self.input_size = options.num_params
        self.hidden_size = options.hidden_size
        self.batch_size = options.batch_size
        self.regtype = options.regtype
        self.decode_position = options.decode_position
        self.sequence_length = options.sequence_length
        self.d = 2 if options.bidirectional else 1
        self.reg = None
        
        # RNN structure
        self.encoder_h = nn.Linear(self.input_size, self.d*self.n_layers*self.hidden_size)
        self.encoder_c = nn.Linear(self.input_size, self.d*self.n_layers*self.hidden_size)
        self.rnn = nn.LSTM(self.input_size, self.hidden_size, self.n_layers, batch_first = True, 
                          bias=False, bidirectional = options.bidirectional)
        self.decoder = nn.Linear(self.d*self.hidden_size, 11, bias=False)
        self.drop = nn.Dropout(.5)
        
        if self.decode_position == 'parallel':
            self.classifier = nn.Linear(11*self.sequence_length, 11)
        elif self.decode_position == 'all':
            self.classifier = nn.Linear(self.d*self.hidden_size * self.sequence_length, 11)
        elif self.decode_position == 'attention':
            self.attn = nn.Linear(self.d*self.hidden_size, 1)
        elif self.decode_position == 'conv':
            self.conv = nn.Conv1d(self.d*self.hidden_size, 11, kernel_size = 7, padding='same')
            self.classifier = nn.Linear(11*int(self.sequence_length), 11)
        elif self.decode_position == 'conv_attn':
            self.conv = nn.Conv1d(self.d*self.hidden_size, 1, kernel_size = 7, padding = 'same')
        # functions to run at init
        self.get_regularizer(options)
        # functions to run at init
        self.get_regularizer(options)
        
    def forward(self, w, x):
        hidden = (self.encoder_h(w).view(self.batch_size, self.hidden_size, self.d*self.n_layers).permute(2, 0, 1).contiguous(),
                  self.encoder_c(w).view(self.batch_size, self.hidden_size, self.d*self.n_layers).permute(2, 0, 1).contiguous())
        out, hidden = self.rnn(x, hidden)
        out = out.contiguous().view(self.batch_size, -1, self.d*self.hidden_size)
        out = self.drop(out)
        #out = self.decoder(out)
        if self.decode_position == 'final':
            out = self.decoder(out)[:,-1,:]
        elif self.decode_position == 'parallel':
            out = F.elu(self.decoder(out).view(self.batch_size, -1))
            out = self.classifier(out)
        elif self.decode_position == 'all':
            out = self.classifier(F.elu(out.view(self.batch_size, -1)))
        elif self.decode_position == 'attention':
            imp = F.softmax(self.attn(out), dim = 1)
            out = F.elu(self.decoder(out))
            out = torch.sum(imp*out, dim=1)
        elif self.decode_position == 'conv':
            out = out.permute(0, 2, 1)
            out = F.elu(self.conv(out))
            out = self.classifier(out.view(self.batch_size, -1))
        elif self.decode_position == 'conv_attn':
            imp = F.softmax(self.conv(out.permute(0, 2, 1)), dim = 1).permute(0, 2, 1)
            out = F.elu(self.decoder(out))
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