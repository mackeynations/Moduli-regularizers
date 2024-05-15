# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
import random
import regularizer



class RNN(torch.nn.Module):
    def __init__(self, options, place_cells):
        super(RNN, self).__init__()
        self.Ng = options.Ng
        self.Np = options.Np
        self.sequence_length = options.sequence_length
        self.weight_decay = options.weight_decay
        self.regtype = int(options.regtype)
        self.device = options.device
        self.place_cells = place_cells
        self.embed_dim = 2
        self.moduli = options.regularizer
        

        # Input weights
        self.encoder = torch.nn.Linear(self.Np, self.Ng, bias=False)
        self.RNN = torch.nn.RNN(input_size=2,
                                hidden_size=self.Ng,
                                nonlinearity=options.activation,
                                bias=False)
        # Linear read-out weights
        self.decoder = torch.nn.Linear(self.Ng, self.Np, bias=False)
        
        self.softmax = torch.nn.Softmax(dim=-1)
        self.logsoftmax = torch.nn.LogSoftmax(dim=-1)
        
        self.trainembed = options.trainembed
        
        
        
        if self.trainembed == False:
            if self.moduli == 'torus':
                self.embed = 10*torch.rand(self.Ng, 2)
                self.reg = regularizer.regularizer(options, embed = self.embed)
            elif self.moduli == 'circle':
                self.embed = torch.randn(self.Ng, 2)
                self.embed = self.embed/torch.linalg.norm(self.embed, dim=1, keepdim=True)
                self.reg = regularizer.regularizer(options, embed = self.embed)
            elif self.moduli == 'sphere':
                self.embed = torch.randn(self.Ng, 3)
                self.embed = self.embed/torch.linalg.norm(self.embed, dim=1, keepdim=True)
                self.reg  = regularizer.regularizer(options, embed = self.embed)
            elif self.moduli == 'klein':
                self.embed = 10*torch.rand(self.Ng, 2)
                self.reg = regularizer.regularizer(options, embed = self.embed)
            else:
                self.reg = regularizer.regularizer(options)
        else:
            if self.moduli == 'torus':
                self.embed = nn.Parameter(10*torch.rand(self.Ng, 2))
            elif self.moduli == 's3':
                emb = torch.randn(self.Ng, 4)
                emb = emb/torch.linalg.norm(emb, dim=1, keepdim=True)
                self.embed = nn.Parameter(emb)
            else:
                raise NotImplementedError
            self.reg = regularizer.regularizer(options, self.embed)
            
        
        

    def g(self, inputs):
        '''
        Compute grid cell activations.
        Args:
            inputs: Batch of 2d velocity inputs with shape [batch_size, sequence_length, 2].

        Returns: 
            g: Batch of grid cell activations with shape [batch_size, sequence_length, Ng].
        '''
        v, p0 = inputs
        init_state = self.encoder(p0)[None]
        g,_ = self.RNN(v, init_state)
        return g
    

    def predict(self, inputs):
        '''
        Predict place cell code.
        Args:
            inputs: Batch of 2d velocity inputs with shape [batch_size, sequence_length, 2].

        Returns: 
            place_preds: Predicted place cell activations with shape 
                [batch_size, sequence_length, Np].
        '''
        place_preds = self.decoder(self.g(inputs))
        
        return place_preds


    def compute_loss(self, inputs, pc_outputs, pos, options):
        '''
        Compute avg. loss and decoding error.
        Args:
            inputs: Batch of 2d velocity inputs with shape [batch_size, sequence_length, 2].
            pc_outputs: Ground truth place cell activations with shape 
                [batch_size, sequence_length, Np].
            pos: Ground truth 2d position with shape [batch_size, sequence_length, 2].

        Returns:
            loss: Avg. loss for this training batch.
            err: Avg. decoded position error in cm.
        '''
        y = pc_outputs
        preds = self.predict(inputs)
        yhat = self.softmax(self.predict(inputs))
        yhat2 = self.logsoftmax(self.predict(inputs))
        #loss = -(y*torch.log(yhat)).sum(-1).mean()
        loss = -(y*yhat2).sum(-1).mean()
        #criterion = torch.nn.NLLLoss()
        #loss = criterion(yhat, y)

        # Weight regularization 
        if self.trainembed:
            self.reg = regularizer.regularizer(options, self.embed)
        loss += (self.weight_decay*self.reg.reg.to(self.device)*(torch.abs(self.RNN.weight_hh_l0)**self.regtype)).mean()
        #loss += .001*torch.mean(torch.abs(self.RNN.weight_ih_l0))
        #loss += .001*torch.mean(torch.abs(self.encoder.weight))
        #loss += .001*torch.mean(torch.abs(self.decoder.weight))

        
        # Compute decoding error
        pred_pos = self.place_cells.get_nearest_cell_pos(preds)
        err = torch.sqrt(((pos - pred_pos)**2).sum(-1))
        cat10 = (err >= .10).int().sum()
        cat20 = (err >= .20).int().sum()
        cat50 = (err >= .50).int().sum()
        err = err.mean()

        return loss, err, cat10, cat20, cat50