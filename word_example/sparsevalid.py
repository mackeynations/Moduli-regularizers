# -*- coding: utf-8 -*-
import torch
import numpy as np
import torch.nn.utils.prune as prune
import torch.nn.functional as F

import os


class SparseValidator(object):
    def __init__(self, options, model, testloader, restore=False):
        self.options = options
        self.model = model
        self.hidden_size = options.hidden_size
        self.savefile = options.savefile
        self.testloader = testloader
        self.criterion = torch.nn.MSELoss()
        self.device = 'cpu'

        self.loss = []
        self.err = []


    def val_step(self):
        self.model.eval()
        h = self.model.init_hidden()
        av_loss = 0
        for x, y in self.testloader:
            x, y = x.to(self.device), y.to(self.device)
            
            # This is to stop us backpropagating into infinity
            #h = tuple(each.data for each in h)
            
            output, h = self.model(x, h)
            loss = self.criterion(output[:, -1, :], y)
            
            av_loss += loss.item()*x.size(0)
        av_loss = av_loss/len(self.testloader)
        return av_loss

    def test(self):
        tloss = self.val_step()
        sparsity = (torch.sum(torch.where(torch.abs(self.model.rnn.weight_hh_l0.data) < .001, 1.0, 0.0))/(self.model.hidden_size**2)).item()
        print("Test Loss: {:.3f}. Sparsity: {:.3f}".format(tloss, sparsity))
        with open('Graphs/traindata/' + self.savefile + '.txt', 'a') as the_file:
            the_file.write("Test Loss: {:.3f}. Sparsity: {:.3f}\n".format(tloss, sparsity))

                
                
                
class PercentileSparse(object):
    def __init__(self, options, model, testloader, percentile, restore=False):
        self.options = options
        self.model = model
        self.hidden_size = options.hidden_size
        self.percentile = percentile
        self.savefile = options.savefile
        self.testloader = testloader
        self.percentile = percentile
        lr = self.options.learning_rate
        self.device = 'cpu'
        self.criterion = torch.nn.MSELoss()

        self.loss = []
        self.err = []
        
    def get_mask(self):
        prune.l1_unstructured(self.model.rnn, 'weight_hh_l0', amount = self.percentile/100)

    
    def val_step(self):
        self.model.eval()
        
        self.get_mask()
        
        h = self.model.init_hidden()
        av_loss = 0
        for x, y in self.testloader:
            x, y = x.to(self.device), y.to(self.device)
            
            # This is to stop us backpropagating into infinity
            #h = tuple(each.data for each in h)
            
            output, h = self.model(x, h)
            loss = self.criterion(output[:, -1, :], y)
            
            av_loss += loss.item()*x.size(0)
        av_loss = av_loss/len(self.testloader)


        return av_loss

    def test(self):
        tloss = self.val_step()
        #sparsity = (torch.sum(torch.where(self.model.rnn.weight_hh_l0.data == 0, 1.0, 0.0))/(self.model.hidden_size**2)).item()
        print('Percentile {}: Loss: {:.3f}'.format(
               self.percentile, tloss))
        with open('Graphs/traindata/' + self.savefile + '.txt', 'a') as the_file:
               the_file.write('Percentile {}: Loss: {:.3f}\n'.format(
               self.percentile, tloss))
