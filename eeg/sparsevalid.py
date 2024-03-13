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
        self.device = options.device
        self.batch_size = options.batch_size

        self.loss = []
        self.err = []


    def val_step(self):
        self.model.eval()
        h = self.model.init_hidden()
        av_loss = 0
        num_correct = 0
        for w, x, y in self.testloader:
            w, x, y = w.to(self.device), x.to(self.device), y.to(self.device)
            
            # This is to stop us backpropagating into infinity
            #h = tuple(each.data for each in h)
            
            output, h = self.model(w, x)
            loss = self.criterion(output, F.one_hot(y, num_classes=11).type(torch.float32))
            _, pred = torch.max(output, dim=1)
            
            num_correct += torch.sum(pred == y.data)
            
            av_loss += loss.item()*x.size(0)
        av_loss = av_loss/(len(self.testloader)*self.batch_size)
        acc = num_correct/(len(self.testloader)*self.batch_size)
        return av_loss, acc

    def test(self):
        tloss, acc = self.val_step()
        sparsity = (torch.sum(torch.where(torch.abs(self.model.rnn.weight_hh_l0.data) < .001, 1.0, 0.0))/(self.model.hidden_size**2)).item()
        print("Test Loss: {:.3f}. Accuracy: {:.3f}. Sparsity: {:.3f}".format(tloss, acc, sparsity))
        with open('Graphs/traindata/' + self.savefile + '.txt', 'a') as the_file:
            the_file.write("Test Loss: {:.3f}. Accuracy: {:.3f}. Sparsity: {:.3f}\n".format(tloss, acc, sparsity))

                
                
                
class PercentileSparse(object):
    def __init__(self, options, model, testloader, percentile, restore=False):
        self.options = options
        self.model = model
        self.hidden_size = options.hidden_size
        self.percentile = percentile
        self.batch_size = options.batch_size
        self.savefile = options.savefile
        self.testloader = testloader
        self.percentile = percentile
        lr = self.options.learning_rate
        self.device = options.device
        self.criterion = torch.nn.MSELoss()

        self.loss = []
        self.err = []
        
    def get_mask(self):
        prune.l1_unstructured(self.model.rnn, 'weight_hh_l0', amount = int(self.percentile*self.hidden_size**2/100))
        prune.remove(self.model.rnn, 'weight_hh_l0')

    
    def val_step(self):
        self.model.eval()
        
        self.get_mask()
        
        h = self.model.init_hidden()
        av_loss = 0
        num_correct = 0
        for w, x, y in self.testloader:
            w, x, y = w.to(self.device), x.to(self.device), y.to(self.device)
            
            # This is to stop us backpropagating into infinity
            #h = tuple(each.data for each in h)
            
            output, h = self.model(w, x)
            loss = self.criterion(output, F.one_hot(y, num_classes=11).type(torch.float32))
            _, pred = torch.max(output, dim=1)
            
            av_loss += loss.item()*x.size(0)
            num_correct += torch.sum(pred == y.data)
        av_loss = av_loss/(len(self.testloader)*self.batch_size)
        acc = num_correct/(len(self.testloader)*self.batch_size)


        return av_loss, acc

    def test(self):
        tloss, acc = self.val_step()
        #sparsity = (torch.sum(torch.where(self.model.rnn.weight_hh_l0.data == 0, 1.0, 0.0))/(self.model.hidden_size**2)).item()
        print('Percentile {}: Loss: {:.3f} Accuracy: {:.3f}'.format(
               self.percentile, tloss, acc))
        with open('Graphs/traindata/' + self.savefile + '.txt', 'a') as the_file:
               the_file.write('Percentile {}: Loss: {:.3f} Accuracy: {:.3f}\n'.format(
               self.percentile, tloss, acc))
