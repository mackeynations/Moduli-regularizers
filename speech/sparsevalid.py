# -*- coding: utf-8 -*-
import torch
import numpy as np
import torch.nn.utils.prune as prune
import torch.nn.functional as F
import torch.nn as nn
from processing import data_processing, GreedyDecoder, cer, wer


import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def test(model, device, test_loader, criterion):
    print('\nevaluating...')
    model.eval()
    test_loss = 0
    test_cer, test_wer = [], []
    #with model.test():
    with torch.no_grad():
        for i, _data in enumerate(test_loader):
            spectrograms, labels, input_lengths, label_lengths = _data 
            spectrograms, labels = spectrograms.to(device), labels.to(device)

            output = model(spectrograms)  # (batch, time, n_class)
            output = F.log_softmax(output, dim=2)
            output = output.transpose(0, 1) # (time, batch, n_class)

            loss = criterion(output, labels, input_lengths, label_lengths)
            test_loss += loss.item() / len(test_loader)

            decoded_preds, decoded_targets = GreedyDecoder(output.transpose(0, 1), labels, label_lengths)
            for j in range(len(decoded_preds)):
                test_cer.append(cer(decoded_targets[j], decoded_preds[j]))
                test_wer.append(wer(decoded_targets[j], decoded_preds[j]))


    avg_cer = sum(test_cer)/len(test_cer)
    avg_wer = sum(test_wer)/len(test_wer)
    #experiment.log_metric('test_loss', test_loss, step=iter_meter.get())
    #experiment.log_metric('cer', avg_cer, step=iter_meter.get())
    #experiment.log_metric('wer', avg_wer, step=iter_meter.get())

    #print('Test set: Average loss: {:.4f}, Average CER: {:4f} Average WER: {:.4f}\n'.format(test_loss, avg_cer, avg_wer))
    return test_loss, avg_cer, avg_wer
                
                
                
class PercentileSparse(object):
    def __init__(self, options, model, testloader, percentile, restore=False):
        self.options = options
        self.model = model
        self.hidden_size = options.rnn_dim
        self.percentile = percentile
        self.bidirectional = options.bidirectional
        self.savefile = options.savefile
        self.testloader = testloader
        self.n_layers = options.n_layers
        self.device = options.device #'cpu'
        self.criterion = nn.CTCLoss(blank=28).to(self.device)
        self.lens = model.birnn_layers[0].rnn.weight_hh_l0.shape
        self.area = self.lens[0]*self.lens[1]
        
    def get_mask(self):
        for i in range(self.n_layers):
            prune.l1_unstructured(self.model.birnn_layers[i].rnn, 'weight_hh_l0', amount = float(self.percentile)/100)
            prune.remove(self.model.birnn_layers[i].rnn, 'weight_hh_l0')
            if self.bidirectional:
                prune.l1_unstructured(self.model.birnn_layers[i].rnn, 'weight_hh_l0_reverse', amount = self.percentile)
                prune.remove(self.model.birnn_layers[i].rnn, 'weight_hh_l0_reverse')

    def run(self):
        criterion = nn.CTCLoss(blank=28).to(device)
        tloss, tcer, twer = test(self.model, device, self.testloader, criterion)
        #sparsity = (torch.sum(torch.where(self.model.rnn.weight_hh_l0.data == 0, 1.0, 0.0))/(self.model.hidden_size**2)).item()
        print('Percentile {}: Loss: {:.4f} CER: {:4f} WER: {:4f}'.format(
               self.percentile, tloss, tcer, twer))
        with open('Graphs/traindata/' + self.savefile + '.txt', 'a') as the_file:
               the_file.write('Percentile {}: Loss: {:.4} CER: {:4f} WER: {:4f}'.format(
               self.percentile, tloss, tcer, twer))