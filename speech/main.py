import os
import argparse
#from comet_ml import Experiment
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import torch.nn.functional as F
import torchaudio
import numpy as np

import model1
import trainer
import regularizer
import sparsevalid

parser = argparse.ArgumentParser(prog='Moduli Speech Recognition')
parser.add_argument('--rnn_dim',
                    type=int,
                    default=512,
                    help='dimension of the hidden state of the RNN')
parser.add_argument('--n_layers',
                    type=int,
                    default=5,
                    help='number of stacked recurrent layers')
parser.add_argument('--RNN_type',
                    choices=['LSTM', 'RNN_tanh', 'RNN_relu', 'GRU'],
                    default='GRU',
                   help='underlying recurrent structure')
parser.add_argument('--bidirectional',
                    default=True)
parser.add_argument('--weight_decay',
                    type=float,
                    default=1e-1,
                    help='strength of weight decay on recurrent weights')
parser.add_argument('--regularizer',
                    choices=['none', 'torus', 'klein', 'circle', 'sphere', 'torus6'],
                    default='none',
                    help='type of (moduli) regularizer applied. \n Try standard, torus, klein, circle, sphere, torus6.')
parser.add_argument('--regpower',
                    choices=['square', 'none', 'gauss', 'DoG', 'mean'],
                    default='DoG',
                    help='inhibitor function applied to distances. \n Try square, none, gauss, DoG, mean')
parser.add_argument('--permute',
                    default=False,
                    action='store_true',
                    help='whether weights of regularizer are permuted')
parser.add_argument('--changeembed',
                    type=bool,
                    default=False,
                    #action='store_true',
                    help='whether the same embedding in the manifold is used for both input and output')
parser.add_argument('--regtype',
                    type=int,
                    default=1,
                    help='power weights are raised to. Input 1 for L1 regularization, 2 for L2 regularization')
parser.add_argument('--invert',
                    default = False,
                    type=bool,
                    #action='store_true',
                    help='Use opposite inhibitor function')
parser.add_argument('--trainembed',
                    default = False,
                    action='store_true',
                    help = 'Whether embedding is registered, trained parameters. Not compatible with changeembed.')
parser.add_argument('--seed_change',
                    type=int,
                    default=0)
parser.add_argument('--savefile',
                    default='_loss_sets')
parser.add_argument('--device',
                    default='cuda' if torch.cuda.is_available() else 'cpu')
parser.add_argument('--n_epochs',
                    type=int,
                    default=10)
args = parser.parse_args()



learning_rate = 5e-4
batch_size = 10
epochs = args.n_epochs
libri_train_set = "train-clean-100"
libri_test_set = "test-clean"

trainer.main(args, learning_rate, batch_size, epochs, libri_train_set, libri_test_set)
