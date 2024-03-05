# conda env mnisteeg
# sample terminal command:
# for i in {0..4}; do python3 main.py --invert True --regularizer torus --regtype 1 --regpower DoG --savefile newtorusl1dog$i; done

import numpy as np
import pandas as pd
import torch
import torch.cuda
import torch.nn as nn
import xarray as xr
import gc

#PYTORCH_NO_CUDA_MEMORY_CACHING=1
#export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

from model import Chrysalis
from trainer import Trainer
import sparsevalid
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.data.sampler import SubsetRandomSampler

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs',
                    default=10,
                    help='number of training epochs')
parser.add_argument('--batch_size',
                    default=20,
                    help='number of sequences used for each gradient step')
parser.add_argument('--learning_rate',
                    default=1e-4,
                    help='gradient descent learning rate')
parser.add_argument('--hidden_size',
                    default=2048,
                    help='number of grid cells')
parser.add_argument('--RNN_type',
                    default='RNN',
                    help='RNN or LSTM')
parser.add_argument('--activation',
                    default='relu',
                    help='recurrent nonlinearity')
parser.add_argument('--csv_file',
                   default='data/tocsv.csv',
                   help='path to data used')
parser.add_argument('--csv_classes',
                    default='data/sols.csv')
parser.add_argument('--window',
                    default=130,
                    help='sequence length for RNN')
parser.add_argument('--weight_decay',
                    default=1e-1,
                    help='strength of weight decay on recurrent weights')
parser.add_argument('--periodic',
                    default=False,
                    help='trajectories with periodic boundary conditions')
parser.add_argument('--device',
                    default='cuda' if torch.cuda.is_available() else 'cpu',
                    help='device to use for training')
parser.add_argument('--regularizer',
                    default='none',
                    help='type of (moduli) regularizer applied. \n Try standard, torus, klein, circle, sphere, torus6.')
parser.add_argument('--regpower',
                    default='square',
                    help='inhibitor function applied to distances. \n Try square, none, gauss, DoG, mean')
parser.add_argument('--permute',
                    default=False,
                    help='whether weights of regularizer are permuted')
parser.add_argument('--changeembed',
                    default=False,
                    help='whether the same embedding in the manifold is used for both input and output')
parser.add_argument('--regtype',
                    default=2,
                    help='power weights are raised to. Input 1 for L1 regularization, 2 for L2 regularization')
parser.add_argument('--savefile',
                    default='_loss_sets')
parser.add_argument('--invert',
                    default = False,
                    help='Use opposite inhibitor function')
parser.add_argument('--trainembed',
                    default = False,
                    help = 'Whether embedding is registered, trained parameters. Not compatible with changeembed.')
parser.add_argument('--num_params',
                    default=14)

options = parser.parse_args()

print(f'Using device: {options.device}')



if options.RNN_type == 'RNN':
    model = Chrysalis(options)
elif options.RNN_type == 'LSTM':
    # model = LSTM(options)
    raise NotImplementedError

# Put model on GPU if using GPU
model = model.to(options.device)

def one_hot_encode(arr, n_labels):
    # Initialize the the encoded array
    one_hot = np.zeros((arr.size, n_labels), dtype=np.float32)
    
    # Fill the appropriate elements with ones
    one_hot[np.arange(one_hot.shape[0]), arr.flatten()] = 1.
    
    # Finally reshape it to get back to the original array
    one_hot = one_hot.reshape((*arr.shape, n_labels))
    
    return one_hot

# Create data splits and load data
class EEGData(Dataset):
    def __init__(self, csv_file, sols_file, transform = None):
        super().__init__()
        self.data_framed = pd.read_csv(csv_file, header = None)
        self.data = np.asarray(self.data_framed.iloc[:,:])
        self.sols_framed = pd.read_csv(sols_file, sep = '\t', header = None, dtype = int)
        self.sols = np.asarray(self.sols_framed.iloc[:,:])
        self.transform = transform
        
    def __getitem__(self, index):
        index = int(14*index)
        if torch.is_tensor(index):
            index = index.tolist()
        x, y = (torch.from_numpy(self.data[index:index+14, :-1]).float().transpose(0,1), one_hot_encode(self.sols[index, -1], 11))
        if self.transform != None:
            x, y = self.transform(x), y
        return x, y
    
    def __len__(self):
        return len(self.data)//14
    
class ReNormEEG(object):
    def __init__(self, norm_file):
        self.df = pd.read_csv(norm_file, header = None)
        self.means = torch.from_numpy(np.asarray(self.df.iloc[:,1])).float()
        self.stds = torch.from_numpy(np.asarray(self.df.iloc[:,2])).float()
    def __call__(self, sample):
        sample = sample - self.means.view(1, -1)
        sample = sample/self.stds.view(1, -1)
        return sample
    
    
ren = ReNormEEG('data/prep.csv')

dataset = EEGData(options.csv_file, options.csv_classes, transform = ren)
l = len(dataset)
indices = list(range(l))
split = options.batch_size*int(.9*l/options.batch_size)
split2 = options.batch_size*int(.1*l/options.batch_size)
np.random.seed(66)
np.random.shuffle(indices)
train_indices, val_indices = indices[:split], indices[-split2:]
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

trainloader = DataLoader(dataset, batch_size = options.batch_size, 
                        num_workers=0, sampler = train_sampler)
valloader = DataLoader(dataset, batch_size = options.batch_size, 
                        num_workers=0, sampler = valid_sampler)

                   
                   

# Train                   
trainer = Trainer(options, model, trainloader, valloader)
trainer.train()

del model.reg
del trainloader

with torch.no_grad():
    torch.cuda.empty_cache()

                   
# Validate Sparseness
with torch.no_grad():
    model.load_state_dict(torch.load('models/bestachieved' + options.savefile +  '.pt'))
    model.rnn.flatten_parameters()
    #model.to('cpu')
    valid = sparsevalid.SparseValidator(options, model, valloader)
    valid.test()
    for p in range(20, 90, 20):
        #model.load_state_dict(torch.load('models/bestachieved' + options.savefile +  '.pt'))
        validp = sparsevalid.PercentileSparse(options, model, valloader, p)
        validp.test()
    #model.load_state_dict(torch.load('models/bestachieved' + options.savefile +  '.pt'))
    validp = sparsevalid.PercentileSparse(options, model, valloader, 90)
    validp.test()
    #model.load_state_dict(torch.load('models/bestachieved' + options.savefile +  '.pt'))
    validp = sparsevalid.PercentileSparse(options, model, valloader, 95)
    validp.test()

                            
#with open('Graphs/' + options.savefile, 'a') as the_file:
#    the_file.write(str(compute_sparsity(model.RNN.weight_hh_l0.data).item()) + '\n')
#print(compute_sparsity(model.RNN.weight_hh_l0.data).item())
                            

