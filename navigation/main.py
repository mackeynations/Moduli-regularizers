# conda env: navigation
# sample terminal command:
# for i in {2..7}; do python3 main.py --invert True --regularizer torus --regtype 1 --regpower DoG --dog_ablation $i --savefile dogablation$i; done
# for i in {0..7}; do python3 main.py --regularizer standard --regtype 1 --regpower mean --savefile newmeanl1$i; done
# for i in {0..7}; do python3 main.py --savefile newnoreg$i; do python3 main.py --invert True --regularizer torus6 --regtype 1 --regpower DoG --savefile newtorus6l1dog$i; do python3 main.py --invert True --regularizer circle --regtype 1 --regpower DoG --savefile newcirclel1dog$i; done



import numpy as np
import torch.cuda
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F
import os
import time



from utils import generate_run_ID
from place_cells import PlaceCells
from trajectory_generator import TrajectoryGenerator
from model1 import RNN
from trainer import Trainer
import sparsevalid_total as sparsevalid

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--save_dir',
                    default='models/',
                    help='directory to save trained models')
parser.add_argument('--n_epochs',
                    default=30,
                    type=int,
                    help='number of training epochs')
parser.add_argument('--n_steps',
                    default=1000,
                    help='batches per epoch')
parser.add_argument('--batch_size',
                    default=200,
                    help='number of trajectories per batch')
parser.add_argument('--sequence_length',
                    default=50,
                    help='number of steps in trajectory')
parser.add_argument('--learning_rate',
                    default=1e-4,
                    type=float,
                    help='gradient descent learning rate')
parser.add_argument('--Np',
                    default=512,
                    help='number of place cells')
parser.add_argument('--Ng',
                    default=4096,
                    help='number of grid cells')
parser.add_argument('--place_cell_rf',
                    default=0.12,
                    help='width of place cell center tuning curve (m)')
parser.add_argument('--surround_scale',
                    default=2,
                    help='if DoG, ratio of sigma2^2 to sigma1^2')
parser.add_argument('--RNN_type',
                    default='RNN',
                    help='RNN or LSTM')
parser.add_argument('--activation',
                    default='relu',
                    help='recurrent nonlinearity')
parser.add_argument('--weight_decay',
                    type=float,
                    default=1e-1,
                    help='strength of weight decay on recurrent weights')
parser.add_argument('--DoG',
                    default=True,
                    help='use difference of gaussians tuning curves')
parser.add_argument('--periodic',
                    default=False,
                    help='trajectories with periodic boundary conditions')
parser.add_argument('--box_width',
                    default=2.2,
                    help='width of training environment')
parser.add_argument('--box_height',
                    default=2.2,
                    help='height of training environment')
parser.add_argument('--device',
                    default='cuda' if torch.cuda.is_available() else 'cpu',
                    help='device to use for training')
parser.add_argument('--regularizer',
                    default='none',
                    help='type of (moduli) regularizer applied. \n Try standard, torus, klein, circle, sphere, torus6, s3.')
parser.add_argument('--regpower',
                    default='square',
                    help='inhibitor function applied to distances. \n Try square, none, gauss, DoG, ripple, mean(only intended for standard reg)')
parser.add_argument('--permute',
                    default=False,
                    type=bool,
                    help='whether weights of regularizer are permuted')
parser.add_argument('--changeembed',
                    default=False,
                    help='whether the same embedding in the manifold is used for both input and output')
parser.add_argument('--regtype',
                    default=1,
                    type=int,
                    help='power weights are raised to. Input 1 for L1 regularization, 2 for L2 regularization')
parser.add_argument('--savefile',
                    default='_loss_sets')
parser.add_argument('--save_repo',
                    default='Graphs/trainembed/',
                    help='Folder the save file goes into')
parser.add_argument('--invert',
                    default = False,
                    help='Use opposite inhibitor function')
parser.add_argument('--trainembed',
                    default = False,
                    type=bool,
                    help = 'Whether embedding is registered, trained parameters. Not compatible with changeembed.')
parser.add_argument('--target_perc',
                    type=float,
                    default=90)


options = parser.parse_args()
options.run_ID = generate_run_ID(options)

print(f'Using device: {options.device}')

def compute_sparsity(x):
    #return torch.sum(x**2)/(torch.sum(torch.abs(x))**2)
    return torch.sum(torch.where(torch.abs(x) < .001, 1.0, 0.0))/(4096**2)



place_cells = PlaceCells(options)
if options.RNN_type == 'RNN':
    model = RNN(options, place_cells)
elif options.RNN_type == 'LSTM':
    # model = LSTM(options, place_cells)
    raise NotImplementedError

# Put model on GPU if using GPU
model = model.to(options.device)

trajectory_generator = TrajectoryGenerator(options, place_cells)

trainer = Trainer(options, model, trajectory_generator)

# Train
start_train = time.time()
trainer.train(options, n_epochs=options.n_epochs, n_steps=options.n_steps)
elapsed = time.time() - start_train
with open(options.save_repo + options.savefile + '.txt', 'a') as the_file:
    the_file.write('TOTAL TRAINING TIME: {}\n'.format(elapsed))
if options.trainembed:
    torch.save(model.reg, 'models/embeddings/' + options.savefile + '.pt')
    

# Validate Sparseness
trajectory_generator2 = TrajectoryGenerator(options, place_cells)
#model.load_state_dict(torch.load('models/bestachieved' + options.savefile + '.pt'))
valid = sparsevalid.SparseValidator(options, model, trajectory_generator2)
valid.test(options, n_epochs = 1, n_steps = 5)


ckpt_dir = os.path.join(options.save_dir, options.run_ID)
savefile = os.path.join(ckpt_dir, 'most_recent_model.pth')
model = RNN(options, place_cells).to(options.device)
myparams = ((model.encoder, 'weight'),
            (model.RNN, 'weight_hh_l0'),
            (model.RNN, 'weight_ih_l0'),
            (model.decoder, 'weight'))
prune.global_unstructured(myparams,
                          pruning_method = prune.L1Unstructured,
                          amount=.5)
wts = torch.load(savefile)
wts_trimmed = {k:v for k, v in wts.items() if k.endswith('mask')}
model.load_state_dict(wts_trimmed, strict=False)
model.RNN.flatten_parameters()
trainer = Trainer(options, model, trajectory_generator)

start_lottery = time.time()
trainer.train_sp(options, n_epochs=options.n_epochs, n_steps=options.n_steps)
elapsed = time.time() - start_lottery
with open(options.save_repo + options.savefile + '.txt', 'a') as the_file:
    the_file.write('TOTAL LOTTERY TIME: {}\n'.format(elapsed))

valid = sparsevalid.SparseValidator(options, model, trajectory_generator2)
valid.test(options, n_epochs = 1, n_steps = 5)