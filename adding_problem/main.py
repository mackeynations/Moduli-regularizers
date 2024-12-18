import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch.nn.utils.prune as prune
import time


import argparse
import regularizer

parser = argparse.ArgumentParser()
parser.add_argument('--save_dir',
                    default='models/',
                    help='directory to save trained models')
parser.add_argument('--n_epochs',
                    default=1500,
                    type=int,
                    help='number of training epochs')
parser.add_argument('--n_steps',
                    default=1000,
                    help='batches per epoch')
parser.add_argument('--batch_size',
                    default=32,
                    type=int,
                    help='number of trajectories per batch')
parser.add_argument('--seq_length',
                    default=50,
                    type=int,
                    help='number of steps in trajectory')
parser.add_argument('--learning_rate',
                    default=1e-4,
                    type=float,
                    help='gradient descent learning rate')
parser.add_argument('--nhid',
                    default=128, # works with 32
                    help='number of hidden components')
parser.add_argument('--RNN_type',
                    default='RNN',
                    help='RNN or LSTM')
parser.add_argument('--activation',
                    default='relu',
                    help='recurrent nonlinearity')
parser.add_argument('--weight_decay',
                    type=float,
                    default=1e-3,
                    help='strength of weight decay on recurrent weights')
parser.add_argument('--device',
                    default='cuda:1' if torch.cuda.is_available() else 'cpu',
                    help='device to use for training')
parser.add_argument('--regularizer',
                    default='none',
                    help='type of (moduli) regularizer applied. \n Try standard, torus, klein, circle, sphere, torus6, s3.')
parser.add_argument('--regpower',
                    default='square',
                    help='inhibitor function applied to distances. \n Try square, none, gauss, DoG, ripple, mean(only intended for standard reg)')
parser.add_argument('--permute', action='store_true',
                    default=False,
                    help='whether weights of regularizer are permuted')
parser.add_argument('--changeembed', action='store_true',
                    default=False,
                    help='whether the same embedding in the manifold is used for both input and output')
parser.add_argument('--regtype',
                    default=1,
                    type=int,
                    help='power weights are raised to. Input 1 for L1 regularization, 2 for L2 regularization')
parser.add_argument('--save',
                    default='models/model.pt')
parser.add_argument('--savefile',
                    default='_loss_sets')
parser.add_argument('--save_repo',
                    default='graphs/scaletests/',
                    help='Folder the save file goes into')
parser.add_argument('--invert', action='store_true',
                    default = False,
                    help='Use opposite inhibitor function')
parser.add_argument('--trainembed', action='store_true', default=False,
                    help = 'Whether embedding is registered, trained parameters. Not compatible with changeembed.')
parser.add_argument('--target_perc',
                    type=float,
                    default=95)
parser.add_argument('--do_lottery', action='store_true',
                    default=False)
parser.add_argument('--scaletest', action='store_true', default=False,
                    help='ignore, here for testing purposes')

args = parser.parse_args()




# Define hyperparameters
input_size = 2
hidden_size = int(args.nhid)
if args.scaletest:
    hidden_size *= 100
    args.nhid = hidden_size
output_size = 1
sequence_length = args.seq_length
batch_size = args.batch_size
learning_rate = args.learning_rate
num_epochs = 100
device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

# Define the Adding Problem Dataset
class AddingProblemDataset(Dataset):
    def __init__(self, num_samples, sequence_length):
        self.num_samples = num_samples
        self.sequence_length = sequence_length

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        values = np.random.rand(self.sequence_length, 1)
        indicators = np.zeros((self.sequence_length, 1))
        indices = np.random.choice(self.sequence_length, 2, replace=False)
        indicators[indices] = 1
        target = values[indices].sum()
        sample = np.hstack((values, indicators))
        return torch.tensor(sample, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)

# Define the Elman RNN model
class ElmanRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ElmanRNN, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
        self.activation = nn.Tanh() if args.activation == 'tanh' else nn.ReLU()
        
        self.moduli = args.regularizer
        
        #self.reg = None
        #self.get_regularizer(args)
        #self.regtype = args.regtype
        
        self.trainembed = args.trainembed
        if not self.trainembed:
            self.reg = regularizer.regularizer(args)
        else:
            if self.moduli == 'torus':
                self.embed = nn.Parameter(10*torch.rand(hidden_size, 2))
            elif self.moduli == 's3':
                emb = torch.randn(hidden_size, 4)
                emb = emb/torch.linalg.norm(emb, dim=1, keepdim=True)
                self.embed = nn.Parameter(emb)
                #raise NotImplementedError('need to adjust embeddings at every grad step before this will work')
            elif self.moduli == 'r3':
                self.embed = nn.Parameter(10*torch.randn(hidden_size, 3))
            elif self.moduli == 'r9':
                self.embed = nn.Parameter(10*torch.randn(hidden_size, 9))
            else:
                raise NotImplementedError
            self.reg = regularizer.regularizer(args, self.embed)

    def forward(self, x):
        h = torch.zeros(x.size(0), self.hidden_size).to(device)  # Initial hidden state
        for t in range(x.size(1)):
            input_and_hidden = torch.cat((x[:, t, :], h), dim=1)
            h = self.activation(self.i2h(input_and_hidden))
        output = self.h2o(h)
        return output
    
    def get_regularizer(self, args):
        self.reg = regularizer.regularizer(args)
        #self.reg = reg.reg
        #self.reg = self.reg.to(device)
        
    def regularizer(self):
        if self.trainembed:
            self.reg = regularizer.regularizer(args, self.embed)
            # For things like S3 we have to correct embeddings back onto the manifold, but as long as we only use R3 and torus we're clear to ignore that step
        return torch.mean(self.reg.reg*torch.abs(self.i2h.weight[:,self.input_size:])**int(args.regtype))

# Create dataset and dataloader
dataset = AddingProblemDataset(num_samples=10000, sequence_length=sequence_length)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Instantiate the model, loss function, and optimizer
model = ElmanRNN(input_size, hidden_size, output_size)
model = model.to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
params_to_prune = ((model.i2h, 'weight'), (model.h2o, 'weight'))

# Training loop
def main():
    start_train = time.time()
    torch.save(model.embed, 'embed_start.pt')
    is_first = True
    for epoch in range(num_epochs):
        total_loss = 0
        current_sparsity = 0
        if epoch > 1:
            prune.remove(model.i2h, 'weight')
            prune.remove(model.h2o, 'weight')
        if epoch > 0:
            current_sparsity = args.target_perc*(epoch)/(100*(num_epochs-1))
            prune.global_unstructured(params_to_prune, 
                                      pruning_method = prune.L1Unstructured,
                                      amount=current_sparsity)
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets.unsqueeze(1)) 
            total_loss += loss.item()
            loss += args.weight_decay*model.regularizer()
            loss.backward()
            if args.trainembed:
                model.embed.grad *= .99**epoch
                if is_first:
                    torch.save(model.embed.grad, 'embed_grad.pt')
                    is_first = False
                #if epoch > num_epochs/2:
                #    model.embed.requires_grad = False
            torch.nn.utils.clip_grad_norm_(model.parameters(), .5)
            optimizer.step()
            if args.trainembed and args.regularizer == 's3':
                model.embed = model.embed/torch.linalg.norm(model.embed, dim=1, keepdim=True)
    
        if True: #(epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(dataloader):.4f}, Sparsity: {current_sparsity}')
            with open(args.save_repo + args.savefile + '.txt', 'a') as the_file:
                the_file.write(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(dataloader):.4f}, Sparsity: {current_sparsity:.4f}\n')
                
    torch.save(model.embed, 'final_embed.pt')
    model.embed.requires_grad = False
    for epoch in range(10):
        total_loss = 0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets.unsqueeze(1)) 
            total_loss += loss.item()
            loss += args.weight_decay*model.regularizer()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), .5)
            optimizer.step()
            
        print(f'Stabilizer [{num_epochs+epoch+1}/{num_epochs+3}], Loss: {total_loss/len(dataloader):.4f}, Sparsity: {current_sparsity}')
        with open(args.save_repo + args.savefile + '.txt', 'a') as the_file:
            the_file.write(f'Stabilizer [{epoch+1}/{num_epochs}], Loss: {total_loss/len(dataloader):.4f}, Sparsity: {current_sparsity}\n')
    train_time = time.time() - start_train
                
    with open(args.save, 'wb') as f:
        torch.save(model.state_dict(), f)
            
    if args.do_lottery:
        print('Begin Lottery ticket training')
        args.trainembed = False
        with open(args.save, 'rb') as f:
            model2 = ElmanRNN(input_size, hidden_size, output_size)
            params_to_prune2 = ((model2.i2h, 'weight'), (model2.h2o, 'weight'))
            prune.global_unstructured(params_to_prune2,
                                      pruning_method = prune.L1Unstructured,
                                      amount=.5)
            wts = torch.load(f)
            #model_dict = model.state_dict()
            wts_trimmed = {k:v for k, v in wts.items() if k.endswith('mask')}
            model2.load_state_dict(wts_trimmed, strict=False)
        model2 = model2.to(device)
        model2.reg = model.reg
                
        optimizer2 = optim.Adam(model2.parameters(), lr=learning_rate)


                
        for epoch in range(num_epochs):
            total_loss = 0
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer2.zero_grad()
                outputs = model2(inputs)
                loss = criterion(outputs, targets.unsqueeze(1)) 
                total_loss += loss.item()
                loss += args.weight_decay*model2.regularizer()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model2.parameters(), .5)
                optimizer2.step()
                    
            if True: #(epoch + 1) % 10 == 0:
                print(f'Lottery [{epoch+1}/{num_epochs}], Loss: {total_loss/len(dataloader):.4f}, Sparsity: {current_sparsity}')
                with open(args.save_repo + args.savefile + '.txt', 'a') as the_file:
                    the_file.write(f'Lottery [{epoch+1}/{num_epochs}], Loss: {total_loss/len(dataloader):.4f}, Sparsity: {current_sparsity}\n')

    # Testing on a single sample
    test_sample, test_target = dataset[0]
    test_sample, test_target = test_sample.to(device), test_target.to(device)
    test_output = model(test_sample.unsqueeze(0))
    print(f'Test Input: {test_sample}')
    print(f'True Sum: {test_target.item()}, Predicted Sum: {test_output.item()}')
    print(f'Train time: {train_time}')
    with open(args.save_repo + args.savefile + '.txt', 'a') as the_file:
        the_file.write(f'Train time: {train_time}\n')

if __name__ == "__main__":
    main()