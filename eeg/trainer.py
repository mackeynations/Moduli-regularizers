import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def one_hot_encode(arr, n_labels):
    # Initialize the the encoded array
    one_hot = np.zeros((arr.size, n_labels), dtype=np.float32)
    
    # Fill the appropriate elements with ones
    one_hot[np.arange(one_hot.shape[0]), arr.flatten()] = 1.
    
    # Finally reshape it to get back to the original array
    one_hot = one_hot.reshape((*arr.shape, n_labels))
    
    return one_hot

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""

    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

class Trainer(object):
    def __init__(self, options, model, dataloader, valloader):
        self.model = model
        self.device = options.device
        self.dataloader = dataloader
        self.valloader = valloader
        self.batch_size = options.batch_size
        self.savefile = options.savefile
        model.to(self.device)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=options.learning_rate)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.batchsize = options.batch_size
        self.epochs = options.n_epochs
        self.savefile = options.savefile
        self.weight_decay = options.weight_decay
        self.besterr = np.Inf 

        
    def train_step(self, e):
        self.model.train()
        #h = self.model.init_hidden()
        av_loss = 0
        counter = 0
        for w, x, y in self.dataloader:
            counter+=1
            w, x, y = w.to(self.device), x.to(self.device), y.to(self.device)
            
            # This is to stop us backpropagating into infinity
            #h = tuple(each.data for each in h)
            
            
            self.model.zero_grad()
            output, h = self.model(w, x)
            
            #Clone hidden values to prevent infinite backprop?
            #I've never fully understood this part
            h = repackage_hidden(h)
            
            loss = self.criterion(output, F.one_hot(y, num_classes=11).type(torch.float32)) + self.weight_decay*self.model.regularizer()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 5)
            self.optimizer.step()
            
            
            
            av_loss += loss.item()*x.size(0)
            if counter % 100 == 0:
                sparsity = (torch.sum(torch.where(torch.abs(self.model.rnn.weight_hh_l0.data) < .001, 1.0, 0.0))/(self.model.hidden_size**2)).item()
                print("Epoch: {}. Step: {}. Train Loss: {:.3f}. Sparsity: {:.3f}".format(e, counter, av_loss/(counter*self.batch_size), sparsity))
                #with open('/Graphs/traindata/' + self.savefile + '.txt', 'a') as thefile:
                #    thefile.write("Epoch: {}. Step: {}. Train Loss: {}. Sparsity: {}".format(e, counter, l, sparsity))
        return av_loss/len(self.dataloader)
    
    def val_step(self):
        self.model.eval()
        h = self.model.init_hidden()
        av_loss = 0
        num_correct = 0
        for w, x, y in self.valloader:
            w, x, y = w.to(self.device), x.to(self.device), y.to(self.device)
            
            # This is to stop us backpropagating into infinity
            #h = tuple(each.data for each in h)
            
            output, h = self.model(w, x)
            loss = self.criterion(output, F.one_hot(y, num_classes=11).type(torch.float32))
            
            _, pred = torch.max(output, dim=1)
            
            num_correct += torch.sum(pred == y.data)
            av_loss += loss.item()*x.size(0)
        av_loss = av_loss/(len(self.valloader)*self.batch_size)
        accuracy = num_correct/(len(self.valloader)*self.batch_size)
        if av_loss < self.besterr:
            self.besterr = av_loss
            torch.save(self.model.state_dict(), 'models/bestachieved' + self.savefile +  '.pt')
        return av_loss, accuracy
    
    def train(self):
        for e in range(self.epochs):
            l = self.train_step(e)/len(self.dataloader)
            with torch.no_grad():
                v, acc = self.val_step()
            sparsity = (torch.sum(torch.where(torch.abs(self.model.rnn.weight_hh_l0.data) < .001, 1.0, 0.0))/(self.model.hidden_size**2)).item()
            print('=' * 81)
            print("Epoch: {}. Train Loss: {:.3f}. Validation: {:.3f}. Accuracy: {:.3f}. Sparsity: {:.3f}".format(e, l, v, acc, sparsity))
            print('=' * 81)
            with open('Graphs/traindata/' + self.savefile + '.txt', 'a') as thefile:
                thefile.write("Epoch: {}. Loss: {}. Validation: {}. Accuracy: {:.3f}. Sparsity: {}\n".format(e, l, v, acc, sparsity))