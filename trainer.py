import torch
import torch.nn as nn
import numpy as np

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
        h = self.model.init_hidden()
        av_loss = 0
        counter = 0
        for x, y in self.dataloader:
            counter+=1
            x, y = x.to(self.device), y.to(self.device)
            
            # This is to stop us backpropagating into infinity
            #h = tuple(each.data for each in h)
            
            
            self.model.zero_grad()
            output, h = self.model(x, h)
            
            #Clone hidden values to prevent infinite backprop?
            #I've never fully understood this part
            h = h.data
            
            loss = self.criterion(output[:, -1, :], y) + self.weight_decay*self.model.regularizer()
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
        for x, y in self.valloader:
            x, y = x.to(self.device), y.to(self.device)
            
            # This is to stop us backpropagating into infinity
            #h = tuple(each.data for each in h)
            
            output, h = self.model(x, h)
            loss = self.criterion(output[:, -1, :], y)
            
            av_loss += loss.item()*x.size(0)
        av_loss = av_loss/len(self.valloader)
        if av_loss < self.besterr:
            self.besterr = av_loss
            torch.save(self.model.state_dict(), 'models/bestachieved' + self.savefile +  '.pt')
        return av_loss
    
    def train(self):
        for e in range(self.epochs):
            l = self.train_step(e)/len(self.dataloader)
            with torch.no_grad():
                v = self.val_step()
            sparsity = (torch.sum(torch.where(torch.abs(self.model.rnn.weight_hh_l0.data) < .001, 1.0, 0.0))/(self.model.hidden_size**2)).item()
            print(''.join(['=' for i in range(60)]))
            print("Epoch: {}. Train Loss: {:.3f}. Validation: {:.3f}. Sparsity: {:.3f}".format(e, l, v, sparsity))
            print(''.join(['=' for i in range(60)]))
            with open('Graphs/traindata/' + self.savefile + '.txt', 'a') as thefile:
                thefile.write("Epoch: {}. Loss: {}. Validation: {}. Sparsity: {}\n".format(e, l, v, sparsity))