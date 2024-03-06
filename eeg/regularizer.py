import torch
import torch.nn as nn
import numpy as np
import random

def torus_distance(a, b, embed_dim):
    a = a.view(embed_dim, -1, 1)
    b = b.view(embed_dim, 1, -1)
    return torch.linalg.norm(torch.min(torch.remainder(a - b, 10), torch.remainder(b-a, 10)), dim=0)


M1 = torch.tensor([[1, 0, 10],
                           [ 0, -1, 10],
                           [ 0, 0, 1]], dtype=torch.float32)
M2 = torch.tensor(np.array([[1, 0, -10],
                           [ 0, -1, 10],
                           [ 0, 0, 1]]), dtype=torch.float32)
M3 = torch.tensor(np.array([[1, 0, 0],
                           [ 0, 1, 10],
                           [ 0, 0, 1]]), dtype=torch.float32)
M4 = torch.tensor(np.array([[1, 0, 0],
                           [ 0, 1, -10],
                           [ 0, 0, 1]]), dtype=torch.float32)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

M1 = M1.to(device)
M2 = M2.to(device)
M3 = M3.to(device)
M4 = M4.to(device)


def klein_distance(a, b):
    padder = nn.ConstantPad1d((0, 1), 1)
    a = torch.remainder(a, 10)
    b = torch.remainder(b, 10)
    a = padder(a).transpose(0, 1)
    b = padder(b).transpose(0, 1)
    a = a.view(3, -1, 1)
    #b = b.view(3, 1, -1)
    return torch.min(torch.cat((torch.unsqueeze(torch.linalg.norm(a - b.view(3, 1, -1), dim = 0), 0),
                    torch.unsqueeze(torch.linalg.norm(a - torch.matmul(M1, b).view(3, 1, -1), dim=0), 0),
                    torch.unsqueeze(torch.linalg.norm(a - torch.matmul(M2, b).view(3, 1, -1), dim=0), 0),
                    torch.unsqueeze(torch.linalg.norm(a - torch.matmul(M3, b).view(3, 1, -1), dim=0), 0),
                    torch.unsqueeze(torch.linalg.norm(a - torch.matmul(M4, b).view(3, 1, -1), dim=0), 0),
                    torch.unsqueeze(torch.linalg.norm(a - torch.matmul(torch.matmul(M1, M3), b).view(3, 1, -1), dim=0), 0),
                    torch.unsqueeze(torch.linalg.norm(a - torch.matmul(torch.matmul(M1, M4), b).view(3, 1, -1), dim=0), 0),
                    torch.unsqueeze(torch.linalg.norm(a - torch.matmul(torch.matmul(M2, M3), b).view(3, 1, -1), dim=0), 0),
                    torch.unsqueeze(torch.linalg.norm(a - torch.matmul(torch.matmul(M2, M4), b).view(3, 1, -1), dim=0), 0))), dim=0)

def sphere_distance(a, b):
    return 5 * torch.nan_to_num(torch.acos(torch.inner(a, b)))

def small_gauss(x):
    return 100*torch.exp(-x**2/3)

def DoG(x):
    return 100*(torch.exp(-x**2/5) - torch.exp(-4*x**2))


class regularizer(object):
    def __init__(self, options, embed=None):
        
        self.hidden_size = options.hidden_size
        self.reg = torch.zeros(self.hidden_size, self.hidden_size).to(options.device)
        self.moduli = options.regularizer
        self.regpower = options.regpower
        self.changeembed = options.changeembed
        self.trainembed = options.trainembed
        self.embed = embed
        if self.trainembed:
            self.generate2()
        else:
            self.generate()
        self.reg = self.reg.to(options.device)
        self.regfunc()
        if options.permute:
            self.perm()
        if options.invert:
            self.invert()
        self.reg.to(options.device)
        
    def perm(self):
        trans1 = list(range(self.hidden_size**2))
        random.shuffle(trans1)
        self.reg = self.reg.view(-1)[torch.tensor(trans1)].view(self.hidden_size, self.hidden_size)
        
    
    def invert(self):
        m = torch.max(self.reg)
        self.reg = m - self.reg
        
    def regfunc(self):
        if self.regpower == 'square':
            self.reg = self.reg**2
        elif self.regpower =='none':
            pass
        elif self.regpower == 'gauss':
            self.reg = small_gauss(self.reg)
        elif self.regpower == 'DoG':
            self.reg = DoG(self.reg)
        elif self.regpower == 'mean':
            embed1 = 10*torch.rand(self.hidden_size, 2)
            m = torch.mean(DoG(torus_distance(embed1, embed1, 2))).item()
            self.reg = m * self.reg
        
    def generate(self):
        if self.moduli == 'none':
            pass
        elif self.moduli == 'torus':
            if self.changeembed:
                embed1 = 10*torch.rand(self.hidden_size, 2)
                embed2 = 10*torch.rand(self.hidden_size, 2)
            else:
                embed1 = 10*torch.rand(self.hidden_size, 2)
                embed2 = embed1
            self.reg = torus_distance(embed1, embed2, 2)
        elif self.moduli == 'klein':
            if self.changeembed:
                embed1 = 10*torch.rand(self.hidden_size, 2).to(device)
                embed2 = 10*torch.rand(self.hidden_size, 2).to(device)
            else:
                embed1 = 10*torch.rand(self.hidden_size, 2).to(device)
                embed2 = embed1
            self.reg, _ = klein_distance(embed1, embed2) #TODO edits here
        elif self.moduli == 'torus6':
            if self.changeembed:
                embed1 = 10*torch.rand(self.hidden_size, 6)
                embed2 = 10*torch.rand(self.hidden_size, 6)
            else:
                embed1 = 10*torch.rand(self.hidden_size, 6)
                embed2 = embed1
            self.reg = torus_distance(embed1, embed2, 6)
        elif self.moduli == 'sphere':
            if self.changeembed:
                embed1 = 10*torch.rand(self.hidden_size, 3)
                embed2 = 10*torch.rand(self.hidden_size, 3)
            else:
                embed1 = 10*torch.rand(self.hidden_size, 3)
                embed2 = embed1
            self.reg = sphere_distance(embed1, embed2)
        elif self.moduli == 'circle':
            if self.changeembed:
                embed1 = 10*torch.rand(self.hidden_size, 2)
                embed2 = 10*torch.rand(self.hidden_size, 2)
            else:
                embed1 = 10*torch.rand(self.hidden_size, 2)
                embed2 = embed1
            self.reg = sphere_distance(embed1, embed2)
        elif self.moduli == 'standard':
            self.reg = torch.ones(self.hidden_size, self.hidden_size)
        
    def generate2(self):
        if self.moduli == 'none':
            pass
        elif self.moduli == 'torus':
            self.reg = torus_distance(self.embed, self.embed, 2)
        elif self.moduli == 'klein':
            self.reg = klein_distance(self.embed, self.embed)
        elif self.moduli == 'torus6':
            self.reg = torus_distance(self.embed, self.embed, 6)
        elif self.moduli == 'sphere':
            self.reg = sphere_distance(self.embed, self.embed)
        elif self.moduli == 'circle':
            self.reg = sphere_distance(self.embed, self.embed)
        elif self.moduli == 'standard':
            self.reg = torch.ones(self.hidden_size, self.hidden_size)
            
            
            