# Creata a Heatmap of trained model
import numpy as np
import torch
import matplotlib.pyplot as plt
from model1 import RNN
from place_cells import PlaceCells
import scipy

class Options:
    pass

options = Options()

#%% Neccesary functions

def tor_points(U, V):
    c = 1
    a = 1
    X = 2*(c+a*np.cos(V))*np.cos(U)
    Y = 2*(c+a*np.cos(V))*np.sin(U)
    Z = a*np.sin(V)
    return X, Y, Z

def klein_points(U, V):
    x = np.cos(U)*(np.cos(.5*U)*(np.sqrt(2) + np.cos(V)) + np.sin(.5*U)*np.sin(V)*np.cos(V))
    y = np.sin(U)*(np.cos(.5*U)*(np.sqrt(2) + np.cos(V)) + np.sin(.5*U)*np.sin(V)*np.cos(V))
    z = -np.sin(.5*U)*(np.sqrt(2) + np.cos(V)) + np.cos(.5*U)*np.sin(V)*np.cos(V)
    return x, y, z

def invert(a):
    a = np.max(a) - a
    return (a - np.min(a))/(np.max(a) - np.min(a))

def normalize(a):
    return (a - np.min(a))/(np.max(a) - np.min(a))

#%% Load model and embedding

# Path to folder with model, options. Feel free to use a fancy os.path.join and a for loop to run this code for multiple models
path = 'models/steps_50_batch_200_RNN_512_relu_rf_012_DoG_True_periodic_False_lr_00001_weight_decay_01_torus'

interpolate = False   # Use interpolation for smoothness

n_pts = 8000 # number of points to interpolate at

options = np.load(path + '/options.npy', allow_pickle=True).item()
place_cells = PlaceCells(options)

model  = RNN(options, place_cells)

#############################
# this may not be neccesary for fully trained models and you can swap this with --> model.load_state_dict(torch.load(path + 'most_recent_model.pth'))
# 

loaded_state_dict = torch.load(path + '/most_recent_model.pth') # swap wiht epoch_n if you'd like

# Create a new state dict with the modified keys
new_state_dict = {}
for key, value in loaded_state_dict.items():
    new_key = key.replace('_orig', '')
    new_state_dict[new_key] = value

# Load the new state dict into your model
model.load_state_dict(new_state_dict, strict = False)

##################################


model.eval()

embedding = model.embed
hidden_weights = torch.abs(model.RNN.weight_hh_l0.detach())
moduli = options.regularizer

#%%

if moduli == 'torus':
    mypoint = embedding.T[:,0]
    
    xys = embedding.T 
    w = torch.tensor([hidden_weights[i, 0] for i in range(xys.shape[1])])
    
    if interpolate:
        # Scipy interpolate
        pts = 10*torch.rand(2, n_pts)
        
        w_terp = scipy.interpolate.griddata(xys.T, w, pts.T, method = 'nearest')
    
        n_xys = np.concatenate((xys.numpy(), pts.numpy()), axis = 1)
        w = np.concatenate((w, w_terp))
        
        bx = []
        by = []
        bz = []
    
        for i in range(n_xys.shape[1]):
            u, v = n_xys[:,i]*np.pi/5
            x, y, z = tor_points(u, v)
            bx.append(x)
            by.append(y)
            bz.append(z)
    else:
        n_xys = xys.numpy()
        w = w.numpy()
        
        bx = []
        by = []
        bz = []
        
        for i in range(n_xys.shape[1]):
            u, v = n_xys[:,i]*np.pi/5
            x, y, z = tor_points(u, v)
            bx.append(x)
            by.append(y)
            bz.append(z)
    
    fig = plt.figure(figsize=(6, 12))  # Adjust the figure size to ensure both plots have a good aspect ratio
    
    # Add a 2D subplot in the first position
    ax1 = fig.add_subplot(2, 1, 1)  # 2 rows, 1 column, 1st subplot
    ax1.scatter(n_xys[0,:], n_xys[1,:], s=1, c = normalize(w), cmap='jet')
    ax1.set_title('Torus', fontsize = 18)
    ax1.axis('off')  # Turn off the axis
    
    # Add a 3D subplot in the second position
    ax2 = fig.add_subplot(2, 1, 2, projection='3d')  # 2 rows, 1 column, 2nd subplot (3D)
    ax2.scatter(bx, by, bz, s=1, c = normalize(w), cmap='jet')
    ax2.set_xlim(-4, 4)
    ax2.set_ylim(-4, 4)
    ax2.set_zlim(-2, 2)
    ax2.set_axis_off()
    
elif moduli == 'circle':
    mypoint = embedding.T[:,0]
    
    xys = embedding.T
    
    zs = xys[0].numpy() + xys[1].numpy()
    
    w = torch.tensor([hidden_weights[i, 0] for i in range(xys.shape[1])])
    
    if interpolate:
        pts = torch.randn(2, n_pts)
        pts = pts/torch.linalg.norm(pts, dim=1, keepdim=True)
        
        w_terp = scipy.interpolate.griddata(xys.T, w, pts.T, method = 'nearest')
        n_xys = np.concatenate((xys.numpy(), pts.numpy()), axis = 1)
        w = np.concatenate((w, w_terp))
    else:
        n_xys = xys.numpy()
        w = w.numpy()
    
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.scatter(n_xys[0], n_xys[1], zs, s = 1, alpha = 1, c = normalize(w), cmap = 'jet')
    ax.set_title('Circle', fontsize = 18)
    ax.set_axis_off()
    
elif moduli == 'sphere':
    mypoint = embedding.T[:,0]
    
    xys = embedding.T 
    w = torch.tensor([hidden_weights[i, 0] for i in range(xys.shape[1])])
    if interpolate:
        pts = torch.rand(3, n_pts)
        pts = pts/torch.linalg.norm(pts, dim=1, keepdim=True)
        
        w_terp = scipy.interpolate.griddata(xys.T, w, pts.T, method = 'nearest')
        n_xys = np.concatenate((xys.numpy(), pts.numpy()), axis = 1)
        w = np.concatenate((w, w_terp))
    else:
        n_xys = xys.numpy()
        w = w.numpy()
    
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.scatter(n_xys[0], n_xys[1], n_xys[1], s = 1, alpha = 1, c = normalize(w), cmap = 'jet')
    ax.set_title('Sphere', fontsize = 18)
    ax.set_axis_off()
    
elif moduli == 'klein':
    mypoint = embedding.T[:,0]
    
    xys = embedding.T 
    w = torch.tensor([hidden_weights[i, 0] for i in range(xys.shape[1])])
    
    if interpolate:
        # Scipy interpolate
        pts = 10*torch.rand(2, n_pts)
        
        w_terp = scipy.interpolate.griddata(xys.T, w, pts.T, method = 'nearest')
    
        n_xys = np.concatenate((xys.numpy(), pts.numpy()), axis = 1)
        w = np.concatenate((w, w_terp))
        
        bx = []
        by = []
        bz = []
    
        for i in range(n_xys.shape[1]):
            u, v = n_xys[:,i]*np.pi/5
            x, y, z = klein_points(u, v)
            bx.append(x)
            by.append(y)
            bz.append(z)
    else:
        n_xys = xys.numpy()
        w = w.numpy()
        
        bx = []
        by = []
        bz = []
        
        for i in range(n_xys.shape[1]):
            u, v = n_xys[:,i]*np.pi/5
            x, y, z = klein_points(u, v)
            bx.append(x)
            by.append(y)
            bz.append(z)
            
    fig = plt.figure(figsize=(6, 12))  # Adjust the figure size to ensure both plots have a good aspect ratio
    
    # Add a 2D subplot in the first position
    ax1 = fig.add_subplot(2, 1, 1)  # 2 rows, 1 column, 1st subplot
    ax1.scatter(n_xys[0,:], n_xys[1,:], s=1, c = normalize(w), cmap='jet')
    ax1.set_title('Klein Bottle', fontsize = 18)
    ax1.axis('off')  # Turn off the axis
    
    # Add a 3D subplot in the second position
    ax2 = fig.add_subplot(2, 1, 2, projection='3d')  # 2 rows, 1 column, 2nd subplot (3D)
    ax2.scatter(bx, by, bz, s=1, c = normalize(w), cmap='jet')
    ax2.set_xlim(-4, 4)
    ax2.set_ylim(-4, 4)
    ax2.set_zlim(-2, 2)
    ax2.set_axis_off()