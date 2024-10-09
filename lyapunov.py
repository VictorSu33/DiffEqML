import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import neural_utils as neural
from mpl_toolkits import mplot3d

class Lyapunov_Known(nn.Module):

    '''
    Partition state vector and create subnetworks to avoid curse of dimensionality 

    Architecture solution options:
    Conv1D
    ModuleList
    Masking
    '''
    #create fully connected neural network with ReLU(?) to return only nonnegative values

    def __init__(self, N_INPUT, N_HIDDEN,  decomposition):
        #hidden should be a multiple of input?
        super().__init__()
        
        self.decomposition = decomposition

        self.subfuncs = nn.ModuleList([])
        if np.sum(decomposition) != N_INPUT:
            raise ValueError('Invalid state decomposition')

        #create subsystems
        sub_hidden = N_HIDDEN // len(self.decomposition)
        for i in range(len(self.decomposition)):
            self.subfuncs.append( nn.Sequential(nn.Linear(self.decomposition[i],sub_hidden), nn.ReLU(), nn.Linear(sub_hidden,sub_hidden), nn.ReLU()))

        self.fce = nn.Sequential(nn.Linear(N_HIDDEN, 1, bias=False),nn.Tanh())
        
    #try to eliminate for loops

    def forward(self, x):
        outputs = []
        if np.sum(self.decomposition) != x.shape[-1]:
            raise ValueError('Invalid state decomposition')
        sub_states = torch.split(x,self.decomposition, dim=-1)
        for i in range(len(sub_states)):
            outputs.append(self.subfuncs[i](sub_states[i]))

        x = torch.cat(outputs, dim=-1)
        x= self.fce(x)

        return x

class Lyapunov_Unkown(nn.Module):
    def __init__(self, N_INPUT, N_HIDDEN, d_max =1):
        #hidden should be a multiple of input?
        super().__init__()
        
        self.d_max = d_max

        self.fcs = nn.Linear(N_INPUT, N_HIDDEN)
        self.conv = nn.Sequential(nn.Conv1d(in_channels=N_HIDDEN, out_channels=N_HIDDEN, kernel_size=1, groups= N_HIDDEN // self.d_max), nn.ReLU())

        #is ReLU okay?
        self.fce = nn.Sequential(
                        nn.Linear(N_HIDDEN, 1),
                        nn.ReLU())
        
    def forward(self, x):
        x = self.fcs(x)
        x = x.unsqueeze(-1)
        x = self.conv(x).squeeze(-1)
        x= self.fce(x)

        return x

class Lyapunov_Loss(nn.Module):
    

    def __init__(self, loss_weight=1):
        super(Lyapunov_Loss, self).__init__()
        self.loss_weight= loss_weight

    def bound_loss(self, y_pred, upper, lower,zeroes):
        # boundary loss of zubov pde
        lower_loss = torch.mean(torch.minimum(y_pred - lower,zeroes)**2)
        upper_loss = torch.mean(torch.maximum(y_pred - upper,zeroes)**2)

        return lower_loss + upper_loss

    def gradient_loss(self,f_data, x,V):
        #gradient loss of zubov pde
        dV_dx = torch.func.vmap (torch.func.jacfwd(V), in_dims = (0,)) (x).squeeze()
        loss = torch.mean((torch.einsum('ij,ij->i',  dV_dx, f_data) + torch.norm(x, dim=1)**2)**2)
        return loss

    def forward(self, y_pred, upper,lower,zeroes,f_data, x,V):
        return self.loss_weight * self.bound_loss(y_pred, upper, lower,zeroes) + self.gradient_loss(f_data, x,V)
