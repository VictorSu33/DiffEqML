import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class FCN(nn.Module):
    "Defines a standard fully-connected network in PyTorch"
    
    def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS):
        super().__init__()
        activation = nn.ReLU
        self.fcs = nn.Sequential(*[
                        nn.Linear(N_INPUT, N_HIDDEN),
                        activation()])
        self.fch = nn.Sequential(*[
                        nn.Sequential(*[
                            nn.Linear(N_HIDDEN, N_HIDDEN),
                            activation()]) for _ in range(N_LAYERS-1)])
        self.fce = nn.Linear(N_HIDDEN, N_OUTPUT)

    def forward(self, x):
        x = self.fcs(x)
        x = self.fch(x)
        x = self.fce(x)
        return x
    

def training_step(optimiser, f,t, t0, x0, model, lambda1=1, lambda2=1):
    '''note higher order problems need to be converted into first order vector problems'''
    optimiser.zero_grad()
    x = model(t)

    #dy_dt = torch.autograd.grad(y,t, torch.ones_like(y), create_graph=True)[0]
    #dy_dt = torch.autograd.functional.jacobian(model, t)
    dx_dt =  torch.func.vmap (torch.func.jacfwd (model), in_dims = (0,)) (t).squeeze()

    loss1 = lambda1 * torch.mean(torch.norm((dx_dt - f(x).squeeze()))**2) 
    loss2 = lambda2 * torch.norm(x0 - model(t0))**2
    loss = loss1 + loss2
    loss.backward()
    optimiser.step()
    return loss

