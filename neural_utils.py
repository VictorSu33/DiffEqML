import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class FCN(nn.Module):
    "Defines a standard fully-connected network in PyTorch"
    
    def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS):
        super().__init__()
        activation = nn.Tanh
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
    

def training_step(optimiser, f,t, t_0, y_0, model, lambda1=1, lambda2=1):
    optimiser.zero_grad()
    y = model(t)
    dy_dt = torch.autograd.grad(y,t, torch.ones_like(t), create_graph=True)[0]
    loss = lambda1 * torch.mean((dy_dt - f(y))**2) +  lambda2 * (y_0 - model(t_0))**2
    loss.backward()
    optimiser.step()
    return loss


