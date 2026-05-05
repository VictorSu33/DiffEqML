import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class Sin(nn.Module):
    def forward(self, x):
        return torch.sin(x)

class FCN(nn.Module):
    "Defines a standard fully-connected network in PyTorch"
    
    def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS, SIREN = False):
        super().__init__()
        activation = Sin if SIREN else nn.Tanh
        layers = [nn.Linear(N_INPUT, N_HIDDEN), activation()]
        for _ in range(N_LAYERS - 1):
            layers.extend([nn.Linear(N_HIDDEN, N_HIDDEN), activation()])
        layers.append(nn.Linear(N_HIDDEN, N_OUTPUT))
        
        self.net = nn.Sequential(*layers)

        if SIREN:
            self._init_siren()

    def _init_siren(self):
        for name, m in self.net.named_modules():
            if isinstance(m, nn.Linear):
                # SIREN initialization scheme (from the Sitzmann et al. paper)
                # Weights are drawn from U(-sqrt(6/n), sqrt(6/n))
                num_input = m.weight.size(1)
                with torch.no_grad():
                    m.weight.uniform_(-np.sqrt(6 / num_input), np.sqrt(6 / num_input))

    def forward(self, x):
        return self.net(x)

def data_loss(model, X_data, u_data):
    u_pred = model(X_data)
    return torch.mean((u_pred - u_data) ** 2)


def gradients(y, X):
    return torch.autograd.grad(
        y, X,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
        retain_graph=True
    )[0]


def physics_loss(model, X_col, residual_fn):
    X_col = X_col.clone().detach().requires_grad_(True)

    u = model(X_col)

    grad_u = gradients(u, X_col)   # shape (N, d)

    residual = residual_fn(X_col, u, grad_u)

    return torch.mean(residual ** 2)

from scipy.stats import qmc

def sample(dim,N, seed = None):
    if seed:
        sampler = qmc.LatinHypercube(d=dim, seed = seed)
    else:
        sampler = qmc.LatinHypercube(d=dim)
    sample = sampler.random(n=N)   # shape (N, dim)

    return torch.tensor(sample, dtype=torch.float32)

def train(epochs, optimizer, X_data, U_data, model, f, lambda_phys, dim, N, evo = False):
    losses = []
    losses_data = []
    losses_phys = []
    frame = []
    frame_collocation = []


    if evo:
        N_grid = 100
        x = torch.linspace(0, 1, N_grid)
        y = torch.linspace(0, 1, N_grid)
        X, Y = torch.meshgrid(x, y, indexing="ij")
        XY = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)
    
    for epoch in range(epochs):
        optimizer.zero_grad()

        X_col = sample(dim, N)

        loss_data = data_loss(model, X_data, U_data)
        loss_phys = physics_loss(model, X_col, f)

        loss = loss_data + lambda_phys * loss_phys

        losses.append(loss.item())
        losses_data.append(loss_data.item())
        losses_phys.append(loss_phys.item())
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(epoch, loss.item(), loss_data.item(), loss_phys.item())

            if evo:
                model.eval()
                with torch.no_grad():
                    U_pred = model(XY).reshape(N_grid, N_grid).detach().cpu().numpy()
                
                frame.append(U_pred)
                frame_collocation.append(X_col.detach().cpu().numpy())
                
                model.train()

    return losses, losses_data, losses_phys, frame, frame_collocation