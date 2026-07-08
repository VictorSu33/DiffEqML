import os
from pathlib import Path
from typing import Optional
import torch
import torch.nn as nn
import numpy as np
import copy

class Sin(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x)

class FCN(nn.Module):
    """Fully Connected Network with optional SIREN initialization.
    
    Args:
        N_INPUT: number of input features
        N_OUTPUT: number of output features
        N_HIDDEN: number of hidden units per layer
        N_LAYERS: total number of layers (including output layer)
        SIREN: whether to use SIREN initialization and sine activations
    """
    
    def __init__(self, N_INPUT: int, N_OUTPUT: int, N_HIDDEN: int, N_LAYERS: int, SIREN: bool = False) -> None:
        super().__init__()
        activation = Sin if SIREN else nn.Tanh
        layers = [nn.Linear(N_INPUT, N_HIDDEN), activation()]
        for _ in range(N_LAYERS - 1):
            layers.extend([nn.Linear(N_HIDDEN, N_HIDDEN), activation()])
        layers.append(nn.Linear(N_HIDDEN, N_OUTPUT))
        
        self.net = nn.Sequential(*layers)

        if SIREN:
            self._init_siren()

    def _init_siren(self) -> None:
        for name, m in self.net.named_modules():
            if isinstance(m, nn.Linear):
                # SIREN initialization scheme (from the Sitzmann et al. paper)
                # Weights are drawn from U(-sqrt(6/n), sqrt(6/n))
                num_input = m.weight.size(1)
                with torch.no_grad():
                    m.weight.uniform_(-np.sqrt(6 / num_input), np.sqrt(6 / num_input))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        Args:
            x: input tensor of shape (N, N_INPUT)
        Returns:
            u: output tensor of shape (N, N_OUTPUT)
        """
        return self.net(x)

def data_loss(model: nn.Module, X_data: torch.Tensor, u_data: torch.Tensor) -> torch.Tensor:
    u_pred = model(X_data)
    return torch.mean((u_pred - u_data) ** 2)


def gradients(y: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(
        y, X,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
        retain_graph=True
    )[0]


def physics_loss(model: nn.Module, X_col: torch.Tensor, residual_fn: callable) -> torch.Tensor:
    X_col = X_col.clone().detach().requires_grad_(True)

    u = model(X_col)

    grad_u = gradients(u, X_col)   # shape (N, d)

    residual = residual_fn(X_col, u, grad_u)

    return torch.mean(residual ** 2)

from scipy.stats import qmc

def identity_map(x: torch.Tensor) -> torch.Tensor:
    return x


class Callback:
    """Base callback class for training hooks."""
    
    def on_epoch_start(self, epoch: int) -> None:
        pass
    
    def on_epoch_end(self, epoch: int, loss: float, loss_data: float, loss_phys: float) -> None:
        pass
    
    def on_train_end(self) -> None:
        pass
    
    def get_result(self):
        """Return any results collected by this callback."""
        return None


class LoggingCallback(Callback):
    """Logs training progress at regular intervals."""
    
    def __init__(self, log_interval: int = 100):
        self.log_interval = log_interval
    
    def on_epoch_end(self, epoch: int, loss: float, loss_data: float, loss_phys: float) -> None:
        if epoch % self.log_interval == 0:
            print(f"Epoch {epoch:5d} | Total Loss: {loss:.4e} | Data Loss: {loss_data:.4e} | Physics Loss: {loss_phys:.4e}")


class CheckpointCallback(Callback):
    """Saves model checkpoints when loss improves."""
    
    def __init__(self):
        self.best_loss = float('inf')
        self.checkpoint_data = None
    
    def on_epoch_end(self, epoch: int, loss: float, loss_data: float, loss_phys: float) -> None:
        if loss < self.best_loss:
            self.best_loss = loss
            # Store checkpoint data to be populated by trainer
            self.checkpoint_data = {
                'epoch': epoch,
                'loss': loss
            }
    
    def set_checkpoint_data(self, model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, loss: float) -> None:
        """Called by trainer to populate checkpoint with model/optimizer state."""
        if loss < self.best_loss:
            self.best_loss = loss
            self.checkpoint_data = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss
            }
    
    def get_result(self):
        return self.checkpoint_data


class FramesCallback(Callback):
    """Captures model predictions on a grid for visualization."""
    
    def __init__(self, N_grid: int = 100, domain_map: callable = identity_map, 
                 save_interval: int = 100, device: str = 'cpu'):
        self.N_grid = N_grid
        self.domain_map = domain_map
        self.save_interval = save_interval
        self.device = device
        self.frames = []
        self.grid = None
    
    def setup(self, model: nn.Module) -> None:
        """Initialize grid for evaluation."""
        x = torch.linspace(0, 1, self.N_grid, device=self.device)
        y = torch.linspace(0, 1, self.N_grid, device=self.device)
        X, Y = torch.meshgrid(x, y, indexing="ij")
        self.grid = self.domain_map(torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1))
    
    def on_epoch_end(self, epoch: int, loss: float, loss_data: float, loss_phys: float, 
                     model: nn.Module = None) -> None:
        if epoch % self.save_interval == 0 and model is not None:
            model.eval()
            with torch.no_grad():
                U_pred = model(self.grid).reshape(self.N_grid, self.N_grid).detach().cpu().numpy()
            self.frames.append(U_pred)
            model.train()
    
    def get_result(self):
        return self.frames


def sample(dim: int, N: int, domain_map: callable = identity_map, device: str = 'cpu') -> torch.Tensor:
    """
    Generate collocation points using Latin Hypercube sampling.
    
    Args:
        dim: dimension of the domain
        N: number of samples
        domain_map: function to transform sampled points to the domain
        device: device to place tensors on
        
    Returns:
        X_col: tensor of shape (N, dim) with sampled collocation points
    """
    sampler = qmc.LatinHypercube(d=dim)
    samples = sampler.random(n=N)   # shape (N, dim)
    return domain_map(torch.tensor(samples, dtype=torch.float32, device=device))


def train(epochs: int, optimizer: torch.optim.Optimizer, X_data: torch.Tensor, U_data: torch.Tensor, 
          model: nn.Module, f: callable, lambda_phys: float, X_col: torch.Tensor, 
          callbacks: list = None) -> tuple:
    """
    Train a physics-informed neural network with callback support.
    
    Args:
        epochs: number of training epochs
        optimizer: optimizer instance
        X_data: boundary and initial condition data points
        U_data: corresponding values for X_data
        model: neural network model
        f: residual function for physics loss
        lambda_phys: weight for physics loss
        X_col: collocation points for physics loss
        callbacks: list of Callback instances for logging, checkpointing, storing frames.
        
    Returns:
        losses: list of total losses per epoch
        losses_data: list of data losses per epoch
        losses_phys: list of physics losses per epoch
        callback_results: dict mapping callback to its results
    """
    if callbacks is None:
        callbacks = []
    
    # Setup callbacks that need initialization
    for callback in callbacks:
        if isinstance(callback, FramesCallback):
            callback.setup(model)
    
    losses = []
    losses_data = []
    losses_phys = []

    for epoch in range(epochs):
        optimizer.zero_grad()

        loss_data = data_loss(model, X_data, U_data)
        loss_phys = physics_loss(model, X_col, f)
        loss = loss_data + lambda_phys * loss_phys

        losses.append(loss.item())
        losses_data.append(loss_data.item())
        losses_phys.append(loss_phys.item())
        
        loss.backward()
        optimizer.step()

        # Call epoch end callbacks
        for callback in callbacks:
            if isinstance(callback, FramesCallback):
                callback.on_epoch_end(epoch, loss.item(), loss_data.item(), loss_phys.item(), model=model)
            elif isinstance(callback, CheckpointCallback):
                callback.set_checkpoint_data(model, optimizer, epoch, loss.item())
            else:
                callback.on_epoch_end(epoch, loss.item(), loss_data.item(), loss_phys.item())

    # Call train end callbacks
    for callback in callbacks:
        callback.on_train_end()

    # Collect results from callbacks
    callback_results = {type(cb).__name__: cb.get_result() for cb in callbacks}
    
    return losses, losses_data, losses_phys, callback_results

def L2RE(u_pred: torch.Tensor, u_true: torch.Tensor) -> float:

    return np.linalg.norm(u_pred - u_true) / np.linalg.norm(u_true)


class FeatureExtractor:
    def __init__(self, model: nn.Module, layer_names: list) -> None:
        self.model = model
        self.layer_names = layer_names
        self.activations = {name: [] for name in layer_names}
        self.hooks = []
        
        self._register_hooks()

    def _register_hooks(self) -> None:
        """Registers forward hooks on the specified layers. Automatically runs on initialization."""

        # A dictionary mapping string names to actual submodules
        named_modules = dict(self.model.named_modules())
        
        for name in self.layer_names:
            if name in named_modules:
                layer = named_modules[name]
                # Use a default argument in the lambda to lock in the current 'name'
                hook = layer.register_forward_hook(
                    lambda module, input, output, n=name: self._hook_fn(n, output)
                )
                self.hooks.append(hook)
            else:
                print(f"Warning: Layer '{name}' not found in this model.")

    def _hook_fn(self, layer_name: str, output: torch.Tensor) -> None:

        self.activations[layer_name].append(output.detach().cpu())

    def get_layer_data(self, layer_name: str) -> torch.Tensor:
        """Combines all accumulated batches for a given layer into a single tensor."""
        if layer_name not in self.activations or not self.activations[layer_name]:
            return None
        return torch.cat(self.activations[layer_name], dim=0)

    def clear(self) -> None:
        """Clears accumulated activations between different data collection loops."""
        for name in self.layer_names:
            self.activations[name] = []

    def close(self) -> None:
        """Removes the hooks completely from the model to prevent memory leaks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []