from pathlib import Path
from typing import Optional
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import copy

class Callback:
    """Base callback class for training hooks."""
    
    def on_epoch_start(self, epoch: int) -> None:
        pass
    
    def on_epoch_end(self, epoch: int, loss: float, model: nn.Module) -> None:
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
    
    def on_epoch_end(self, epoch: int, loss: float, loss_data: float, loss_phys: float, model: nn.Module = None) -> None:
        if (epoch + 1) % self.log_interval == 0:
            tqdm.write(f"Epoch {epoch:5d} | Total Loss: {loss:.4e} | Data Loss: {loss_data:.4e} | Physics Loss: {loss_phys:.4e}")

class SaveTrajectoryCallback(Callback):
    """Saves model trajectory at regular intervals."""
    
    def __init__(self, save_interval: int = 100, save_path: Optional[Path] = None):
        self.save_interval = save_interval
        if save_path is not None:
            self.save_path = Path(save_path)
            self.save_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            self.save_path = None
        self.trajectory_vectors = []
        self.recorded_epochs = []
        
    def on_epoch_end(self, epoch: int, loss: float, model: nn.Module = None) -> None:
        if (epoch + 1) % self.save_interval == 0 and model is not None:
            with torch.no_grad():

                flat_weights = torch.cat([p.detach().clone().cpu().flatten() for p in model.parameters()])
                
                self.trajectory_vectors.append(flat_weights)
                self.recorded_epochs.append(epoch)
                
    def get_result(self) -> Optional[Path]:
        """
        Save the trajectory data to disk if a save path is provided, and return the path.
        """
        if not self.trajectory_vectors:
            print("[Warning] SaveTrajectoryCallback: No trajectory data was captured.")
            return self.save_path
            
        trajectory_matrix = torch.stack(self.trajectory_vectors)
        epochs = torch.tensor(self.recorded_epochs)
        
        trajectory_dict = {
            "trajectory": trajectory_matrix,
            "epochs": epochs,
        }
        
        if self.save_path is not None:
            torch.save(trajectory_dict, self.save_path)
            print(f"\n[Callback] Successfully saved single trajectory file to: {self.save_path}")
        else:
            print("\n[Callback] No save path provided. Trajectory data not saved to disk.")
            
        self.trajectory_vectors = []
        self.recorded_epochs = []
        
        return self.save_path


class CheckpointCallback(Callback):
    """Saves model checkpoints when loss improves."""
    
    def __init__(self):
        self.best_loss = float('inf')
        self.checkpoint_data = None
    
    def on_epoch_end(self, epoch: int, loss: float, model: nn.Module = None) -> None:
        if loss < self.best_loss:
            self.best_loss = loss
            self.checkpoint_data = {
                'epoch': epoch,
                'model_state_dict': copy.deepcopy(model.state_dict()),
                'loss': loss
            }
    
    def get_result(self):
        return self.checkpoint_data

def identity_map(x: torch.Tensor) -> torch.Tensor:
    return x

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
    
    def on_epoch_end(self, epoch: int, loss: float, model: nn.Module = None) -> None:
        if (epoch + 1) % self.save_interval == 0 and model is not None:
            model.eval()
            with torch.no_grad():
                U_pred = model(self.grid).reshape(self.N_grid, self.N_grid).detach().cpu().numpy()
            self.frames.append(U_pred)
            model.train()
    
    def get_result(self):
        return self.frames

class SpectrumCallback(Callback):
    """Captures the spectrum of model for analysis."""
    
    def __init__(self, data: torch.tensor, transform: callable = torch.fft.rfft, save_path: Optional[Path] = None, save_interval: int = 100, **transform_kwargs):
        self.save_interval = save_interval
        self.data = data
        self.spectra = {}
        self.transform = transform
        self.transform_kwargs = transform_kwargs
        if save_path is not None:
            self.save_path = Path(save_path)
            self.save_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            self.save_path = None
    
    def on_epoch_end(self, epoch: int, loss: float, model: nn.Module = None) -> None:
        
        if (epoch + 1) % self.save_interval == 0 and model is not None:
            with torch.no_grad():
                model_device = next(model.parameters()).device
                data = self.data.to(model_device)

                pred = model(data).detach().squeeze(-1)
                spectrum = self.transform(pred, **self.transform_kwargs)
                self.spectra[epoch] = spectrum.cpu()
    
    def on_train_end(self) -> None:
        if self.save_path is not None:
            torch.save(self.spectra, self.save_path)
            print(f"\n[Callback] Successfully saved spectra to: {self.save_path}")
        else:
            print("\n[Callback] No save path provided. Spectra data not saved to disk.")

    def get_result(self):
        results = self.spectra
        self.spectra = {}
        return results
    