import torch
import torch.nn as nn
import numpy

class Deep_BS():
    
    def __init__(self):
        pass

class Sin(nn.module):
    def forward(self, x):
        return torch.sin(x)
    

class RFF_PINN(nn.module):

    def __init__(self, d_input: int, d_output: int, h_layers: list[int]):
        super().__init__()
        activation = Sin
        layers = [nn.Linear(d_input, h_layers[0]), activation()]
        for i in range(len(h_layers) - 1):
            layers.extend([nn.Linear(h_layers[i], h_layers[i+1]), activation()])
        layers.append(nn.Linear(h_layers[-1], d_output))

        self.net = nn.Sequential(*layers)
        self.siren()
        
    def siren(self,):
        for name, m in self.net.named_modules():
            if isinstance(m, nn.Linear):
                # SIREN initialization scheme (from the Sitzmann et al. paper)
                # Weights are drawn from U(-sqrt(6/n), sqrt(6/n))
                num_input = m.weight.size(1)
                with torch.no_grad():
                    m.weight.uniform_(-torch.sqrt(6 / num_input), torch.sqrt(6 / num_input))