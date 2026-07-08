# define solvers

import numpy as np
import torch
import torch.nn as nn


class monteCarlo():
    def __init__(self, model, num_paths=1000):
        self.model = model
        self.num_paths = num_paths

class deepBS():
    # read up on
    def __init__(self, model, num_paths=1000):
        super(deepBS, self).__init__()
        self.model = model
        self.num_paths = num_paths

class finiteDifference():
    def __init__(self, model, num_paths=1000):
        self.model = model
        self.num_paths = num_paths

class PINN():
    def __init__(self, model, num_paths=1000):
        self.model = model
        self.num_paths = num_paths