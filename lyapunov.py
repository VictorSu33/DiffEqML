import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import neural_utils as neural
from mpl_toolkits import mplot3d

device = torch.device("cuda:0")

class Lyapunov_Known(nn.module):

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
        for sub_input in self.decomposition:
            self.subfuncs.add_module(nn.Sequential(nn.Linear(sub_input,sub_hidden), nn.ReLU()))

        self.fce = nn.Sequential(nn.Linear(N_HIDDEN, 1, bias=False),nn.ReLU)
        
    #try to eliminate for loops

    def forward(self, x):
        outputs = []
        sub_states = torch.split(x,self.decomposition)
        for i in range(len(sub_states)):
            outputs.append(self.subfuncs[i](sub_states[i]))

        x = torch.cat(outputs)
        x= self.fce(x)

        return x

class Lyapunov_Unkown(nn.module):
    #create fully connected neural network with softplus to return only nonnegative values

    def __init__(self, N_INPUT, N_HIDDEN, d_max =None):
        #hidden should be a multiple of input?
        super().__init__()
        
        if not d_max:
            self.d_max = N_HIDDEN
        else:
            self.d_max = d_max

        self.fcs = nn.Linear(N_INPUT, N_HIDDEN)
        self.conv = nn.Sequential(nn.Conv1d(in_channels=N_HIDDEN, out_channels=N_HIDDEN, kernel_size=1, groups= N_HIDDEN // self.d_max), nn.Tanh())

        #is ReLU okay?
        self.fce = nn.Sequential(
                        nn.Linear(N_HIDDEN, 1, bias=False),
                        nn.ReLU())
        
    def forward(self, x):
        x = self.fcs(x)
        x = self.conv(x)


        x = torch.cat(x)
        x= self.fce(x)

        return x

class Lyapunov_Loss(nn.module):
    

    def __init__(self, lambda1=1,lambda2=1):
        super(Lyapunov_Loss, self).__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2

    def bound_loss(self, y_pred, upper, lower,zeroes):
        # boundary loss of zubov pde
        lower_loss = torch.mean(torch.minimum(y_pred - lower,zeroes)**2)
        upper_loss = torch.mean(torch.maximum(y_pred - upper,zeroes)**2)

        return lower_loss + upper_loss

    def gradient_loss(self,f_data, x,V):
        #gradient loss of zubov pde
        dV_dx = torch.func.vmap (torch.func.jacfwd(V), in_dims = (0,)) (x).squeeze()
        loss = torch.mean((torch.dot(dV_dx, f_data) + torch.norm(x)**2)**2)
        return loss

    def forward(self, y_pred, upper,lower,zeroes,f_data, x,V):
        return self.lambda1 * self.bound_loss(self, y_pred, upper, lower,zeroes) + self.lambda2 * self.gradient_loss(self,f_data, x,V)
    
def f(x,b):
    #Nonlinear pendulum system from Morris, Hirsch & Smale
    x_0 = x[:, 1]
    x_1 = -b*x[:, 1]-torch.sin(x[:,0])
    return torch.cat([x_0,x_1])

# define the upper bound for the boundary condition
def upperbound ( x ):
    return 10.*x[:,0]**2 + 10.*x[:,1]**2

# define the lower bound for the boundary condition
def lowerbound ( x ):
    return 0.1*x[:,0]**2 + 0.1*x[:,1]**2

#generate training data
x_train = (2*torch.rand((1000,2))-1).requires_grad_(True).to(device)

upper = upperbound(x_train)
lower = lowerbound(x_train)
f_train = f(x_train)
zeroes = torch.zeros(upper.shape)

epochs = 20
tol = 1e-5
batch_size = 8
gradweight = 1

def train_step():
    pass

train_data = TensorDataset(upper,lower, f_train, x_train, zeroes)
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

#hyperparameters
lyapunov = Lyapunov_Unkown(2,16,4)
lyapunov_loss = Lyapunov_Loss()

lyapunov.to(device)
lyapunov_loss.to(device)

optimiser = torch.optim.Adam(Lyapunov_Unkown.parameters(),lr=1e-3)

loss_values = []

for epoch in range(epochs):
    epoch_loss = 0

    mlv = 0
    slv = 0
    

    for i, (upper_batch, lower_batch, f_batch, x_batch, zeroes_batch) in enumerate(train_dataloader):
        optimiser.zero_grad()
        y_batch = lyapunov(x_batch)

        loss = torch.sum(lyapunov_loss(y_batch, upper_batch,lower_batch,zeroes_batch, f_batch, x_train, lyapunov))
        loss.backward()
        
        optimiser.step()

        epoch_loss += loss

        mlv = torch.maximum(mlv,loss/batch_size)
        slv = slv + loss
    loss_values.append(epoch_loss.detach()/x_train.shape[0])

    print('epoch %2s, samples %7s, loss %10.6f, L1 loss %10.6f, max loss %10.6f' % (epoch, ((i + 1) * batch_size), float(loss), slv/ (i + 1), mlv))

    if mlv < tol:
        break

torch.save(lyapunov.state_dict(), "lyapunov2d.pt")

plt.plot(range(epochs), loss_values, label="Learning Curve")
