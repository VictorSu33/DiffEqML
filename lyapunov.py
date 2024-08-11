import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import neural_utils as neural
from mpl_toolkits import mplot3d

device = torch.device("cuda:0")

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

class Lyapunov_Unkown(nn.Module):
    def __init__(self, N_INPUT, N_HIDDEN, d_max =1):
        #hidden should be a multiple of input?
        super().__init__()
        
        self.d_max = d_max

        self.fcs = nn.Linear(N_INPUT, N_HIDDEN)
        self.conv = nn.Sequential(nn.Conv1d(in_channels=N_HIDDEN, out_channels=N_HIDDEN, kernel_size=1, groups= N_HIDDEN // self.d_max), nn.ReLU())

        #is ReLU okay?
        self.fce = nn.Sequential(
                        nn.Linear(N_HIDDEN, 1, bias=False),
                        nn.ReLU())
        
    def forward(self, x):
        x = self.fcs(x)
        x = x.unsqueeze(-1)
        x = self.conv(x).squeeze(-1)
        x= self.fce(x)

        return x

class Lyapunov_Loss(nn.Module):
    

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
        loss = torch.mean((torch.einsum('ij,ij->i',  dV_dx, f_data) + torch.norm(x, dim=1)**2)**2)
        return loss

    def forward(self, y_pred, upper,lower,zeroes,f_data, x,V):
        return self.lambda1 * self.bound_loss(y_pred, upper, lower,zeroes) + self.lambda2 * self.gradient_loss(f_data, x,V)
    
def f(x,b=0):
    #Nonlinear pendulum system from Morris, Hirsch & Smale
    x_0 = x[:, 1]
    x_1 = -b*x[:, 1]-torch.sin(x[:,0])
    return torch.cat([x_0.reshape(-1,1),x_1.reshape(-1,1)], axis = 1)

# define the upper bound for the boundary condition
def upperbound ( x ):
    return 10.*x[:,0]**2 + 10.*x[:,1]**2

# define the lower bound for the boundary condition
def lowerbound ( x ):
    return 0.1*x[:,0]**2 + 0.1*x[:,1]**2

#generate training data
train_num = 1000
x_train = (2*torch.rand((train_num,2))-1).requires_grad_(True).to(device)

upper = upperbound(x_train)
lower = lowerbound(x_train)
f_train = f(x_train)
zeroes = torch.zeros(upper.shape).to(device)

epochs = 20
tol = 1e-5
batch_size = 8
gradweight = 1

def train_step():
    pass

'''
print("upper shape", upper.shape)
print("lower shape", lower.shape)
print("f_train shape", f_train.shape)
print("x_train shape", x_train.shape)
print("zeroes shape", zeroes.shape)
'''


train_data = TensorDataset(upper,lower, f_train, x_train, zeroes)
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

#hyperparameters

input_dim = 2
hidden_dim = 16
sub_dim = 4

lyapunov = Lyapunov_Unkown(input_dim,hidden_dim,sub_dim).to(device)
lyapunov_loss = Lyapunov_Loss().to(device)

lyapunov.to(device)
lyapunov_loss.to(device)

optimiser = torch.optim.Adam(lyapunov.parameters(),lr=1e-3)

loss_values = []

for epoch in range(epochs):
    epoch_loss = 0

    mlv = 0
    slv = 0
    

    for i, (upper_batch, lower_batch, f_batch, x_batch, zeroes_batch) in enumerate(train_dataloader):
        optimiser.zero_grad()
        y_batch = lyapunov(x_batch)

        loss = lyapunov_loss(y_batch, upper_batch,lower_batch,zeroes_batch, f_batch, x_batch, lyapunov)
        print(loss)
        loss.backward()
        
        optimiser.step()

        epoch_loss += loss.detach()

        mlv = max(mlv,loss/batch_size)
        slv = slv + loss
    loss_values.append(epoch_loss.detach()/x_train.shape[0])

    print('epoch %2s, samples %7s, loss %10.6f, L1 loss %10.6f, max loss %10.6f' % (epoch, ((i + 1) * batch_size), float(loss), slv/ (i + 1), mlv))

    if mlv < tol:
        break

torch.save(lyapunov.state_dict(), "lyapunov2d.pt")

plt.plot(range(epochs), loss_values, label="Learning Curve")

#plot graph

# define resolution
numpoints = 30

# define plotting range and mesh
x = np.linspace(-1, 1, numpoints)
y = np.linspace(-1, 1, numpoints)

X, Y = np.meshgrid(x, y)

s = X.shape

Z_grad = np.zeros(s)
Z_func = np.zeros(s)
DT = torch.zeros((numpoints**2,input_dim)).to(device)

# convert mesh into point vector for which the model can be evaluated
c = 0
for i in range(s[0]):
    for j in range(s[1]):
        DT[c,0] = X[i,j]
        DT[c,1] = Y[i,j]
        c = c+1

# evaluate model (= Lyapunov function values V)
Ep = lyapunov(DT)

# convert point vector to tensor for evaluating x-derivative
tDT = torch.tensor(DT)

# evaluate gradients DV of Lyapunov function
grads = torch.func.vmap (torch.func.jacfwd(lyapunov), in_dims = (0,)) (DT)

# compute orbital derivative DVf
Ee = torch.sum(torch.dot(grads, f(DT).T))

# copy V and DVf values into plottable format
c = 0
for i in range(s[0]):
    for j in range(s[1]):
        Z_grad[i,j] = Ee[c]
        Z_func[i,j] = Ep[c]
        c = c+1

# define figure
fig = plt.figure()
ax = plt.axes(projection='3d')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

# plot values V
ax.plot_surface(X, Y, Z_func, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')

# plot orbital derivative DVf
ax.plot_wireframe(X, Y, Z_grad, rstride=1, cstride=1)

plt.show()
