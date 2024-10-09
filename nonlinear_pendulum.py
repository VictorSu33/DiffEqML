import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import neural_utils as neural
from mpl_toolkits import mplot3d
from Lyapunov import *

device = torch.device("cuda:0")

def f(x,b=0):
    #Nonlinear pendulum system from Morris, Hirsch & Smale
    x_0 = x[:, 1]
    x_1 = -b*x[:, 1]-torch.sin(x[:,0])
    return torch.cat([x_0.reshape(-1,1),x_1.reshape(-1,1)], axis = 1)

# define the upper bound for the boundary condition
def upperbound ( x , p = 100.):
    return p*x[:,0]**2 + p*x[:,1]**2


# define the lower bound for the boundary condition
def lowerbound ( x, p =0.01 ):
    return p*x[:,0]**2 + p*x[:,1]**2

#generate training data
train_num = 10
x_train = (2*torch.rand((train_num,2))-1).requires_grad_(True).to(device)

upper = upperbound(x_train)
lower = lowerbound(x_train)
f_train = f(x_train)
zeroes = torch.zeros(upper.shape).to(device)

epochs = 5000
total_loss = 1e-5
batch_size = 10

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
hidden_dim = 4
sub_dim = 2

#lyapunov = Lyapunov_Unkown(input_dim,hidden_dim,sub_dim).to(device)

lyapunov = Lyapunov_Known(input_dim, hidden_dim, [1,1]).to(device)
lyapunov_loss = Lyapunov_Loss().to(device)

lyapunov.to(device)
lyapunov_loss.to(device)

from prettytable import PrettyTable

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params
    
count_parameters(lyapunov)

optimiser = torch.optim.Adam(lyapunov.parameters(),lr=1e-5)

loss_values = []

for epoch in range(epochs):

    mlv = 0
    slv = 0
    

    for i, (upper_batch, lower_batch, f_batch, x_batch, zeroes_batch) in enumerate(train_dataloader):
        optimiser.zero_grad()
        y_batch = lyapunov(x_batch)
        #print("breakpoint", x_batch.shape)
        loss = lyapunov_loss(y_batch, upper_batch,lower_batch,zeroes_batch, f_batch, x_batch, lyapunov)
        loss.backward(retain_graph = True)
        
        optimiser.step()


        mlv = max(mlv,loss)
        slv = slv + loss
    loss_values.append(slv/(i + 1))

    print('epoch %2s, average loss %10.6f, max loss %10.6f' % (epoch, slv/ (i + 1), mlv))

    if mlv < total_loss:
        break

#torch.save(lyapunov.state_dict(), "lyapunov2d.pt")

plt.plot(range(epochs), loss_values)
plt.title("Loss Curve")

#plot graph

# define resolution
numpoints = 30

# define plotting range and mesh
x = np.linspace(-1, 1, numpoints)
y = np.linspace(-1, 1, numpoints)

X, Y = np.meshgrid(x, y)

s = X.shape

#define known Lyapunov function
def energy(data):
    return 0.5*data[:,1]**2 + 1 - torch.cos(data[:,0])

Z_grad = np.zeros(s)
Z_func = np.zeros(s)
Z_energy = np.zeros(s)
DT = torch.zeros((numpoints**2,input_dim)).requires_grad_(True).to(device)

# convert mesh into point vector for which the model can be evaluated
c = 0
for i in range(s[0]):
    for j in range(s[1]):
        DT[c,0] = X[i,j]
        DT[c,1] = Y[i,j]
        c = c+1

# evaluate model (= Lyapunov function values V)
Ep = lyapunov(DT)
Energy = energy(DT)

# evaluate gradients DV of Lyapunov function
grads = torch.func.vmap (torch.func.jacfwd(lyapunov), in_dims = (0,)) (DT).squeeze()

print(grads.shape)
print(f(DT).shape   )
# compute orbital derivative DVf
Ee =torch.einsum('ij,ij->i',  grads, f(DT))

# copy V and DVf values into plottable format
c = 0
for i in range(s[0]):
    for j in range(s[1]):
        Z_grad[i,j] = Ee[c]
        Z_func[i,j] = Ep[c]
        Z_energy[i,j] = Energy[c]
        c = c+1

# define figure
fig = plt.figure()
ax = plt.axes(projection='3d')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

# plot known Lyapunov function



# plot values V
ax.plot_surface(X, Y, Z_func, rstride=1, cstride=1,
                cmap='Reds', edgecolor='none')

ax.plot_surface(X, Y, Z_energy, rstride=1, cstride=1,
                cmap='Greens', edgecolor='none')

# plot orbital derivative DVf
ax.plot_wireframe(X, Y, Z_grad, rstride=1, cstride=1)

plt.show()
