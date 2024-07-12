import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import neural_utils as neural
from mpl_toolkits import mplot3d

def f(x_hat):
    '''requires 2d input tensor'''
    A = torch.tensor([[0.,-1.], [1.,0.]])
    return torch.matmul( x_hat, A.T)

def exact(t, x0):
    t = t.numpy()
    e1 = np.array([1,0])
    e2 = np.array([0,1])
    return torch.from_numpy(x0[0] * (np.cos(t) * e2 - np.sin(t) * e1) + x0[1]*(np.cos(t) * e1 + np.sin(t) * e2))

t0 = torch.tensor(0.).view(-1,1).requires_grad_(True)
x0 = np.array([1,1])

t_train = torch.rand(100).view(-1,1).requires_grad_(True)

t_test = torch.linspace(0,1,300).view(-1,1)
y_exact = exact(t_test, x0)

model = neural.FCN(1,2,20,3)

optimiser = torch.optim.Adam(model.parameters(),lr=1e-3)

loss_history = []
#begin training
for i in range(1000):
    loss = neural.training_step(optimiser, f, t_train,t0,x0, model)
    loss_history.append(torch.flatten(loss.detach()))

    #display results as training progresses
    if i % 200 == 0: 
        y = model(t_test).detach()

        plt.figure(figsize=(6,2.5))
        plt.scatter(t_train.detach()[:,0], 
                    torch.zeros_like(t_train)[:,0], s=20, lw=0, color="tab:green", alpha=0.6)

        plt.scatter(t0.detach()[:,0], 
                    0, s=40, lw=0, color="tab:orange", alpha=0.6)
                    
        plt.plot(y_exact[:,0], label="Exact solution", linewidth=2,color="tab:blue", alpha=0.6)
        plt.plot(y[:,0], '--', label="PINN solution", linewidth=2, color="tab:red")
        plt.title("Comparison")
        plt.legend()
        plt.show()

plt.plot(range(1000), loss_history, label="Learning Curve")
plt.show()

    
