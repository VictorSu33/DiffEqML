import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import neural_utils as neural

def f(x):
    A = torch.tensor([[0.,-1.], [1.,0.]])
    return torch.matmul( x, A.T)

def exact(t, x0):
    t = t.numpy()
    e1 = np.array([1,0])
    e2 = np.array([0,1])
    return x0[0] * (np.cos(t) * e2 - np.sin(t) * e1) + x0[1]*(np.cos(t) * e1 + np.sin(t) * e2)

t0 = torch.tensor(0.).view(-1,1).requires_grad_(True)
x0 = torch.tensor([1,1])

t_train = (2 * np.pi*(torch.rand(100) - 0.5)).view(-1,1).requires_grad_(True)

t_test = torch.linspace(-np.pi,np.pi,1000).view(-1,1)

x_exact = exact(t_test, x0)

x_exact1 = x_exact[:,0]
x_exact2 = x_exact[:,1]

#watch out for overfitting

model = neural.FCN(1,2,10,3)

optimiser = torch.optim.Adam(model.parameters(),lr=1e-3)

#define minibatch size
minibatch = 8
#define epochs 
epochs = 1000

num_batches = np.ceil(epochs/minibatch)

loss_history = []

#begin training

for epoch in range(epochs):

    # X is a torch Variable
    permutation = torch.randperm(t_train.shape[0])
    epoch_loss = 0

    for i in range(0,t_train.shape[0], minibatch):

            indices = permutation[i:i+minibatch]

            batch_t = t_train[indices]

            epoch_loss += neural.training_step(optimiser, f, batch_t,t0,x0, model).detach()
    
    loss_history.append(epoch_loss)

    if epoch % 100 == 0:

        x = model(t_test).detach()
        x_1 = x[:,0]
        x_2 = x[:,1]

        x_train = model(t_train).detach()
        x_train1 = x_train[:,0]
        x_train2 = x_train[:,1]

        plt.figure(figsize=(6,2.5))
        plt.scatter(t_train.detach()[:,0], 
                    torch.zeros_like(t_train)[:,0], s=20, lw=0, color="tab:green", alpha=0.6)

        plt.scatter(t0.detach()[:,0], 
                    0, s=40, lw=0, color="tab:orange", alpha=0.6)
                    
        plt.plot(x_exact1,x_exact2, label="Exact solution", linewidth=2,color="tab:blue", alpha=0.6)
        plt.plot(x_1,x_2, '--', label="PINN solution test", linewidth=2, color="tab:red")
        plt.scatter(x_train1,x_train2, s = 20, label="PINN solution train", color="tab:purple",alpha=0.6)
        plt.title(f"Comparison after {epoch} epochs")
        plt.legend()
        plt.show()

plt.plot(range(epochs), loss_history, label="Learning Curve")
plt.show()

print("final training loss:", loss_history[-1])

x = model(t_test).detach()

test_loss = torch.mean(torch.norm(x_exact - x)**2)

print("Test loss:", test_loss)

#test for generalization outside of interval.

#def generalization(epsilon,lower,upper,model, exact):
    
    
