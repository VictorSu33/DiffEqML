import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import neural_utils as neural

#define differential equation 
def f(x):
    return x

def exact(t):
    return np.exp(t)

#boundary points
t0 = torch.tensor(0.).view(-1,1).requires_grad_(True)
x0 = 1

#training points
t_train = torch.rand(33).view(-1,1).requires_grad_(True)

t_test = torch.linspace(-2,2,300).view(-1,1)
x_exact = exact(t_test)

model = neural.FCN(1,1,20,3)

optimiser = torch.optim.Adam(model.parameters(),lr=1e-3)

#define minibatch size
minibatch = 5
#define epochs 
epochs = 200


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

    if epoch % 50 == 0:

        x = model(t_test).detach()

        plt.figure(figsize=(6,2.5))
        plt.scatter(t_train.detach()[:,0], 
                    torch.zeros_like(t_train)[:,0], s=20, lw=0, color="tab:green", alpha=0.6)

        plt.scatter(t0.detach()[:,0], 
                    0, s=40, lw=0, color="tab:orange", alpha=0.6)
                    
        plt.plot(t_test[:,0], x_exact[:,0], label="Exact solution", linewidth=2,color="tab:blue", alpha=0.6)
        plt.plot(t_test[:,0], x[:,0], '--', label="PINN solution", linewidth=2, color="tab:red")
        plt.title("Comparison")
        plt.legend()
        plt.show()


plt.plot(range(epochs), loss_history, label="Training Loss")
plt.show()

x = model(t_test).detach()

test_loss = torch.mean((x_exact - x)**2)

print("Test loss:", test_loss)

#test generalization outside of [0,1]