import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import neural_utils as neural

#define differential equation 
def f(y):
    return y

def exact(y):
    return np.exp(y)

#boundary points
t0 = torch.tensor(0.).view(-1,1).requires_grad_(True)
y0 = 1

#training points
t_train = torch.rand(33).view(-1,1).requires_grad_(True)

t_test = torch.linspace(0,1,300).view(-1,1)
y_exact = exact(t_test)

model = neural.FCN(1,1,20,3)

print(model.parameters)

optimiser = torch.optim.Adam(model.parameters(),lr=1e-3)

loss_history = []
#begin training
for i in range(1000):
    loss = neural.training_step(optimiser, f, t_train,t0,y0, model)
    loss_history.append(torch.flatten(loss.detach()))

    #display results as training progresses
    if i % 200 == 0: 
        y = model(t_test).detach()

        plt.figure(figsize=(6,2.5))
        plt.scatter(t_train.detach()[:,0], 
                    torch.zeros_like(t_train)[:,0], s=20, lw=0, color="tab:green", alpha=0.6)

        plt.scatter(t0.detach()[:,0], 
                    0, s=40, lw=0, color="tab:orange", alpha=0.6)
                    
        plt.plot(t_test[:,0], y_exact[:,0], label="Exact solution", linewidth=2,color="tab:blue", alpha=0.6)
        plt.plot(t_test[:,0], y[:,0], '--', label="PINN solution", linewidth=2, color="tab:red")
        plt.title("Comparison")
        plt.legend()
        plt.show()

plt.plot(range(1000), loss_history, label="Learning Curve")
plt.show()