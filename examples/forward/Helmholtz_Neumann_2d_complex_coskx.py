"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch"""
import deepxde as dde
import numpy as np

# General parameters
n = 2
precision_train = 10
precision_test = 10
hard_constraint = True
weights = 100  # if hard_constraint == False
epochs = 20000 # tested with 50.000 epochs.
parameters = [1e-3, 4, 80, "sin"]# learning rate, depth, width, activation function
k0 = 1  
wave_len = 2*np.pi / k0
dim_x = 2 * np.pi
n_wave = 20
h_elem = wave_len / n_wave
nx = int(dim_x / h_elem)

learning_rate, num_dense_layers, num_dense_nodes, activation = parameters

def pde(x, y):
    yRe, yIm = y[:, 0:1], y[:, 1:2]
    
    dyRe_xx = dde.grad.hessian(y, x, component=0, i=0, j=0)
    dyRe_yy = dde.grad.hessian(y, x, component=0, i=1, j=1)
    
    dyIm_xx = dde.grad.hessian(y, x, component=1, i=0, j=0)
    dyIm_yy = dde.grad.hessian(y, x, component=1, i=1, j=1)
    
    return [-dyRe_xx - dyRe_yy - k0 ** 2 * yRe,
            -dyIm_xx - dyIm_yy - k0 ** 2 * yIm]

def sol(x):
    return np.cos(k0 * x[:, 0:1])

def func(x):
    real = np.real(np.cos(k0 * x[:, 0:1]))
    imag = np.imag(np.cos(k0 * x[:, 0:1]))
    return np.hstack((real, imag))

def boundary(_, on_boundary):
    return on_boundary

def func0(x):
    normal = geom.boundary_normal(x)
    result = k0 * np.cos(k0 * x[:, 0:1]) * normal[:, 0:1]
    return(result)

geom = dde.geometry.Rectangle([0, 0], [dim_x, dim_x])

bcRe = dde.icbc.NeumannBC(geom, func0, boundary, component=0)
bcIm = dde.icbc.NeumannBC(geom, func0, boundary, component=1)

bcs = [bcRe, bcIm]

data = dde.data.PDE(
    geom,
    pde,
    bcs,
    num_domain=nx ** 2,
    num_boundary=4 * nx,
    solution=func,
    num_test=10 * nx ** 2,
)

net = dde.nn.FNN(
    [2] + [num_dense_nodes] * num_dense_layers + [2], activation, "Glorot uniform"
)

model = dde.Model(data, net)
loss_weights = [1, 1, weights, weights]
model.compile(
    "adam", lr=learning_rate, metrics=["l2 relative error"], 
    loss_weights=loss_weights
)

losshistory, train_state = model.train(epochs=epochs)
dde.saveplot(losshistory, train_state, issave=True, isplot=True)
