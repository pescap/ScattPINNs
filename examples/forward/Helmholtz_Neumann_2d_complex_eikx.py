"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch"""
import deepxde as dde
import numpy as np

# General parameters
n = 2
precision_train = 10
precision_test = 10
hard_constraint = True
weights = 100  # if hard_constraint == False
epochs = 5000 #TODO, testing, 25000 for worflow testing
parameters = [1e-3, 4, 80, "sin"]

# learning rate
# depth
# width
# activation function

# Define sine function
if dde.backend.backend_name == "pytorch":
    cos = dde.backend.pytorch.cos
    sin = dde.backend.pytorch.sin
else:
    from deepxde.backend import tf

    cos = tf.cos
    sin = tf.sin

learning_rate, num_dense_layers, num_dense_nodes, activation = parameters


def pde(x, y):
    yRe, yIm = y[:, 0:1], y[:, 1:2]
    
    
    dyRe_xx = dde.grad.hessian(y, x, component=0, i=0, j=0)
    dyRe_yy = dde.grad.hessian(y, x, component=0, i=1, j=1)
    
    dyIm_xx = dde.grad.hessian(y, x, component=1, i=0, j=0)
    dyIm_yy = dde.grad.hessian(y, x, component=1, i=1, j=1)
    

    fRe = k0 ** 2 * cos(k0 * x[:, 0:1])
    fIm = k0 ** 2 * sin(k0 * x[:, 1:2])
    
    return [-dyRe_xx - dyRe_yy - k0 ** 2 * yRe - fRe,
            -dyIm_xx - dyIm_yy - k0 ** 2 * yIm - fIm]


def func(x):
    real = np.real(np.exp(1j * k0 * x[:, 0:1]))
    imag = np.imag(np.exp(1j * k0 * x[:, 0:1]))
    return np.hstack((real, imag))


#def func(x):
#    real = np.cos(k0 * x[:, 0:1]) * np.cos(k0 * x[:, 1:2])
#   imag = np.sin(k0 * x[:, 0:1]) * np.sin(k0 * x[:, 1:2])
#    return np.hstack((real, imag))


def boundary(_, on_boundary):
    return on_boundary

def func0(x):
    normal = geom.boundary_normal(x)
    g0 = np.real(1j * k0 * np.exp(1j * k0 * x[:, 0:1]) * (normal[:, 0:1]-1))
    return g0

def func1(x):
    normal = geom.boundary_normal(x)
    g1 = np.imag(1j * k0 * np.exp(1j * k0 * x[:, 0:1]) * (normal[:, 0:1]-1))
    return g1



geom = dde.geometry.Rectangle([0, 0], [1, 1])
k0 = 2 * np.pi * n
wave_len = 1 / n

hx_train = wave_len / precision_train
nx_train = int(1 / hx_train)

hx_test = wave_len / precision_test
nx_test = int(1 / hx_test)

bcRe = dde.icbc.NeumannBC(geom, func0, boundary, component=0)
bcIm = dde.icbc.NeumannBC(geom, func1, boundary, component=1)

bcs = [bcRe, bcIm]

data = dde.data.PDE(
    geom,
    pde,
    bcs,
    num_domain=nx_train ** 2,
    num_boundary=4 * nx_train,
    solution=func,
    num_test=nx_test ** 2,
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
