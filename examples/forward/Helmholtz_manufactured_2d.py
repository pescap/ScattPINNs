"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch"""
import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt

# General parameters
n = 1
dim_x = 1
R = 1/4.

precision_train = 15
precision_test = 30

weight_inner = 10
weight_outer = 100
epochs = 5000
parameters = [1e-3, 3, 350, "sin"]

k0 = 2 * np.pi * n
wave_len = 1 / n

# Define sine function
if dde.backend.backend_name == "pytorch":
    sin = dde.backend.pytorch.sin
else:
    from deepxde.backend import tf

    sin = tf.sin

learning_rate, num_dense_layers, num_dense_nodes, activation = parameters


def pde(x, y):
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    dy_yy = dde.grad.hessian(y, x, i=1, j=1)
    f = k0**2 * sin(k0 * x[:, 0:1]) * sin(k0 * x[:, 1:2])
    return -dy_xx - dy_yy - k0**2 * y - f


def func(x):
    return np.sin(k0 * x[:, 0:1]) * np.sin(k0 * x[:, 1:2])


def boundary(_, on_boundary):
    return on_boundary


def neumann(x):
    grad = np.array(
        [
            k0 * np.cos(k0 * x[:, 0:1]) * np.sin(k0 * x[:, 1:2]),
            k0 * np.sin(k0 * x[:, 0:1]) * np.cos(k0 * x[:, 1:2]),
        ]
    )

    normal = -inner.boundary_normal(x)
    normal = np.array([normal]).T

    result = tf.math.reduce_sum(grad * normal, axis=0)
    return result


outer = dde.geometry.Rectangle([-dim_x / 2.0, -dim_x / 2.0], [dim_x / 2.0, dim_x / 2.0])
inner = dde.geometry.Disk([0, 0], R)


def boundary_outer(_, on_boundary):
    return on_boundary and outer.on_boundary(_)


def boundary_inner(_, on_boundary):
    return on_boundary and inner.on_boundary(_)


geom = dde.geometry.CSGDifference(outer, inner)

hx_train = wave_len / precision_train
nx_train = int(1 / hx_train)

hx_test = wave_len / precision_test
nx_test = int(1 / hx_test)

bc_inner = dde.icbc.NeumannBC(geom, neumann, boundary_inner)
bc_outer = dde.icbc.DirichletBC(geom, func, boundary_outer)

data = dde.data.PDE(
    geom,
    pde,
    [bc_inner, bc_outer],
    num_domain=nx_train**2,
    num_boundary=16 * nx_train,
    solution=func,
    num_test=nx_test**2,
)

net = dde.nn.FNN(
    [2] + [num_dense_nodes] * num_dense_layers + [1], activation, "Glorot uniform"
)

model = dde.Model(data, net)

loss_weights = [1, weight_inner, weight_outer]
model.compile(
    "adam", lr=learning_rate, metrics=["l2 relative error"], loss_weights=loss_weights
)

losshistory, train_state = model.train(epochs=epochs)
dde.saveplot(losshistory, train_state, issave=True, isplot=True)

# Plot the solution over a square grid with 100 points per wavelength in each direction
Nx = int(np.ceil(wave_len * 100))
Ny = Nx

# Grid points
xmin, xmax, ymin, ymax = [-dim_x / 2.0, dim_x / 2.0, -dim_x / 2.0, dim_x / 2.0]
plot_grid = np.mgrid[xmin : xmax : Nx * 1j, ymin : ymax : Ny * 1j]
points = np.vstack(
    (plot_grid[0].ravel(), plot_grid[1].ravel(), np.zeros(plot_grid[0].size))
)

points_2d = points[:2, :]
u = model.predict(points[:2, :].T)
u = u.reshape((Nx, Ny))

ide = np.sqrt(points_2d[0, :] ** 2 + points_2d[1, :] ** 2) < R
ide = ide.reshape((Nx, Nx))

u_exact = func(points.T)
u_exact = u_exact.reshape((Nx, Ny))
diff = u_exact - u
error = np.linalg.norm(diff) / np.linalg.norm(u_exact)
print("Relative error = ", error)

plt.rc("font", family="serif", size=22)

fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(24, 12))

matrix = np.fliplr(u).T
matrix = np.ma.masked_where(ide, matrix)
pcm = ax1.imshow(
    matrix,
    extent=[-dim_x / 2.0, dim_x / 2.0, -dim_x / 2.0, dim_x / 2.0],
    cmap=plt.cm.get_cmap("seismic"),
    interpolation="spline16",
    label="PINN",
)

fig.colorbar(pcm, ax=ax1)

matrix = np.fliplr(u_exact).T
matrix = np.ma.masked_where(ide, matrix)
pcm = ax2.imshow(
    matrix,
    extent=[-dim_x / 2.0, dim_x / 2.0, -dim_x / 2.0, dim_x / 2.0],
    cmap=plt.cm.get_cmap("seismic"),
    interpolation="spline16",
    label="Exact",
)

ax1.set_title("PINNs")
ax2.set_title("Exact")
fig.colorbar(pcm, ax=ax2)

# Add the boundary normal vectors
p = inner.random_boundary_points(16 * nx_train)
px, py = p.T
nx, ny = inner.boundary_normal(p).T
ax1.quiver(px, py, nx, ny)
ax2.quiver(px, py, nx, ny)
plt.savefig("plot_manufactured.pdf")