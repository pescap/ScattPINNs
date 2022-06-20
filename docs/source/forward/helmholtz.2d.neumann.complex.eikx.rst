Complex Helmholtz equation with Euler term
==========================================

Problem setup
--------------

Work in progress. 


For a wavenumber :math:`k_0 = 2\pi n` with :math:`n = 2`, we will solve a Helmholtz equation for :math:`u = uRe + 1j * uIm:``

.. math:: - u_{xx}-u_{yy} - k_0^2 u = f, \qquad  \Omega = [0,1]^2


with the Neumann boundary conditions

.. math:: \nabla u(x,y) \cdot n =0, \qquad (x,y)\in \partial \Omega

with :math:`n` the normal exterior vector and a source term 

.. math:: f(x,y) = k_0^2 (k_0 x)\cos(k_0 y) = fRe + 1j fIm.

Remark that the exact solution reads:

.. math:: u(x,y)=(1 + 1j) \cos(k_0 x)\cos(k_0 y)

Projection to the real and imaginary axes for :math:`u = uR + 1j * uIm` leads to:

.. math:: - uRe_{xx}-uRe_{yy} - k_0^2 uRe = fRe, \qquad  \Omega = [0,1]^2

and

.. math:: - uIm_{xx}-uIm_{yy} - k_0^2 uIm = fIm, \qquad  \Omega = [0,1]^2

This example is the Neumann boundary condition conterpart to `this Dolfinx tutorial <https://github.com/FEniCS/dolfinx/blob/main/python/demo/helmholtz2D/demo_helmholtz_2d.py>`_. One can refer to Ihlenburg\'s book \"Finite Element Analysis of Acoustic Scattering\" p138-139 for more details.

Implementation
--------------

This description goes through the implementation of a solver for the above described Helmholtz equation step-by-step.

First, the DeepXDE and Numpy modules are imported:

.. code-block:: python

  import deepxde as dde
  import numpy as np

We begin by definying the general parameters for the problem. We use a collocation points density of 10 (resp. 30) points per wavelength for the training (resp. testing) data along each direction.
This code allows to use both soft and hard boundary conditions. 

.. code-block:: python

  n = 2
  precision_train = 10
  precision_test = 10
  weights = 100

The PINN will be trained over 50000 epochs. We define the learning rate, the number of dense layers and nodes, and the activation function. Moreover, we import the cosine function.

.. code-block:: python

  epochs = 50000
  parameters = [1e-3, 3, 150, "sin"]

  # Define sine function
  if dde.backend.backend_name == "pytorch":
      cos = dde.backend.pytorch.cos
  else:
      from deepxde.backend import tf

      cos = tf.cos
      
  learning_rate, num_dense_layers, num_dense_nodes, activation = parameters

Next, we express the PDE residual of the Helmholtz equation:

.. code-block:: python

  def pde(x, y):
    yRe, yIm = y[:, 0:1], y[:, 1:2]
    
    
    dyRe_xx = dde.grad.hessian(y, x, component=0, i=0, j=0)
    dyRe_yy = dde.grad.hessian(y, x, component=0, i=1, j=1)
    
    dyIm_xx = dde.grad.hessian(y, x, component=1, i=0, j=0)
    dyIm_yy = dde.grad.hessian(y, x, component=1, i=1, j=1)
    

    fRe = k0 ** 2 * cos(k0 * x[:, 0:1]) * cos(k0 * x[:, 1:2])
    fIm = k0 ** 2 * cos(k0 * x[:, 0:1]) * cos(k0 * x[:, 1:2])
    
    return [-dyRe_xx - dyRe_yy - k0 ** 2 * yRe - fRe,
            -dyIm_xx - dyIm_yy - k0 ** 2 * yIm - fIm]



The first argument to ``pde`` is the network input, i.e., the :math:`x`-coordinate and :math:`y`-coordinate. The second argument is the network output, i.e., the solution :math:`u(x)`, but here we use ``y`` as the name of the variable.

Next, we introduce the exact solution and the Neumann boundary condition for a complex domain. 

.. code-block:: python

  def func(x):
    real = np.cos(k0 * x[:, 0:1]) * np.cos(k0 * x[:, 1:2])
    imag = np.cos(k0 * x[:, 0:1]) * np.cos(k0 * x[:, 1:2])
    return np.hstack((real, imag))

  def boundary(_, on_boundary):
      return on_boundary

Now, we define the geometry and evaluate the number of training and test random collocation points. The values allow to obtain collocation points density of 10 (resp. 30) points per wavelength along each direction.
We define the boundary and the Neumann boundary conditions. 

.. code-block:: python

  geom = dde.geometry.Rectangle([0, 0], [1, 1])
  k0 = 2 * np.pi * n
  wave_len = 1 / n

  hx_train = wave_len / precision_train
  nx_train = int(1 / hx_train)

  hx_test = wave_len / precision_test
  nx_test = int(1 / hx_test)

  bcRe = dde.icbc.NeumannBC(geom, lambda x: 0, boundary, component=0)
  bcIm = dde.icbc.NeumannBC(geom, lambda x: 0, boundary, component=1)


Next, we generate the training and testing points.

.. code-block:: python

  data = dde.data.PDE(
      geom,
      pde,
      bc,
      num_domain=nx_train ** 2,
      num_boundary=4 * nx_train,
      solution=func,
      num_test=nx_test ** 2,
  )

Next, we choose the network. Here, we use a fully connected neural network of depth 4 (i.e., 3 hidden layers) and width 150. Besides, we choose sin as activation function and Glorot uniform as initializer :

.. code-block:: python

  net = dde.nn.FNN(
    [2] + [num_dense_nodes] * num_dense_layers + [1], activation, "Glorot uniform"
  )


Now, we have the PDE problem and the network. We build a ``Model`` and define the optimizer and learning rate.

.. code-block:: python

  model = dde.Model(data, net)

  if hard_constraint == True:
      model.compile("adam", lr=learning_rate, metrics=["l2 relative error"])
  else:
      loss_weights = [1, 1, weights, weights]
      model.compile(
          "adam",
          lr=learning_rate,
          metrics=["l2 relative error"],
          loss_weights=loss_weights,
      )

We first train the model for 5000 iterations with Adam optimizer:

.. code-block:: python

    losshistory, train_state = model.train(epochs=epochs)

Complete code
--------------

.. literalinclude:: ../../../examples/forward/Helmholtz_Neumann_2d_complex_eikx.py
  :language: python
