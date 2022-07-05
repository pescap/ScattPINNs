Helmholtz sound-hard problem over a 2D circle with absorbing boundary conditions
===================================================================================

Problem setup
--------------

For a wavenumber :math:`k_0 = 2\pi n` with :math:`n = 2`, we will solve a sound-hard scattering problem for :math:`u = u^{scat} =  uRe + 1j * uIm:``

.. math:: - u_{xx}-u_{yy} - k_0^2 u = 0, \qquad  \Omega = B(0,R)^c

with the Neumann boundary conditions

.. math:: \gamma_1 u :=\nabla u(x,y) \cdot n =0, \qquad (x,y)\in \Gamma : =\partial \Omega

with :math:`n`, and suitable radiation conditions at infinity. The analytical formula for the scattered field is given by Bessel function.

We decide to approximate the radiation conditions by absorbing boundary condition, on a :math:`dim_x` square :math:`\Gamma^{out}`. 

Projection to the real and imaginary axes for :math:`u = uR + 1j * uIm` leads to:

.. math:: - uRe_{xx}-uRe_{yy} - k_0^2 uRe = 0, \qquad  \Omega \cap D^{out}

and

.. math:: - uIm_{xx}-uIm_{yy} - k_0^2 uIm = 0, \qquad  \Omega \cap D^{out}

The boundary conditions read:

.. math::\gamma_1 u =  - \gamma_1 u^{inc}, \qquad \Gamma
.. math:: \gamma_1 u - \imath k_0 \gamma_0 = 0, \qquad \Gamma^{out}.

Absorbing boundary conditions rewrite:

.. math:: \gamma_1 [uRe + \imath  uIm] - \imath k_0 [uRe + \imath uIm] = 0, \qquad \Gamma^{out}

i.e.

.. math:: \gamma_1 uRe + k_0 uIm = 0, \qquad \Gamma^{out}
.. math:: \gamma_1 uIm - k_0 uRe = 0, \qquad \Gamma^{out}.


This example is inspired by `this Dolfinx tutorial <https://github.com/samuelpgroth/waves-fenicsx/tree/master/frequency>`_.

Implementation
--------------

This description goes through the implementation of a solver for the above scattering problem step-by-step.

First, the DeepXDE and required modules are imported:

.. code-block:: python

  import numpy as np
  import matplotlib.pyplot as plt
  import deepxde as dde
  from scipy.special import jv, hankel1

Then, we begin by definying the general parameters for the problem. The PINN will be trained over 5000 iterations, we also define the learning rate, the number of dense layers and nodes, and the activation function.

.. code-block:: python

  #General parameters
  weights = 1
  epochs = 5000 
  parameters = [1e-2, 4, 50, "tanh"]
  learning_rate, num_dense_layers, num_dense_nodes, activation = parameters

We set the physical parameters for the problem.

.. code-block:: python

  #Problem parameters
  k0 = 1
  wave_len = np.pi / k0
  dim_x = 2 * np.pi
  R = np.pi / 2.
  n_wave = 10
  h_elem = wave_len / n_wave
  nx = int(dim_x / h_elem)

We define the geometry (inner and outer domains):

.. code-block:: python

  #Computational domain
  outer = dde.geometry.Rectangle([-dim_x/2., -dim_x/2.], [dim_x/2., dim_x/2.])
  inner = dde.geometry.Disk([0,0], R)
  geom = dde.geometry.CSGDifference(outer, inner)


We define the analytic solution:

.. code-block:: python

  def sound_hard_circle_deepxde(k0, a, points):
    
    fem_xx = points[:, 0:1]
    fem_xy = points[:, 1:2]
    r = np.sqrt(fem_xx * fem_xx + fem_xy * fem_xy)
    theta = np.arctan2(fem_xy, fem_xx)
    npts = np.size(fem_xx, 0)
    n_terms = np.int(30 + (k0 * a)**1.01)
    u_sc = np.zeros((npts), dtype=np.complex128)
    for n in range(-n_terms, n_terms):
        bessel_deriv = jv(n-1, k0*a) - n/(k0*a) * jv(n, k0*a)
        hankel_deriv = n/(k0*a)*hankel1(n, k0*a) - hankel1(n+1, k0*a)
        u_sc += (-(1j)**(n) * (bessel_deriv/hankel_deriv) * hankel1(n, k0*r) * \
            np.exp(1j*n*theta)).ravel()
    return u_sc

Next, we express the PDE residual of the Helmholtz equation:

.. code-block:: python

  #Definition of the pde
  def pde(x, y):
      y0, y1 = y[:, 0:1], y[:, 1:2]
      
      y0_xx = dde.grad.hessian(y, x, component=0, i=0, j=0)
      y0_yy = dde.grad.hessian(y, x, component=0, i=1, j=1)

      y1_xx = dde.grad.hessian(y, x,component=1, i=0, j=0)
      y1_yy = dde.grad.hessian(y, x,component=1, i=1, j=1)

      return [-y0_xx - y0_yy - k0 ** 2 * y0,
              -y1_xx - y1_yy - k0 ** 2 * y1]


Then, we introduce the exact solution and both Neumann and Robin boundary conditions:

.. code-block:: python

  def sol(x):
      result = sound_hard_circle_deepxde(k0, R, x).reshape((x.shape[0],1))
      real = np.real(result)
      imag = np.imag(result)
      return np.hstack((real, imag))

  #Boundary conditions
  def boundary(_, on_boundary):
      return on_boundary

  def boundary_outer(_, on_boundary):
      return on_boundary and outer.on_boundary(_)

  def boundary_inner(_, on_boundary):
      return on_boundary and inner.on_boundary(_)

  def func0_inner(x):
      normal = -inner.boundary_normal(x)
      g = 1j * k0 * np.exp(1j * k0 * x[:, 0:1]) * normal[:, 0:1]
      return np.real(-g)

  def func1_inner(x):
      normal = -inner.boundary_normal(x)
      g = 1j * k0 * np.exp(1j * k0 * x[:, 0:1]) * normal[:, 0:1]
      return np.imag(-g)

  def func0_outer(x, y):
      normal = outer.boundary_normal(x)
      result = - k0 * y[:, 1:2]
      return result

  def func1_outer(x, y):
      normal = outer.boundary_normal(x)
      result =  k0 * y[:, 0:1]
      return result
   
  #ABC 
  bc0_inner = dde.NeumannBC(geom, func0_inner, boundary_inner, component = 0)
  bc1_inner = dde.NeumannBC(geom, func1_inner, boundary_inner, component = 1)

  bc0_outer = dde.RobinBC(geom, func0_outer, boundary_outer, component = 0)
  bc1_outer = dde.RobinBC(geom, func1_outer, boundary_outer, component = 1)

  bcs = [bc0_inner, bc1_inner, bc0_outer, bc1_outer]


Next, we define the weights for the loss function and generate the training and testing points.

.. code-block:: python

  loss_weights = [1, 1, weights, weights, weights, weights]
  data = dde.data.PDE(geom, pde, bcs, num_domain= nx**2, num_boundary= 8 * nx, num_test= 5 * nx ** 2, solution = sol)


Next, we choose the network. Here, we use a fully connected neural network of depth 4 (i.e., 3 hidden layers) and width 50. Besides, we choose sin as activation function and Glorot uniform as initializer :

.. code-block:: python

  net = dde.maps.FNN([2] + [num_dense_nodes] * num_dense_layers + [2], activation, "Glorot uniform")

Now, we have the PDE problem and the network. We build a ``Model`` and define the optimizer and learning rate.

.. code-block:: python

  model.compile("adam", lr=learning_rate, loss_weights=loss_weights , metrics=["l2 relative error"])

We first train the model for 5000 iterations with Adam optimizer:

.. code-block:: python

    losshistory, train_state = model.train(epochs=epochs)


Complete code
--------------

.. literalinclude:: ../../../examples/forward/Helmholtz_Sound_hard_ABC_2d.py
  :language: python
