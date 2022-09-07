from matplotlib import pyplot as plt 
import numpy as np
import deepxde as dde
from deepxde.backend import tf

dde.config.real.set_float64()

# En nuestro caso:
#BOX = np.array([[-length /2, - length /2], [length/2, length/2]])
BOX = np.array([[-2, -2], [2, 3]])
DPML = 1

OMEGA = 2 * np.pi
SIGMA0 = -np.log(1e-20) / (4 * DPML ** 3 / 3)
N = 1

def PML(X):
	def sigma(x, a, b):
		def _sigma(d):
			#return SIGMA0 * d ** 2 * np.heaviside(d, 0)
			heav = tf.numpy_function(np.heaviside, [d,0], tf.float64)
			return SIGMA0 * d ** 2 * heav

		return tf.cast(_sigma(a - x) + _sigma(x - b), tf.complex128)

	def dsigma(x, a, b):
		def _sigma(d):
			heav = tf.numpy_function(np.heaviside, [d,0], tf.float64)
			return 2 * SIGMA0 * d * heav

		return tf.cast(-_sigma(a - x) + _sigma(x - b), tf.complex128)

	sigma_x = sigma(X[:, :1], BOX[0][0], BOX[1][0])
	AB1 = 1 / (1 + 1j / OMEGA * sigma_x) ** 2
	A1, B1 = tf.math.real(AB1), tf.math.imag(AB1)

	dsigma_x = dsigma(X[:, :1], BOX[0][0], BOX[1][0])
	AB2 = -1j / OMEGA * dsigma_x * AB1 / (1 + 1j / OMEGA * sigma_x)
	A2, B2 = tf.math.real(AB2), tf.math.imag(AB2)

	sigma_y = sigma(X[:, 1:], BOX[0][1], BOX[1][1])
	AB3 = 1 / (1 + 1j / OMEGA * sigma_y) ** 2
	A3, B3 = tf.math.real(AB3), tf.imag(AB3)

	dsigma_y = dsigma(X[:, 1:], BOX[0][1], BOX[1][1])
	AB4 = -1j / OMEGA * dsigma_y * AB3 / (1 + 1j / OMEGA * sigma_y)
	A4, B4 = tf.math.real(AB4), tf.math.imag(AB4)
	return A1, B1, A2, B2, A3, B3, A4, B4


geom = dde.geometry.Rectangle(BOX[0] - DPML, BOX[1] + DPML)
geom = dde.geometry.Rectangle([0,0], [1,1])


k0 = OMEGA * N
wave_len = 1 / N

hx = wave_len / 10
nx = int(1 / hx)

print(hx, nx)

hx_test = wave_len / 20
nx_test = int(1/hx_test)
print(nx, 'nx')
print(nx **2, 'nx **2')

def boundary(_, on_boundary):
        return on_boundary

bc0 = dde.DirichletBC(geom, lambda x: 0, boundary, component=0)
bc1 = dde.DirichletBC(geom, lambda x: 0, boundary, component=1)

def pde(x, y):
		#Apply the PML
		A1, B1, A2, B2, A3, B3, A4, B4 = PML(x)

		y0, y1 = y[:, 0:1], y[:, 1:2]
		y0_x = dde.grad.jacobian(y, x, i=0, j=0)
		y0_y = dde.grad.jacobian(y, x, i=0, j=1)

		y0_xx = dde.grad.hessian(y, x, component=0, i=0, j=0)
		y0_yy = dde.grad.hessian(y, x, component=0, i=1, j=1)

		y1_x = dde.grad.jacobian(y, x, i=1, j=0)
		y1_y = dde.grad.jacobian(y, x, i=1, j=1)
		
		y1_xx = dde.grad.hessian(y, x,component=1, i=0, j=0)
		y1_yy = dde.grad.hessian(y, x,component=1, i=1, j=1)

		f0 = k0 ** 2 * tf.sin(k0 * x[:, 0:1]) * tf.sin(k0 * x[:, 1:2]) 
		f1 = k0 ** 2 * tf.sin(k0 * x[:, 0:1]) * tf.sin(k0 * x[:, 1:2]) 

		loss_y0 = -(A1 * y0_xx + A2 * y0_x + A3 * y0_yy + A4 * y0_y) / OMEGA+(B1 * y1_xx + B2 * y1_x + B3 * y1_yy + B4 * y1_y) / OMEGA + OMEGA * y0

		loss_y1 = -(A1 * y1_xx + A2 * y1_x + A3 * y1_yy + A4 * y1_y) / OMEGA- (B1 * y0_xx + B2 * y0_x + B3 * y0_yy + B4 * y0_y) / OMEGA- OMEGA * y1

		return [loss_y0,
		        loss_y1]


data = dde.data.PDE(geom, pde, [bc0, bc1], num_domain= nx ** 2, num_boundary= 4 * nx, num_test = nx_test ** 2)#,  solution = func)


net = dde.maps.FNN([2] + [50] * 4 + [2], "tanh", "Glorot uniform")
model = dde.Model(data, net)

weights = [1, 1, 1000, 1000]
model.compile("adam", lr=0.005,  metrics=["l2 relative error"])#, loss_weights = weights)
losshistory, train_state = model.train(epochs=10000)
#model.compile("L-BFGS", lr=0.001,  metrics=["l2 relative error"], loss_weights = weights)
#losshistory, train_state = model.train(epochs = 5000)

X = data.train_points()
K = PML(X)