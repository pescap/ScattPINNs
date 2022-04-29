"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch"""
import deepxde as dde

#dde.backend.backend_name = 'tensorflow'
#dde.backend.backend_name = 'tensorflow.compat.v1'
dde.config.set_default_float('float32')

def pde(x, y):
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    dy_yy = dde.grad.hessian(y, x, i=1, j=1)
    print(x,y)
    return -dy_xx -dy_yy - 1


def boundary(_, on_boundary):
    return on_boundary


#geom = dde.geometry.Rectangle([0,0], [1,1])
#geom = dde.geometry.Disk([0,0], 1)
geom = dde.geometry.Polygon([[0, 0], [1, 0], [1, -1], [-1, -1], [-1, 1], [0, 1]])
bc = dde.DirichletBC(geom, lambda x: 0, boundary)

data = dde.data.PDE(geom, pde, bc, num_domain=2000, num_boundary=200, num_test=2000)
net = dde.maps.FNN([2] + [200] * 8 + [1], "tanh", "Glorot uniform")
model = dde.Model(data, net)
import time 
ta = time.time()
model.compile("adam", lr=0.001)
model.train(epochs=10000)
#model.compile("L-BFGS")
#model.train(epochs=1000)

print(time.time()-ta, 'time')
