import deepxde as dde

#Computational domain
outer = dde.geometry
inner = dde.geometry

#Definition of the pde
def pde(x, y):
    y0, y1 = y[:, 0:1], y[:, 1:2]
    
    y0_xx = dde.grad.hessian(y, x, component=0, i=0, j=0)
    y0_yy = dde.grad.hessian(y, x, component=0, i=1, j=1)

    y1_xx = dde.grad.hessian(y, x,component=1, i=0, j=0)
    y1_yy = dde.grad.hessian(y, x,component=1, i=1, j=1)

    return [-y0_xx - y0_yy - k0 ** 2 * y0,
            -y1_xx - y1_yy - k0 ** 2 * y1]
 #Boundaries
 def boundary(_, on_boundary):
    return on_boundary

def boundary_outer(_, on_boundary):
    return on_boundary and outer.on_boundary(_)

def boundary_inner(_, on_boundary):
    return on_boundary and inner.on_boundary(_)
 
#ABC 
bc0_inner = dde.NeumannBC(geom, , boundary_inner, component = 0)
bc1_inner = dde.NeumannBC(geom, , boundary_inner, component = 1)

bc0_outer = dde.RobinBC(geom, , boundary_outer, component = 0)
bc1_outer = dde.RobinBC(geom, , boundary_outer, component = 1)
