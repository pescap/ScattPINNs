from dolfin import *
import numpy

a=0.1;
t=a/10;

def compProd(a1,b1,a2,b2):

        x1=a1*a2-b1*b2;
        x2=a1*b2+a2*b1;
        return (x1, x2)


class LR(UserExpression):
    def eval(self, value, x):
        a=0.1;t=a/10;

        b1 = numpy.abs(x[0])-a/2; f1=10*(b1+numpy.abs(b1))/(2*t);
        b2 = numpy.abs(x[1])-a/2; f2=10*(b2+numpy.abs(b2))/(2*t);
        l1R=1+f1;l1I=-f1;
        l2R=1+f2;l2I=-f2;

        value[0] = l1R/(pow(l1R,2)+pow(l1I,2));
        value[3] = l2R/(pow(l2R,2)+pow(l2I,2));
        value[1] = 0;
        value[2] = 0;

    def value_shape(self):
        return (2,2)

class LI(UserExpression):
    def eval(self, value, x):
        a=0.1;t=a/10;

        b1 = numpy.abs(x[0])-a/2; f1=10*(b1+numpy.abs(b1))/(2*t);
        b2 = numpy.abs(x[1])-a/2; f2=10*(b2+numpy.abs(b2))/(2*t);
        l1R=1+f1;l1I=-f1;
        l2R=1+f2;l2I=-f2;

        value[0] = -l1I/(pow(l1R,2)+pow(l1I,2));
        value[3] = -l2I/(pow(l2R,2)+pow(l2I,2));
        value[1] = 0;
        value[2] = 0;

    def value_shape(self):
        return (2,2)


class lR(UserExpression):
    def eval(self, value, x):
        a=0.1;t=a/10;

        b1 = numpy.abs(x[0])-a/2; f1=10*(b1+numpy.abs(b1))/(2*t);
        b2 = numpy.abs(x[1])-a/2; f2=10*(b2+numpy.abs(b2))/(2*t);
        l1R=1+f1;l1I=-f1;
        l2R=1+f2;l2I=-f2;
        value[0], temp = compProd(l1R,l1I,l2R,l2I);

#    def value_shape(self):
#        return (1,)

class lI(UserExpression):
    def eval(self, value, x):
        a=0.1;t=a/10;

        b1 = numpy.abs(x[0])-a/2; f1=10*(b1+numpy.abs(b1))/(2*t);
        b2 = numpy.abs(x[1])-a/2; f2=10*(b2+numpy.abs(b2))/(2*t);
        l1R=1+f1;l1I=-f1;
        l2R=1+f2;l2I=-f2;
        temp, value[0] = compProd(l1R,l1I,l2R,l2I);

#    def value_shape(self):
#        return (1,)

LambdaR=LR(degree=1)
LambdaI=LI(degree=1)
LLR=lR(degree=1)
LLI=lI(degree=1)


frequency=50000
omega=2*numpy.pi*frequency



#Mesh information
##########################################################
mesh = Mesh("pml-mesh.xml");
#cd=MeshFunction('size_t',mesh,"geom_physical_region.xml");
cd=MeshFunction('size_t',mesh, 2)#"geom_physical.xml");
fd=MeshFunction('size_t',mesh, 2)
#fd=MeshFunction('size_t',mesh,"geom_facet_region.xml");

#Function space definition over the mesh
#########################################################
V = VectorFunctionSpace(mesh, "CG", 1) ##Continuous Galerkin for the displacement field
Vcomplex = V*V
V0=FunctionSpace(mesh, 'DG', 0) ##Discontinuous Galerkin for the material property fields
M0=TensorFunctionSpace(mesh, 'DG', 0, shape=(2,2,2,2))


#f0=LR(element=M0.ufl_element())
#print f0(0,0)
####Material properties
############################################################
E, rho, nu = 300E+8, 8000, 0.3

lam=E*nu/((1+nu)*(1-2*nu))
mu=E/(2*(1+nu))
i,j,k,l=indices(4)
delta=Identity(2)


C=as_tensor((lam*(delta[i,j]*delta[k,l])+mu*(delta[i,k]*delta[j,l]+delta[i,l]*delta[j,k])),(i,j,k,l))
#############################################################



#Applying the Boundary conditions
###########################################################
zero=Constant((0.0, 0.0))
one=Constant((1.0, 0.0))

#boundary = [DirichletBC(Vcomplex.sub(0), zero, fd, 12), DirichletBC(Vcomplex.sub(1), zero, fd, 12)]
source = [DirichletBC(Vcomplex.sub(0), one, fd, 27), DirichletBC(Vcomplex.sub(1), one, fd, 27)]

#bc=boundary+source
bc=source

###########################################################



#Real and Imaginary parts of the trial and test functions
uR, uI = TrialFunctions(Vcomplex)
wR, wI = TestFunctions(Vcomplex)

strainR=0.5*(as_tensor((Dx(uR[i],k)*LambdaR[k,j]),(i,j))  +  as_tensor((Dx(uR[k],i)*LambdaR[j,k]),(i,j)))  -  0.5*(as_tensor((Dx(uI[i],k)*LambdaI[k,j]),(i,j))  +  as_tensor((Dx(uI[k],i)*LambdaI[j,k]),(i,j)))
strainI=0.5*(as_tensor((Dx(uR[i],k)*LambdaI[k,j]),(i,j))  +  as_tensor((Dx(uR[k],i)*LambdaI[j,k]),(i,j)))  +  0.5*(as_tensor((Dx(uI[i],k)*LambdaR[k,j]),(i,j))  +  as_tensor((Dx(uI[k],i)*LambdaR[j,k]),(i,j)))


tempR=LLR*LambdaR-LLI*LambdaI
tempI=LLR*LambdaI+LLI*LambdaR
sR=as_tensor((C[i,j,k,l]*strainR[k,l]),[i,j])
sI=as_tensor((C[i,j,k,l]*strainI[k,l]),[i,j])
stressR=as_tensor((sR[i,j]*tempR[j,k]),(i,k))  -  as_tensor((sI[i,j]*tempI[j,k]),(i,k))
stressI=as_tensor((sR[i,j]*tempI[j,k]),(i,k))  +  as_tensor((sI[i,j]*tempR[j,k]),(i,k))


F=omega**2*rho*(dot((LLR*uR-LLI*uI),wR)+dot((LLR*uI+LLI*uR),wI))*dx-(inner(stressR, grad(wR))+inner(stressI, grad(wI)))*dx
a, L = lhs(F), rhs(F)

# Set up the PDE
solnU = Function(Vcomplex)
problem = LinearVariationalProblem(a, L, solnU, bcs=bc)
solver = LinearVariationalSolver(problem)
solver.solve()
solnUR, solnUI=solnU.split()
temp=solnUI[0]**2+solnUI[1]**2
#plot(temp, mode = "displacement", wireframe=False,rescale=True)
plot(solnUI)
interactive()
                                                                                                  
