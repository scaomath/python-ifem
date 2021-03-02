#%% imports
import numpy as np
from libs.mesh import *
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

#%% Finite element methods

def Poisson2DLite(mesh, pde): 
    '''
    A lightweight port of the Poisson
    from Long Chen's iFEM library

    Linear Lagrange element on triangulations
    '''
    node = mesh.node
    elem = mesh.elem
    isBdNode = mesh.isBdNode
    Dphi = mesh.Dlambda
    area = mesh.area

    N = len(node)
    NT = len(elem)

    phi, weight = quadpts()
    nQuad = len(phi)

    # diffusion coeff
    K = np.zeros(NT)
    for p in range(nQuad):
        # quadrature points in the x-y coordinate
        pxy = phi[p,0]*node[elem[:,0]] + phi[p,1]*node[elem[:,1]] + phi[p,2]*node[elem[:,2]]
        K += weight[p]*pde.d(pxy)

    # stiffness matrix
    A = csr_matrix((N, N))
    for i in range(3):
        for j in range(3):
            # $A_{ij}|_{\tau} = \int_{\tau}K\nabla \phi_i\cdot \nabla \phi_j dxdy$ 
            Aij = area*K*(Dphi[...,i]*Dphi[...,j]).sum(axis=-1)
            A += csr_matrix((Aij, (elem[:,i],elem[:,j])), shape=(N,N))     

    # right hand side
    b = np.zeros(N)
    bt = np.zeros((NT,3))

    for p in range(nQuad):
        # quadrature points in the x-y coordinate
        pxy = phi[p,0]*node[elem[:,0]] + phi[p,1]*node[elem[:,1]] + phi[p,2]*node[elem[:,2]]
        fp = pde.f(pxy)
        for i in range(3):
            bt[:,i] += weight[p]*phi[p,i]*fp

    bt *= area.reshape(-1,1)
    b = np.bincount(elem.ravel(), weights=bt.ravel())

    # Dirichlet
    u = np.zeros(N)
    u[isBdNode] = pde.g_D(node[isBdNode])
    b -= A.dot(u) 

    # Direct solve
    freeNode = ~isBdNode
    u[freeNode] = spsolve(A[freeNode,:][:,freeNode], b[freeNode])

    # compute Du
    dudx =  (u[elem]*Dphi[:,0,:]).sum(axis=-1)
    dudy =  (u[elem]*Dphi[:,1,:]).sum(axis=-1)      
    Du = np.c_[dudx, dudy]

    soln = {'u': u,
            'Du': Du}
    eqn = {'A': A,
           'b': b,
           'freeNode': freeNode}

    return soln, eqn