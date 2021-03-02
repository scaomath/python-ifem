#%%
import os, sys

from numpy.core.arrayprint import dtype_short_repr
current_path = os.path.dirname(os.path.abspath(__file__))
HOME = os.path.dirname(current_path)
sys.path.append(HOME)
from utils import *
from example.data import *
from libs.mesh import *
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.tri as tri
import plotly.figure_factory as ff
import plotly.io as pio
import plotly.graph_objects as go
plotly_template = pio.templates["plotly_dark"]

import math


# %%

def sparse_matlab(i, j, v, m, n):
    """
    Create and compressing a matrix that have many zeros
    Parameters:
        i: 1-D array representing the index 1 values 
            Size n1
        j: 1-D array representing the index 2 values 
            Size n1
        v: 1-D array representing the values 
            Size n1
        m: integer representing x size of the matrix >= n1
        n: integer representing y size of the matrix >= n1
    Returns:
        s: 2-D array
            Matrix full of zeros excepting values v at indexes i, j
    """
    return csr_matrix((v, (i, j)), shape=(m, n))

def setboundary(elem):
    '''
    Set up auxiliary data structures
    Type of bundary conditions:
    - Dirichlet
    - Neumann (todo)
    - Robin (todo)
    
    ported from Long Chen's iFEM
    '''

    nT = len(elem)
    allEdge = np.r_[elem[:,[1,2]], elem[:,[2,0]], elem[:,[0,1]]]

    allEdge = np.sort(allEdge, axis=1)
    nEdgeAll = len(allEdge)
    edge, E2e, e2E, counts = np.unique(allEdge, 
                            return_index=True, 
                            return_inverse=True, 
                            return_counts=True,
                            axis=0)
    # allEdge[E2e] = edge
    # edge[e2E] = allEdge
    elem2edge = e2E.reshape(3,-1).T
    isBdEdge = (counts==1)
    bdFlag = isBdEdge[e2E].reshape(3,-1).T     

    return bdFlag       


def quadpts(order=2):
    '''
    ported from iFEM's quadpts
    '''

    if order==1:     # Order 1, nQuad 1
        baryCoords = [1/3, 1/3, 1/3]
        weight = 1
    elif order==2:    # Order 2, nQuad 3
        baryCoords = [[2/3, 1/6, 1/6],
                  [1/6, 2/3, 1/6],
                  [1/6, 1/6, 2/3]]
        weight = [1/3, 1/3, 1/3]
    elif order==3:     # Order 3, nQuad 4
        baryCoords = [[1/3, 1/3, 1/3],
                        [0.6, 0.2, 0.2],
                        [0.2, 0.6, 0.2],
                         [0.2, 0.2, 0.6]]
        weight = [-27/48, 25/48, 25/48, 25/48]
    elif order==4:     # Order 4, nQuad 6
        baryCoords = [[0.108103018168070, 0.445948490915965, 0.445948490915965 ],
                  [0.445948490915965, 0.108103018168070, 0.445948490915965 ],
                  [0.445948490915965, 0.445948490915965, 0.108103018168070] ,
                  [0.816847572980459, 0.091576213509771, 0.091576213509771],
                  [0.091576213509771, 0.816847572980459, 0.091576213509771 ],
                  [0.091576213509771, 0.091576213509771, 0.816847572980459],]
        weight = [0.223381589678011, 0.223381589678011, 0.223381589678011,
                  0.109951743655322, 0.109951743655322, 0.109951743655322]
    return np.array(baryCoords), np.array(weight)
# %%
class TriMesh2D:
    '''
    Set up auxiliary data structures for Dirichlet boundary condition

    Combined setboundary, auxstructure, gradbasis
    from Long Chen's iFEM
    '''

    def __init__(self, node=None, elem=None) -> None:
        self.elem = elem
        self.node = node
    
    def update_auxstructure(self):
        node = self.node
        elem = self.elem
        N = len(node)
        allEdge = np.r_[elem[:,[1,2]], elem[:,[2,0]], elem[:,[0,1]]]

        allEdge = np.sort(allEdge, axis=1)
        self.edge, E2e, e2E, counts = np.unique(allEdge, 
                                return_index=True, 
                                return_inverse=True, 
                                return_counts=True,
                                axis=0)
        self.elem2edge = e2E.reshape(3,-1).T
        isBdEdge = (counts==1)
        self.bdFlag = isBdEdge[e2E].reshape(3,-1).T
        Dirichlet = self.edge[isBdEdge]
        self.isBdNode = np.zeros(N, dtype=bool) 
        self.isBdNode[Dirichlet.ravel()] = True

    def get_elem2edge(self):
        return self.elem2edge

    def get_bdFlag(self):
        return self.bdFlag

    def get_edge(self):
        return self.edge

    def compute_gradbasis(self):
        node = self.node
        elem = self.elem
        ve1 = node[elem[:,2]]-node[elem[:,1]]
        ve2 = node[elem[:,0]]-node[elem[:,2]]
        ve3 = node[elem[:,1]]-node[elem[:,0]]
        area = 0.5*(-ve3[:,0]*ve2[:,1] + ve3[:,1]*ve2[:,0])
        nT = len(elem)
        Dlambda = np.zeros((nT, 2, 3)) #(# elem, 2-dim vector, 3 vertices)

        Dlambda[...,2] = np.c_[-ve3[:,1]/(2*area), ve3[:,0]/(2*area)]
        Dlambda[...,0] = np.c_[-ve1[:,1]/(2*area), ve1[:,0]/(2*area)]
        Dlambda[...,1] = np.c_[-ve2[:,1]/(2*area), ve2[:,0]/(2*area)]

        self.area = area
        self.Dlambda = Dlambda

# %%
node, elem = rectangleMesh(x_range=(0,1), y_range=(0,1), h=1/8)
triangulation = tri.Triangulation(node[:,0], node[:,1], elem)
plt.triplot(triangulation, 'b-h')
plt.scatter(node[:,0], node[:,1], color='red')
plt.show()
# %%
pde = DataSinCos()

phi, weight = quadpts()
nQuad = len(phi)
# allEdge = np.r_[elem[:,[1,2]], elem[:,[2,0]], elem[:,[0,1]]]
# allEdge = np.sort(allEdge, axis=1)
# edge, E2e, e2E, counts = np.unique(allEdge, 
#                             return_index=True, 
#                             return_inverse=True, 
#                             return_counts=True,
#                             axis=0)
T = TriMesh2D(node,elem)
T.update_auxstructure()
T.compute_gradbasis()
isBdNode = T.isBdNode
Dphi = T.Dlambda
area = T.area
N = nDoF = len(node)
NT = len(elem)

K = np.zeros(NT)
for p in range(nQuad):
    # quadrature points in the x-y coordinate
    pxy = phi[p,0]*node[elem[:,0]] + phi[p,1]*node[elem[:,1]] + phi[p,2]*node[elem[:,2]]
    K += weight[p]*pde.d(pxy)
#%%
A = csr_matrix((N, N))
for i in range(3):
    for j in range(3):
        # $A_{ij}|_{\tau} = \int_{\tau}K\nabla \phi_i\cdot \nabla \phi_j dxdy$ 
        Aij = area*K*(Dphi[...,i]*Dphi[...,j]).sum(axis=-1)
        A += csr_matrix((Aij, (elem[:,i],elem[:,j])), shape=(nDoF,nDoF))     
# %%
b = np.zeros(nDoF)
bt = np.zeros((NT,3))

for p in range(nQuad):
    # quadrature points in the x-y coordinate
    pxy = phi[p,0]*node[elem[:,0]] + phi[p,1]*node[elem[:,1]] + phi[p,2]*node[elem[:,2]]
    fp = pde.f(pxy)
    for i in range(3):
        bt[:,i] += weight[p]*phi[p,i]*fp

bt *= area.reshape(-1,1)
b = np.bincount(elem.ravel(), weights=bt.ravel())
# %% Dirichlet
u = np.zeros(nDoF)
u[isBdNode] = pde.g_D(node[isBdNode])
b -= A.dot(u) 
# %%
freeNode = ~isBdNode
u[freeNode] = spsolve(A[freeNode,:][:,freeNode], b[freeNode])
# %%
surf = ff.create_trisurf(x=node[:,0], y=node[:,1], z=u,
                         simplices=elem,
                         colormap="Viridis",
                         showbackground=False,
                         title="FE approx", 
                         aspectratio=dict(x=1, y=1, z=1))
fig = go.Figure(data=surf)
fig.update_layout(template='plotly_dark')
fig.show()
# %%
