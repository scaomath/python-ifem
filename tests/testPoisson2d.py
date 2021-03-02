#%%
import os, sys

current_path = os.path.dirname(os.path.abspath(__file__))
HOME = os.path.dirname(current_path)
sys.path.append(HOME)
from libs.utils import *
from libs.mesh import *
from libs.fem import *
from example.data import *
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.tri as tri
import plotly.figure_factory as ff
import plotly.io as pio
import plotly.graph_objects as go

import math


# %%
node, elem = rectangleMesh(x_range=(0,1), y_range=(0,1), h=1/16)
triangulation = tri.Triangulation(node[:,0], node[:,1], elem)
plt.triplot(triangulation, 'b-h')
# plt.scatter(node[:,0], node[:,1], color='red')
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

# %% comparison
soln, _ = Poisson2DLite(T, pde)
# %%
surf = ff.create_trisurf(x=node[:,0], y=node[:,1], z=soln['u'],
                         simplices=elem,
                         colormap="Viridis", # similar to matlab's default colormap
                         showbackground=False,
                         title="FE approx", 
                         aspectratio=dict(x=1, y=1, z=1))
fig = go.Figure(data=surf)
fig.update_layout(template='plotly_dark')
fig.show()
# %%
center = node[elem].mean(axis=1)

DuPlot = ff.create_quiver(x=center[:,0], y=center[:,1], 
                       u=soln['Du'][:,0], v=soln['Du'][:,1],
                       scale=.02,
                       arrow_scale=.1,
                       name='quiver',
                       line_width=1.5)

DuPlot.add_trace(go.Scatter(x=node[:,0], y=node[:,1],
                    mode='markers',
                    marker_size=5,
                    name='vertices'))

fig = go.Figure(data=DuPlot)
fig.update_layout(template='plotly_dark',
                  margin=dict(l=20, r=20, t=20, b=20),)
fig.show()
 # %%
