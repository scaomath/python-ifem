import numpy as np
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import matplotlib.tri as tri

try:
    import plotly.figure_factory as ff
    import plotly.io as pio
    import plotly.graph_objects as go
except ImportError as e:
    print('Please install Plotly for showing mesh and solutions.')

def rectangleMesh(x_range=(0,1), y_range=(0,1), h=0.25):
    """ 
    Input: 
    - x's range, (x_min, x_max)
    - y's range, (y_min, y_max)
    - h, mesh size, can be a tuple
    Return the element matrix (NT, 3)
    of the mesh a np.meshgrid 
    """
    try:
        hx, hy = h[0], h[1]
    except:
        hx, hy = h, h

    # need to add h because arange is not inclusive
    xp = np.arange(x_range[0], x_range[1]+hx, hx)
    yp = np.arange(y_range[0], y_range[1]+hy, hy)
    nx, ny = len(xp), len(yp)

    x, y = np.meshgrid(xp, yp)
    
    elem = []
    for j in range(ny-1):
        for i in range(nx-1):      
            a = i + j*nx
            b = (i+1) + j*nx
            d = i + (j+1)*nx
            c = (i+1) + (j+1)*nx
            elem += [[a, c, d], [b, c, a]]

    node = np.c_[x.ravel(), y.ravel()]
    elem = np.asarray(elem, dtype=np.int32)
    return node, elem

def showmesh(node,elem, **kwargs):
    triangulation = tri.Triangulation(node[:,0], node[:,1], elem)
    markersize = 3000/len(node)
    if kwargs.items():
        h = plt.triplot(triangulation, 'b-h', **kwargs)
    else:
        h = plt.triplot(triangulation, 'b-h', linewidth=0.5, alpha=0.5, markersize=markersize)
    return h

def showsolution(node,elem,u,**kwargs):
    '''
    show 2D solution either of a scalar function or a vector field
    '''
    markersize = 300/len(node)
    u /= (np.abs(u)).max()
    if u.ndim == 1:
        us = ff.create_trisurf(x=node[:,0], y=node[:,1], z=u,
                            simplices=elem,
                            colormap="Viridis", # similar to matlab's default colormap
                            showbackground=False,
                            aspectratio=dict(x=1, y=1, z=1),
                            **kwargs)
        fig = go.Figure(data=us)
        fig.update_layout(template='plotly_dark')
        fig.show()
    elif u.ndim == 2 and u.shape[-1] == 2:
        center = node[elem].mean(axis=1)
        uvec = ff.create_quiver(x=center[:,0], y=center[:,1], 
                            u=u[:,0], v=u[:,1],
                            scale=.2,
                            arrow_scale=.1,
                            name='quiver',
                            line_width=1.5,
                            **kwargs)

        uvec.add_trace(go.Scatter(x=node[:,0], y=node[:,1],
                            mode='markers',
                            marker_size=markersize,
                            name='vertices'))

        fig = go.Figure(data=uvec)
        fig.update_layout(template='plotly_dark',
                        margin=dict(l=20, r=20, t=20, b=20),)
        fig.show()


def setboundary(elem):
    '''
    unused: for debugging purposes
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

class TriMesh2D:
    '''
    Set up auxiliary data structures for Dirichlet boundary condition

    To-do:
        - Add Neumann boundary.

    Combined the following routine from Long Chen's iFEM 
        - setboundary: get a boundary bool matrix according to elem
        - delmesh: delete mesh by eval()
        - auxstructure: edge-based auxiliary data structure
        - gradbasis: compute the gradient of local barycentric coords

    Input:
        - node: (N, 2)
        - elem: (NT, 3)
    
    Outputs:
        - edge: (NE, 2) global indexing of edges
        - elem2edge: (NT, 3) local to global indexing
        - edge2edge: (NE, 4)
          edge2elem[e,:2] are the global indexes of two elements sharing the e-th edge
          edge2elem[e,-2:] are the local indices of e to edge2elem[e,:2]
        - neighbor: (NT, 3) the local to global indices map of neighbor of elements
          neighbor[t,i] is the global index of the element opposite to the i-th vertex of the t-th element. 
    
    Notes: 
        1. Python assigns the first appeared entry's index in unique; Matlab assigns the last appeared entry's index in unique.
        2. Matlab uses columns as natural indexing, reshape(NT, 3) in Matlab should be changed to
        reshape(3, -1).T in Python if initially the data is concatenated along axis=0 using np.r_[].

    '''

    def __init__(self, node=None, elem=None) -> None:
        self.elem = elem
        self.node = node
    
    def update_auxstructure(self):
        node, elem = self.node, self.elem
        numElem = len(elem)
        numNode = len(node)

        # every edge's sign
        allEdge = np.r_[elem[:,[1,2]], elem[:,[2,0]], elem[:,[0,1]]]
        elem2edgeSign = np.ones(3*numElem, dtype=int)
        elem2edgeSign[allEdge[:,0] > allEdge[:, 1]] = -1
        self.elem2edgeSign = elem2edgeSign.reshape(3,-1).T
        allEdge = np.sort(allEdge, axis=1)

        # edge structures
        self.edge, E2e, e2E, counts = np.unique(allEdge, 
                                return_index=True, 
                                return_inverse=True, 
                                return_counts=True,
                                axis=0)
        self.elem2edge = e2E.reshape(3,-1).T
        isBdEdge = (counts==1)
        self.bdFlag = isBdEdge[e2E].reshape(3,-1).T
        Dirichlet = self.edge[isBdEdge]
        self.isBdNode = np.zeros(numNode, dtype=bool) 
        self.isBdNode[Dirichlet.ravel()] = True

        # neighbor structures
        E2e_reverse = np.zeros_like(E2e)
        E2e_reverse[e2E] = np.arange(3*numElem)

        k1 = E2e//numElem
        k2 = E2e_reverse//numElem
        t1 = E2e - numElem*k1
        t2 = E2e_reverse - numElem*k2
        ix = (counts == 2) # interior edge indicator

        self.neighbor = np.zeros((numElem, 3), dtype=int)
        ixElemLocalEdge1 = np.c_[t1[ix], k1[ix]]
        ixElemLocalEdge2 = np.c_[t2, k2]
        ixElemLocalEdge = np.r_[ixElemLocalEdge1, ixElemLocalEdge2]
        ixElem = np.r_[t2[ix],t1]
        for i in range(3):
            ix = (ixElemLocalEdge[:,1]==i) # i-th edge's neighbor
            self.neighbor[:,i] = np.bincount(ixElemLocalEdge[ix, 0], 
                                             weights=ixElem[ix], minlength=numElem)
        
        # edge to elem
        self.edge2elem = np.c_[t1, t2, k1, k2]

    def delete_mesh(self, expr=None):
        '''
        Update the mesh by deleting the eval(expr)
        '''
        assert expr is not None
        node, elem = self.node, self.elem
        center = node[elem].mean(axis=1)
        x, y = center[:,0], center[:,1]
        
        # delete element
        idx = eval(expr)
        mask = np.ones(len(elem), dtype=bool)
        mask[idx] = False
        elem = elem[mask]

        # re-mapping the indices of vertices
        # to remove the unused ones
        isValidNode = np.zeros(len(node), dtype=bool)
        indexMap = np.zeros(len(node), dtype=int)

        isValidNode[elem.ravel()] = True
        self.node = node[isValidNode]
    
        indexMap[isValidNode] = np.arange(len(self.node))
        self.elem = indexMap[elem]

    def update_gradbasis(self):
        node, elem = self.node, self.elem

        ve1 = node[elem[:,2]]-node[elem[:,1]]
        ve2 = node[elem[:,0]]-node[elem[:,2]]
        ve3 = node[elem[:,1]]-node[elem[:,0]]
        area = 0.5*(-ve3[:,0]*ve2[:,1] + ve3[:,1]*ve2[:,0])
        Dlambda = np.zeros((len(elem), 2, 3)) #(# elem, 2-dim vector, 3 vertices)

        Dlambda[...,2] = np.c_[-ve3[:,1]/(2*area), ve3[:,0]/(2*area)]
        Dlambda[...,0] = np.c_[-ve1[:,1]/(2*area), ve1[:,0]/(2*area)]
        Dlambda[...,1] = np.c_[-ve2[:,1]/(2*area), ve2[:,0]/(2*area)]

        self.area = area
        self.Dlambda = Dlambda

    def get_elem2edge(self):
        return self.elem2edge

    def get_bdFlag(self):
        return self.bdFlag

    def get_edge(self):
        return self.edge

    def get_gradbasis(self):
        try:
            return self.Dlambda
        except NameError:
            print("Run meshObj.update_gradbasis() first.")