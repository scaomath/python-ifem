#%%
import os, sys
current_path = os.path.dirname(os.path.abspath(__file__))
HOME = os.path.dirname(current_path)
sys.path.append(HOME)
from utils import *

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.tri as mtri

#%%
def rectangleMesh(n, m):
    """ 
    Input: 
    - x's range, (x_min, x_max)
    - y's range, (y_min, y_max)
    - h, mesh size, if 
    triangles to mesh a np.meshgrid of n x m points 
    """
    triangles = []
    for i in range(n-1):
        for j in range(m-1):
            a = i + j*(n)
            b = (i+1) + j*n
            d = i + (j+1)*n
            c = (i+1) + (j+1)*n
            triangles += [[a, b, c], [a, c, d]]
    return np.asarray(triangles, dtype=np.int32)
# %%
xp = np.arange(-1, 1, 1/4)
yp = np.arange(-1, 1, 1/4)
x, y = np.meshgrid(xp, yp)
triangles = rectangleMesh(len(xp), len(yp))
triangulation = mtri.Triangulation(x.ravel(), y.ravel(), triangles)
plt.triplot(triangulation, 'b-h')
plt.scatter(x, y, color='red')
plt.show()
# %%
