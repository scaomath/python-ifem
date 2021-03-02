import numpy as np


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