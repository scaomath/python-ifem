import math
import numpy as np

class DataSinCos:
    '''
    Trigonometric data for Poisson equation

        f = 2*pi^2*cos(pi*x)*cos(pi*y);
        u = cos(pi*x)*cos(pi*y);
        Du = (-pi*sin(pi*x)*cos(pi*y), -pi*cos(pi*x)*sin(pi*y));

    The u satisfies the zero flux condition du/dn = 0 on boundary of [0,1]^2
    and thus g_N is not assigned.
    
    Ported from Long Chen's iFEM package to Python
    '''
    def __init__(self):
        self.pi = math.pi

    def f(self, p):
        x = p[:,0]; y = p[:,1]
        return 2*self.pi**2*np.cos(self.pi*x)*np.cos(self.pi*y)

    def exactu(self, p):
        x = p[:,0]; y = p[:,1]
        return np.cos(self.pi*x)*np.cos(self.pi*y)
    
    def g_D(self,p):
        return self.exactu(p)
    
    def Du(self,p):
        x = p[:,0]; y = p[:,1]
        Dux = -self.pi*np.sin(self.pi*x)*np.cos(self.pi*y)
        Duy = -self.pi*np.cos(self.pi*x)*np.sin(self.pi*y)
        return np.c_[Dux, Duy]

    def d(self,p):
        return np.ones(p.shape[0], dtype=float)