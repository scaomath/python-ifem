import math
import numpy as np

class DataSinCos:
    '''
    Trigonometric data for Poisson equation

        f = 2*pi^2*np.cos(pi*x)*np.cos(pi*y);
        u = cos(pi*x)*np.cos(pi*y);
        Du = (-pi*np.sin(pi*x)*np.cos(pi*y), -pi*np.cos(pi*x)*np.sin(pi*y));

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


class DataKellogg:
    '''
    Kellogg data for elliptic interface problem

        u = r^{gamma} sin(mu \theta)
        theta is the polar angle either in cylindrical or polar coords
        Reference Z. Chen and S. Dai 2002 SISC
        "On the efficiency of adaptive finite element methods for elliptic problems
        with discontinuous coefficients"
    
    Ported from Long Chen's iFEM package to Python
    '''

    def __init__(self, gamma=0.1):
        self.pi = math.pi
        self.rho = self.pi/4
        self.gamma = gamma
        if gamma == 0.1:
            self.sigma = -14.92256510455152
            self.R = 161.4476387975881
        elif gamma == 0.5:
            self.sigma = -2.3561944901923448
            self.R = 5.8284271247461907
        elif gamma == 0.02:
            self.sigma = -77.754418176347386
            self.R = 4052.1806954768103
        self.eps = 1e-17

    def f(self, p):
        return np.zeros_like(p[:,0])

    def d(self,p):
        K = np.ones(p.shape[0], dtype=float)
        K[p[:,0]*p[:,1] > 0] = self.R
        return K

    def exactu(self, p):
        pi = self.pi
        sigma = self.sigma
        rho = self.rho
        gamma = self.gamma

        x = p[:,0]; y = p[:,1]
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        theta[theta<0] += 2*pi
        mu = (theta>=0) * (theta<pi/2) * np.cos((pi/2-sigma)*gamma)*np.cos((theta-pi/2+rho)*gamma)\
           + (theta>=pi/2) * (theta<pi) * np.cos(rho*gamma)*np.cos((theta-pi+sigma)*gamma)\
           + (theta>=pi) * (theta<1.5*pi) * np.cos(sigma*gamma)*np.cos((theta-pi-rho)*gamma)\
           + (theta>=1.5*pi) * (theta<2*pi) * np.cos((pi/2-rho)*gamma)*np.cos((theta-1.5*pi-sigma)*gamma)
        return mu*r**gamma
    
    def g_D(self,p):
        return self.exactu(p)
    
    def Du(self,p):
        pi = self.pi
        sigma = self.sigma
        rho = self.rho
        gamma = self.gamma
        x = p[:,0]; y = p[:,1]
        r = np.sqrt(x**2 + y**2)
        rg = r**gamma
        theta = np.arctan2(y, x)
        theta[theta<0] += 2*pi
        x[x==0] += self.eps
        t = 1+y**2/(x**2)

        Dux1 = (x > 0.0) * (y>= 0.0)\
            *(rg*gamma/r*np.cos((pi/2-sigma)*gamma)/r*x\
            *np.cos((theta-pi/2+rho)*gamma)\
            +rg*np.cos((pi/2-sigma)*gamma)*np.sin((theta-pi/2+rho)*gamma)\
            *gamma*y/(x**2)/t)

        Duy1 = (x> 0.0 )*(y>= 0.0)*\
            (rg*gamma/r*np.cos((pi/2-sigma)*gamma)\
            *np.cos((theta-pi/2+rho)*gamma)/r*y\
            -rg*np.cos((pi/2-sigma)*gamma)*np.sin((theta-pi/2+rho)*gamma)\
            *gamma/x/t)
        
        Dux2 = (x<= 0.0 )*(y> 0.0)*\
            (rg*gamma/r*np.cos(rho*gamma)/r*x\
            *np.cos((theta-pi+sigma)*gamma)\
            +rg*np.cos(rho*gamma)*np.sin((theta-pi+sigma)*gamma)*gamma\
            *y/(x**2)/t)
        
        Duy2 = (x<= 0.0 )*(y> 0.0)*\
            (rg*gamma/r*np.cos(rho*gamma)/r*y\
            *np.cos((theta-pi+sigma)*gamma)-rg*np.cos(rho*gamma)\
            *np.sin((theta-pi+sigma)*gamma)*gamma/x/t)
        
        Dux3 = (x< 0.0 )*(y<= 0.0)*\
            (rg*gamma/r*np.cos(sigma*gamma)/r*x\
            *np.cos((theta-pi-rho)*gamma) \
            +rg*np.cos(sigma*gamma)*np.sin((theta-pi-rho)*gamma)*gamma\
            *y/(x**2)/t)
        
        Duy3 = (x< 0.0 )*(y<= 0.0)*\
            (rg*gamma/r*np.cos(sigma*gamma)/r*y\
            *np.cos((theta-pi-rho)*gamma) -rg*np.cos(sigma*gamma)\
            *np.sin((theta-pi-rho)*gamma)*gamma/x/t)
        
        Dux4 = (x>= 0.0)*(y< 0.0)*\
            (rg*gamma/r*np.cos((pi/2-rho)*gamma)/r*x\
            *np.cos((theta-3*pi/2-sigma)*gamma) \
            +rg*np.cos((pi/2-rho)*gamma)*np.sin((theta-3*pi/2-sigma)*gamma)\
            *gamma*y/(x**2)/t)
        
        Duy4 = (x>= 0.0 )*(y< 0.0)*\
            (rg*gamma/r*np.cos((pi/2-rho)*gamma)/r*y\
            *np.cos((theta-3*pi/2-sigma)*gamma)\
            -rg*np.cos((pi/2-rho)*gamma)*np.sin((theta-3*pi/2-sigma)*gamma)\
            *gamma/x/t)

        Dux = Dux1+Dux2+Dux3+Dux4
        Duy = Duy1+Duy2+Duy3+Duy4
        return np.c_[Dux, Duy]

    