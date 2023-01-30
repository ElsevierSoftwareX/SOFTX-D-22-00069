import numpy as np
from numpy.matlib     import repmat
class Chebyshev:
    """ Class for discretization of sub-domain
        - N    : polynomial order
        - mn   : physical lower bound
        - mx   : physical upper bound
        - y    : physical nodes(i)
        - yc   : computational nodes(i)
        - sc   : scaling for derivative (dyc/dy)
        - D    : 1st derivative matrix
    """
    def __init__(self, N, mn, mx):
        self.N = N
        self.y  = np.zeros(N)
        self.mn = mn
        self.mx = mx
        self.getPoints()
        self.c = np.ones(N+1)*2.
        self.c[1:N] = (-1)**(np.arange(N-1)+1)
        self.c.shape = (N+1,1)

    def getPoints(self):
        # Gauss-Lebatto distribution
        N = self.N
        self.yc = -np.cos( (np.pi*np.arange(N+1)/N))
        self.y = (self.yc + 1. )/2. * ( self.mx - self.mn ) + self.mn
        self.sc = np.gradient(self.yc)/np.gradient(self.y)
        return self.y

    def getDiffMatrix(self,Dealiasing=False):
        N = self.N
        c = self.c
        x = repmat(self.yc, N+1, 1 )
        dx = x.T - x
        self.D = (np.matmul(c, 1/c.T))/(dx+np.eye(N+1))
        self.D = self.D - np.eye(N+1)*np.sum(self.D.T, axis = 0)
        self.D = self.D*self.sc[0]
        return self.D

    def getSecondDiffMatrix(self):
        self.getDiffMatrix()
        return self.D.dot(self.D)

    def getIntWeight(self):
        N = self.N
        D = self.D
        self.w = np.zeros(N+1)
        self.w = np.hstack([np.linalg.inv(-D[0:-1,0:-1])[0,:],0.])
        return self.w

    def evalFunctionAt(self, xq, f):
        x = (xq-self.mn)/(self.mx-self.mn)*2. - 1
        if x == 1.:
            return f[-1]
        elif x == -1:
            return f[0]
        else:
            N = self.N
            T = N*np.sin(N*np.arccos(x))/(1.-x**2)**0.5
            c = self.c
            c.shape = (N+1,)
            g = (self.yc**2-1.)/(x-self.yc)*T/N**2/c
        return g.dot(f)

