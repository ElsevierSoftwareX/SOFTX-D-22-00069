import sys
import numpy as np
import scipy as sc

from Polynomial import Polynomial
from Chebyshev  import Chebyshev

class SpectralElem:
    def __init__(self, Npol, yelm):
        self.Nelm = yelm.shape[0]-1
        self.Npol = Npol
        self.N = self.Nelm*Npol+1
        self.yelm = yelm
        self.y    = np.zeros(self.N)
        self.elem = [None]*self.Nelm
        self.scaledElem = [None]*self.Nelm
        for i in range(self.Nelm):
            self.elem[i] = Polynomial(yelm[i:i+2],  Npol, dist='Chebyshev')

    def getPoints(self):
        k = 0
        for i in range(self.Nelm):
            for j in range(0,self.Npol+1):
                self.y[k] = self.elem[i].getPoints()[j]
                k = k + 1
            k = k - 1
        return self.y

    def getDiffMatrix(self,sparse=False,Dealiasing=False):
        self.getPoints()
        if sparse:
            self.D = sc.sparse.lil_matrix((self.N,self.N))
        else:
            self.D = np.zeros((self.N, self.N))
        for i in range(self.Nelm):
            self.D[i*self.Npol:(i+1)*self.Npol+1,i*self.Npol:(i+1)*self.Npol+1] += self.elem[i].getDiffMatrix(Dealiasing=Dealiasing)
        for i in range(1,self.Nelm):
            self.D[i*self.Npol,:] = 0.5*self.D[i*self.Npol,:]
        return self.D

    def getSecondDiffMatrix(self,sparse=False):
        return self.D.dot(self.D)

    def getIntWeight(self,sparse=False):
        if sc.sparse.issparse(self.D):
            self.w = np.zeros(self.N)
            self.w[0:self.N-1] = np.linalg.inv(-self.D.todense()[0:-1,0:-1])[0,:]
        else:
            self.w = np.zeros(self.N)
            self.w[0:self.N-1] = np.linalg.inv(-self.D[0:-1,0:-1])[0,:]
        return self.w

    def findElem(self, x, scale=1):
        ans = None
        for i in range(0, self.Nelm):
            if x <= self.y[(i+1)*self.Npol]*scale:
                ans = i
                return ans

        if ans == None:
            ans = self.Nelm
            return ans

    def scaleFunction(self, scale, f, updateL=False, Dealiasing=False):
        # Interpolate f on the new grid x
        if updateL:
            for i in range(self.Nelm):
                self.scaledElem[i] = Polynomial(self.yelm[i:i+2]*scale,  self.Npol, dist='Chebyshev')

        ans = np.zeros(f.size)
        for i in range(f.size):
            k  = self.findElem(self.y[i], scale=scale)
            if k == self.Nelm:
                ans[i] = f[-1]
            else:
                ff = f[k*self.Npol:(k+1)*self.Npol+1]
                ans[i] = self.scaledElem[k].evalFunctionAt(self.y[i]*scale,ff, Dealiasing=Dealiasing)
        return ans

    def evalFunction(self, x, f, Dealiasing=False):
        xx = self.getPoints()
        if x.shape == xx.shape:
            if np.allclose(xx, x, atol=1e-13):
                ans = f
            else:
                ans = np.zeros(x.size)
                for i in range(0,x.size):
                    k  = self.findElem(x[i])
                    if k == self.Nelm:
                        ans[i] = f[-1]
                    else:
                        ff = f[k*self.Npol:(k+1)*self.Npol+1]
                        ans[i] = self.elem[k].evalFunctionAt(x[i],ff, Dealiasing=Dealiasing)
        else:
             ans = np.zeros(x.size)
             for i in range(0,x.size):
                 k  = self.findElem(x[i])
                 if k == self.Nelm:
                     ans[i] = f[-1]
                 else:
                     ff = f[k*self.Npol:(k+1)*self.Npol+1]
                     ans[i] = self.elem[k].evalFunctionAt(x[i],ff, Dealiasing=Dealiasing)
        return ans
