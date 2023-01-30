import numpy as np
from Chebyshev import Chebyshev
import scipy as sc

class Polynomial(object):
    # This class for the WENO scheme
    # numerical methods.
    def __init__(self, x, Order, dist='Chebyshev'):
        self.O = Order
        self.x = x
        if dist == 'Chebyshev':
            if x.shape[0] > 2:
               self.x = Chebyshev(x.shape[0]-1, x[0], x[-1]).y
               self.N = self.x.shape[0]
            else:
               self.x = Chebyshev(Order, x[0], x[-1]).y
               self.N = self.x.shape[0]

    def getPoints(self):
        return self.x

    def findRange(self, x, order, xx):
        # Find which points to use for the
        # Piecewise polynomial interpolate
        # centered at index (the closest x value)
        # ---------------------------------------
        # If not points supplied, use self.x
        # Closest point
        i = (np.abs(x-xx)).argmin()
        
        # Range
        if i <= order//2:
            off0 = 0
            offf = order + 1
        elif i >= x.shape[0] - order//2 - 1:
            off0 = x.shape[0] - order - 1
            offf = x.shape[0]
        else:
            off0 = i - order//2
            offf = i + order//2 + 1

        # Indices
        Indices = range(off0,offf)
        return Indices

    def getDiffMatrix(self, x=np.array([]), Dealiasing=False,sparse=False):
        try:
            self.D = self.getDiffMatrixAt(x=x,Dealiasing=Dealiasing,sparse=sparse) 
        except:
            self.D = self.getDiffMatrixAt(Dealiasing=Dealiasing,sparse=sparse) 
        return self.D

    def getSecondDiffMatrix(self,x=np.array([]),Dealiasing=False,sparse=False):
        try:
            if not Dealiasing:
                return self.D2
            else:
                self.D2 = self.getSecondDiffMatrixAt(x=x,Dealiasing=Dealiasing,sparse=sparse) 
                return self.D2
        except:
            self.D2 = self.getSecondDiffMatrixAt(x=x,Dealiasing=Dealiasing,sparse=sparse) 
            return self.D2

    def getSecondDiffMatrixAt(self, x=np.array([]), Dealiasing=False, sparse=False):
        # If not points supplied, use self.x
#        N = x.shape[0]
#        if N == 0:
#            self.getPoints()
#            x = self.x
#            N = x.shape[0]
#
#        # Allocating memory
#        if sparse:
#            self.D2 = sc.sparse.lil_matrix((N,N))
#        else:
#            self.D2 = np.zeros((N,N))
#
#        # Building D2 matrix
#        for i in range(N):
#            if Dealiasing:
#                Indices = self.findRange(x, self.O, x[i])[::2]
#            else:
#                Indices = self.findRange(x, self.O, x[i])
#            self.D2[i,Indices] = self.evald2P(x[Indices], x[i])
        self.D2 = self.D @ self.D
        return self.D2

    def getDiffMatrixAt(self,x=np.array([]),Dealiasing=False,sparse=False):
        # If not points supplied, use self.x
        N = x.shape[0]
        if N == 0:
            self.getPoints()
            x = self.x
            N = x.shape[0]

        # Allocating memory
        if sparse:
            self.D = sc.sparse.lil_matrix((N,N))
        else:
            self.D = np.zeros((N,N))

        # Building D matrix
        for i in range(N):
            if Dealiasing:
                Indices = self.findRange(x, self.O, x[i])[::2]
            else:
                Indices = self.findRange(x, self.O, x[i])
            self.D[i,Indices] = self.evaldP(x[Indices], x[i])

        return self.D

    def getInterpMatrix(self, xx, x=np.array([]),Dealiasing=False):
        # If no points supplied, use self.x
        N = x.shape[0]
        if N == 0:
            self.getPoints()
            x = self.x
            N = x.shape[0]

        self.L = np.zeros((xx.shape[0],N))
        for i in range(xx.shape[0]):
            if Dealiasing:
                Indices = self.findRange(x, self.O, x[i])[::2]
            else:
                Indices = self.findRange(x, self.O, x[i])
            self.L[i,Indices] = self.evalP(x[Indices], xx[i])
        return self.L

    def evalFunctionAt(self, xx, f, x=np.array([]), Dealiasing=False):
        # If no points supplied, use self.x
        N = x.shape[0]
        if N == 0:
            self.getPoints()
            x = self.x
            N = x.shape[0]

        L = np.zeros(N)
        if Dealiasing:
            Indices = self.findRange(x, self.O, xx)[::2]
        else:
            Indices = self.findRange(x, self.O, xx)
        L[Indices] = self.evalP(x[Indices], xx)
        return L @ f

    def evalFunction(self, xx, f, x=np.array([]), Dealiasing=False):
        # If no points supplied, use self.x
        if x.shape[0] == 0:
            self.getPoints()
            x = self.x
        try:
            return self.L @ f
        except:
            return self.getInterpMatrix(xx,x=x,Dealiasing=Dealiasing) @ f

    def evalP(self,x,xx):
        L = np.ones(x.shape[0])
        for j in range(x.shape[0]):
            for m in range(x.shape[0]):
                if j != m:
                    L[j] *= (xx - x[m]) / (x[j] - x[m])
        return L

    def evaldP(self,x,xx):
        Lp = np.zeros(x.shape[0])
        for j in range(x.shape[0]):
            for i in range(x.shape[0]):
                if i != j:
                    prod = 1
                    for l in range(x.shape[0]):
                        if l != i and l != j:
                            prod *= (xx - x[l]) / (x[j] - x[l])
                    Lp[j] += prod/(x[j] - x[i])
        return Lp

    def evald2P(self,x,xx):
        Lpp = np.zeros(x.shape[0])
        for j in range(x.shape[0]):
            for i in range(x.shape[0]):
                if i != j:
                    summ = 0
                    for m in range(x.shape[0]):
                        if m != i and m != j:
                            prod = 1
                            for l in range(x.shape[0]):
                                if l != i and l != j and l != m:
                                    prod *= (xx - x[l]) / (x[j] - x[l])
                            summ += prod/(x[j] - x[m])
                    Lpp[j] += summ/(x[j] - x[i])
        return Lpp
