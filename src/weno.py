import numpy as np
import scipy as sc
from Lagrange import Lagrange, dLagrange, d2Lagrange

class WENO(object):
    # This class for the WENO scheme
    # numerical methods.
    def __init__(self, x, FDO, sparse=False, gamma=[0, 1, 0], perturb=1e-2):
        self.eps = 1e-4
        gamma[0] += perturb
        gamma[1] -= 2*perturb
        gamma[2] += perturb
        self.N   = x.shape[0]
        self.FDO = FDO
        self.k   = 3
        self.offPol  = np.array([0,FDO//2,FDO-1])
        self.x = x
        self.xn = (x-x[0])/(x[-1]-x[0])
        self.xm = x + np.gradient(x)/2
        self.xnm = self.xn + np.gradient(self.xn)/2
        self.getDiff1Matrix()
        self.getDiff2Matrix()
        self.getDiff3Matrix()
        self.getSecondDiff1Matrix()
        self.getSecondDiff2Matrix()
        self.getSecondDiff3Matrix()
        self.gamma = gamma #np.array([1/4, 1/2, 1/4])
        self.gammaBND = gamma #np.array([0.5, 0.4, 0.1])
        self.w  = np.tile(self.gamma,(self.N,1))

    def getPoints(self):
        return self.x
    def off0(self,i, FDO, p, N, offset=0):
#        return max(min(i - FDO + 1 + p, N-1), 0)
        return max(min(i - FDO + 1 + p+offset, N-FDO), 0)
    def offf(self,i, FDO, p, N, offset=0):
#        return max(min(i + FDO + p + 1, N),1)
        return max(min(self.off0(i, FDO, p, N,offset=offset) + FDO, N),0)

    def getIntWeight(self,sparse=False):
        if sc.sparse.issparse(self.Dx):
            self.iw = np.zeros(self.N)
            self.iw[0:self.N-1] = np.linalg.inv(-self.Dx.todense()[0:-1,0:-1])[0,:]
        else:
            self.iw = np.zeros(self.N)
            self.iw[0:self.N-1] = np.linalg.inv(-self.Dx[0:-1,0:-1])[0,:]
        return self.iw

    def SameOmagnitude(self,x):
        return all(np.floor(np.log10(x))==np.floor(np.log10(x[0])))


    def getSecondDiff1Matrix(self,sparse=False):
        # Compute the first differentiation matrix
        if sparse:
            self.D2x1 = sc.sparse.lil_matrix((self.N, self.N))
        else:
            self.D2x1 = np.zeros((self.N,self.N))
        for i in range(self.N):
            p = self.offPol[0]
            off0 = self.off0(i, self.FDO, p, self.N) #max(min(i - self.FDO + 1 + p, self.N-self.FDO), 0)
            offf = self.offf(i, self.FDO, p, self.N) #max(min(off0 + self.FDO, self.N),0)
            xloc = self.x[off0:offf]
            self.D2x1[i,off0:offf] = d2Lagrange(xloc, self.x[i])

        return self.D2x1

    def getSecondDiff2Matrix(self,sparse=False):
        # Compute the first differentiation matrix
        if sparse:
            self.D2x2 = sc.sparse.lil_matrix((self.N, self.N))
        else:
            self.D2x2 = np.zeros((self.N,self.N))
        for i in range(self.N):
            p = self.offPol[1]
            off0 = self.off0(i, self.FDO, p, self.N)
            offf = self.offf(i, self.FDO, p, self.N)
            xloc = self.x[off0:offf]
            self.D2x2[i,off0:offf] = d2Lagrange(xloc, self.x[i])

        return self.D2x2

    def getSecondDiff3Matrix(self,sparse=False):
        # Compute the first differentiation matrix
        if sparse:
            self.D2x3 = sc.sparse.lil_matrix((self.N, self.N))
        else:
            self.D2x3 = np.zeros((self.N,self.N))
        for i in range(self.N):
            p = self.offPol[2]
            off0 = self.off0(i, self.FDO, p, self.N)
            offf = self.offf(i, self.FDO, p, self.N)
            xloc = self.x[off0:offf]
            self.D2x3[i,off0:offf] = d2Lagrange(xloc, self.x[i])

        return self.D2x3
    
    def getDiff1Matrix(self,sparse=False):
        # Compute the first differentiation matrix
        if sparse:
            self.Dx1 = sc.sparse.lil_matrix((self.N, self.N))
        else:
            self.Dx1 = np.zeros((self.N,self.N))
        for i in range(self.N):
            p = self.offPol[0]
            off0 = self.off0(i, self.FDO, p, self.N)
            offf = self.offf(i, self.FDO, p, self.N)
            xloc = self.x[off0:offf]
            self.Dx1[i,off0:offf] = dLagrange(xloc, self.x[i])

        return self.Dx1

    def getDiff2Matrix(self,sparse=False):
        # Compute the first differentiation matrix
        if sparse:
            self.Dx2 = sc.sparse.lil_matrix((self.N, self.N))
        else:
            self.Dx2 = np.zeros((self.N,self.N))
        for i in range(self.N):
            p = self.offPol[1]
            off0 = self.off0(i, self.FDO, p, self.N)
            offf = self.offf(i, self.FDO, p, self.N)
            xloc = self.x[off0:offf]
            self.Dx2[i,off0:offf] = dLagrange(xloc, self.x[i])

        return self.Dx2

    def getDiff3Matrix(self,sparse=False):
        # Compute the first differentiation matrix
        if sparse:
            self.Dx3 = sc.sparse.lil_matrix((self.N, self.N))
        else:
            self.Dx3 = np.zeros((self.N,self.N))
        for i in range(self.N):
            p = self.offPol[2]
            off0 = self.off0(i, self.FDO, p, self.N)
            offf = self.offf(i, self.FDO, p, self.N)
            xloc = self.x[off0:offf]
            self.Dx3[i,off0:offf] = dLagrange(xloc, self.x[i])

        return self.Dx3

    def WENOweight(self, phi):
        self.w  = np.zeros((self.N,self.k))
        for i in range(self.N):
            if i < self.FDO:
                self.w[i,:] = self.gammaBND
            elif i > self.N-self.FDO - 1:
                self.w[i,:] = np.flip(self.gammaBND)
            else:
                for k in range(self.k):
                    p = self.offPol[k]
                    off0 = self.off0(i,self.FDO, p, self.N) #max(min(i - self.FDO + 1 + p, self.N-self.FDO), 0)
                    offf = self.off0(i,self.FDO, p, self.N) #max(min(off0 + self.FDO, self.N),0)
                    xloc = self.xn[off0:offf]
                    philoc = phi[off0:offf]
#                    philoc = philoc/max(np.max(philoc),1e-8)
                    # Gaussian quadrature for integration
                    dx   = self.xnm[i] - self.xnm[i-1]
                    pt1  = self.xnm[i-1] + dx/2 * 0.2254033307585166
                    pt2  = self.xn[i]
                    pt3  = self.xnm[i-1] + dx/2 * 1.7745966692414834
                    beta = dx**2 * (  5/9*d2Lagrange(xloc, pt1).dot(philoc)**2  \
                                    + 8/9*d2Lagrange(xloc, pt2).dot(philoc)**2  \
                                    + 5/9*d2Lagrange(xloc, pt3).dot(philoc)**2)
                    self.w[i,k] = self.gamma[k]/(self.eps + beta.real)**2
                self.w[i,:] = self.w[i,:] / np.sum(self.w[i,:])
        
#                if self.SameOmagnitude(self.w[i,:]):
#                    self.w[i,:] = 1/self.k

    def buildDiffMatrix(self,sparse=False):
        if sparse:
            self.Dx  = sc.sparse.lil_matrix((self.N, self.N))
        else:
            self.Dx  = np.zeros((self.N,self.N))
        for i in range(self.N):
            if self.w[i,0] != 0: 
                p = self.offPol[0]
                off0 = self.off0(i, self.FDO, p, self.N)
                offf = self.offf(i, self.FDO, p, self.N)
                self.Dx[i,off0:offf] += self.Dx1[i,off0:offf] * self.w[i,0]
            if self.w[i,1] != 0: 
                p = self.offPol[1]
                off0 = self.off0(i, self.FDO, p, self.N)
                offf = self.offf(i, self.FDO, p, self.N)
                self.Dx[i,off0:offf] += self.Dx2[i,off0:offf] * self.w[i,1]
            if self.w[i,2] != 0: 
                p = self.offPol[2]
                off0 = self.off0(i, self.FDO, p, self.N)
                offf = self.offf(i, self.FDO, p, self.N)
                self.Dx[i,off0:offf] += self.Dx3[i,off0:offf] * self.w[i,2]

    def buildSecondDiffMatrix(self,sparse=False):
        if sparse:
            self.D2x = sc.sparse.lil_matrix((self.N, self.N))
        else:
            self.D2x = np.zeros((self.N,self.N))
        for i in range(self.N):
            if self.w[i,0] != 0: 
                p = self.offPol[0]
                off0 = self.off0(i, self.FDO, p, self.N)
                offf = self.offf(i, self.FDO, p, self.N)
                self.D2x[i,off0:offf] += self.D2x1[i,off0:offf] * self.w[i,0]
            if self.w[i,1] != 0: 
                p = self.offPol[1]
                off0 = self.off0(i, self.FDO, p, self.N)
                offf = self.offf(i, self.FDO, p, self.N)
                self.D2x[i,off0:offf] += self.D2x2[i,off0:offf] * self.w[i,1]
            if self.w[i,2] != 0: 
                p = self.offPol[2]
                off0 = self.off0(i, self.FDO, p, self.N)
                offf = self.offf(i, self.FDO, p, self.N)
                self.D2x[i,off0:offf] += self.D2x3[i,off0:offf] * self.w[i,2]

    def getSecondDiffMatrix(self,updateWENO=False, phi=None):
        # Compute the first differentiation matrix
        if updateWENO:
            self.WENOweight(phi)
            self.buildSecondDiffMatrix()
            return self.D2x
        else:
            try:
#                self.w   = np.ones((self.N,self.k))/self.k
                self.buildSecondDiffMatrix()
                return self.D2x
            except:
                return self.D2x2

    def getDiffMatrix(self,sparse=False, updateWENO=False, phi=None, Dealiasing=False):
        # Compute the first differentiation matrix
        if updateWENO:
            self.WENOweight(phi)
            self.buildDiffMatrix()
            return self.Dx
        else:
            try:
                self.buildDiffMatrix()
                return self.Dx
            except:
                return self.Dx2

    def simpleLagrange(self, x, xx, FDO, sparse=False):
        # Compute the Lagrange interpolation matrix
        N = x.shape[0]
        NN = xx.shape[0]
        if sparse:
            self.Ls = sc.sparse.lil_matrix((NN, N))
        else:
            self.Ls = np.zeros((NN,N))
        for i in range(NN):
            p = FDO//2
            j = np.argmin(abs(x - xx[i]))
            off0 = max(min(j - FDO + 1 + p, N-FDO), 0)
            offf = max(min(off0 + FDO, N),0)
            xloc = x[off0:offf]
            self.Ls[i,off0:offf] = Lagrange(xloc, xx[i])


    def buildL1Matrix(self, xx, Dealiasing=False, sparse=False):
        # Compute the first differentiation matrix
        NN = xx.shape[0]
        if sparse:
            self.L1 = sc.sparse.lil_matrix((NN, self.N))
        else:
            self.L1 = np.zeros((NN,self.N))
        for i in range(NN):
            p = self.offPol[0]
            j = np.argmin(abs(self.x - xx[i]))
            off0 = self.off0(j, self.FDO, p, self.N)
            offf = self.offf(j, self.FDO, p, self.N)
            if Dealiasing:
                xloc = self.x[off0:offf:2]
                self.L1[i,off0:offf:2] = Lagrange(xloc, xx[i])
            else:
                xloc = self.x[off0:offf]
                self.L1[i,off0:offf] = Lagrange(xloc, xx[i])

    def buildL2Matrix(self, xx, Dealiasing=False, sparse=False):
        # Compute the first differentiation matrix
        NN = xx.shape[0]
        if sparse:
            self.L2 = sc.sparse.lil_matrix((NN, self.N))
        else:
            self.L2 = np.zeros((NN,self.N))
        for i in range(NN):
            p = self.offPol[1]
            j = np.argmin(abs(self.x - xx[i]))
            off0 = self.off0(j, self.FDO, p, self.N)
            offf = self.offf(j, self.FDO, p, self.N)
            if Dealiasing:
                xloc = self.x[off0:offf:2]
                self.L2[i,off0:offf:2] = Lagrange(xloc, xx[i])
            else:
                xloc = self.x[off0:offf]
                self.L2[i,off0:offf] = Lagrange(xloc, xx[i])

    def buildL3Matrix(self, xx, Dealiasing=False, sparse=False):
        # Compute the first differentiation matrix
        NN = xx.shape[0]
        if sparse:
            self.L3 = sc.sparse.lil_matrix((NN, self.N))
        else:
            self.L3 = np.zeros((NN,self.N))
        for i in range(NN):
            p = self.offPol[2]
            j = np.argmin(abs(self.x - xx[i]))
 #           off0 = max(min(j - self.FDO + 1 + p, self.N-self.FDO), 0)
#            offf = max(min(off0 + self.FDO, self.N),0)
            off0 = self.off0(j, self.FDO, p, self.N)
            offf = self.offf(j, self.FDO, p, self.N)
            if Dealiasing:
                xloc = self.x[off0:offf:2]
                self.L3[i,off0:offf:2] = Lagrange(xloc, xx[i])
            else:
                xloc = self.x[off0:offf]
                self.L3[i,off0:offf] = Lagrange(xloc, xx[i])

    def buildLagrangeInterpolator(self, xx, sparse=False, Dealiasing=False, updateL=False):
        # Compute the interpolation matrix
        # f(x) = L(x).dot(f_i)
        NN = xx.shape[0]
        if Dealiasing:
            FDO = self.FDO//2 + 1
        else:
            FDO = self.FDO
        if sparse:
            L = sc.sparse.lil_matrix((NN, self.N))
        else:
            L = np.zeros((NN,self.N))

        if updateL:
            self.buildL1Matrix(xx, Dealiasing=Dealiasing)
            self.buildL2Matrix(xx, Dealiasing=Dealiasing)
            self.buildL3Matrix(xx, Dealiasing=Dealiasing)
        try:
            self.L1
            self.L2
            self.L3
        except:
            self.buildL1Matrix(xx, Dealiasing=Dealiasing)
            self.buildL2Matrix(xx, Dealiasing=Dealiasing)
            self.buildL3Matrix(xx, Dealiasing=Dealiasing)

        if self.L1.shape[0] != xx.shape[0]:
            self.buildL1Matrix(xx, Dealiasing=Dealiasing)
            self.buildL2Matrix(xx, Dealiasing=Dealiasing)
            self.buildL3Matrix(xx, Dealiasing=Dealiasing)

        for i in range(NN):
            j = np.argmin(abs(self.x - xx[i]))
            if self.w[j,0] != 0: 
                p = self.offPol[0]
                off0 = self.off0(j, self.FDO, p, self.N)
                offf = self.offf(j, self.FDO, p, self.N)
                L[i,off0:offf] += self.L1[i,off0:offf] * self.w[j,0]
            if self.w[j,1] != 0: 
                p = self.offPol[1]
                off0 = self.off0(j, self.FDO, p, self.N)
                offf = self.offf(j, self.FDO, p, self.N)
                L[i,off0:offf] += self.L2[i,off0:offf] * self.w[j,1]
            if self.w[j,2] != 0: 
                p = self.offPol[2]
                off0 = self.off0(j, self.FDO, p, self.N)
                offf = self.offf(j, self.FDO, p, self.N)
                L[i,off0:offf] += self.L3[i,off0:offf] * self.w[j,2]

        return L

    def scaleFunction(self, scale, f, updateWENO=False, Dealiasing=False, updateL=False):
        # Interpolate f on the new grid x
        if updateL:
            self.simpleLagrange(self.x*scale, self.x, self.FDO, sparse=False)
        return self.Ls @ f

    def evalFunction(self, xx, f, updateWENO=False, Dealiasing=True):
        # Interpolate f on the new grid xx
        if updateWENO:
            self.WENOweight(f)
            
        L = self.buildLagrangeInterpolator(xx, Dealiasing=Dealiasing)
        return L @ f

    def evalFunction2(self, xx, f, Dealiasing=True):
        # Interpolate f on the new grid x
        # using second order piecewise
        # polynomial
        if Dealiasing:
            X = self.getPoints()[::2]
            F = f[::2]
        else:
            X = self.getPoints()
            F = f
        if X.shape == xx.shape:
            if np.allclose(xx, X, atol=1e-13):
                ans = F
            else:
                ans = sc.interpolate.interp1d(X, F, kind='quadratic',fill_value='extrapolate')(xx)
        else:
            ans = sc.interpolate.interp1d(X, F, kind='quadratic', fill_value='extrapolate')(xx)
        return ans
