import sys
import numpy as np
import scipy as sc
from numpy.matlib     import repmat
from weno             import WENO
from Spectral         import SpectralElem
from Polynomial       import Polynomial

class Numerical:
    def __init__(self, Name, MethodNameX, MethodNameY, Geom, tol=1e-4, relaxU=1, relaxP=1, relaxT=1, perturb=0, stabCoeff=1):
        # Name and methods
        self.ProbName = Name
        self.MethodNameX = MethodNameX
        self.MethodNameY = MethodNameY
        # Geometry
        self.geom = Geom
        # Discretization
        self.Nx  = 40
        self.Ny  = 40
        self.Neq = 5
        self.Nelmx = 40
        self.Nelmy = 40
        # Wall distribution (Spectral only)
        self.epsY = 0.1
        self.epsX = 1.
        # Smoothing (FD only)
        self.SmoothX = False
        self.SmoothY = False
        # upwinding
        self.upwind = False
        # Finite difference order and scheme
        self.FDOX = 4
        self.FDOY = 4
        self.FDSX = 'central'
        self.FDSY = 'central'
        self.perturb = perturb
        # Backward differentiation order
        self.BDF = 1
        # Limiter (for NS only)
        self.limU    = (-999,999)
        self.limP    = ( 0.01,999.)
        self.limT    = ( 0.01,999.)
        # Relaxation
        self.relaxU = relaxU
        self.relaxP = relaxP
        self.relaxT = relaxT
        # Tolerance
        self.tol = tol
        self.stabCoeff = stabCoeff

    def _setMethods(self):
        # Generate numerical methods in X and Y
        # X
        if self.MethodNameX == 'weno':
            self.Nx = self.geom.x.shape[0]
            self.MethodX = WENO(self.geom.x, self.FDOX, perturb=self.perturb)
        elif self.MethodNameX == 'Collocation':
            self.Nx = self.geom.x.shape[0]
            self.MethodX = Polynomial(self.geom.x, self.FDOX)
        elif self.MethodNameX == 'Spectral-Elem':
            self.Nelmx = self.Nx//self.FDOX
            self.MethodX = SpectralElem(self.Nelmx, self.FDOX, self.geom.yelm)
            self.MethodX.getDiffMatrix()
            self.MethodX.getSecondDiffMatrix()
        # Y
        if self.MethodNameY == 'weno':
            self.Ny = self.geom.Ny
            self.MethodY = WENO(self.geom.yelm, self.FDOY, perturb=self.perturb)
        elif self.MethodNameY == 'Collocation':
            self.Ny = self.geom.Ny
            self.MethodY = Polynomial(self.geom.yelm, self.FDOY)
        elif self.MethodNameY == 'Spectral-Elem':
            self.Ny = (self.geom.Ny-1)*self.FDOY+1
            self.MethodY = SpectralElem(self.FDOY, self.geom.yelm)
            self.MethodY.getDiffMatrix()
            self.MethodY.getSecondDiffMatrix()

        # Compute diff. matrix
        # X
        self.getX()
        # Y
        self.getY()

    def _computeMetrics(self):
        # Compute the metric associated with the 
        # curvilinear coordinates
        self.Jxx, self.Jxy, self.Jyx, self.Jyy, self.theta, self.thetax = self.geom.computeJac(self.X, self.Y)

    def getX(self):
        # return X in mesh format (2D matrix)
        self.getX1D()
        self.X = np.zeros((self.Nx, self.Ny))
        for i in range(0, self.Nx):
            self.X[i,:] = self.X1D[i]
        return self.X

    def getY(self):
        # return Y in mesh format (2D matrix)
        self.getY1D()
        self.Y = np.zeros((self.Nx, self.Ny))
        for i in range(self.Nx):
            self.Y[i,:] = self.Y1D
        return self.Y

    def getX1D(self):
        # return X in vector format (1D vector)
        self.X1D = self.MethodX.getPoints()
        return self.X1D

    def getY1D(self):
        # return X in vector format (1D vector)
        self.Y1D = self.MethodY.getPoints()
        return self.X1D

    def SetUp(self):
        self._setMethods()
        self._computeMetrics()


    def buildDX(self,phi=None,updateWENO=False,Dealiasing=False):
        icr = self.Ny*self.Neq
        dim = self.Nx*self.Ny*self.Neq
        self.DX = sc.sparse.lil_matrix((dim,dim))
        for n in range(self.Neq):
            for j in range(self.Ny):
                v0 = n * self.Ny + j
                w0 = 0 * self.Ny + j
                if self.MethodNameX == 'WENO':
                    try:
                        if n == 0 and updateWENO:
                            Dx = self.MethodX.getDiffMatrix(updateWENO=updateWENO,phi=phi[w0::icr])
                        else:
                            Dx = self.MethodX.getDiffMatrix(updateWENO=False)
                        Dx = self.MethodX.getDiffMatrix(updateWENO=updateWENO,phi=phi[v0::icr])
                    except TypeError:
                        Dx = self.MethodX.getDiffMatrix(updateWENO=False)
                else:
                    Dx = self.MethodX.getDiffMatrix(Dealiasing=Dealiasing)
                for i in range(self.Nx):
                    l  = i*self.Neq*self.Ny + n*self.Ny + j
                    self.DX[l,v0::icr] = Dx[i,:]

    def buildD2X(self,phi=None,updateWENO=False):
        icr = self.Ny*self.Neq
        dim = self.Nx*self.Ny*self.Neq
        self.D2X = sc.sparse.lil_matrix((dim,dim))
        for n in range(self.Neq):
            for j in range(self.Ny):
                v0 = n * self.Ny + j
                w0 = 0 * self.Ny + j
                if self.MethodNameX == 'WENO':
                    try:
                        if n == 0 and updateWENO:
                            D2x = self.MethodX.getSecondDiffMatrix(updateWENO=updateWENO,phi=phi[w0::icr])
                        else:
                            D2x = self.MethodX.getSecondDiffMatrix(updateWENO=False)
                    except TypeError:
                        D2x = self.MethodX.getSecondDiffMatrix(updateWENO=False)
                else:
                    D2x = self.MethodX.getSecondDiffMatrix()
                for i in range(self.Nx):
                    l  = i*self.Neq*self.Ny + n*self.Ny + j
                    self.D2X[l,v0::icr] = D2x[i,:]

    def buildDX_AND_D2X(self,phi=None,updateWENO=False, Dealiasing=False):
        icr = self.Ny*self.Neq
        dim = self.Nx*self.Ny*self.Neq
        self.DX  = sc.sparse.lil_matrix((dim,dim))
        self.D2X = sc.sparse.lil_matrix((dim,dim))
        for j in range(self.Ny):
            for n in range(self.Neq):
                v0 = n * self.Ny + j
                w0 = 0 * self.Ny + j
                if self.MethodNameX == 'WENO':
                    try:
                        if n == 0 and updateWENO:
                            Dx  = self.MethodX.getDiffMatrix(updateWENO=True, phi=phi[w0::icr], Dealiasing=Dealiasing)
                            D2x = self.MethodX.getSecondDiffMatrix(updateWENO=False)
                        else:
                            Dx  = self.MethodX.getDiffMatrix(updateWENO=False, Dealiasing=Dealiasing)
                            D2x = self.MethodX.getSecondDiffMatrix(updateWENO=False)
                    except TypeError:
                        Dx  = self.MethodX.getDiffMatrix(Dealiasing=Dealiasing)
                        D2x = self.MethodX.getSecondDiffMatrix()
                else:
                    Dx  = self.MethodX.getDiffMatrix(Dealiasing=Dealiasing)
                    D2x = self.MethodX.getSecondDiffMatrix()
                for i in range(self.Nx):
                    l  = i*self.Neq*self.Ny + n*self.Ny + j
                    self.DX[l,v0::icr]  =  Dx[i,:]
                    self.D2X[l,v0::icr] = D2x[i,:]

    def getDX(self,new=True, updateWENO=True, phi=None, Dealiasing=False):
        # return the global derivative matrix
        # (streamwise direction) in case of
        # multiple variables (Neq)
        try:
            if new:
                self.buildDX(updateWENO=updateWENO,phi=phi, Dealiasing=Dealiasing)
            return self.DX
        except:
            self.buildDX(updateWENO=updateWENO,phi=phi, Dealiasing=Dealiasing)
            return self.DX

    def getD2X(self,new=True, updateWENO=True, phi=None):
        # return the global derivative matrix
        # (streamwise direction) in case of
        # multiple variables (Neq)
        try:
            if new:
                self.buildD2X(updateWENO=updateWENO,phi=phi)
            return self.D2X
        except:
            self.buildD2X(updateWENO=updateWENO,phi=phi)
            return self.D2X

    def getDX_AND_D2X(self,new=True, updateWENO=False, phi=None, Dealiasing=False):
        # return the global derivatives matrices
        # (streamwise direction) in case of
        # multiple variables (Neq)
        try:
            if new:
                self.buildDX_AND_D2X(updateWENO=updateWENO, phi=phi, Dealiasing=Dealiasing)
            return self.DX, self.D2X
        except:
            self.buildDX_AND_D2X(updateWENO=updateWENO, phi=phi, Dealiasing=Dealiasing)
            return self.DX, self.D2X

    def getDY(self,phi=None, Dealiasing=False):
        # return the global derivative matrix
        # (normal direction)
        Dy = self.MethodY.getDiffMatrix(Dealiasing=Dealiasing)
        self.DY = sc.sparse.kron(sc.sparse.eye(self.Neq*self.Nx, format='lil'), sc.sparse.lil_matrix(Dy), format='lil')
        return self.DY

    def getD2Y(self,phi=None):
        # return the global derivative matrix
        # (normal direction)
        D2y = self.MethodY.getSecondDiffMatrix()
        self.D2Y = sc.sparse.kron(sc.sparse.eye(self.Neq*self.Nx, format='lil'), sc.sparse.lil_matrix(D2y), format='lil')
        return self.D2Y

    def buildDY_AND_D2Y(self,phi=None,updateWENO=False, Dealiasing=False):
        dim = self.Nx*self.Ny*self.Neq
        self.DY  = sc.sparse.lil_matrix((dim,dim))
        for i in range(self.Nx):
            for n in range(self.Neq):
                v0 = (self.Neq * i + n + 0) * self.Ny
                vf = (self.Neq * i + n + 1) * self.Ny
                w0 = (self.Neq * i + 0 + 0) * self.Ny
                wf = (self.Neq * i + 0 + 1) * self.Ny
                try:
                    if n==0 and updateWENO:
                        Dy  = self.MethodY.getDiffMatrix(updateWENO=True, phi=phi[w0:wf], Dealiasing=Dealiasing)
                    else:
                        Dy  = self.MethodY.getDiffMatrix(updateWENO=False, Dealiasing=Dealiasing)
                except TypeError:
                    Dy  = self.MethodY.getDiffMatrix(Dealiasing=Dealiasing)
                self.DY[v0:vf,v0:vf] = Dy
        self.D2Y = self.DY @ self.DY

    def getDY_AND_D2Y(self,new=True, updateWENO=False, phi=None, Dealiasing=False):
        # return the global derivatives matrices
        # (streamwise direction) in case of
        # multiple variables (Neq)
        try:
            if new:
                self.buildDY_AND_D2Y(updateWENO=updateWENO, phi=phi, Dealiasing=Dealiasing)
            return self.DY, self.D2Y
        except:
            self.buildDY_AND_D2Y(updateWENO=updateWENO, phi=phi, Dealiasing=Dealiasing)
            return self.DY, self.D2Y

    @staticmethod
    def SmoothConv(X,N,kind='linear', l=3,y=None):
        from scipy.interpolate import interp1d
        # Function that smooth a 1D vector
        # using a convolution algorithm
        # x : input vector
        # N : Number of smoothing iterations
        if N == 0:
            return X
        else:
            #
            # Averaging radius
            d = l//2+1

            # Mirror Boundaries to avoid boundary effects
            x = np.hstack([2*X[0]-X[1:d], X, 2*X[-1]-X[-d:-1]])
            xx = np.arange(-(l//2),X.shape[0]+l//2)

            for i in range(N):
                x  = np.convolve(x,  np.ones((l,))/l, mode='valid')
                xx = np.convolve(xx, np.ones((l,))/l, mode='valid')
                x  = np.hstack([ 2*x[0]- x[1:d],  x,  2*x[-1]- x[-d:-1]])
                xx = np.hstack([2*xx[0]-xx[1:d], xx, 2*xx[-1]-xx[-d:-1]])
            x = interp1d(xx, x, kind=kind,fill_value='extrapolate')(np.arange(X.shape[0]))
            try:
                return interp1d(y[::2],x[::2],kind=kind,fill_value='extrapolate')(y)
            except:
                return x
    def eval2DFunc(self, x, y, F, ymax=999999):
        # allocating memory
        tempFx = np.zeros((x.shape[0], F.shape[1]))
        xx = self.MethodX.getPoints()
        yy = self.MethodY.getPoints()

        if xx.shape == x.shape:
            if not np.allclose(x, xx, atol=1e-13):
                for i in range(self.Ny):
                    tempFx[:,i] = self.MethodX.evalFunction(x[:], F[:,i])
                    for ii in range(len(x)):
                        if x[ii] < xx[0]:
                            tempFx[ii,i] = F[ 0,i]
                        elif x[ii] > xx[-1]:
                            tempFx[ii,i] = F[-1,i]
            else:
                tempFx = F
        else:
            for i in range(self.Ny):
                tempFx[:,i] = self.MethodX.evalFunction(x[:], F[:,i])
                for ii in range(len(x)):
                    if x[ii] < xx[0]:
                        tempFx[ii,i] = F[ 0,i]
                    elif x[ii] > xx[-1]:
                        tempFx[ii,i] = F[-1,i]

        if y.shape == yy.shape:
            if not np.allclose(yy, y, atol=1e-13):
                tempFy = np.zeros((x.shape[0], y.shape[0]))
                for i in range(x.shape[0]):
                    tempFy[i,:] = self.MethodY.evalFunction(y[:], tempFx[i,:])
                    for ii in range(len(y)):
                        if y[ii] < yy[0]:
                            tempFy[i,ii] = tempFx[i, 0]
                        elif y[ii] > yy[-1]:
                            tempFy[i,ii] = tempFx[i,-1]
            else:
                return tempFx
        else:
            tempFy = np.zeros((x.shape[0], y.shape[0]))
            for i in range(x.shape[0]):
                tempFy[i,:] = self.MethodY.evalFunction(y[:], tempFx[i,:])
                for ii in range(len(y)):
                    if y[ii] < yy[0]:
                        tempFy[i,ii] = tempFx[i, 0]
                    elif y[ii] > yy[-1]:
                        tempFy[i,ii] = tempFx[i,-1]

        ymax = min(ymax, y[-1])
        jj = np.argmin(abs(y-ymax))
        for j in range(jj-10, tempFy.shape[1]):
            Nsmooth = min(max(j - jj, 1),50)
            tempFy[:,j] = self.SmoothConv(tempFy[:,j], Nsmooth)
#        for i in range(tempFy.shape[0]):
#            for j in range(jj-20, jj+20):
#                tempFy[i,j:] = self.SmoothConv(tempFy[i,j:],3)

        return tempFy


class NumericalMethod:
    def __init__(self, GeomNSE, GeomPSE, FDOX=3, FDOY=3, MethodNSE=('weno','weno'), MethodPSE=('weno','Spectral-Elem'), tol=1e-6,relaxU=1,relaxP=1,relaxT=1, stabCoeff=1):
        self.NSE = Numerical('NSE', MethodNSE[0], MethodNSE[1], GeomNSE, tol=tol,relaxU=relaxU,relaxP=relaxP,relaxT=relaxT,perturb=0.25,stabCoeff=0)
        self.NSE.FDOX = 3
        self.NSE.FDOY = 3
        self.PSE = Numerical('PSE', MethodPSE[0], MethodPSE[1], GeomPSE, tol=tol,relaxU=relaxU,relaxP=relaxP,relaxT=relaxT,perturb=0,stabCoeff=stabCoeff)
#        self.PSE = Numerical('PSE', MethodX, 'Spectral-Elem', GeomPSE, tol=tol,relaxU=relaxU,relaxP=relaxP,relaxT=relaxT,perturb=0,stabCoeff=stabCoeff)
        self.PSE.FDOX = FDOX
        self.PSE.FDOY = FDOY

    def SetUp(self):
        # Setup everything
        self.NSE.SetUp()
        self.PSE.SetUp()
