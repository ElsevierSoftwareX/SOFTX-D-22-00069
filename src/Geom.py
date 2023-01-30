import os
import numpy as np
import math as mt
import scipy as sc
from scipy import optimize
from scipy.interpolate import interp1d
from scipy.integrate import simps
from scipy.interpolate import CubicSpline

class Curve:
    def __init__(self,t,x,y):
        # x, y coordinates
        self.x = x
        self.y = y
        self.t = t
        # Parametrization of the 3D curve
        self._xsp = sc.interpolate.CubicSpline(t,x)
        self._ysp = sc.interpolate.CubicSpline(t,y)
        
        # Computing x'(t), y'(t), x''(t), y''(t)
        self._dxsp  = self._xsp.derivative(1)
        self._dysp  = self._ysp.derivative(1)
        self._d2xsp = self._xsp.derivative(2)
        self._d2ysp = self._ysp.derivative(2)
        
        # Vectorial r(t), r'(t) and r''(t)
        # corresponding to the position,
        # velocity and acceleration in
        # vectorial form
        self._r   = np.vstack([  self._xsp(t),   self._ysp(t)])
        self._rp  = np.vstack([ self._dxsp(t),  self._dysp(t)])
        self._rpp = np.vstack([self._d2xsp(t), self._d2ysp(t)])
        
        # Length of the 3D curve
        self.length = sc.integrate.simps((self._rp[0,:]**2+self._rp[1,:]**2)**0.5,t)
        # angle of the 3D curve
        self.theta  = np.arctan(self._rp[1,:]/self._rp[0,:])
        #Curvature of the 3D curve
        self.K = np.gradient(self.theta, t/t[-1]*self.length)
        self.Ksp = CubicSpline(t,self.K)

class Geom:
    # Class for creating geometry and metric
    def __init__(self, coords, Lref, Nx, Ny, Y0=0, Yf=200, Yffs=None, xelm=None, yelm=None, Smooth=0, opt=0.1, exp=0.5, epsY=1, AoA=0, equiX=False, equiRex=False, Optimal=False, disp=False, Growth=False, dywall=10, FDO=None):
        # File containing the points
        # Smoothing
        self.Smooth = Smooth
        # Optimal mesh based on curvature
        self.Optimal = Optimal
        # Reference length
        self.Lref = Lref
        # Normal height
        self.yelm  = yelm
        # Number of points in x
        self.Nx = Nx
        self.Ny = Ny
        self.opt = opt
        self.exp = exp
        self.disp = disp
        self.xelm = np.zeros(Nx)

        if yelm is None:
            self.yelm = np.zeros(Ny)
            if Yffs is None:
                yf_lambda = lambda epsY: dywall * (epsY**(Ny-2) + np.log(epsY) - 1)/np.log(epsY) - Yf
                epsY = optimize.fsolve(yf_lambda,1.8)
                self.yelm[1:] = dywall * (epsY**np.arange(0,Ny-1) + np.log(epsY) - 1)/np.log(epsY)
            elif Yffs is not None:
                yf_lambda = lambda epsY: dywall * (epsY**(Ny-2) + np.log(epsY) - 1)/np.log(epsY) - Yf
                epsY = optimize.fsolve(yf_lambda,1.8)
                self.yelm[1:] = dywall * (epsY**np.arange(0,Ny-1) + np.log(epsY) - 1)/np.log(epsY)
                self.yelm[-1] = Yffs


        # angle of attack
        self.AoA = AoA
        self.equiX = equiX
        self.equiRex = equiRex
        try:
            if isinstance(coords, str):
                path = os.path.join(os.path.dirname(__file__), '_geo/',coords+'.xy')
                XY   = np.loadtxt(path)
                txy  = np.arange(XY.shape[0])
                if XY.shape[0] < 500:
                    n = 10*XY.shape[0]
                    self.coords = np.zeros((n,2))
                    # Minimum number of points = 5000
                    t = np.linspace(txy[0],txy[-1],n)
                    self.coords[:,0] = interp1d(txy,XY[:,0],kind=3)(t)
                    self.coords[:,1] = interp1d(txy,XY[:,1],kind=3)(t)
                else:
                    self.coords = np.zeros((XY.shape[0],2))
                    self.coords[:,0] = XY[:,0]
                    self.coords[:,1] = XY[:,1]
            else:
                self.coords = coords
        except NameError:
            print('Could not find file : _geo/'+coords)

        # Generating 1D curvilinear points 
        self.x = self.genX()

    @staticmethod
    def SmoothConv(X,N,kind='linear', l=3,y=None):
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

    def error(self,t0):
        t = np.sort(t0)
        x = self.curve1._xsp(t)
        y = self.curve1._ysp(t)
        tt = self.curve1.tt
        curve_opt = Curve(t,x,y)
        err = np.linalg.norm(curve_opt.Ksp(tt) - self.curve1.Ksp(tt))
        return err

    def genX(self):
        # Generate the 1D curvilinear coordinates
        # from the boundary coordinates
        # Smoothing geometry (if requested)
        t = (self.coords[:,0]-self.coords[0,0])/(self.coords[-1,0] - self.coords[0,0])
        if self.Smooth>0:
            x = self.SmoothConv(self.coords[:,0],min(self.Smooth,5))
            y = self.SmoothConv(self.coords[:,1],min(self.Smooth,5))
        else:
            x = self.coords[:,0]
            y = self.coords[:,1]

        if self.Optimal:
            fac  = max(self.coords.shape[0]//self.Nx,1)
            tsm  = np.linspace(t[0],t[-1],fac*self.Nx)
        else:
            tsm = t

        # Parametrization of the curve
        xsm = interp1d(t,x/self.Lref, kind=5)(tsm)
        ysm = interp1d(t,y/self.Lref, kind=5)(tsm)
        self.curve = Curve(tsm,xsm,ysm)

        # Computing optimal point distribution. 
        if self.Optimal:
            NN=10
            Nrand = 250
            Nsamples = NN*self.Nx
            tprob = np.linspace(tsm[0],tsm[-1], NN*Nsamples)
            prob  = (abs(self.curve.Ksp(tprob))**self.exp/np.max(abs(self.curve.Ksp(tprob))**self.exp)) + self.opt
            prob = prob/np.sum(prob)
            t_opt = np.zeros(self.Nx)
            np.random.seed(10092899)
            for i in range(0,Nrand):
                tt = np.sort(np.random.choice(tprob,Nsamples,p=prob))[::NN]
                tt[0] = tsm[0]
                t_opt += self.SmoothConv(tt,1)
            t_opt = t_opt/Nrand
            t_opt = self.SmoothConv(t_opt,self.Smooth)

            # Computing curvature, length, angle
            x_opt = interp1d(t,x/self.Lref, kind=3)(t_opt)
            y_opt = interp1d(t,y/self.Lref, kind=3)(t_opt)
            y_opt[x_opt<1/self.Lref] = 0
            self.curve = Curve(t_opt,x_opt,y_opt)
            self.x = self.coords[0,0]/self.Lref + t_opt/t_opt[-1] * self.curve.length
#            self.x[np.argmin(abs(self.x))] = 0
#            self.x[np.argmin(abs(self.x-1/self.Lref))] = 1/self.Lref
        else:
            if self.equiX:
                self.x = 1/self.Lref + np.linspace(0,self.curve.length,self.Nx)
                t_eq = (self.x-self.x[0])/(self.x[-1] - self.x[0])
                y_eq = self.curve._ysp(t_eq)
                x_eq = self.curve._xsp(t_eq)
                self.curve = Curve(t_eq,x_eq,y_eq)
            elif self.equiRex:
                Re0 = 1/self.Lref
                Ref = (1 + self.curve.length*self.Lref)**0.5/self.Lref
                self.Rex = np.linspace(Re0, Ref, self.Nx)
                self.x = 1/self.Lref*(self.Rex*self.Lref)**2
                t_eq = (self.x-self.x[0])/(self.x[-1] - self.x[0])
                y_eq = self.curve._ysp(t_eq)
                x_eq = self.curve._xsp(t_eq)
                self.curve = Curve(t_eq,x_eq,y_eq)
            else:
                self.x = self.coords[0,0]/self.Lref + t * self.curve.length

        if self.disp:
            from matplotlib import pyplot as plt
            plt.figure(1)
            plt.plot(x/self.Lref,y/self.Lref)
            plt.plot(self.curve.x,self.curve.y,'.')
            plt.figure(2)
            plt.plot((self.x[:-1]+self.x[1:])/2,np.diff(self.x))
            plt.show()
            
        self.theta1D = self.curve.theta
        self.length  = self.curve.length
        self.K1D     = self.curve.K
        return self.x


    def computeJac(self, X, Y):
        # Check if grid will overlap
        if 1/max(np.max(self.K1D),1e-8) < Y[-1,-1]:
            print('Overlapping grid: Reduce the curvature of the boundary or decrease Yf in main.py')
            print('Yf = ', Y[-1,-1])
            print('Max Y (curvature)', 1/np.max(self.K1D))
        # Compute the Jacobian from X, Y (in meshgrid format)
        # Local orientation
        self.theta = np.tile(self.theta1D.reshape((X.shape[0],1)), (1,Y.shape[1])) + self.AoA/180*np.pi
        # Curvature
        self.K  = np.tile(self.K1D.reshape((X.shape[0],1)), (1,Y.shape[1]))
       
        # Computing metrics (Jacobian)
        self.Jxx = np.cos(self.theta)/(1+Y*self.K)
        self.Jyx = -np.sin(self.theta)/(1+Y*self.K)
        self.Jxy = np.sin(self.theta)
        self.Jyy = np.cos(self.theta)


        return [self.Jxx, self.Jxy, self.Jyx, self.Jyy, self.theta, self.K]




