import numpy as np
from Param import Parameter
class Field(object):
    """ Class for the mean flow
        U    Streamwise Velocity
        V    Normal to Wall velocity
        W    Crossflow Velocity
        Ux   dU/dx
        Uy   dU/dy
        Uxx  d2U/dx2
        Uxy  d2U/dxdy
        Uyy  d2U/dy2
    """
    def __init__(self, Nx, Ny, M=1, N=1, typ = float, path='default', NumMethod=None, minimal=False):
        try:
            self.load(path,minimal=minimal)
        except:
            if M == 1 and N == 1:
                # Field dimension
                self.Ny = Ny
                self.Nx = Nx
                self.M  = 1
                self.N  = 1
                self.y  = np.zeros((Nx,Ny), dtype=typ)
                self.yc = np.zeros((Nx,Ny), dtype=typ)

                # Numerical Method
                self.NumMethod = NumMethod
            
                # Position X
                self.x  = np.zeros((Nx,Ny), dtype=typ)
                self.xc = np.zeros((Nx,Ny), dtype=typ)
            
                # Geometric angle
                self.theta = np.zeros((Nx,Ny), dtype=typ)
            
                # Local Reynolds number
                self.Rex = np.zeros((Nx,Ny), dtype=typ)
            
                # Non-dim Y direction
                self.eta = np.zeros((Nx,Ny), dtype=typ)
            
                # Field Properties
                self.Cp     = np.zeros((Nx,Ny), dtype=typ)
                self.mu     = np.zeros((Nx,Ny), dtype=typ)
                self.muT    = np.zeros((Nx,Ny), dtype=typ)
                self.muTT   = np.zeros((Nx,Ny), dtype=typ)
                self.mux    = np.zeros((Nx,Ny), dtype=typ)
                self.muxx   = np.zeros((Nx,Ny), dtype=typ)
                self.muxy   = np.zeros((Nx,Ny), dtype=typ)
                self.muy    = np.zeros((Nx,Ny), dtype=typ)
                self.muyy   = np.zeros((Nx,Ny), dtype=typ)
                self.lamb   = np.zeros((Nx,Ny), dtype=typ)
                self.lambx  = np.zeros((Nx,Ny), dtype=typ)
                self.lambxx = np.zeros((Nx,Ny), dtype=typ)
                self.lamby  = np.zeros((Nx,Ny), dtype=typ)
                self.lambyy = np.zeros((Nx,Ny), dtype=typ)
                self.lambxy = np.zeros((Nx,Ny), dtype=typ)
                self.lambT  = np.zeros((Nx,Ny), dtype=typ)
                self.lambTT = np.zeros((Nx,Ny), dtype=typ)
                self.rho    = np.zeros((Nx,Ny), dtype=typ)
                self.rhox   = np.zeros((Nx,Ny), dtype=typ)
                self.rhoxx  = np.zeros((Nx,Ny), dtype=typ)
                self.rhoy   = np.zeros((Nx,Ny), dtype=typ)
                self.rhoyy  = np.zeros((Nx,Ny), dtype=typ)
                self.rhoxy  = np.zeros((Nx,Ny), dtype=typ)
            
                # Dimensionless quantities
                self.delta   =  np.zeros(Nx, dtype=typ)
            
                # Disturbance kinetic energy (DKE)
                self.DKE  = np.zeros(Nx, dtype=typ)
                self.DKEx = np.zeros(Nx, dtype=typ)

                # Amplitude
                self.amplitude = np.zeros(Nx, dtype=typ)
            
                # Growth rate & N-factor
                self.sigma   = np.zeros(Nx, dtype=typ)
                self.nfactor = np.zeros(Nx, dtype=typ)
                self.Nfactor = np.zeros(Nx, dtype=typ)
            
                # Growth rate parameter alpha
                self.omega   = np.zeros(Nx, dtype=typ)
                self.F       = np.zeros(Nx, dtype=typ)
                self.beta    = np.zeros(Nx, dtype=typ)
                self.B       = np.zeros(Nx, dtype=typ)
                self.alpha   = np.zeros(Nx, dtype=typ)
                self.alphax  = np.zeros(Nx, dtype=typ)
            
                # U
                self.U   = np.zeros((Nx,Ny), dtype=typ)
                self.Ux  = np.zeros((Nx,Ny), dtype=typ)
                self.Uxx = np.zeros((Nx,Ny), dtype=typ)
            
                self.Uy  = np.zeros((Nx,Ny), dtype=typ)
                self.Uyy = np.zeros((Nx,Ny), dtype=typ)
            
                self.Uz  = np.zeros((Nx,Ny), dtype=typ)
                self.Uzz = np.zeros((Nx,Ny), dtype=typ)
            
                self.Uxy = np.zeros((Nx,Ny), dtype=typ)
                self.Uxz = np.zeros((Nx,Ny), dtype=typ)
                self.Uyz = np.zeros((Nx,Ny), dtype=typ)
            
                # V
                self.V   = np.zeros((Nx,Ny), dtype=typ)
                self.Vx  = np.zeros((Nx,Ny), dtype=typ)
                self.Vxx = np.zeros((Nx,Ny), dtype=typ)
            
                self.Vy  = np.zeros((Nx,Ny), dtype=typ)
                self.Vyy = np.zeros((Nx,Ny), dtype=typ)
            
                self.Vz  = np.zeros((Nx,Ny), dtype=typ)
                self.Vzz = np.zeros((Nx,Ny), dtype=typ)
            
                self.Vxy = np.zeros((Nx,Ny), dtype=typ)
                self.Vxz = np.zeros((Nx,Ny), dtype=typ)
                self.Vyz = np.zeros((Nx,Ny), dtype=typ)
            
                # W
                self.W   = np.zeros((Nx,Ny), dtype=typ)
                self.Wx  = np.zeros((Nx,Ny), dtype=typ)
                self.Wxx = np.zeros((Nx,Ny), dtype=typ)
            
                self.Wy  = np.zeros((Nx,Ny), dtype=typ)
                self.Wyy = np.zeros((Nx,Ny), dtype=typ)
            
                self.Wz  = np.zeros((Nx,Ny), dtype=typ)
                self.Wzz = np.zeros((Nx,Ny), dtype=typ)
            
                self.Wxy = np.zeros((Nx,Ny), dtype=typ)
                self.Wxz = np.zeros((Nx,Ny), dtype=typ)
                self.Wyz = np.zeros((Nx,Ny), dtype=typ)
            
                # P
                self.P   = np.zeros((Nx,Ny), dtype=typ)
                self.Px  = np.zeros((Nx,Ny), dtype=typ)
                self.Pxx = np.zeros((Nx,Ny), dtype=typ)
            
                self.Py  = np.zeros((Nx,Ny), dtype=typ)
                self.Pyy = np.zeros((Nx,Ny), dtype=typ)
            
                self.Pz  = np.zeros((Nx,Ny), dtype=typ)
                self.Pzz = np.zeros((Nx,Ny), dtype=typ)
            
                self.Pxy = np.zeros((Nx,Ny), dtype=typ)
                self.Pxz = np.zeros((Nx,Ny), dtype=typ)
                self.Pyz = np.zeros((Nx,Ny), dtype=typ)
            
                # T
                self.T   = np.zeros((Nx,Ny), dtype=typ)
                self.Tx  = np.zeros((Nx,Ny), dtype=typ)
                self.Txx = np.zeros((Nx,Ny), dtype=typ)
            
                self.T   = np.zeros((Nx,Ny), dtype=typ)
                self.Ty  = np.zeros((Nx,Ny), dtype=typ)
                self.Tyy = np.zeros((Nx,Ny), dtype=typ)
            
                self.Tz  = np.zeros((Nx,Ny), dtype=typ)
                self.Tzz = np.zeros((Nx,Ny), dtype=typ)
            
                self.Txy = np.zeros((Nx,Ny), dtype=typ)
                self.Txz = np.zeros((Nx,Ny), dtype=typ)
                self.Tyz = np.zeros((Nx,Ny), dtype=typ)
            
                # Conservative quantities
                self.E    = np.zeros((Nx,Ny), dtype=typ)
                self.rhoU = np.zeros((Nx,Ny), dtype=typ)
                self.rhoV = np.zeros((Nx,Ny), dtype=typ)
                self.rhoW = np.zeros((Nx,Ny), dtype=typ)
            
            
            else:
                # Field dimension
                self.Ny  = Ny
                self.Nx  = Nx
                self.M   = M
                self.N   = N
                self.x   = np.zeros((M,N,Nx,Ny), dtype=typ)
                self.xc  = np.zeros((M,N,Nx,Ny), dtype=typ)
                self.y   = np.zeros((M,N,Nx,Ny), dtype=typ)
                self.yc  = np.zeros((M,N,Nx,Ny), dtype=typ)
                self.Rex = np.zeros((M,N,Nx,Ny), dtype=typ)

                # wall angle
                self.theta = np.zeros((M,N,Nx,Ny), dtype=typ)
            
                # Non-dim Y direction
                self.eta = np.zeros((M,N,Nx,Ny), dtype=typ)
            
                # Field Properties
                self.Cp     = np.zeros((M,N,Nx,Ny), dtype=typ)
                self.mu     = np.zeros((M,N,Nx,Ny), dtype=typ)
                self.muT    = np.zeros((M,N,Nx,Ny), dtype=typ)
                self.muTT   = np.zeros((M,N,Nx,Ny), dtype=typ)
                self.mux    = np.zeros((M,N,Nx,Ny), dtype=typ)
                self.muxx   = np.zeros((M,N,Nx,Ny), dtype=typ)
                self.muxy   = np.zeros((M,N,Nx,Ny), dtype=typ)
                self.muy    = np.zeros((M,N,Nx,Ny), dtype=typ)
                self.muyy   = np.zeros((M,N,Nx,Ny), dtype=typ)
                self.lamb   = np.zeros((M,N,Nx,Ny), dtype=typ)
                self.lambx  = np.zeros((M,N,Nx,Ny), dtype=typ)
                self.lambxx = np.zeros((M,N,Nx,Ny), dtype=typ)
                self.lamby  = np.zeros((M,N,Nx,Ny), dtype=typ)
                self.lambyy = np.zeros((M,N,Nx,Ny), dtype=typ)
                self.lambxy = np.zeros((M,N,Nx,Ny), dtype=typ)
                self.lambT  = np.zeros((M,N,Nx,Ny), dtype=typ)
                self.lambTT = np.zeros((M,N,Nx,Ny), dtype=typ)
                self.rho    = np.zeros((M,N,Nx,Ny), dtype=typ)
                self.rhox   = np.zeros((M,N,Nx,Ny), dtype=typ)
                self.rhoxx  = np.zeros((M,N,Nx,Ny), dtype=typ)
                self.rhoy   = np.zeros((M,N,Nx,Ny), dtype=typ)
                self.rhoyy  = np.zeros((M,N,Nx,Ny), dtype=typ)
                self.rhoxy  = np.zeros((M,N,Nx,Ny), dtype=typ)
            
                # Dimensionless quantities
                self.delta   =  np.zeros(Nx, dtype=typ)
            
                # Disturbance kinetic energy (DKE)
                self.DKE  = np.zeros((M,N,Nx), dtype=typ)
                self.DKEx = np.zeros((M,N,Nx), dtype=typ)

                # Amplitude
                self.amplitude = np.zeros((M,N,Nx), dtype=typ)
            
                # Wavenumbers
                self.omega = np.zeros((M,N,Nx), dtype=typ)
                self.F     = np.zeros((M,N,Nx), dtype=typ)
                self.B     = np.zeros((M,N,Nx), dtype=typ)
                self.beta  = np.zeros((M,N,Nx), dtype=typ)
            
                # Growth rate
                self.sigma   = np.zeros((M,N,Nx), dtype=typ)
                self.nfactor = np.zeros((M,N,Nx), dtype=typ)
                self.Nfactor = np.zeros(      Nx, dtype=typ)
            
                # Growth rate parameter alpha
                self.alpha   = np.zeros((M,N,Nx), dtype=typ)
                self.alphax  = np.zeros((M,N,Nx), dtype=typ)
            
                # U
                self.U   = np.zeros((M,N,Nx,Ny), dtype=typ)
                self.Ux  = np.zeros((M,N,Nx,Ny), dtype=typ)
                self.Uxx = np.zeros((M,N,Nx,Ny), dtype=typ)
            
                self.Uy  = np.zeros((M,N,Nx,Ny), dtype=typ)
                self.Uyy = np.zeros((M,N,Nx,Ny), dtype=typ)
            
                self.Uz  = np.zeros((M,N,Nx,Ny), dtype=typ)
                self.Uzz = np.zeros((M,N,Nx,Ny), dtype=typ)
            
                self.Uxy = np.zeros((M,N,Nx,Ny), dtype=typ)
                self.Uxz = np.zeros((M,N,Nx,Ny), dtype=typ)
                self.Uyz = np.zeros((M,N,Nx,Ny), dtype=typ)
            
                # V
                self.V   = np.zeros((M,N,Nx,Ny), dtype=typ)
                self.Vx  = np.zeros((M,N,Nx,Ny), dtype=typ)
                self.Vxx = np.zeros((M,N,Nx,Ny), dtype=typ)
            
                self.Vy  = np.zeros((M,N,Nx,Ny), dtype=typ)
                self.Vyy = np.zeros((M,N,Nx,Ny), dtype=typ)
            
                self.Vz  = np.zeros((M,N,Nx,Ny), dtype=typ)
                self.Vzz = np.zeros((M,N,Nx,Ny), dtype=typ)
            
                self.Vxy = np.zeros((M,N,Nx,Ny), dtype=typ)
                self.Vxz = np.zeros((M,N,Nx,Ny), dtype=typ)
                self.Vyz = np.zeros((M,N,Nx,Ny), dtype=typ)
            
                # W
                self.W   = np.zeros((M,N,Nx,Ny), dtype=typ)
                self.Wx  = np.zeros((M,N,Nx,Ny), dtype=typ)
                self.Wxx = np.zeros((M,N,Nx,Ny), dtype=typ)
            
                self.Wy  = np.zeros((M,N,Nx,Ny), dtype=typ)
                self.Wyy = np.zeros((M,N,Nx,Ny), dtype=typ)
            
                self.Wz  = np.zeros((M,N,Nx,Ny), dtype=typ)
                self.Wzz = np.zeros((M,N,Nx,Ny), dtype=typ)
            
                self.Wxy = np.zeros((M,N,Nx,Ny), dtype=typ)
                self.Wxz = np.zeros((M,N,Nx,Ny), dtype=typ)
                self.Wyz = np.zeros((M,N,Nx,Ny), dtype=typ)
            
                # P
                self.P   = np.zeros((M,N,Nx,Ny), dtype=typ)
                self.Px  = np.zeros((M,N,Nx,Ny), dtype=typ)
                self.Pxx = np.zeros((M,N,Nx,Ny), dtype=typ)
            
                self.Py  = np.zeros((M,N,Nx,Ny), dtype=typ)
                self.Pyy = np.zeros((M,N,Nx,Ny), dtype=typ)
            
                self.Pz  = np.zeros((M,N,Nx,Ny), dtype=typ)
                self.Pzz = np.zeros((M,N,Nx,Ny), dtype=typ)
            
                self.Pxy = np.zeros((M,N,Nx,Ny), dtype=typ)
                self.Pxz = np.zeros((M,N,Nx,Ny), dtype=typ)
                self.Pyz = np.zeros((M,N,Nx,Ny), dtype=typ)
            
                # T
                self.T   = np.zeros((M,N,Nx,Ny), dtype=typ)
                self.Tx  = np.zeros((M,N,Nx,Ny), dtype=typ)
                self.Txx = np.zeros((M,N,Nx,Ny), dtype=typ)
            
                self.T   = np.zeros((M,N,Nx,Ny), dtype=typ)
                self.Ty  = np.zeros((M,N,Nx,Ny), dtype=typ)
                self.Tyy = np.zeros((M,N,Nx,Ny), dtype=typ)
            
                self.Tz  = np.zeros((M,N,Nx,Ny), dtype=typ)
                self.Tzz = np.zeros((M,N,Nx,Ny), dtype=typ)
            
                self.Txy = np.zeros((M,N,Nx,Ny), dtype=typ)
                self.Txz = np.zeros((M,N,Nx,Ny), dtype=typ)
                self.Tyz = np.zeros((M,N,Nx,Ny), dtype=typ)
            
                # Conservative quantities
                self.E    = np.zeros((M,N,Nx,Ny), dtype=typ)
                self.rhoU = np.zeros((M,N,Nx,Ny), dtype=typ)
                self.rhoV = np.zeros((M,N,Nx,Ny), dtype=typ)
                self.rhoW = np.zeros((M,N,Nx,Ny), dtype=typ)

    def interpVar(self, xx, yy, var, _method='linear', grid='curvilinear', mode=(0,0)):
        # Interpolate the variable Var on a new grid (xx,yy)
        from scipy.interpolate import griddata
        if mode == (0,0):
            x = self.x ; xc = self.xc
            y = self.y ; yc = self.yc
            values = eval('self.'+var+'.reshape(x.shape[0]*x.shape[1])')

            if grid == 'curvilinear':
                coords = np.stack((x.reshape(x.shape[0]*x.shape[1]), \
                                   y.reshape(y.shape[0]*y.shape[1])), axis=1)
                xx = np.clip(xx, np.min(x), np.max(x))
                yy = np.clip(yy, np.min(y), np.max(y))
            elif grid == 'cartesian':
                coords = np.stack((xc.reshape(xc.shape[0]*xc.shape[1]), \
                                   yc.reshape(yc.shape[0]*yc.shape[1])), axis=1)
                xx = np.clip(xx, np.min(xc), np.max(xc))
                yy = np.clip(yy, np.min(yc), np.max(yc))

            coords_i = np.stack((xx.reshape(xx.shape[0]*xx.shape[1]), \
                                 yy.reshape(yy.shape[0]*yy.shape[1])), axis=1)
        else:
            (m,n) = mode
            x = self.x[m,n,:,:] ; xc = self.xc[m,n,:,:]
            y = self.y[m,n,:,:] ; yc = self.yc[m,n,:,:]
            values = eval('self.'+var+'[m,n,:,:].reshape(x.shape[0]*x.shape[1])')

            if grid == 'curvilinear':
                coords = np.stack((x.reshape(x.shape[0]*x.shape[1]), \
                                   y.reshape(y.shape[0]*y.shape[1])), axis=1)
                xx = np.clip(xx, np.min(x), np.max(x))
                yy = np.clip(yy, np.min(y), np.max(y))
            elif grid == 'cartesian':
                coords = np.stack((xc.reshape(xc.shape[0]*xc.shape[1]), \
                                   yc.reshape(yc.shape[0]*yc.shape[1])), axis=1)
                xx = np.clip(xx, np.min(xc), np.max(xc))
                yy = np.clip(yy, np.min(yc), np.max(yc))

            coords_i = np.stack((xx.reshape(xx.shape[0]*xx.shape[1]), \
                                 yy.reshape(yy.shape[0]*yy.shape[1])), axis=1)

        return griddata(coords, values,coords_i, method=_method).reshape(xx.shape[0], xx.shape[1])

    def interp(self,this,method='linear'):
        from scipy.interpolate import interp2d
        # interp self into this
        # Minimum requirement, this.x, this.y must be defined
        # wall angle
        if (this.M,this.N) == (0,0):
            x  = self.x[:,0].real
            y  = self.y[0,:].real
            xx = this.x[:,0].real
            yy = this.y[0,:].real

            # Field Properties
            this.Cp     = interp2d(y,x,     self.Cp.real, kind=method)(yy,xx) \
                        + interp2d(y,x,     self.Cp.imag, kind=method)(yy,xx)*1j
            this.mu     = interp2d(y,x,     self.mu.real, kind=method)(yy,xx) \
                        + interp2d(y,x,     self.mu.imag, kind=method)(yy,xx)*1j
            this.muT    = interp2d(y,x,    self.muT.real, kind=method)(yy,xx) \
                        + interp2d(y,x,    self.muT.imag, kind=method)(yy,xx)*1j
            this.muTT   = interp2d(y,x,   self.muTT.real, kind=method)(yy,xx) \
                        + interp2d(y,x,   self.muTT.imag, kind=method)(yy,xx)*1j
            this.mux    = interp2d(y,x,    self.mux.real, kind=method)(yy,xx) \
                        + interp2d(y,x,    self.mux.imag, kind=method)(yy,xx)*1j
            this.muxx   = interp2d(y,x,   self.muxx.real, kind=method)(yy,xx) \
                        + interp2d(y,x,   self.muxx.imag, kind=method)(yy,xx)*1j
            this.muxy   = interp2d(y,x,   self.muxy.real, kind=method)(yy,xx) \
                        + interp2d(y,x,   self.muxy.imag, kind=method)(yy,xx)*1j
            this.muy    = interp2d(y,x,   self.muxx.real, kind=method)(yy,xx) \
                        + interp2d(y,x,   self.muxx.imag, kind=method)(yy,xx)*1j
            this.muyy   = interp2d(y,x,   self.muyy.real, kind=method)(yy,xx) \
                        + interp2d(y,x,   self.muyy.imag, kind=method)(yy,xx)*1j
            this.lamb   = interp2d(y,x,   self.lamb.real, kind=method)(yy,xx) \
                        + interp2d(y,x,   self.lamb.imag, kind=method)(yy,xx)*1j
            this.lambx  = interp2d(y,x,  self.lambx.real, kind=method)(yy,xx) \
                        + interp2d(y,x,  self.lambx.imag, kind=method)(yy,xx)*1j
            this.lambxx = interp2d(y,x, self.lambxx.real, kind=method)(yy,xx) \
                        + interp2d(y,x, self.lambxx.imag, kind=method)(yy,xx)*1j
            this.lambxy = interp2d(y,x, self.lambxy.real, kind=method)(yy,xx) \
                        + interp2d(y,x, self.lambxy.imag, kind=method)(yy,xx)*1j
            this.lamby  = interp2d(y,x,  self.lamby.real, kind=method)(yy,xx) \
                        + interp2d(y,x,  self.lamby.imag, kind=method)(yy,xx)*1j
            this.lambyy = interp2d(y,x, self.lambyy.real, kind=method)(yy,xx) \
                        + interp2d(y,x, self.lambyy.imag, kind=method)(yy,xx)*1j
            this.lambT  = interp2d(y,x,  self.lambT.real, kind=method)(yy,xx) \
                        + interp2d(y,x,  self.lambT.imag, kind=method)(yy,xx)*1j
            this.lambTT = interp2d(y,x, self.lambTT.real, kind=method)(yy,xx) \
                        + interp2d(y,x, self.lambTT.imag, kind=method)(yy,xx)*1j
                                                                                           
            this.rho    = interp2d(y,x,    self.rho.real, kind=method)(yy,xx) \
                        + interp2d(y,x,    self.rho.imag, kind=method)(yy,xx)*1j
            this.rhox   = interp2d(y,x,   self.rhox.real, kind=method)(yy,xx) \
                        + interp2d(y,x,   self.rhox.imag, kind=method)(yy,xx)*1j
            this.rhoxx  = interp2d(y,x,  self.rhoxx.real, kind=method)(yy,xx) \
                        + interp2d(y,x,  self.rhoxx.imag, kind=method)(yy,xx)*1j
            this.rhoy   = interp2d(y,x,   self.rhoy.real, kind=method)(yy,xx) \
                        + interp2d(y,x,   self.rhoy.imag, kind=method)(yy,xx)*1j
            this.rhoyy  = interp2d(y,x,  self.rhoyy.real, kind=method)(yy,xx) \
                        + interp2d(y,x,  self.rhoyy.imag, kind=method)(yy,xx)*1j
            this.rhoxy  = interp2d(y,x,  self.rhoxy.real, kind=method)(yy,xx) \
                        + interp2d(y,x,  self.rhoxy.imag, kind=method)(yy,xx)*1j
            
            # U
            this.U   = interp2d(y,x,   self.U.real, kind=method)(yy,xx) \
                     + interp2d(y,x,   self.U.imag, kind=method)(yy,xx)*1j
            this.Ux  = interp2d(y,x,  self.Ux.real, kind=method)(yy,xx) \
                     + interp2d(y,x,  self.Ux.imag, kind=method)(yy,xx)*1j
            this.Uxx = interp2d(y,x, self.Uxx.real, kind=method)(yy,xx) \
                     + interp2d(y,x, self.Uxx.imag, kind=method)(yy,xx)*1j
            
            this.Uy  = interp2d(y,x,  self.Uy.real, kind=method)(yy,xx) \
                     + interp2d(y,x,  self.Uy.imag, kind=method)(yy,xx)*1j
            this.Uyy = interp2d(y,x, self.Uyy.real, kind=method)(yy,xx) \
                     + interp2d(y,x, self.Uyy.imag, kind=method)(yy,xx)*1j
            
            this.Uz  = interp2d(y,x,  self.Uz.real, kind=method)(yy,xx) \
                     + interp2d(y,x,  self.Uz.imag, kind=method)(yy,xx)*1j
            this.Uzz = interp2d(y,x, self.Uzz.real, kind=method)(yy,xx) \
                     + interp2d(y,x, self.Uzz.imag, kind=method)(yy,xx)*1j
            
            this.Uxy = interp2d(y,x, self.Uxy.real, kind=method)(yy,xx) \
                     + interp2d(y,x, self.Uxy.imag, kind=method)(yy,xx)*1j
            this.Uxz = interp2d(y,x, self.Uxz.real, kind=method)(yy,xx) \
                     + interp2d(y,x, self.Uxz.imag, kind=method)(yy,xx)*1j
            this.Uyz = interp2d(y,x, self.Uyz.real, kind=method)(yy,xx) \
                     + interp2d(y,x, self.Uyz.imag, kind=method)(yy,xx)*1j
            
            # V
            this.V   = interp2d(y,x,   self.V.real, kind=method)(yy,xx) \
                     + interp2d(y,x,   self.V.imag, kind=method)(yy,xx)*1j
            this.Vx  = interp2d(y,x,  self.Vx.real, kind=method)(yy,xx) \
                     + interp2d(y,x,  self.Vx.imag, kind=method)(yy,xx)*1j
            this.Vxx = interp2d(y,x, self.Vxx.real, kind=method)(yy,xx) \
                     + interp2d(y,x, self.Vxx.imag, kind=method)(yy,xx)*1j
            
            this.Vy  = interp2d(y,x,  self.Vy.real, kind=method)(yy,xx) \
                     + interp2d(y,x,  self.Vy.imag, kind=method)(yy,xx)*1j
            this.Vyy = interp2d(y,x, self.Vyy.real, kind=method)(yy,xx) \
                     + interp2d(y,x, self.Vyy.imag, kind=method)(yy,xx)*1j
            
            this.Vz  = interp2d(y,x,  self.Vz.real, kind=method)(yy,xx) \
                     + interp2d(y,x,  self.Vz.imag, kind=method)(yy,xx)*1j
            this.Vzz = interp2d(y,x, self.Vzz.real, kind=method)(yy,xx) \
                     + interp2d(y,x, self.Vzz.imag, kind=method)(yy,xx)*1j
            
            this.Vxy = interp2d(y,x, self.Vxy.real, kind=method)(yy,xx) \
                     + interp2d(y,x, self.Vxy.imag, kind=method)(yy,xx)*1j
            this.Vxz = interp2d(y,x, self.Vxz.real, kind=method)(yy,xx) \
                     + interp2d(y,x, self.Vxz.imag, kind=method)(yy,xx)*1j
            this.Vyz = interp2d(y,x, self.Vyz.real, kind=method)(yy,xx) \
                     + interp2d(y,x, self.Vyz.imag, kind=method)(yy,xx)*1j
            
            # W
            this.W   = interp2d(y,x,   self.W.real, kind=method)(yy,xx) \
                     + interp2d(y,x,   self.W.imag, kind=method)(yy,xx)*1j
            this.Wx  = interp2d(y,x,  self.Wx.real, kind=method)(yy,xx) \
                     + interp2d(y,x,  self.Wx.imag, kind=method)(yy,xx)*1j
            this.Wxx = interp2d(y,x, self.Wxx.real, kind=method)(yy,xx) \
                     + interp2d(y,x, self.Wxx.imag, kind=method)(yy,xx)*1j
            
            this.Wy  = interp2d(y,x,  self.Wy.real, kind=method)(yy,xx) \
                     + interp2d(y,x,  self.Wy.imag, kind=method)(yy,xx)*1j
            this.Wyy = interp2d(y,x, self.Wyy.real, kind=method)(yy,xx) \
                     + interp2d(y,x, self.Wyy.imag, kind=method)(yy,xx)*1j
            
            this.Wz  = interp2d(y,x,  self.Wz.real, kind=method)(yy,xx) \
                     + interp2d(y,x,  self.Wz.imag, kind=method)(yy,xx)*1j
            this.Wzz = interp2d(y,x, self.Wzz.real, kind=method)(yy,xx) \
                     + interp2d(y,x, self.Wzz.imag, kind=method)(yy,xx)*1j
            
            this.Wxy = interp2d(y,x, self.Wxy.real, kind=method)(yy,xx) \
                     + interp2d(y,x, self.Wxy.imag, kind=method)(yy,xx)*1j
            this.Wxz = interp2d(y,x, self.Wxz.real, kind=method)(yy,xx) \
                     + interp2d(y,x, self.Wxz.imag, kind=method)(yy,xx)*1j
            this.Wyz = interp2d(y,x, self.Wyz.real, kind=method)(yy,xx) \
                     + interp2d(y,x, self.Wyz.imag, kind=method)(yy,xx)*1j
            
            # P
            this.P   = interp2d(y,x,   self.P.real, kind=method)(yy,xx) \
                     + interp2d(y,x,   self.P.imag, kind=method)(yy,xx)*1j
            this.Px  = interp2d(y,x,  self.Px.real, kind=method)(yy,xx) \
                     + interp2d(y,x,  self.Px.imag, kind=method)(yy,xx)*1j
            this.Pxx = interp2d(y,x, self.Pxx.real, kind=method)(yy,xx) \
                     + interp2d(y,x, self.Pxx.imag, kind=method)(yy,xx)*1j
            
            this.Py  = interp2d(y,x,  self.Py.real, kind=method)(yy,xx) \
                     + interp2d(y,x,  self.Py.imag, kind=method)(yy,xx)*1j
            this.Pyy = interp2d(y,x, self.Pyy.real, kind=method)(yy,xx) \
                     + interp2d(y,x, self.Pyy.imag, kind=method)(yy,xx)*1j
            
            this.Pz  = interp2d(y,x,  self.Pz.real, kind=method)(yy,xx) \
                     + interp2d(y,x,  self.Pz.imag, kind=method)(yy,xx)*1j
            this.Pzz = interp2d(y,x, self.Pzz.real, kind=method)(yy,xx) \
                     + interp2d(y,x, self.Pzz.imag, kind=method)(yy,xx)*1j
            
            this.Pxy = interp2d(y,x, self.Pxy.real, kind=method)(yy,xx) \
                     + interp2d(y,x, self.Pxy.imag, kind=method)(yy,xx)*1j
            this.Pxz = interp2d(y,x, self.Pxz.real, kind=method)(yy,xx) \
                     + interp2d(y,x, self.Pxz.imag, kind=method)(yy,xx)*1j
            this.Pyz = interp2d(y,x, self.Pyz.real, kind=method)(yy,xx) \
                     + interp2d(y,x, self.Pyz.imag, kind=method)(yy,xx)*1j
            
            # T
            this.T   = interp2d(y,x,   self.T.real, kind=method)(yy,xx) \
                     + interp2d(y,x,   self.T.imag, kind=method)(yy,xx)*1j
            this.Tx  = interp2d(y,x,  self.Tx.real, kind=method)(yy,xx) \
                     + interp2d(y,x,  self.Tx.imag, kind=method)(yy,xx)*1j
            this.Txx = interp2d(y,x, self.Txx.real, kind=method)(yy,xx) \
                     + interp2d(y,x, self.Txx.imag, kind=method)(yy,xx)*1j
            
            this.Ty  = interp2d(y,x,  self.Ty.real, kind=method)(yy,xx) \
                     + interp2d(y,x,  self.Ty.imag, kind=method)(yy,xx)*1j
            this.Tyy = interp2d(y,x, self.Tyy.real, kind=method)(yy,xx) \
                     + interp2d(y,x, self.Tyy.imag, kind=method)(yy,xx)*1j
            
            this.Tz  = interp2d(y,x,  self.Tz.real, kind=method)(yy,xx) \
                     + interp2d(y,x,  self.Tz.imag, kind=method)(yy,xx)*1j
            this.Tzz = interp2d(y,x, self.Tzz.real, kind=method)(yy,xx) \
                     + interp2d(y,x, self.Tzz.imag, kind=method)(yy,xx)*1j
            
            this.Txy = interp2d(y,x, self.Txy.real, kind=method)(yy,xx) \
                     + interp2d(y,x, self.Txy.imag, kind=method)(yy,xx)*1j
            this.Txz = interp2d(y,x, self.Txz.real, kind=method)(yy,xx) \
                     + interp2d(y,x, self.Txz.imag, kind=method)(yy,xx)*1j
            this.Tyz = interp2d(y,x, self.Tyz.real, kind=method)(yy,xx) \
                     + interp2d(y,x, self.Tyz.imag, kind=method)(yy,xx)*1j
            
            # Conservative quantities
            this.E    = interp2d(y,x,    self.E.real, kind=method)(yy,xx) \
                      + interp2d(y,x,    self.E.imag, kind=method)(yy,xx)*1j
            this.rhoU = interp2d(y,x, self.rhoU.real, kind=method)(yy,xx) \
                      + interp2d(y,x, self.rhoU.imag, kind=method)(yy,xx)*1j
            this.rhoV = interp2d(y,x, self.rhoV.real, kind=method)(yy,xx) \
                      + interp2d(y,x, self.rhoV.imag, kind=method)(yy,xx)*1j
            this.rhoW = interp2d(y,x, self.rhoW.real, kind=method)(yy,xx) \
                      + interp2d(y,x, self.rhoW.imag, kind=method)(yy,xx)*1j
        else:
            for m in range(0, this.M):
                # Growth rate parameter alpha
                this.alpha   = interp2d(self.x[m,:,:].real,self.F[m,:,:].real,   self.alpha[m,:,:].real, kind=method)(this.x[m,:,:].real,this.F[m,:,:].real) \
                             + interp2d(self.x[m,:,:].real,self.F[m,:,:].real,   self.alpha[m,:,:].imag, kind=method)(this.x[m,:,:].real,this.F[m,:,:].real)*1j
                this.alphax  = interp2d(self.x[m,:,:].real,self.F[m,:,:].real,  self.alphax[m,:,:].real, kind=method)(this.x[m,:,:].real,this.F[m,:,:].real) \
                             + interp2d(self.x[m,:,:].real,self.F[m,:,:].real,  self.alphax[m,:,:].imag, kind=method)(this.x[m,:,:].real,this.F[m,:,:].real)*1j
                # Growth rate
                this.sigma[m,:,:]   = interp2d(self.x[m,:,:].real,self.F[m,:,:].real,   self.sigma[m,:,:].real, kind=method)(this.x[m,:,:].real,this.F[m,:,:].real) \
                                    + interp2d(self.x[m,:,:].real,self.F[m,:,:].real,   self.sigma[m,:,:].imag, kind=method)(this.x[m,:,:].real,this.F[m,:,:].real)*1j
                this.nfactor[m,:,:] = interp2d(self.x[m,:,:].real,self.F[m,:,:].real, self.nfactor[m,:,:].real, kind=method)(this.x[m,:,:].real,this.F[m,:,:].real) \
                                    + interp2d(self.x[m,:,:].real,self.F[m,:,:].real, self.nfactor[m,:,:].imag, kind=method)(this.x[m,:,:].real,this.F[m,:,:].real)*1j
                this.Nfactor[m,:]   = interp1d(self.x[m,:,:].real, self.Nfactor.real, kind=method)(this.x[m,:,:].real) \
                                    + interp1d(self.x[m,:,:].real, self.Nfactor.imag, kind=method)(this.x[m,:,:].real)*1j

                for n in range(0,this.N):
                    # Field Properties
                    this.Cp[m,n,:,:]     = interp2d(self.x[m,n,:,:] .real,self.y.real,    self.Cp[m,n,:,:].real, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real) \
                                         + interp2d(self.x[m,n,:,:] .real,self.y.real,    self.Cp[m,n,:,:].imag, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real)*1j
                    this.mu[m,n,:,:]     = interp2d(self.x[m,n,:,:] .real,self.y.real,    self.mu[m,n,:,:].real, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real) \
                                         + interp2d(self.x[m,n,:,:] .real,self.y.real,    self.mu[m,n,:,:].imag, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real)*1j
                    this.muT[m,n,:,:]    = interp2d(self.x[m,n,:,:] .real,self.y.real,   self.muT[m,n,:,:].real, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real) \
                                         + interp2d(self.x[m,n,:,:].real,self.y.real,    self.muT[m,n,:,:].imag, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real)*1j
                    this.muTT[m,n,:,:]   = interp2d(self.x[m,n,:,:].real,self.y.real,   self.muTT[m,n,:,:].real, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real) \
                                         + interp2d(self.x[m,n,:,:].real,self.y.real,   self.muTT[m,n,:,:].imag, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real)*1j
                    this.mux[m,n,:,:]    = interp2d(self.x[m,n,:,:].real,self.y.real,    self.mux[m,n,:,:].real, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real) \
                                         + interp2d(self.x[m,n,:,:].real,self.y.real,    self.mux[m,n,:,:].imag, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real)*1j
                    this.muxx[m,n,:,:]   = interp2d(self.x[m,n,:,:].real,self.y.real,   self.muxx[m,n,:,:].real, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real) \
                                         + interp2d(self.x[m,n,:,:].real,self.y.real,   self.muxx[m,n,:,:].imag, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real)*1j
                    this.muxy[m,n,:,:]   = interp2d(self.x[m,n,:,:].real,self.y.real,   self.muxy[m,n,:,:].real, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real) \
                                         + interp2d(self.x[m,n,:,:].real,self.y.real,   self.muxy[m,n,:,:].imag, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real)*1j
                    this.muy[m,n,:,:]    = interp2d(self.x[m,n,:,:].real,self.y.real,   self.muxx[m,n,:,:].real, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real) \
                                         + interp2d(self.x[m,n,:,:].real,self.y.real,   self.muxx[m,n,:,:].imag, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real)*1j
                    this.muyy[m,n,:,:]   = interp2d(self.x[m,n,:,:].real,self.y.real,   self.muyy[m,n,:,:].real, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real) \
                                         + interp2d(self.x[m,n,:,:].real,self.y.real,   self.muyy[m,n,:,:].imag, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real)*1j
                    this.lamb[m,n,:,:]   = interp2d(self.x[m,n,:,:].real,self.y.real,   self.lamb[m,n,:,:].real, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real) \
                                         + interp2d(self.x[m,n,:,:].real,self.y.real,   self.lamb[m,n,:,:].imag, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real)*1j
                    this.lambx[m,n,:,:]  = interp2d(self.x[m,n,:,:].real,self.y.real,  self.lambx[m,n,:,:].real, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real) \
                                         + interp2d(self.x[m,n,:,:].real,self.y.real,  self.lambx[m,n,:,:].imag, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real)*1j
                    this.lambxx[m,n,:,:] = interp2d(self.x[m,n,:,:].real,self.y.real, self.lambxx[m,n,:,:].real, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real) \
                                         + interp2d(self.x[m,n,:,:].real,self.y.real, self.lambxx[m,n,:,:].imag, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real)*1j
                    this.lambxy[m,n,:,:] = interp2d(self.x[m,n,:,:].real,self.y.real, self.lambxy[m,n,:,:].real, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real) \
                                         + interp2d(self.x[m,n,:,:].real,self.y.real, self.lambxy[m,n,:,:].imag, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real)*1j
                    this.lamby[m,n,:,:]  = interp2d(self.x[m,n,:,:].real,self.y.real,  self.lamby[m,n,:,:].real, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real) \
                                         + interp2d(self.x[m,n,:,:].real,self.y.real,  self.lamby[m,n,:,:].imag, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real)*1j
                    this.lambyy[m,n,:,:] = interp2d(self.x[m,n,:,:].real,self.y.real, self.lambyy[m,n,:,:].real, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real) \
                                         + interp2d(self.x[m,n,:,:].real,self.y.real, self.lambyy[m,n,:,:].imag, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real)*1j
                    this.lambT[m,n,:,:]  = interp2d(self.x[m,n,:,:].real,self.y.real,  self.lambT[m,n,:,:].real, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real) \
                                         + interp2d(self.x[m,n,:,:].real,self.y.real,  self.lambT[m,n,:,:].imag, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real)*1j
                    this.lambTT[m,n,:,:] = interp2d(self.x[m,n,:,:].real,self.y.real, self.lambTT[m,n,:,:].real, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real) \
                                         + interp2d(self.x[m,n,:,:].real,self.y.real, self.lambTT[m,n,:,:].imag, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real)*1j

                    this.rho[m,n,:,:]    = interp2d(self.x[m,n,:,:].real,self.y[m,n,:,:].real,    self.rho[m,n,:,:].real, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real) \
                                         + interp2d(self.x[m,n,:,:].real,self.y[m,n,:,:].real,    self.rho[m,n,:,:].imag, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real)*1j
                    this.rhox[m,n,:,:]   = interp2d(self.x[m,n,:,:].real,self.y[m,n,:,:].real,   self.rhox[m,n,:,:].real, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real) \
                                         + interp2d(self.x[m,n,:,:].real,self.y[m,n,:,:].real,   self.rhox[m,n,:,:].imag, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real)*1j
                    this.rhoxx[m,n,:,:]  = interp2d(self.x[m,n,:,:].real,self.y[m,n,:,:].real,  self.rhoxx[m,n,:,:].real, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real) \
                                         + interp2d(self.x[m,n,:,:].real,self.y[m,n,:,:].real,  self.rhoxx[m,n,:,:].imag, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real)*1j
                    this.rhoy[m,n,:,:]   = interp2d(self.x[m,n,:,:].real,self.y[m,n,:,:].real,   self.rhoy[m,n,:,:].real, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real) \
                                         + interp2d(self.x[m,n,:,:].real,self.y[m,n,:,:].real,   self.rhoy[m,n,:,:].imag, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real)*1j
                    this.rhoyy[m,n,:,:]  = interp2d(self.x[m,n,:,:].real,self.y[m,n,:,:].real,  self.rhoyy[m,n,:,:].real, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real) \
                                         + interp2d(self.x[m,n,:,:].real,self.y[m,n,:,:].real,  self.rhoyy[m,n,:,:].imag, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real)*1j
                    this.rhoxy[m,n,:,:]  = interp2d(self.x[m,n,:,:].real,self.y[m,n,:,:].real,  self.rhoxy[m,n,:,:].real, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real) \
                                         + interp2d(self.x[m,n,:,:].real,self.y[m,n,:,:].real,  self.rhoxy[m,n,:,:].imag, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real)*1j
                    # Disturbance kinetic energy (DKE)
                    this.DKE[m,n,:,:]    = interp2d(self.x[m,n,:,:].real,self.y[m,n,:,:].real,    self.DKE[m,n,:,:].real, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real) \
                                         + interp2d(self.x[m,n,:,:].real,self.y[m,n,:,:].real,    self.DKE[m,n,:,:].imag, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real)*1j
                    this.DKEx[m,n,:,:]   = interp2d(self.x[m,n,:,:].real,self.y[m,n,:,:].real,   self.DKEx[m,n,:,:].real, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real) \
                                         + interp2d(self.x[m,n,:,:].real,self.y[m,n,:,:].real,   self.DKEx[m,n,:,:].imag, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real)*1j
                    
                    this.amplitude[m,n,:,:] = interp2d(self.x[m,n,:,:].real,self.y[m,n,:,:].real,   self.amplitude[m,n,:,:].real, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real) \
                                         + interp2d(self.x[m,n,:,:].real,self.y[m,n,:,:].real,   self.amplitude[m,n,:,:].imag, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real)*1j
                    
                    # U
                    this.U[m,n,:,:]   = interp2d(self.x[m,n,:,:].real,self.y[m,n,:,:].real,   self.U[m,n,:,:].real, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real) \
                                      + interp2d(self.x[m,n,:,:].real,self.y[m,n,:,:].real,   self.U[m,n,:,:].imag, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real)*1j
                    this.Ux[m,n,:,:]  = interp2d(self.x[m,n,:,:].real,self.y[m,n,:,:].real,  self.Ux[m,n,:,:].real, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real) \
                                      + interp2d(self.x[m,n,:,:].real,self.y[m,n,:,:].real,  self.Ux[m,n,:,:].imag, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real)*1j
                    this.Uxx[m,n,:,:] = interp2d(self.x[m,n,:,:].real,self.y[m,n,:,:].real, self.Uxx[m,n,:,:].real, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real) \
                                      + interp2d(self.x[m,n,:,:].real,self.y[m,n,:,:].real, self.Uxx[m,n,:,:].imag, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real)*1j
                    this.Uy[m,n,:,:]  = interp2d(self.x[m,n,:,:].real,self.y[m,n,:,:].real,  self.Uy[m,n,:,:].real, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real) \
                                      + interp2d(self.x[m,n,:,:].real,self.y[m,n,:,:].real,  self.Uy[m,n,:,:].imag, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real)*1j
                    this.Uyy[m,n,:,:] = interp2d(self.x[m,n,:,:].real,self.y[m,n,:,:].real, self.Uyy[m,n,:,:].real, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real) \
                                      + interp2d(self.x[m,n,:,:].real,self.y[m,n,:,:].real, self.Uyy[m,n,:,:].imag, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real)*1j
                    this.Uz[m,n,:,:]  = interp2d(self.x[m,n,:,:].real,self.y[m,n,:,:].real,  self.Uz[m,n,:,:].real, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real) \
                                      + interp2d(self.x[m,n,:,:].real,self.y[m,n,:,:].real,  self.Uz[m,n,:,:].imag, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real)*1j
                    this.Uzz[m,n,:,:] = interp2d(self.x[m,n,:,:].real,self.y[m,n,:,:].real, self.Uzz[m,n,:,:].real, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real) \
                                      + interp2d(self.x[m,n,:,:].real,self.y[m,n,:,:].real, self.Uzz[m,n,:,:].imag, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real)*1j
                    this.Uxy[m,n,:,:] = interp2d(self.x[m,n,:,:].real,self.y[m,n,:,:].real, self.Uxy[m,n,:,:].real, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real) \
                                      + interp2d(self.x[m,n,:,:].real,self.y[m,n,:,:].real, self.Uxy[m,n,:,:].imag, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real)*1j
                    this.Uxz[m,n,:,:] = interp2d(self.x[m,n,:,:].real,self.y[m,n,:,:].real, self.Uxz[m,n,:,:].real, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real) \
                                      + interp2d(self.x[m,n,:,:].real,self.y[m,n,:,:].real, self.Uxz[m,n,:,:].imag, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real)*1j
                    this.Uyz[m,n,:,:] = interp2d(self.x[m,n,:,:].real,self.y[m,n,:,:].real, self.Uyz[m,n,:,:].real, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real) \
                                      + interp2d(self.x[m,n,:,:].real,self.y[m,n,:,:].real, self.Uyz[m,n,:,:].imag, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real)*1j
                    
                    # V
                    this.V[m,n,:,:]   = interp2d(self.x[m,n,:,:].real,self.y[m,n,:,:].real,   self.V[m,n,:,:].real, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real) \
                                      + interp2d(self.x[m,n,:,:].real,self.y[m,n,:,:].real,   self.V[m,n,:,:].imag, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real)*1j
                    this.Vx[m,n,:,:]  = interp2d(self.x[m,n,:,:].real,self.y[m,n,:,:].real,  self.Vx[m,n,:,:].real, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real) \
                                      + interp2d(self.x[m,n,:,:].real,self.y[m,n,:,:].real,  self.Vx[m,n,:,:].imag, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real)*1j
                    this.Vxx[m,n,:,:] = interp2d(self.x[m,n,:,:].real,self.y[m,n,:,:].real, self.Vxx[m,n,:,:].real, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real) \
                                      + interp2d(self.x[m,n,:,:].real,self.y[m,n,:,:].real, self.Vxx[m,n,:,:].imag, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real)*1j
                    this.Vy[m,n,:,:]  = interp2d(self.x[m,n,:,:].real,self.y[m,n,:,:].real,  self.Vy[m,n,:,:].real, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real) \
                                      + interp2d(self.x[m,n,:,:].real,self.y[m,n,:,:].real,  self.Vy[m,n,:,:].imag, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real)*1j
                    this.Vyy[m,n,:,:] = interp2d(self.x[m,n,:,:].real,self.y[m,n,:,:].real, self.Vyy[m,n,:,:].real, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real) \
                                      + interp2d(self.x[m,n,:,:].real,self.y[m,n,:,:].real, self.Vyy[m,n,:,:].imag, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real)*1j
                    this.Vz[m,n,:,:]  = interp2d(self.x[m,n,:,:].real,self.y[m,n,:,:].real,  self.Vz[m,n,:,:].real, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real) \
                                      + interp2d(self.x[m,n,:,:].real,self.y[m,n,:,:].real,  self.Vz[m,n,:,:].imag, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real)*1j
                    this.Vzz[m,n,:,:] = interp2d(self.x[m,n,:,:].real,self.y[m,n,:,:].real, self.Vzz[m,n,:,:].real, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real) \
                                      + interp2d(self.x[m,n,:,:].real,self.y[m,n,:,:].real, self.Vzz[m,n,:,:].imag, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real)*1j
                    this.Vxy[m,n,:,:] = interp2d(self.x[m,n,:,:].real,self.y[m,n,:,:].real, self.Vxy[m,n,:,:].real, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real) \
                                      + interp2d(self.x[m,n,:,:].real,self.y[m,n,:,:].real, self.Vxy[m,n,:,:].imag, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real)*1j
                    this.Vxz[m,n,:,:] = interp2d(self.x[m,n,:,:].real,self.y[m,n,:,:].real, self.Vxz[m,n,:,:].real, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real) \
                                      + interp2d(self.x[m,n,:,:].real,self.y[m,n,:,:].real, self.Vxz[m,n,:,:].imag, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real)*1j
                    this.Vyz[m,n,:,:] = interp2d(self.x[m,n,:,:].real,self.y[m,n,:,:].real, self.Vyz[m,n,:,:].real, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real) \
                                      + interp2d(self.x[m,n,:,:].real,self.y[m,n,:,:].real, self.Vyz[m,n,:,:].imag, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real)*1j
                    
                    # W
                    this.W[m,n,:,:]   = interp2d(self.x[m,n,:,:].real,self.y[m,n,:,:].real,   self.W[m,n,:,:].real, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real) \
                                      + interp2d(self.x[m,n,:,:].real,self.y[m,n,:,:].real,   self.W[m,n,:,:].imag, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real)*1j
                    this.Wx[m,n,:,:]  = interp2d(self.x[m,n,:,:].real,self.y[m,n,:,:].real,  self.Wx[m,n,:,:].real, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real) \
                                      + interp2d(self.x[m,n,:,:].real,self.y[m,n,:,:].real,  self.Wx[m,n,:,:].imag, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real)*1j
                    this.Wxx[m,n,:,:] = interp2d(self.x[m,n,:,:].real,self.y[m,n,:,:].real, self.Wxx[m,n,:,:].real, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real) \
                                      + interp2d(self.x[m,n,:,:].real,self.y[m,n,:,:].real, self.Wxx[m,n,:,:].imag, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real)*1j
                    this.Wy[m,n,:,:]  = interp2d(self.x[m,n,:,:].real,self.y[m,n,:,:].real,  self.Wy[m,n,:,:].real, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real) \
                                      + interp2d(self.x[m,n,:,:].real,self.y[m,n,:,:].real,  self.Wy[m,n,:,:].imag, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real)*1j
                    this.Wyy[m,n,:,:] = interp2d(self.x[m,n,:,:].real,self.y[m,n,:,:].real, self.Wyy[m,n,:,:].real, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real) \
                                      + interp2d(self.x[m,n,:,:].real,self.y[m,n,:,:].real, self.Wyy[m,n,:,:].imag, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real)*1j
                    this.Wz[m,n,:,:]  = interp2d(self.x[m,n,:,:].real,self.y[m,n,:,:].real,  self.Wz[m,n,:,:].real, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real) \
                                      + interp2d(self.x[m,n,:,:].real,self.y[m,n,:,:].real,  self.Wz[m,n,:,:].imag, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real)*1j
                    this.Wzz[m,n,:,:] = interp2d(self.x[m,n,:,:].real,self.y[m,n,:,:].real, self.Wzz[m,n,:,:].real, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real) \
                                      + interp2d(self.x[m,n,:,:].real,self.y[m,n,:,:].real, self.Wzz[m,n,:,:].imag, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real)*1j
                    this.Wxy[m,n,:,:] = interp2d(self.x[m,n,:,:].real,self.y[m,n,:,:].real, self.Wxy[m,n,:,:].real, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real) \
                                      + interp2d(self.x[m,n,:,:].real,self.y[m,n,:,:].real, self.Wxy[m,n,:,:].imag, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real)*1j
                    this.Wxz[m,n,:,:] = interp2d(self.x[m,n,:,:].real,self.y[m,n,:,:].real, self.Wxz[m,n,:,:].real, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real) \
                                      + interp2d(self.x[m,n,:,:].real,self.y[m,n,:,:].real, self.Wxz[m,n,:,:].imag, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real)*1j
                    this.Wyz[m,n,:,:] = interp2d(self.x[m,n,:,:].real,self.y[m,n,:,:].real, self.Wyz[m,n,:,:].real, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real) \
                                      + interp2d(self.x[m,n,:,:].real,self.y[m,n,:,:].real, self.Wyz[m,n,:,:].imag, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real)*1j
                    
                    # P
                    this.P[m,n,:,:]   = interp2d(self.x[m,n,:,:].real,self.y[m,n,:,:].real,   self.P[m,n,:,:].real, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real) \
                                      + interp2d(self.x[m,n,:,:].real,self.y[m,n,:,:].real,   self.P[m,n,:,:].imag, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real)*1j
                    this.Px[m,n,:,:]  = interp2d(self.x[m,n,:,:].real,self.y[m,n,:,:].real,  self.Px[m,n,:,:].real, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real) \
                                      + interp2d(self.x[m,n,:,:].real,self.y[m,n,:,:].real,  self.Px[m,n,:,:].imag, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real)*1j
                    this.Pxx[m,n,:,:] = interp2d(self.x[m,n,:,:].real,self.y[m,n,:,:].real, self.Pxx[m,n,:,:].real, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real) \
                                      + interp2d(self.x[m,n,:,:].real,self.y[m,n,:,:].real, self.Pxx[m,n,:,:].imag, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real)*1j
                    this.Py[m,n,:,:]  = interp2d(self.x[m,n,:,:].real,self.y[m,n,:,:].real,  self.Py[m,n,:,:].real, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real) \
                                      + interp2d(self.x[m,n,:,:].real,self.y[m,n,:,:].real,  self.Py[m,n,:,:].imag, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real)*1j
                    this.Pyy[m,n,:,:] = interp2d(self.x[m,n,:,:].real,self.y[m,n,:,:].real, self.Pyy[m,n,:,:].real, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real) \
                                      + interp2d(self.x[m,n,:,:].real,self.y[m,n,:,:].real, self.Pyy[m,n,:,:].imag, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real)*1j
                    this.Pz[m,n,:,:]  = interp2d(self.x[m,n,:,:].real,self.y[m,n,:,:].real,  self.Pz[m,n,:,:].real, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real) \
                                      + interp2d(self.x[m,n,:,:].real,self.y[m,n,:,:].real,  self.Pz[m,n,:,:].imag, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real)*1j
                    this.Pzz[m,n,:,:] = interp2d(self.x[m,n,:,:].real,self.y[m,n,:,:].real, self.Pzz[m,n,:,:].real, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real) \
                                      + interp2d(self.x[m,n,:,:].real,self.y[m,n,:,:].real, self.Pzz[m,n,:,:].imag, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real)*1j
                    this.Pxy[m,n,:,:] = interp2d(self.x[m,n,:,:].real,self.y[m,n,:,:].real, self.Pxy[m,n,:,:].real, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real) \
                                      + interp2d(self.x[m,n,:,:].real,self.y[m,n,:,:].real, self.Pxy[m,n,:,:].imag, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real)*1j
                    this.Pxz[m,n,:,:] = interp2d(self.x[m,n,:,:].real,self.y[m,n,:,:].real, self.Pxz[m,n,:,:].real, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real) \
                                      + interp2d(self.x[m,n,:,:].real,self.y[m,n,:,:].real, self.Pxz[m,n,:,:].imag, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real)*1j
                    this.Pyz[m,n,:,:] = interp2d(self.x[m,n,:,:].real,self.y[m,n,:,:].real, self.Pyz[m,n,:,:].real, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real) \
                                      + interp2d(self.x[m,n,:,:].real,self.y[m,n,:,:].real, self.Pyz[m,n,:,:].imag, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real)*1j
                    
                    # T
                    this.T[m,n,:,:]   = interp2d(self.x[m,n,:,:].real,self.y[m,n,:,:].real,   self.T[m,n,:,:].real, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real) \
                                      + interp2d(self.x[m,n,:,:].real,self.y[m,n,:,:].real,   self.T[m,n,:,:].imag, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real)*1j
                    this.Tx[m,n,:,:]  = interp2d(self.x[m,n,:,:].real,self.y[m,n,:,:].real,  self.Tx[m,n,:,:].real, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real) \
                                      + interp2d(self.x[m,n,:,:].real,self.y[m,n,:,:].real,  self.Tx[m,n,:,:].imag, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real)*1j
                    this.Txx[m,n,:,:] = interp2d(self.x[m,n,:,:].real,self.y[m,n,:,:].real, self.Txx[m,n,:,:].real, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real) \
                                      + interp2d(self.x[m,n,:,:].real,self.y[m,n,:,:].real, self.Txx[m,n,:,:].imag, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real)*1j
                    this.Ty[m,n,:,:]  = interp2d(self.x[m,n,:,:].real,self.y[m,n,:,:].real,  self.Ty[m,n,:,:].real, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real) \
                                      + interp2d(self.x[m,n,:,:].real,self.y[m,n,:,:].real,  self.Ty[m,n,:,:].imag, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real)*1j
                    this.Tyy[m,n,:,:] = interp2d(self.x[m,n,:,:].real,self.y[m,n,:,:].real, self.Tyy[m,n,:,:].real, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real) \
                                      + interp2d(self.x[m,n,:,:].real,self.y[m,n,:,:].real, self.Tyy[m,n,:,:].imag, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real)*1j
                    this.Tz[m,n,:,:]  = interp2d(self.x[m,n,:,:].real,self.y[m,n,:,:].real,  self.Tz[m,n,:,:].real, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real) \
                                      + interp2d(self.x[m,n,:,:].real,self.y[m,n,:,:].real,  self.Tz[m,n,:,:].imag, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real)*1j
                    this.Tzz[m,n,:,:] = interp2d(self.x[m,n,:,:].real,self.y[m,n,:,:].real, self.Tzz[m,n,:,:].real, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real) \
                                      + interp2d(self.x[m,n,:,:].real,self.y[m,n,:,:].real, self.Tzz[m,n,:,:].imag, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real)*1j
                    this.Txy[m,n,:,:] = interp2d(self.x[m,n,:,:].real,self.y[m,n,:,:].real, self.Txy[m,n,:,:].real, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real) \
                                      + interp2d(self.x[m,n,:,:].real,self.y[m,n,:,:].real, self.Txy[m,n,:,:].imag, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real)*1j
                    this.Txz[m,n,:,:] = interp2d(self.x[m,n,:,:].real,self.y[m,n,:,:].real, self.Txz[m,n,:,:].real, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real) \
                                      + interp2d(self.x[m,n,:,:].real,self.y[m,n,:,:].real, self.Txz[m,n,:,:].imag, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real)*1j
                    this.Tyz[m,n,:,:] = interp2d(self.x[m,n,:,:].real,self.y[m,n,:,:].real, self.Tyz[m,n,:,:].real, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real) \
                                      + interp2d(self.x[m,n,:,:].real,self.y[m,n,:,:].real, self.Tyz[m,n,:,:].imag, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real)*1j
                    
                    # Conservative quantities
                    this.E[m,n,:,:]    = interp2d(self.x[m,n,:,:].real,self.y[m,n,:,:].real,    self.E[m,n,:,:].real, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real) \
                                       + interp2d(self.x[m,n,:,:].real,self.y[m,n,:,:].real,    self.E[m,n,:,:].imag, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real)*1j
                    this.rhoU[m,n,:,:] = interp2d(self.x[m,n,:,:].real,self.y[m,n,:,:].real, self.rhoU[m,n,:,:].real, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real) \
                                       + interp2d(self.x[m,n,:,:].real,self.y[m,n,:,:].real, self.rhoU[m,n,:,:].imag, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real)*1j
                    this.rhoV[m,n,:,:] = interp2d(self.x[m,n,:,:].real,self.y[m,n,:,:].real, self.rhoV[m,n,:,:].real, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real) \
                                       + interp2d(self.x[m,n,:,:].real,self.y[m,n,:,:].real, self.rhoV[m,n,:,:].imag, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real)*1j
                    this.rhoW[m,n,:,:] = interp2d(self.x[m,n,:,:].real,self.y[m,n,:,:].real, self.rhoW[m,n,:,:].real, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real) \
                                       + interp2d(self.x[m,n,:,:].real,self.y[m,n,:,:].real, self.rhoW[m,n,:,:].imag, kind=method)(this.x[m,n,:,:].real,this.y[m,n,:,:].real)*1j


    def extractMode(self,m,n):
        res = Field(self.Nx, self.Ny, M=0, N=0)
        # Field dimension
        res.x   = np.array(self.x[m,n,:,:].real) 
        res.xc  = np.array(self.xc[m,n,:,:].real)
        res.y   = np.array(self.y[m,n,:,:].real)
        res.yc  = np.array(self.yc[m,n,:,:].real)
        res.Rex = np.array(self.Rex[m,n,:,:].real)

        # wall angle
        res.theta = np.array(self.theta[m,n,:,:].real)
        
        # Field Properties
        res.Cp     = np.array(self.Cp[m,n,:,:].real)     
        res.mu     = np.array(self.mu[m,n,:,:].real)    
        res.muT    = np.array(self.muT[m,n,:,:].real)   
        res.muTT   = np.array(self.muTT[m,n,:,:].real)  
        res.mux    = np.array(self.mux[m,n,:,:].real)   
        res.muxx   = np.array(self.muxx[m,n,:,:].real)  
        res.muxy   = np.array(self.muxy[m,n,:,:].real)  
        res.muy    = np.array(self.muy[m,n,:,:].real)   
        res.muyy   = np.array(self.muyy[m,n,:,:].real)  
        res.lamb   = np.array(self.lamb[m,n,:,:].real)  
        res.lambx  = np.array(self.lambx[m,n,:,:].real) 
        res.lambxx = np.array(self.lambxx[m,n,:,:].real)
        res.lamby  = np.array(self.lamby[m,n,:,:].real) 
        res.lambyy = np.array(self.lambyy[m,n,:,:].real)
        res.lambxy = np.array(self.lambxy[m,n,:,:].real)
        res.lambT  = np.array(self.lambT[m,n,:,:].real) 
        res.lambTT = np.array(self.lambTT[m,n,:,:].real)
        res.rho    = np.array(self.rho[m,n,:,:].real)   
        res.rhox   = np.array(self.rhox[m,n,:,:].real)  
        res.rhoxx  = np.array(self.rhoxx[m,n,:,:].real) 
        res.rhoy   = np.array(self.rhoy[m,n,:,:].real) 
        res.rhoyy  = np.array(self.rhoyy[m,n,:,:].real) 
        res.rhoxy  = np.array(self.rhoxy[m,n,:,:].real) 

        # Disturbance kinetic energy (DKE)
        res.DKE  = self.DKE[m,n,:] 
        res.DKEx = self.DKEx[m,n,:]

        # Amplitude
        res.amplitude = self.amplitude[m,n,:] 
        
        # Wavenumbers
        res.omega = self.omega[m,n,:] 
        res.F     = self.F[m,n,:] 
        res.B     = self.B[m,n,:] 
        res.beta  = self.beta[m,n,:] 
        
        # Growth rate
        res.sigma   = self.sigma[m,n,:]
        res.nfactor = self.nfactor[m,n,:]
        res.Nfactor = self.Nfactor[n,:]
        
        # Growth rate parameter alpha
        res.alpha   = self.alpha[m,n,:] 
        res.alphax  = self.alphax[m,n,:]
        
        # U
        res.U   = np.array(self.U[m,n,:,:].real)   
        res.Ux  = np.array(self.Ux[m,n,:,:].real)  
        res.Uxx = np.array(self.Uxx[m,n,:,:].real) 
               
        res.Uy  = np.array(self.Uy[m,n,:,:].real)  
        res.Uyy = np.array(self.Uyy[m,n,:,:].real) 
               
        res.Uz  = np.array(self.Uz[m,n,:,:].real)  
        res.Uzz = np.array(self.Uzz[m,n,:,:].real) 
               
        res.Uxy = np.array(self.Uxy[m,n,:,:].real) 
        res.Uxz = np.array(self.Uxz[m,n,:,:].real) 
        res.Uyz = np.array(self.Uyz[m,n,:,:].real) 
               
        # V
        res.V   = np.array(self.V[m,n,:,:].real)   
        res.Vx  = np.array(self.Vx[m,n,:,:].real)  
        res.Vxx = np.array(self.Vxx[m,n,:,:].real) 
               
        res.Vy  = np.array(self.Vy[m,n,:,:].real)  
        res.Vyy = np.array(self.Vyy[m,n,:,:].real) 
               
        res.Vz  = np.array(self.Vz[m,n,:,:].real)  
        res.Vzz = np.array(self.Vzz[m,n,:,:].real) 
               
        res.Vxy = np.array(self.Vxy[m,n,:,:].real) 
        res.Vxz = np.array(self.Vxz[m,n,:,:].real) 
        res.Vyz = np.array(self.Vyz[m,n,:,:].real) 
               
        # W
        res.W   = np.array(self.W[m,n,:,:].real)   
        res.Wx  = np.array(self.Wx[m,n,:,:].real)  
        res.Wxx = np.array(self.Wxx[m,n,:,:].real) 
               
        res.Wy  = np.array(self.Wy[m,n,:,:].real)  
        res.Wyy = np.array(self.Wyy[m,n,:,:].real) 
               
        res.Wz  = np.array(self.Wz[m,n,:,:].real)  
        res.Wzz = np.array(self.Wzz[m,n,:,:].real) 
               
        res.Wxy = np.array(self.Wxy[m,n,:,:].real) 
        res.Wxz = np.array(self.Wxz[m,n,:,:].real) 
        res.Wyz = np.array(self.Wyz[m,n,:,:].real) 
               
        # P
        res.P   = np.array(self.P[m,n,:,:].real)   
        res.Px  = np.array(self.Px[m,n,:,:].real)  
        res.Pxx = np.array(self.Pxx[m,n,:,:].real) 
               
        res.Py  = np.array(self.Py[m,n,:,:].real)  
        res.Pyy = np.array(self.Pyy[m,n,:,:].real) 
               
        res.Pz  = np.array(self.Pz[m,n,:,:].real)  
        res.Pzz = np.array(self.Pzz[m,n,:,:].real) 
               
        res.Pxy = np.array(self.Pxy[m,n,:,:].real) 
        res.Pxz = np.array(self.Pxz[m,n,:,:].real) 
        res.Pyz = np.array(self.Pyz[m,n,:,:].real) 
                   
        # T        # T
        res.T   = np.array(self.T[m,n,:,:].real)   
        res.Tx  = np.array(self.Tx[m,n,:,:].real)  
        res.Txx = np.array(self.Txx[m,n,:,:].real) 
               
        res.T   = np.array(self.T[m,n,:,:].real)   
        res.Ty  = np.array(self.Ty[m,n,:,:].real)  
        res.Tyy = np.array(self.Tyy[m,n,:,:].real) 
               
        res.Tz  = np.array(self.Tz[m,n,:,:].real) 
        res.Tzz = np.array(self.Tzz[m,n,:,:].real) 
               
        res.Txy = np.array(self.Txy[m,n,:,:].real) 
        res.Txz = np.array(self.Txz[m,n,:,:].real) 
        res.Tyz = np.array(self.Tyz[m,n,:,:].real) 
        
        # Conservative quantities
        res.E    = np.array(self.E[m,n,:,:].real)
        res.rhoU = np.array(self.rhoU[m,n,:,:].real)
        res.rhoV = np.array(self.rhoV[m,n,:,:].real)
        res.rhoW = np.array(self.rhoW[m,n,:,:].real)
        return res

    def _gridToVTK(self, name, path, outputDeriv, Field):
        from pyevtk.hl import gridToVTK
        # Write a VTS file, can be imported in Paraview, VisiT, etc.
        # VTK only support 3D grids
        Nx = Field.Nx
        Ny = Field.Ny
        x = np.array(Field.xc.real.reshape((Nx,Ny,1),order='F'),dtype='float64')
        y = np.array(Field.yc.real.reshape((Nx,Ny,1),order='F'),dtype='float64')
        z = np.zeros_like(x)

        if not outputDeriv:
            # This step is needed to avoid type error further downstream in PyEVTK
            U = np.zeros_like(x) + Field.U.reshape((Nx,Ny,1),order='F')
            V = np.zeros_like(x) + Field.V.reshape((Nx,Ny,1),order='F')
            W = np.zeros_like(x) + Field.W.reshape((Nx,Ny,1),order='F')
            P = np.zeros_like(x) + Field.P.reshape((Nx,Ny,1),order='F')
            T = np.zeros_like(x) + Field.T.reshape((Nx,Ny,1),order='F')
            E = np.zeros_like(x) + Field.E.reshape((Nx,Ny,1),order='F')
            mu = np.zeros_like(x) + Field.mu.reshape((Nx,Ny,1),order='F')
            lamb = np.zeros_like(x)+ Field.lamb.reshape((Nx,Ny,1),order='F')
            rho = np.zeros_like(x) + Field.rho.reshape((Nx,Ny,1),order='F')
            rhoU = np.zeros_like(x)+ Field.rhoU.reshape((Nx,Ny,1),order='F')
            rhoV = np.zeros_like(x)+ Field.rhoV.reshape((Nx,Ny,1),order='F')
            rhoW = np.zeros_like(x)+ Field.rhoW.reshape((Nx,Ny,1),order='F')

            gridToVTK(path+name, x,y,z, pointData = {'U':U,      \
                                                     'V':V,      \
                                                     'W':W,      \
                                                     'P':P,      \
                                                     'T':T,      \
                                                     'E':E,      \
                                                     'mu':mu,    \
                                                     'lamb':lamb,\
                                                     'rho':rho,  \
                                                     'rhoU':rhoU,\
                                                     'rhoV':rhoV,\
                                                     'rhoW':rhoW,\
                                                     })
        else:
            # This step is needed to avoid type error further downstream in PyEVTK
            U = np.zeros_like(x) + Field.U.reshape((Nx,Ny,1),order='F')
            V = np.zeros_like(x) + Field.V.reshape((Nx,Ny,1),order='F')
            W = np.zeros_like(x) + Field.W.reshape((Nx,Ny,1),order='F')
            P = np.zeros_like(x) + Field.P.reshape((Nx,Ny,1),order='F')
            T = np.zeros_like(x) + Field.T.reshape((Nx,Ny,1),order='F')
            E = np.zeros_like(x) + Field.E.reshape((Nx,Ny,1),order='F')
            mu = np.zeros_like(x) + Field.mu.reshape((Nx,Ny,1),order='F')
            lamb = np.zeros_like(x) + Field.lamb.reshape((Nx,Ny,1),order='F')
            rho = np.zeros_like(x)  + Field.rho.reshape((Nx,Ny,1),order='F')
            rhoU = np.zeros_like(x) + Field.rhoU.reshape((Nx,Ny,1),order='F')
            rhoV = np.zeros_like(x) + Field.rhoV.reshape((Nx,Ny,1),order='F')
            rhoW = np.zeros_like(x) + Field.rhoW.reshape((Nx,Ny,1),order='F')

            Uy  = np.zeros_like(x) + Field.Uy.reshape((Nx,Ny,1),order='F')
            Uyy = np.zeros_like(x) + Field.Uyy.reshape((Nx,Ny,1),order='F')
            Uxy = np.zeros_like(x) + Field.Uxy.reshape((Nx,Ny,1),order='F')
            Uxx = np.zeros_like(x) + Field.Uxx.reshape((Nx,Ny,1),order='F')
            Ux  = np.zeros_like(x) + Field.Ux.reshape((Nx,Ny,1),order='F')
            Vy  = np.zeros_like(x) + Field.Vy.reshape((Nx,Ny,1),order='F')
            Vyy = np.zeros_like(x) + Field.Vyy.reshape((Nx,Ny,1),order='F')
            Vxy = np.zeros_like(x) + Field.Vxy.reshape((Nx,Ny,1),order='F')
            Vxx = np.zeros_like(x) + Field.Vxx.reshape((Nx,Ny,1),order='F')
            Vx  = np.zeros_like(x) + Field.Vx.reshape((Nx,Ny,1),order='F')
            Wy  = np.zeros_like(x) + Field.Wy.reshape((Nx,Ny,1),order='F')
            Wyy = np.zeros_like(x) + Field.Wyy.reshape((Nx,Ny,1),order='F')
            Wxy = np.zeros_like(x) + Field.Wxy.reshape((Nx,Ny,1),order='F')
            Wxx = np.zeros_like(x) + Field.Wxx.reshape((Nx,Ny,1),order='F')
            Wx  = np.zeros_like(x) + Field.Wx.reshape((Nx,Ny,1),order='F')
            Py  = np.zeros_like(x) + Field.Py.reshape((Nx,Ny,1),order='F')
            Px  = np.zeros_like(x) + Field.Px.reshape((Nx,Ny,1),order='F')
            Ty  = np.zeros_like(x) + Field.Ty.reshape((Nx,Ny,1),order='F')
            Tyy = np.zeros_like(x) + Field.Tyy.reshape((Nx,Ny,1),order='F')
            Txy = np.zeros_like(x) + Field.Txy.reshape((Nx,Ny,1),order='F')
            Txx = np.zeros_like(x) + Field.Txx.reshape((Nx,Ny,1),order='F')
            Tx  = np.zeros_like(x) + Field.Tx.reshape((Nx,Ny,1),order='F')

            # Vorticity
            Vortz = Vx - Uy

            gridToVTK(path+name, x,y,z, pointData = {'U':U,      \
                                                     'V':V,      \
                                                     'W':W,      \
                                                     'P':P,      \
                                                     'T':T,      \
                                                     'E':E,      \
                                                     'mu':mu,    \
                                                     'lamb':lamb,\
                                                     'rho':rho,  \
                                                     'rhoU':rhoU,\
                                                     'rhoV':rhoV,\
                                                     'rhoW':rhoW,\
                                                     'Uy':Uy,    \
                                                     'Uyy':Uyy,  \
                                                     'Uxy':Uxy,  \
                                                     'Uxx':Uxx,  \
                                                     'Ux' :Ux,   \
                                                     'Vy':Vy,    \
                                                     'Vyy':Vyy,  \
                                                     'Vxy':Vxy,  \
                                                     'Vxx':Vxx,  \
                                                     'Vx':Vx,    \
                                                     'Wy':Wy,    \
                                                     'Wyy':Wyy,  \
                                                     'Wxy':Wxy,  \
                                                     'Wxx':Wxx,  \
                                                     'Wx':Wx,    \
                                                     'Py':Py,    \
                                                     'Px':Px,    \
                                                     'Ty':Ty,    \
                                                     'Tyy':Tyy,  \
                                                     'Txy':Txy,  \
                                                     'Txx':Txx,  \
                                                     'Tx':Tx,    \
                                                     'Vortz':Vortz\
                                                     })

    def writeVTK(self, path, name='Flow', outputDeriv=False, mode=(0,1)):
        import os
        if not os.path.exists(path):
            os.makedirs(path)
        # Which mode to output
        (m,n) = mode
        
        if len(self.U.shape) == 2:
            self._gridToVTK(name, path, outputDeriv, self)
        else:
            this = self.extractMode(m,n)
            self._gridToVTK(name, path, outputDeriv, this)

    def dump(self, path,minimal=False):
        import os
        import hickle
        if not os.path.exists(path):
            os.makedirs(path)
        hickle.dump(self.Ny,path+'/Ny.bin')
        hickle.dump(self.Nx,path+'/Nx.bin')
        hickle.dump(self.Nx,path+'/M.bin')
        hickle.dump(self.Nx,path+'/N.bin')
        if minimal:
            # coordinates
            hickle.dump(self.y ,path+ '/y.bin')
            hickle.dump(self.yc,path+'/yc.bin')
            hickle.dump(self.x ,path+ '/x.bin')
            hickle.dump(self.xc,path+'/xc.bin')

            # Solution
            hickle.dump(self.U,path+'/U.bin')
            hickle.dump(self.V,path+'/V.bin')
            hickle.dump(self.W,path+'/W.bin')
            hickle.dump(self.P,path+'/P.bin')
            hickle.dump(self.T,path+'/T.bin')

        else:
            # coordinates
            hickle.dump(self.y ,path+ '/y.bin')
            hickle.dump(self.yc,path+'/yc.bin')
            hickle.dump(self.x ,path+ '/x.bin')
            hickle.dump(self.xc,path+'/xc.bin')
            
            # Geometric angle
            hickle.dump(self.theta,path+'/theta.bin')
            
            # Local Reynolds number
            hickle.dump(self.Rex,path+'/Rex.bin')
            
            # Non-dim Y direction
            hickle.dump(self.eta,path+'/eta.bin')
            
            # Field Properties
            hickle.dump(self.Cp,path+'/Cp.bin')    
            hickle.dump(self.mu,path+'/mu.bin')    
            hickle.dump(self.muT,path+'/muT.bin')   
            hickle.dump(self.muTT,path+'/muTT.bin')  
            hickle.dump(self.mux,path+'/mux.bin')   
            hickle.dump(self.muxx,path+'/muxx.bin')  
            hickle.dump(self.muxy,path+'/muxy.bin')  
            hickle.dump(self.muy,path+'/muy.bin')   
            hickle.dump(self.muyy,path+'/muyy.bin')  
            hickle.dump(self.lamb,path+'/lamb.bin')  
            hickle.dump(self.lambx,path+'/lambx.bin') 
            hickle.dump(self.lambxx,path+'/lambxx.bin')
            hickle.dump(self.lamby,path+'/lamby.bin') 
            hickle.dump(self.lambyy,path+'/lambyy.bin')
            hickle.dump(self.lambxy,path+'/lambxy.bin')
            hickle.dump(self.lambT,path+'/lambT.bin') 
            hickle.dump(self.lambTT,path+'/lambTT.bin')
            hickle.dump(self.rho,path+'/rho.bin')   
            hickle.dump(self.rhox,path+'/rhox.bin')  
            hickle.dump(self.rhoxx,path+'/rhoxx.bin') 
            hickle.dump(self.rhoy,path+'/rhoy.bin')  
            hickle.dump(self.rhoyy,path+'/rhoyy.bin') 
            hickle.dump(self.rhoxy,path+'/rhoxy.bin') 
            
            # Dimensionless quantities
            hickle.dump(self.delta,path+'/delta.bin')
            
            # Disturbance kinetic energy (DKE)
            hickle.dump(self.DKE,path+'/DKE.bin')
            hickle.dump(self.DKEx,path+'/DKEx.bin')

            # amplitude
            hickle.dump(self.amplitude,path+'/amplitude.bin')
            
            # Growth rate & N-factor
            hickle.dump(self.sigma,path+'/sigma.bin')
            hickle.dump(self.nfactor,path+'/nfactor.bin')
            hickle.dump(self.Nfactor,path+'/Nfactor.bin')
            
            # Growth rate parameter alpha
            hickle.dump(self.omega,path+'/omega.bin') 
            hickle.dump(self.F,path+'/F.bin') 
            hickle.dump(self.beta,path+'/beta.bin') 
            hickle.dump(self.B,path+'/B.bin') 
            hickle.dump(self.alpha,path+'/alpha.bin') 
            hickle.dump(self.alphax,path+'/alphax.bin')
            
            # U
            hickle.dump(self.U,path+'/U.bin')   
            hickle.dump(self.Ux,path+'/Ux.bin')  
            hickle.dump(self.Uxx,path+'/Uxx.bin') 
            
            hickle.dump(self.Uy,path+'/Uy.bin')  
            hickle.dump(self.Uyy,path+'/Uyy.bin') 
            
            hickle.dump(self.Uz,path+'/Uz.bin')  
            hickle.dump(self.Uzz,path+'/Uzz.bin') 
            
            hickle.dump(self.Uxy,path+'/Uxy.bin') 
            hickle.dump(self.Uxz,path+'/Uxz.bin') 
            hickle.dump(self.Uyz,path+'/Uyz.bin') 
            
            # V
            hickle.dump(self.V,path+'/V.bin')   
            hickle.dump(self.Vx,path+'/Vx.bin')  
            hickle.dump(self.Vxx,path+'/Vxx.bin') 
            
            hickle.dump(self.Vy,path+'/Vy.bin')  
            hickle.dump(self.Vyy,path+'/Vyy.bin') 
            
            hickle.dump(self.Vz,path+'/Vz.bin')  
            hickle.dump(self.Vzz,path+'/Vzz.bin') 
            
            hickle.dump(self.Vxy,path+'/Vxy.bin') 
            hickle.dump(self.Vxz,path+'/Vxz.bin') 
            hickle.dump(self.Vyz,path+'/Vyz.bin') 
            
            # W
            hickle.dump(self.W,path+'/W.bin')   
            hickle.dump(self.Wx,path+'/Wx.bin')  
            hickle.dump(self.Wxx,path+'/Wxx.bin') 
            
            hickle.dump(self.Wy,path+'/Wy.bin')  
            hickle.dump(self.Wyy,path+'/Wyy.bin')
            
            hickle.dump(self.Wz,path+'/Wz.bin')  
            hickle.dump(self.Wzz,path+'/Wzz.bin') 
            
            hickle.dump(self.Wxy,path+'/Wxy.bin') 
            hickle.dump(self.Wxz,path+'/Wxz.bin') 
            hickle.dump(self.Wyz,path+'/Wyz.bin') 
            
            # P
            hickle.dump(self.P,path+'/P.bin')   
            hickle.dump(self.Px,path+'/Px.bin')  
            hickle.dump(self.Pxx,path+'/Pxx.bin') 
            
            hickle.dump(self.Py,path+'/Py.bin')  
            hickle.dump(self.Pyy,path+'/Pyy.bin') 
            
            hickle.dump(self.Pz,path+'/Pz.bin')  
            hickle.dump(self.Pzz,path+'/Pzz.bin') 
            
            hickle.dump(self.Pxy,path+'/Pxy.bin') 
            hickle.dump(self.Pxz,path+'/Pxz.bin') 
            hickle.dump(self.Pyz,path+'/Pyz.bin') 
            
            # T
            hickle.dump(self.T,path+'/T.bin')   
            hickle.dump(self.Tx,path+'/Tx.bin')  
            hickle.dump(self.Txx,path+'/Txx.bin') 
            
            hickle.dump(self.T,path+'/T.bin')   
            hickle.dump(self.Ty,path+'/Ty.bin')  
            hickle.dump(self.Tyy,path+'/Tyy.bin') 
            
            hickle.dump(self.Tz,path+'/Tz.bin')  
            hickle.dump(self.Tzz,path+'/Tzz.bin') 
            
            hickle.dump(self.Txy,path+'/Txy.bin') 
            hickle.dump(self.Txz,path+'/Txz.bin') 
            hickle.dump(self.Tyz,path+'/Tyz.bin') 
            
            # Conservative quantities
            hickle.dump(self.E,path+'/E.bin')   
            hickle.dump(self.rhoU,path+'/rhoU.bin')
            hickle.dump(self.rhoV,path+'/rhoV.bin')
            hickle.dump(self.rhoW,path+'/rhoW.bin')

    def load(self,path,var='default', minimal=False):
        import os
        import hickle
        
        if var != 'default':
            return hickle.load(path+'/'+var+'.bin')
        else:
            if minimal:
                # coordinates
                self.x   = hickle.load(path+'/x.bin')
                self.xc  = hickle.load(path+'/xc.bin')
                self.y   = hickle.load(path+'/y.bin')
                self.yc  = hickle.load(path+'/yc.bin')
                
                self.U  = hickle.load(path+'/U.bin')
                self.V  = hickle.load(path+'/V.bin')
                self.W  = hickle.load(path+'/W.bin')
                self.P  = hickle.load(path+'/P.bin')
                self.T  = hickle.load(path+'/T.bin')
            
            else:
                # Field dimension
                self.Ny  = hickle.load(path+'/Ny.bin')
                self.Nx  = hickle.load(path+'/Nx.bin')
                self.M   = hickle.load(path+'/M.bin')
                self.N   = hickle.load(path+'/N.bin')
                self.x   = hickle.load(path+'/x.bin')
                self.xc  = hickle.load(path+'/xc.bin')
                self.y   = hickle.load(path+'/y.bin')
                self.yc  = hickle.load(path+'/yc.bin')
                self.Rex = hickle.load(path+'/Rex.bin')
            
                # Geometric jacobian
                self.theta =hickle.load(path+'/theta.bin')
                
                # Non-dim Y direction
                self.eta = hickle.load(path+'/eta.bin')
                
                # Field Properties
                self.Cp     = hickle.load(path+'/Cp.bin')
                self.mu     = hickle.load(path+'/mu.bin')
                self.muT    = hickle.load(path+'/muT.bin')
                self.muTT   = hickle.load(path+'/muTT.bin')
                self.mux    = hickle.load(path+'/mux.bin')
                self.muxx   = hickle.load(path+'/muxx.bin')
                self.muxy   = hickle.load(path+'/muxy.bin')
                self.muy    = hickle.load(path+'/muy.bin')
                self.muyy   = hickle.load(path+'/muyy.bin')
                self.lamb   = hickle.load(path+'/lamb.bin')
                self.lambx  = hickle.load(path+'/lambx.bin')
                self.lambxx = hickle.load(path+'/lambxx.bin')
                self.lamby  = hickle.load(path+'/lamby.bin')
                self.lambyy = hickle.load(path+'/lambyy.bin')
                self.lambxy = hickle.load(path+'/lambxy.bin')
                self.lambT  = hickle.load(path+'/lambT.bin')
                self.lambTT = hickle.load(path+'/lambTT.bin')
                self.rho    = hickle.load(path+'/rho.bin')
                self.rhox   = hickle.load(path+'/rhox.bin')
                self.rhoxx  = hickle.load(path+'/rhoxx.bin')
                self.rhoy   = hickle.load(path+'/rhoy.bin')
                self.rhoyy  = hickle.load(path+'/rhoyy.bin')
                self.rhoxy  = hickle.load(path+'/rhoxy.bin')
                
                # Dimensionless quantities
                self.delta  = hickle.load(path+'/delta.bin')
                
                # Disturbance kinetic energy (DKE)
                self.DKE  = hickle.load(path+'/DKE.bin')
                self.DKEx = hickle.load(path+'/DKEx.bin')
                
                # Amplitude
                try:
                    self.amplitude = hickle.load(path+'/amplitude.bin')
                except:
                    self.amplitude = np.zeros_like(self.DKEx)
                    pass
                
                # Wavenumbers
                self.omega = hickle.load(path+'/omega.bin')
                self.beta  = hickle.load(path+'/beta.bin')
                
                # Growth rate
                self.sigma   = hickle.load(path+'/sigma.bin')
                self.F       = hickle.load(path+'/F.bin')
                self.B       = hickle.load(path+'/B.bin')
                self.nfactor = hickle.load(path+'/nfactor.bin')
                self.Nfactor = hickle.load(path+'/Nfactor.bin')
                
                # Growth rate parameter alpha
                self.alpha   = hickle.load(path+'/alpha.bin')
                self.alphax  = hickle.load(path+'/alphax.bin')
                
                # U
                self.U   = hickle.load(path+'/U.bin')
                self.Ux  = hickle.load(path+'/Ux.bin')
                self.Uxx = hickle.load(path+'/Uxx.bin')
                
                self.Uy  = hickle.load(path+'/Uy.bin')
                self.Uyy = hickle.load(path+'/Uyy.bin')
                
                self.Uz  = hickle.load(path+'/Uz.bin')
                self.Uzz = hickle.load(path+'/Uzz.bin')
                
                self.Uxy = hickle.load(path+'/Uxy.bin')
                self.Uxz = hickle.load(path+'/Uxz.bin')
                self.Uyz = hickle.load(path+'/Uyz.bin')
                
                # V
                self.V   = hickle.load(path+'/V.bin')
                self.Vx  = hickle.load(path+'/Vx.bin')
                self.Vxx = hickle.load(path+'/Vxx.bin')
                
                self.Vy  = hickle.load(path+'/Vy.bin')
                self.Vyy = hickle.load(path+'/Vyy.bin')
                
                self.Vz  = hickle.load(path+'/Vz.bin')
                self.Vzz = hickle.load(path+'/Vzz.bin')
                
                self.Vxy = hickle.load(path+'/Vxy.bin')
                self.Vxz = hickle.load(path+'/Vxz.bin')
                self.Vyz = hickle.load(path+'/Vyz.bin')
                
                # W
                self.W   = hickle.load(path+'/W.bin')
                self.Wx  = hickle.load(path+'/Wx.bin')
                self.Wxx = hickle.load(path+'/Wxx.bin')
                
                self.Wy  = hickle.load(path+'/Wy.bin')
                self.Wyy = hickle.load(path+'/Wyy.bin')
                
                self.Wz  = hickle.load(path+'/Wz.bin')
                self.Wzz = hickle.load(path+'/Wzz.bin')
                
                self.Wxy = hickle.load(path+'/Wxy.bin')
                self.Wxz = hickle.load(path+'/Wxz.bin')
                self.Wyz = hickle.load(path+'/Wyz.bin')
                
                # P
                self.P   = hickle.load(path+'/P.bin')
                self.Px  = hickle.load(path+'/Px.bin')
                self.Pxx = hickle.load(path+'/Pxx.bin')
                
                self.Py  = hickle.load(path+'/Py.bin')
                self.Pyy = hickle.load(path+'/Pyy.bin')
                
                self.Pz  = hickle.load(path+'/Pz.bin')
                self.Pzz = hickle.load(path+'/Pzz.bin')
                
                self.Pxy = hickle.load(path+'/Pxy.bin')
                self.Pxz = hickle.load(path+'/Pxz.bin')
                self.Pyz = hickle.load(path+'/Pyz.bin')
                
                # T
                self.T   = hickle.load(path+'/T.bin')
                self.Tx  = hickle.load(path+'/Tx.bin')
                self.Txx = hickle.load(path+'/Txx.bin')
                
                self.T   = hickle.load(path+'/T.bin')
                self.Ty  = hickle.load(path+'/Ty.bin')
                self.Tyy = hickle.load(path+'/Tyy.bin')
                
                self.Tz  = hickle.load(path+'/Tz.bin')
                self.Tzz = hickle.load(path+'/Tzz.bin')
                
                self.Txy = hickle.load(path+'/Txy.bin')
                self.Txz = hickle.load(path+'/Txz.bin')
                self.Tyz = hickle.load(path+'/Tyz.bin')
                
                # Conservative quantities
                self.E    = hickle.load(path+'/E.bin')
                self.rhoU = hickle.load(path+'/rhoU.bin')
                self.rhoV = hickle.load(path+'/rhoV.bin')
                self.rhoW = hickle.load(path+'/rhoW.bin')

    def __add__(self, other):
        try:
            Err = not (self.Nx == other.Nx and self.Ny == other.Ny and self.N == other.N and self.M == other.M)
            if Err:
                raise ValueError('dimension mismatch')
        except:
            pass

        res = Field(self.Nx, self.Ny, M=self.M, N=self.N)

        res.x   = self.y
        res.y   = self.x

        # Non-dim Y direction
        res.eta = self.eta

        # Field Properties
        res.Cp     = self.Cp     + other.Cp
        res.mu     = self.mu     + other.Cp
        res.muT    = self.muT    + other.muT
        res.muTT   = self.muTT   + other.muTT
        res.mux    = self.mux    + other.mux
        res.muxx   = self.muxx   + other.muxx
        res.muxy   = self.muxy   + other.muxy
        res.muy    = self.muy    + other.muy 
        res.muyy   = self.muyy   + other.muyy 
        res.lamb   = self.lamb   + other.lamb
        res.lambx  = self.lambx  + other.lambx
        res.lambxx = self.lambxx + other.lambxx
        res.lamby  = self.lamby  + other.lamby
        res.lambyy = self.lambyy + other.lambyy 
        res.lambxy = self.lambxy + other.lambxy
        res.lambT  = self.lambT  + other.lambT
        res.lambTT = self.lambTT + other.lambTT 
        res.rho    = self.rho    + other.rho
        res.rhox   = self.rhox   + other.rhox
        res.rhoxx  = self.rhoxx  + other.rhoxx 
        res.rhoy   = self.rhoy   + other.rhoy
        res.rhoyy  = self.rhoyy  + other.rhoyy
        res.rhoxy  = self.rhoxy  + other.rhoxy
        
        # Dimensionless quantities
        res.delta = self.delta + other.delta
        
        # Disturbance kinetic energy (DKE)
        res.DKE  = self.DKE + other.DKE
        res.DKEx = self.DKEx + other.DKEx

        # Amplitude
        res.amplitude = self.amplitude + other.amplitude

        # Wavenumbers
        res.omega = self.omega + other.omega
        res.beta  = self.bata + other.beta
        
        # Growth rate
        res.sigma   = self.sigma   + other.sigma
        res.nfactor = self.nfactor + other.nfactor
        res.Nfactor = self.Nfactor + other.Nfactor
        
        # Growth rate parameter alpha
        res.alpha   = self.alpha + other.alpha
        res.alphax  = self.alphax + other.alphax
        
        # U
        res.U   = self.U   + other.U
        res.Ux  = self.Ux  + other.Ux
        res.Uxx = self.Uxx + other.Uxx
        
        res.Uy  = self.Uy  + other.Uy
        res.Uyy = self.Uyy + other.Uyy
        
        res.Uz  = self.Uz  + other.Uz
        res.Uzz = self.Uzz + other.Uzz
        
        res.Uxy = self.Uxy + other.Uxy
        res.Uxz = self.Uxz + other.Uxz
        res.Uyz = self.Uyz + other.Uyz
        
        # V
        res.V   = self.V   + other.V
        res.Vx  = self.Vx  + other.Vx
        res.Vxx = self.Vxx + other.Vxx
        
        res.Vy  = self.Vy  + other.Vy
        res.Vyy = self.Vyy + other.Vyy
        
        res.Vz  = self.Vz  + other.Vz
        res.Vzz = self.Vzz + other.Vzz
        
        res.Vxy = self.Vxy + other.Vxy
        res.Vxz = self.Vxz + other.Vxz
        res.Vyz = self.Vyz + other.Vyz
        
        # W
        res.W   = self.W   + other.W
        res.Wx  = self.Wx  + other.Wx
        res.Wxx = self.Wxx + other.Wxx
        
        res.Wy  = self.Wy  + other.Wy
        res.Wyy = self.Wyy + other.Wyy
        
        res.Wz  = self.Wz  + other.Wz
        res.Wzz = self.Wzz + other.Wzz
        
        res.Wxy = self.Wxy + other.Wxy
        res.Wxz = self.Wxz + other.Wxz
        res.Wyz = self.Wyz + other.Wyz
        
        # P
        res.P   = self.P   + other.P  
        res.Px  = self.Px  + other.Px 
        res.Pxx = self.Pxx + other.Pxx

        res.Py  = self.Py  + other.Py
        res.Pyy = self.Pyy + other.Pyy
        
        res.Pz  = self.Pz  + other.Pz
        res.Pzz = self.Pzz + other.Pzz
        
        res.Pxy = self.Pxy + other.Pxy
        res.Pxz = self.Pxz + other.Pxz
        res.Pyz = self.Pyz + other.Pyz
        
        # T
        res.T   = self.T   + other.T  
        res.Tx  = self.Tx  + other.Tx 
        res.Txx = self.Txx + other.Txx
        
        res.T   = self.T   + other.T  
        res.Ty  = self.Ty  + other.Ty 
        res.Tyy = self.Tyy + other.Tyy
        
        res.Tz  = self.Tz  + other.Tz 
        res.Tzz = self.Tzz + other.Tzz
        
        res.Txy = self.Txy + other.Txy
        res.Txz = self.Txz + other.Txz
        res.Tyz = self.Tyz + other.Tyz

        # conservative quantities
        res.E    = self.E    + other.E
        res.rhoU = self.rhoU + other.rhoU
        res.rhoV = self.rhoV + other.rhoV
        res.rhoW = self.rhoW + other.rhoW

        return res

        
    def __sub__(self, other):
        try:
            Err = not (self.Nx == other.Nx and self.Ny == other.Ny and self.N == other.N  and self.M == other.M)
            if Err:
                raise ValueError('dimension mismatch')
        except:
            pass

        res = Field(self.Nx, self.Ny, M=self.M, N=self.N)

        res.x   = self.y
        res.y   = self.x

        # Non-dim Y direction
        res.eta = self.eta

        # Field Properties
        res.Cp     = self.Cp     - other.Cp
        res.mu     = self.mu     - other.Cp
        res.muT    = self.muT    - other.muT
        res.muTT   = self.muTT   - other.muTT
        res.mux    = self.mux    - other.mux
        res.muxx   = self.muxx   - other.muxx
        res.muxy   = self.muxy   - other.muxy
        res.muy    = self.muy    - other.muy 
        res.muyy   = self.muyy   - other.muyy 
        res.lamb   = self.lamb   - other.lamb
        res.lambx  = self.lambx  - other.lambx
        res.lambxx = self.lambxx - other.lambxx
        res.lamby  = self.lamby  - other.lamby
        res.lambyy = self.lambyy - other.lambyy 
        res.lambxy = self.lambxy - other.lambxy
        res.lambT  = self.lambT  - other.lambT
        res.lambTT = self.lambTT - other.lambTT 
        res.rho    = self.rho    - other.rho
        res.rhox   = self.rhox   - other.rhox
        res.rhoxx  = self.rhoxx  - other.rhoxx 
        res.rhoy   = self.rhoy   - other.rhoy
        res.rhoyy  = self.rhoyy  - other.rhoyy
        res.rhoxy  = self.rhoxy  - other.rhoxy
        
        # Dimensionless quantities
        res.delta = self.delta - other.delta
        
        # Disturbance kinetic energy (DKE)
        res.DKE  = self.DKE - other.DKE
        res.DKEx = self.DKEx - other.DKEx

        # Amplitude
        res.amplitude = self.amplitude - other.amplitude

        # Wavenumbers
        res.omega = self.omega - other.omega
        res.beta  = self.bata - other.beta
        
        # Growth rate
        res.sigma   = self.sigma - other.sigma
        res.nfactor = self.nfactor - other.nfactor
        res.Nfactor = self.Nfactor - other.Nfactor
        
        # Growth rate parameter alpha
        res.alpha   = self.alpha - other.alpha
        res.alphax  = self.alphax - other.alphax
        
        # U
        res.U   = self.U   - other.U
        res.Ux  = self.Ux  - other.Ux
        res.Uxx = self.Uxx - other.Uxx
        
        res.Uy  = self.Uy  - other.Uy
        res.Uyy = self.Uyy - other.Uyy
        
        res.Uz  = self.Uz  - other.Uz
        res.Uzz = self.Uzz - other.Uzz
        
        res.Uxy = self.Uxy - other.Uxy
        res.Uxz = self.Uxz - other.Uxz
        res.Uyz = self.Uyz - other.Uyz
        
        # V
        res.V   = self.V   - other.V
        res.Vx  = self.Vx  - other.Vx
        res.Vxx = self.Vxx - other.Vxx
        
        res.Vy  = self.Vy  - other.Vy
        res.Vyy = self.Vyy - other.Vyy
        
        res.Vz  = self.Vz  - other.Vz
        res.Vzz = self.Vzz - other.Vzz
        
        res.Vxy = self.Vxy - other.Vxy
        res.Vxz = self.Vxz - other.Vxz
        res.Vyz = self.Vyz - other.Vyz
        
        # W
        res.W   = self.W   - other.W
        res.Wx  = self.Wx  - other.Wx
        res.Wxx = self.Wxx - other.Wxx
        
        res.Wy  = self.Wy  - other.Wy
        res.Wyy = self.Wyy - other.Wyy
        
        res.Wz  = self.Wz  - other.Wz
        res.Wzz = self.Wzz - other.Wzz
        
        res.Wxy = self.Wxy - other.Wxy
        res.Wxz = self.Wxz - other.Wxz
        res.Wyz = self.Wyz - other.Wyz
        
        # P
        res.P   = self.P   - other.P  
        res.Px  = self.Px  - other.Px 
        res.Pxx = self.Pxx - other.Pxx

        res.Py  = self.Py  - other.Py
        res.Pyy = self.Pyy - other.Pyy
        
        res.Pz  = self.Pz  - other.Pz
        res.Pzz = self.Pzz - other.Pzz
        
        res.Pxy = self.Pxy - other.Pxy
        res.Pxz = self.Pxz - other.Pxz
        res.Pyz = self.Pyz - other.Pyz
        
        # T
        res.T   = self.T   - other.T  
        res.Tx  = self.Tx  - other.Tx 
        res.Txx = self.Txx - other.Txx
        
        res.T   = self.T   - other.T  
        res.Ty  = self.Ty  - other.Ty 
        res.Tyy = self.Tyy - other.Tyy
        
        res.Tz  = self.Tz  - other.Tz 
        res.Tzz = self.Tzz - other.Tzz
        
        res.Txy = self.Txy - other.Txy
        res.Txz = self.Txz - other.Txz
        res.Tyz = self.Tyz - other.Tyz

        # conservative quantities
        res.E    = self.E    - other.E
        res.rhoU = self.rhoU - other.rhoU
        res.rhoV = self.rhoV - other.rhoV
        res.rhoW = self.rhoW - other.rhoW

        return res

    def __complex__(self):
        res = Field(self.Nx, self.Ny, M=self.M, N=self.N, typ=complex)

        res.x   = complex(self.y)
        res.y   = complex(self.x)

        # Non-dim Y direction
        res.eta = complex(self.eta)

        # Field Properties
        res.Cp     = complex(self.Cp)
        res.mu     = complex(self.mu)
        res.muT    = complex(self.muT)
        res.muTT   = complex(self.muTT)
        res.mux    = complex(self.mux)
        res.muxx   = complex(self.muxx)
        res.muxy   = complex(self.muxy)
        res.muy    = complex(self.muy)
        res.muyy   = complex(self.muyy)
        res.lamb   = complex(self.lamb)
        res.lambx  = complex(self.lambx)
        res.lambxx = complex(self.lambxx)
        res.lamby  = complex(self.lamby)
        res.lambyy = complex(self.lambyy)
        res.lambxy = complex(self.lambxy)
        res.lambT  = complex(self.lambT)
        res.lambTT = complex(self.lambTT)
        res.rho    = complex(self.rho) 
        res.rhox   = complex(self.rhox)
        res.rhoxx  = complex(self.rhoxx) 
        res.rhoy   = complex(self.rhoy)
        res.rhoyy  = complex(self.rhoyy)
        res.rhoxy  = complex(self.rhoxy)
        
        # Dimensionless quantities
        res.delta = complex(self.delta)
        
        # Disturbance kinetic energy (DKE)
        res.DKE  = complex(self.DKE)
        res.DKEx = complex(self.DKEx)

        # Amplitude
        res.amplitude = complex(self.amplitude)

        # Wavenumbers
        res.omega = complex(self.omega)
        res.beta  = complex(self.bata)
        
        # Growth rath
        res.sigma   = complex(self.sigma)
        res.nfactor = complex(self.nfactor)
        res.Nfactor = complex(self.Nfactor)
        
        # Growth rate parameter alpha
        res.alpha   = complex(self.alpha)
        res.alphax  = complex(self.alphax)
        
        # U
        res.U   = complex(self.U)
        res.Ux  = complex(self.Ux)
        res.Uxx = complex(self.Uxx)
        
        res.Uy  = complex(self.Uy)
        res.Uyy = complex(self.Uyy)
        
        res.Uz  = complex(self.Uz)
        res.Uzz = complex(self.Uzz)
        
        res.Uxy = complex(self.Uxy)
        res.Uxz = complex(self.Uxz)
        res.Uyz = complex(self.Uyz)
        
        # V
        res.V   = complex(self.V)
        res.Vx  = complex(self.Vx)
        res.Vxx = complex(self.Vxx)
        
        res.Vy  = complex(self.Vy)
        res.Vyy = complex(self.Vyy)
        
        res.Vz  = complex(self.Vz)
        res.Vzz = complex(self.Vzz)
        
        res.Vxy = complex(self.Vxy)
        res.Vxz = complex(self.Vxz)
        res.Vyz = complex(self.Vyz)
        
        # W
        res.W   = complex(self.W)
        res.Wx  = complex(self.Wx)
        res.Wxx = complex(self.Wxx)
        
        res.Wy  = complex(self.Wy)
        res.Wyy = complex(self.Wyy)
        
        res.Wz  = complex(self.Wz)
        res.Wzz = complex(self.Wzz)
        
        res.Wxy = complex(self.Wxy)
        res.Wxz = complex(self.Wxz)
        res.Wyz = complex(self.Wyz)
        
        # P
        res.P   = complex(self.P)
        res.Px  = complex(self.Px)
        res.Pxx = complex(self.Pxx)

        res.Py  = complex(self.Py)
        res.Pyy = complex(self.Pyy)
        
        res.Pz  = complex(self.Pz)
        res.Pzz = complex(self.Pzz)
        
        res.Pxy = complex(self.Pxy)
        res.Pxz = complex(self.Pxz)
        res.Pyz = complex(self.Pyz)
        
        # T
        res.T   = complex(self.T)
        res.Tx  = complex(self.Tx)
        res.Txx = complex(self.Txx)
        
        res.T   = complex(self.T)
        res.Ty  = complex(self.Ty)
        res.Tyy = complex(self.Tyy)
        
        res.Tz  = complex(self.Tz)
        res.Tzz = complex(self.Tzz)
        
        res.Txy = complex(self.Txy)
        res.Txz = complex(self.Txz)
        res.Tyz = complex(self.Tyz)

        # conservative quantities
        res.E    = complex(self.E)
        res.rhoU = complex(self.rhoU)
        res.rhoV = complex(self.rhoV)
        res.rhoW = complex(self.rhoW)

        return res
    
    def __float__(self):
        res = Field(self.Nx, self.Ny, M=self.M, N=self.N, typ=float)

        res.x   = float(self.y)
        res.y   = float(self.x)

        # Non-dim Y direction
        res.eta = float(self.eta)

        # Field Properties
        res.Cp     = float(self.Cp)
        res.mu     = float(self.mu)
        res.muT    = float(self.muT)
        res.muTT   = float(self.muTT)
        res.mux    = float(self.mux)
        res.muxx   = float(self.muxx)
        res.muxy   = float(self.muxy)
        res.muy    = float(self.muy)
        res.muyy   = float(self.muyy)
        res.lamb   = float(self.lamb)
        res.lambx  = float(self.lambx)
        res.lambxx = float(self.lambxx)
        res.lamby  = float(self.lamby)
        res.lambyy = float(self.lambyy)
        res.lambxy = float(self.lambxy)
        res.lambT  = float(self.lambT)
        res.lambTT = float(self.lambTT)
        res.rho    = float(self.rho) 
        res.rhox   = float(self.rhox)
        res.rhoxx  = float(self.rhoxx) 
        res.rhoy   = float(self.rhoy)
        res.rhoyy  = float(self.rhoyy)
        res.rhoxy  = float(self.rhoxy)
        
        # Dimensionless quantities
        res.delta = float(self.delta)
        
        # Disturbance kinetic energy (DKE)
        res.DKE  = float(self.DKE)
        res.DKEx = float(self.DKEx)

        # Amplitude
        res.amplitude = float(self.amplitude)

        # Wavenumbers
        res.omega = float(self.omega)
        res.beta  = float(self.bata)
        
        # Growth rath
        res.sigma   = float(self.sigma)
        res.nfactor = float(self.nfactor)
        res.Nfactor = float(self.Nfactor)
        
        # Growth rate parameter alpha
        res.alpha   = float(self.alpha)
        res.alphax  = float(self.alphax)
        
        # U
        res.U   = float(self.U)
        res.Ux  = float(self.Ux)
        res.Uxx = float(self.Uxx)
        
        res.Uy  = float(self.Uy)
        res.Uyy = float(self.Uyy)
        
        res.Uz  = float(self.Uz)
        res.Uzz = float(self.Uzz)
        
        res.Uxy = float(self.Uxy)
        res.Uxz = float(self.Uxz)
        res.Uyz = float(self.Uyz)
        
        # V
        res.V   = float(self.V)
        res.Vx  = float(self.Vx)
        res.Vxx = float(self.Vxx)
        
        res.Vy  = float(self.Vy)
        res.Vyy = float(self.Vyy)
        
        res.Vz  = float(self.Vz)
        res.Vzz = float(self.Vzz)
        
        res.Vxy = float(self.Vxy)
        res.Vxz = float(self.Vxz)
        res.Vyz = float(self.Vyz)
        
        # W
        res.W   = float(self.W)
        res.Wx  = float(self.Wx)
        res.Wxx = float(self.Wxx)
        
        res.Wy  = float(self.Wy)
        res.Wyy = float(self.Wyy)
        
        res.Wz  = float(self.Wz)
        res.Wzz = float(self.Wzz)
        
        res.Wxy = float(self.Wxy)
        res.Wxz = float(self.Wxz)
        res.Wyz = float(self.Wyz)
        
        # P
        res.P   = float(self.P)
        res.Px  = float(self.Px)
        res.Pxx = float(self.Pxx)

        res.Py  = float(self.Py)
        res.Pyy = float(self.Pyy)
        
        res.Pz  = float(self.Pz)
        res.Pzz = float(self.Pzz)
        
        res.Pxy = float(self.Pxy)
        res.Pxz = float(self.Pxz)
        res.Pyz = float(self.Pyz)
        
        # T
        res.T   = float(self.T)
        res.Tx  = float(self.Tx)
        res.Txx = float(self.Txx)
        
        res.T   = float(self.T)
        res.Ty  = float(self.Ty)
        res.Tyy = float(self.Tyy)
        
        res.Tz  = float(self.Tz)
        res.Tzz = float(self.Tzz)
        
        res.Txy = float(self.Txy)
        res.Txz = float(self.Txz)
        res.Tyz = float(self.Tyz)

        # conservative quantities
        res.E    = float(self.E)
        res.rhoU = float(self.rhoU)
        res.rhoV = float(self.rhoV)
        res.rhoW = float(self.rhoW)

        return res

    def __int__(self):
        res = Field(self.Nx, self.Ny, M=self.M, N=self.N, typ=int)

        res.x   = int(self.y)
        res.y   = int(self.x)

        # Non-dim Y direction
        res.eta = int(self.eta)

        # Field Properties
        res.Cp     = int(self.Cp)
        res.mu     = int(self.mu)
        res.muT    = int(self.muT)
        res.muTT   = int(self.muTT)
        res.mux    = int(self.mux)
        res.muxx   = int(self.muxx)
        res.muxy   = int(self.muxy)
        res.muy    = int(self.muy)
        res.muyy   = int(self.muyy)
        res.lamb   = int(self.lamb)
        res.lambx  = int(self.lambx)
        res.lambxx = int(self.lambxx)
        res.lamby  = int(self.lamby)
        res.lambyy = int(self.lambyy)
        res.lambxy = int(self.lambxy)
        res.lambT  = int(self.lambT)
        res.lambTT = int(self.lambTT)
        res.rho    = int(self.rho) 
        res.rhox   = int(self.rhox)
        res.rhoxx  = int(self.rhoxx) 
        res.rhoy   = int(self.rhoy)
        res.rhoyy  = int(self.rhoyy)
        res.rhoxy  = int(self.rhoxy)
        
        # Dimensionless quantities
        res.delta = int(self.delta)
        
        # Disturbance kinetic energy (DKE)
        res.DKE  = int(self.DKE)
        res.DKEx = int(self.DKEx)

        # Amplitude
        res.amplitude = int(self.amplitude)

        # Wavenumbers
        res.omega = int(self.omega)
        res.beta  = int(self.bata)
        
        # Growth rath
        res.sigma   = int(self.sigma)
        res.nfactor = int(self.nfactor)
        res.Nfactor = int(self.Nfactor)
        
        # Growth rate parameter alpha
        res.alpha   = int(self.alpha)
        res.alphax  = int(self.alphax)
        
        # U
        res.U   = int(self.U)
        res.Ux  = int(self.Ux)
        res.Uxx = int(self.Uxx)
        
        res.Uy  = int(self.Uy)
        res.Uyy = int(self.Uyy)
        
        res.Uz  = int(self.Uz)
        res.Uzz = int(self.Uzz)
        
        res.Uxy = int(self.Uxy)
        res.Uxz = int(self.Uxz)
        res.Uyz = int(self.Uyz)
        
        # V
        res.V   = int(self.V)
        res.Vx  = int(self.Vx)
        res.Vxx = int(self.Vxx)
        
        res.Vy  = int(self.Vy)
        res.Vyy = int(self.Vyy)
        
        res.Vz  = int(self.Vz)
        res.Vzz = int(self.Vzz)
        
        res.Vxy = int(self.Vxy)
        res.Vxz = int(self.Vxz)
        res.Vyz = int(self.Vyz)
        
        # W
        res.W   = int(self.W)
        res.Wx  = int(self.Wx)
        res.Wxx = int(self.Wxx)
        
        res.Wy  = int(self.Wy)
        res.Wyy = int(self.Wyy)
        
        res.Wz  = int(self.Wz)
        res.Wzz = int(self.Wzz)
        
        res.Wxy = int(self.Wxy)
        res.Wxz = int(self.Wxz)
        res.Wyz = int(self.Wyz)
        
        # P
        res.P   = int(self.P)
        res.Px  = int(self.Px)
        res.Pxx = int(self.Pxx)

        res.Py  = int(self.Py)
        res.Pyy = int(self.Pyy)
        
        res.Pz  = int(self.Pz)
        res.Pzz = int(self.Pzz)
        
        res.Pxy = int(self.Pxy)
        res.Pxz = int(self.Pxz)
        res.Pyz = int(self.Pyz)
        
        # T
        res.T   = int(self.T)
        res.Tx  = int(self.Tx)
        res.Txx = int(self.Txx)
        
        res.T   = int(self.T)
        res.Ty  = int(self.Ty)
        res.Tyy = int(self.Tyy)
        
        res.Tz  = int(self.Tz)
        res.Tzz = int(self.Tzz)
        
        res.Txy = int(self.Txy)
        res.Txz = int(self.Txz)
        res.Tyz = int(self.Tyz)

        # conservative quantities
        res.E    = int(self.E)
        res.rhoU = int(self.rhoU)
        res.rhoV = int(self.rhoV)
        res.rhoW = int(self.rhoW)

        return res

    def __abs__(self):
        res = Field(self.Nx, self.Ny, M=self.M, N=self.N, typ=float)

        res.x   = abs(self.y)
        res.y   = abs(self.x)

        # Non-dim Y direction
        res.eta = abs(self.eta)

        # Field Properties
        res.Cp     = abs(self.Cp)
        res.mu     = abs(self.mu)
        res.muT    = abs(self.muT)
        res.muTT   = abs(self.muTT)
        res.mux    = abs(self.mux)
        res.muxx   = abs(self.muxx)
        res.muxy   = abs(self.muxy)
        res.muy    = abs(self.muy)
        res.muyy   = abs(self.muyy)
        res.lamb   = abs(self.lamb)
        res.lambx  = abs(self.lambx)
        res.lambxx = abs(self.lambxx)
        res.lamby  = abs(self.lamby)
        res.lambyy = abs(self.lambyy)
        res.lambxy = abs(self.lambxy)
        res.lambT  = abs(self.lambT)
        res.lambTT = abs(self.lambTT)
        res.rho    = abs(self.rho) 
        res.rhox   = abs(self.rhox)
        res.rhoxx  = abs(self.rhoxx) 
        res.rhoy   = abs(self.rhoy)
        res.rhoyy  = abs(self.rhoyy)
        res.rhoxy  = abs(self.rhoxy)
        
        # Dimensionless quantities
        res.delta = abs(self.delta)
        
        # Disturbance kinetic energy (DKE)
        res.DKE  = abs(self.DKE)
        res.DKEx = abs(self.DKEx)

        # Amplitude
        res.amplitude = abs(self.amplitude)

        # Wavenumbers
        res.omega = abs(self.omega)
        res.beta  = abs(self.bata)
        
        # Growth rath
        res.sigma   = abs(self.sigma)
        res.nfactor = abs(self.nfactor)
        res.Nfactor = abs(self.Nfactor)
        
        # Growth rate parameter alpha
        res.alpha   = abs(self.alpha)
        res.alphax  = abs(self.alphax)
        
        # U
        res.U   = abs(self.U)
        res.Ux  = abs(self.Ux)
        res.Uxx = abs(self.Uxx)
        
        res.Uy  = abs(self.Uy)
        res.Uyy = abs(self.Uyy)
        
        res.Uz  = abs(self.Uz)
        res.Uzz = abs(self.Uzz)
        
        res.Uxy = abs(self.Uxy)
        res.Uxz = abs(self.Uxz)
        res.Uyz = abs(self.Uyz)
        
        # V
        res.V   = abs(self.V)
        res.Vx  = abs(self.Vx)
        res.Vxx = abs(self.Vxx)
        
        res.Vy  = abs(self.Vy)
        res.Vyy = abs(self.Vyy)
        
        res.Vz  = abs(self.Vz)
        res.Vzz = abs(self.Vzz)
        
        res.Vxy = abs(self.Vxy)
        res.Vxz = abs(self.Vxz)
        res.Vyz = abs(self.Vyz)
        
        # W
        res.W   = abs(self.W)
        res.Wx  = abs(self.Wx)
        res.Wxx = abs(self.Wxx)
        
        res.Wy  = abs(self.Wy)
        res.Wyy = abs(self.Wyy)
        
        res.Wz  = abs(self.Wz)
        res.Wzz = abs(self.Wzz)
        
        res.Wxy = abs(self.Wxy)
        res.Wxz = abs(self.Wxz)
        res.Wyz = abs(self.Wyz)
        
        # P
        res.P   = abs(self.P)
        res.Px  = abs(self.Px)
        res.Pxx = abs(self.Pxx)

        res.Py  = abs(self.Py)
        res.Pyy = abs(self.Pyy)
        
        res.Pz  = abs(self.Pz)
        res.Pzz = abs(self.Pzz)
        
        res.Pxy = abs(self.Pxy)
        res.Pxz = abs(self.Pxz)
        res.Pyz = abs(self.Pyz)
        
        # T
        res.T   = abs(self.T)
        res.Tx  = abs(self.Tx)
        res.Txx = abs(self.Txx)
        
        res.T   = abs(self.T)
        res.Ty  = abs(self.Ty)
        res.Tyy = abs(self.Tyy)
        
        res.Tz  = abs(self.Tz)
        res.Tzz = abs(self.Tzz)
        
        res.Txy = abs(self.Txy)
        res.Txz = abs(self.Txz)
        res.Tyz = abs(self.Tyz)

        # conservative quantities
        res.E    = abs(self.E)
        res.rhoU = abs(self.rhoU)
        res.rhoV = abs(self.rhoV)
        res.rhoW = abs(self.rhoW)

        return res
