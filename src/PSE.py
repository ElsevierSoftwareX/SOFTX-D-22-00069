import os
import numpy as np
from matplotlib import pyplot as plt
from numpy.matlib         import diagflat
from numpy                import vstack
from numpy.fft            import irfft2  as ifft2
from numpy.fft            import fft2   as fft2
from numpy.fft            import irfft  as ifft
from numpy.fft            import fft    as fft
#from tensorflow.signal    import irfft2d
#from tensorflow.signal    import  rfft2d
from matplotlib           import pylab
from scipy.interpolate    import interp1d
from scipy.integrate      import cumtrapz
from scipy.integrate      import simps
from scipy.linalg         import block_diag
from Field                import Field
import scipy as sc
from Lagrange import dLagrange, d2Lagrange, d3Lagrange, d4Lagrange

class PSE(object):
    # Description
    # Handle the PSE calculation. All the subroutines relevant
    # to build the operator, compute the nonlinear forcing term
    # and solve the system.
    def __init__(self, NumMethod, Param, Flow, i0=0):
        # Station Number
        self.i            = i0
        self.i0           = i0
        # Spectral methods routines
        self.NumMethod    = NumMethod
        self.x            = NumMethod.X1D
        self.y            = NumMethod.Y1D
        self.Nx           = NumMethod.Nx
        self.Ny           = NumMethod.Ny
        # Flow parameters
        self.Param        = Param
        # Base Flow
        self.Flow         = Flow
        # Mean Flow Distorsion
        self.MFD          = Field(self.NumMethod.Nx, self.NumMethod.Ny)
        # Differentiation/Integration
        self.Dy  = NumMethod.MethodY.getDiffMatrix()#.todense()
        self.D2y = NumMethod.MethodY.getSecondDiffMatrix()#.todense()
        self.w   = NumMethod.MethodY.getIntWeight()
        # Solution at the current iteration
        self.phi        = np.zeros((Param['(M,N)max'][0], Param['(M,N)max'][1], self.NumMethod.Nx, 5*NumMethod.Ny), dtype=complex)
        self.phix       = np.zeros((Param['(M,N)max'][0], Param['(M,N)max'][1], self.NumMethod.Nx, 5*NumMethod.Ny), dtype=complex)
        self.phi_st     = np.zeros((Param['(M,N)max'][0], Param['(M,N)max'][1], self.NumMethod.Nx, 5*NumMethod.Ny), dtype=complex)
        self.phix_0     = np.zeros((Param['(M,N)max'][0], Param['(M,N)max'][1], self.NumMethod.Nx, 5*NumMethod.Ny), dtype=complex)
        self.alpha      = np.zeros((Param['(M,N)max'][0], Param['(M,N)max'][1], self.NumMethod.Nx), dtype=complex)
        self.alpha_old  = np.zeros((Param['(M,N)max'][0], Param['(M,N)max'][1]), dtype=complex)
        self.alphax     = np.zeros((Param['(M,N)max'][0], Param['(M,N)max'][1], self.NumMethod.Nx), dtype=complex)
        self.phi_old    = np.zeros((Param['(M,N)max'][0], Param['(M,N)max'][1],  5*NumMethod.Ny), dtype=complex)
        self.NormCond   = np.zeros((Param['(M,N)max'][0], Param['(M,N)max'][1], self.NumMethod.Nx, 600, 2), dtype=complex)
        self.amplitude  = np.ones((Param['(M,N)max'][0], Param['(M,N)max'][1], self.NumMethod.Nx), dtype=float)
        # Growth rate
        self.sigma      = np.zeros((Param['(M,N)max'][0], Param['(M,N)max'][1], self.NumMethod.Nx))
        self.DKE        = np.zeros((Param['(M,N)max'][0], Param['(M,N)max'][1], self.NumMethod.Nx))
        self.DKEx       = np.zeros((Param['(M,N)max'][0], Param['(M,N)max'][1], self.NumMethod.Nx))
        # PDE operator
        self.L = np.zeros((Param['(M,N)max'][0], Param['(M,N)max'][1], 5*self.NumMethod.Ny,5*self.NumMethod.Ny), dtype=complex)

    def RMS(self, MFDcoupling=False):
        #-----------------------------------------------------------
        # Description
        #
        # ifft[u(m,n)] -> u' 
        #-----------------------------------------------------------

        # Station number
        i = self.i
        i0 = self.i0

        # Numerical methods
        (M,N) = self.Param['(M,N)max']
        D     = self.Dy

        # FFT in time only
        padM = 100*M
        padN = 100*N

        # Dimensions
        Nx    = self.NumMethod.Nx
        Ny    = self.NumMethod.Ny

        # Offset
        u = Ny*1
        v = Ny*2
        w = Ny*3
        p = Ny*4
        T = Ny*5

        # Initializing memory
        phi   = np.zeros((self.Param['(M,N)max'][0], self.Param['(M,N)max'][1], 5*self.NumMethod.Ny), dtype=complex)
        phix  = np.zeros((self.Param['(M,N)max'][0], self.Param['(M,N)max'][1], 5*self.NumMethod.Ny), dtype=complex)
        phiz  = np.zeros((self.Param['(M,N)max'][0], self.Param['(M,N)max'][1], 5*self.NumMethod.Ny), dtype=complex)
        phit  = np.zeros((self.Param['(M,N)max'][0], self.Param['(M,N)max'][1], 5*self.NumMethod.Ny), dtype=complex)

        # Initializing memory
        RMS    = Field(Nx, Ny, M=self.Param['(M,N)max'][0], N=self.Param['(M,N)max'][1])

        # Loop on x stations
        for i in range(i0, self.NumMethod.Nx):
            # Streamwise/spanwise derivatives
            for m,n in self.Param['modes'][i]:
                # Mode (0,0) is not considered for forcing if MFD is updated,
                if not MFDcoupling or (m,n) != (0,0):
                    # Retrieving Mean Flow and properties
                    if self.Param[(m,n)]['local'][i]:
                        beta   = self.Param[(m,n)]['beta'][i]
                        omega  = self.Param[(m,n)]['omega'][i]
                    else:
                        beta  = self.Param[(m,n)]['beta'][self.i0]
                        omega = self.Param[(m,n)]['omega'][self.i0]
            
                    phi[m,n,:] = self.phi[m,n,i,:]*self.amplitude[m,n,i]
                    phix[m,n,:]  = (self.phix[m,n,i,:] + self.phi[m,n,i,:]*self.alpha[m,n,i])*self.amplitude[m,n,i]
                    phiz[m,n,:]  = (self.phi[m,n,i,:]*beta*1j)*self.amplitude[m,n,i]

            # Spectral space -> physical space
            # Each column m and n correspond to a different
            # location on Z and t.
            # 1D FFT
            ifftPHI   = ifft2(phi  , axes=(0,1),norm='ortho',s=(M+padM, N+padN))[0:self.Param['(M,N)max'][0], 0:self.Param['(M,N)max'][1],:]
            ifftPHIX  = ifft2(phix , axes=(0,1),norm='ortho',s=(M+padM, N+padN))[0:self.Param['(M,N)max'][0], 0:self.Param['(M,N)max'][1],:]
            ifftPHIZ  = ifft2(phiz , axes=(0,1),norm='ortho',s=(M+padM, N+padN))[0:self.Param['(M,N)max'][0], 0:self.Param['(M,N)max'][1],:]
            
            # Physical fluctuation
            # (u',v',w',p',T')
            RMS.U[:,:,i,:] = ifftPHI[:,:,0:u]
            RMS.V[:,:,i,:]= ifftPHI[:,:,u:v]
            RMS.W[:,:,i,:] = ifftPHI[:,:,v:w]
            RMS.P[:,:,i,:]= ifftPHI[:,:,w:p]
            RMS.T[:,:,i,:] = ifftPHI[:,:,p:T]
            
            # x and z Derivatives
            RMS.Ux[:,:,i,:]= ifftPHIX[:,:,0:u]
            RMS.Vx[:,:,i,:] = ifftPHIX[:,:,u:v]
            RMS.Wx[:,:,i,:] = ifftPHIX[:,:,v:w]
            RMS.Px[:,:,i,:] = ifftPHIX[:,:,w:p]
            RMS.Tx[:,:,i,:] = ifftPHIX[:,:,p:T]
            
            RMS.Uz[:,:,i,:] = ifftPHIZ[:,:,0:u]
            RMS.Vz[:,:,i,:] = ifftPHIZ[:,:,u:v]
            RMS.Wz[:,:,i,:] = ifftPHIZ[:,:,v:w]
            RMS.Pz[:,:,i,:] = ifftPHIZ[:,:,w:p]
            RMS.Tz[:,:,i,:] = ifftPHIZ[:,:,p:T]
            
            # y-derivatives (evaluated numerically)
            for m,n in self.Param['modes'][i]:
                RMS.Uy[m,n,i,:] = D.dot(RMS.U[m,n,i,:])
                RMS.Vy[m,n,i,:] = D.dot(RMS.V[m,n,i,:])
                RMS.Wy[m,n,i,:] = D.dot(RMS.W[m,n,i,:])
                RMS.Py[m,n,i,:] = D.dot(RMS.P[m,n,i,:])
                RMS.Ty[m,n,i,:] = D.dot(RMS.T[m,n,i,:])

            # Storing Results
            RMS.xc[:,:,i,:]  = self.Flow.xc[i,:]
            RMS.yc[:,:,i,:]  = self.Flow.yc[i,:]
            RMS.x[:,:,i,:]   = self.Flow.x[i,:]
            RMS.y[:,:,i,:]   = self.Flow.y[i,:]
            
            # conservatives quantities
            RMS.rho[:,:,i,:] = self.Flow.rho[i,:]/self.Flow.P[i,:]*RMS.P[:,:,i,:]    \
                    - self.Flow.rho[i,:]/self.Flow.T[i,:]*RMS.T[:,:,i,:]
            
            RMS.rhoU[:,:,i,:] = self.Flow.rho[i,:]*RMS.U[:,:,i,:]                    \
                    + self.Flow.U[i,:]*RMS.rho[:,:,i,:]
            
            RMS.rhoV[:,:,i,:] = self.Flow.rho[i,:]*RMS.V[:,:,i,:]                    \
                    + self.Flow.V[i,:]*RMS.rho[:,:,i,:]
            
            RMS.rhoW[:,:,i,:] = self.Flow.rho[i,:]*RMS.W[:,:,i,:]                    \
                    + self.Flow.W[i,:]*RMS.rho[:,:,i,:]
            
            RMS.E[:,:,i,:]  = 1/(self.Param['prop']['gamma']-1)*RMS.P[:,:,i,:]          \
                            + 0.5*( self.Flow.U[i,:]**2                                 \
                                  + self.Flow.V[i,:]**2                                 \
                                  + self.Flow.W[i,:]**2)*RMS.rho[:,:,i,:]               \
                                  + self.Flow.rho[i,:]*(self.Flow.U[i,:]*RMS.U[:,:,i,:] \
                                  + self.Flow.V[i,:]*RMS.V[:,:,i,:] \
                                  + self.Flow.W[i,:]*RMS.W[:,:,i,:] )
        return RMS

    def updateMFD(self, relax=1):
        #-----------------------------------------------------------
        # Description
        # Update the mean flow distorsion field from the mode (0,0)
        # The derivatives (x and y) are computed numerically
        #-----------------------------------------------------------
        # Station number
        i = self.i

        # Number of points
        Ny = self.NumMethod.Ny

        # Offset
        u = Ny*1
        v = Ny*2
        w = Ny*3
        p = Ny*4
        T = Ny*5

        # Meanflow distorsion
        self.MFD.U[i] = (1-relax)*self.MFD.U[i] + relax*self.phi[0,0,i,0:u].real
        self.MFD.V[i] = (1-relax)*self.MFD.V[i] + relax*self.phi[0,0,i,u:v].real
        self.MFD.W[i] = (1-relax)*self.MFD.W[i] + relax*self.phi[0,0,i,v:w].real
        self.MFD.P[i] = (1-relax)*self.MFD.P[i] + relax*self.phi[0,0,i,w:p].real
        self.MFD.T[i] = (1-relax)*self.MFD.T[i] + relax*self.phi[0,0,i,p:T].real

        self.MFD.Ux[i] = (1-relax)*self.MFD.Ux[i] + relax*self.phix[0,0,i,0:u].real
        self.MFD.Vx[i] = (1-relax)*self.MFD.Vx[i] + relax*self.phix[0,0,i,u:v].real
        self.MFD.Wx[i] = (1-relax)*self.MFD.Wx[i] + relax*self.phix[0,0,i,v:w].real
        self.MFD.Px[i] = (1-relax)*self.MFD.Px[i] + relax*self.phix[0,0,i,w:p].real
        self.MFD.Tx[i] = (1-relax)*self.MFD.Tx[i] + relax*self.phix[0,0,i,p:T].real
                                            
        self.MFD.Uy[i] = (1-relax)*self.MFD.Uy[i] + relax*self.Dy.dot(self.phi[0,0,i,0:u].real)
        self.MFD.Vy[i] = (1-relax)*self.MFD.Vy[i] + relax*self.Dy.dot(self.phi[0,0,i,u:v].real)
        self.MFD.Wy[i] = (1-relax)*self.MFD.Wy[i] + relax*self.Dy.dot(self.phi[0,0,i,v:w].real)
        self.MFD.Py[i] = (1-relax)*self.MFD.Py[i] + relax*self.Dy.dot(self.phi[0,0,i,w:p].real)
        self.MFD.Ty[i] = (1-relax)*self.MFD.Ty[i] + relax*self.Dy.dot(self.phi[0,0,i,p:T].real)
                                            
        self.MFD.Uyy[i] = (1-relax)*self.MFD.Uyy[i] + relax*self.D2y.dot(self.phi[0,0,i,0:u].real)
        self.MFD.Vyy[i] = (1-relax)*self.MFD.Vyy[i] + relax*self.D2y.dot(self.phi[0,0,i,u:v].real)
        self.MFD.Wyy[i] = (1-relax)*self.MFD.Wyy[i] + relax*self.D2y.dot(self.phi[0,0,i,v:w].real)
        self.MFD.Pyy[i] = (1-relax)*self.MFD.Pyy[i] + relax*self.D2y.dot(self.phi[0,0,i,w:p].real)
        self.MFD.Tyy[i] = (1-relax)*self.MFD.Tyy[i] + relax*self.D2y.dot(self.phi[0,0,i,p:T].real)
                                            
        self.MFD.Uxy[i] = (1-relax)*self.MFD.Uxy[i] + relax*self.Dy.dot(self.phix[0,0,i,0:u].real)
        self.MFD.Vxy[i] = (1-relax)*self.MFD.Vxy[i] + relax*self.Dy.dot(self.phix[0,0,i,u:v].real)
        self.MFD.Wxy[i] = (1-relax)*self.MFD.Wxy[i] + relax*self.Dy.dot(self.phix[0,0,i,v:w].real)
        self.MFD.Pxy[i] = (1-relax)*self.MFD.Pxy[i] + relax*self.Dy.dot(self.phix[0,0,i,w:p].real)
        self.MFD.Txy[i] = (1-relax)*self.MFD.Txy[i] + relax*self.Dy.dot(self.phix[0,0,i,p:T].real)

    def forcingTerm(self, itr, MFDcoupling=False):
        #-----------------------------------------------------------
        # Description
        # Compute the nonlinear forcing term (for NPSE calculations)
        # The NLT are computed in the physical space and converted
        # in the spectral space using ifft/fft.
        #
        # Basic steps:
        #
        # ifft[u(m,n)] -> u' 
        # NLT' = f(u')
        # NLT(m,n) = fft(NLT')
        #
        #-----------------------------------------------------------

        # Station number
        i = self.i
        i0 = self.i0

        # Numerical methods
        (M,N) = self.Param['(M,N)max']

        # FFT in time only
        padM = 25 #5*(M-1)
        padN = 25 #5*(N-1)

        Ny    = self.NumMethod.Ny
        D     = self.Dy
        D2    = self.D2y

        # Flow properties
        Pr     = self.Param['prop']['Pr']
        Cp     = self.Param['prop']['Cp']
        gamma  = self.Param['prop']['gamma']
        Ma     = self.Param['prop']['Ma']

        # Geometric Jacobian
        Jxx = self.NumMethod.Jxx[i,:]
        Jxy = self.NumMethod.Jxy[i,:]
        Jyx = self.NumMethod.Jyx[i,:]
        Jyy = self.NumMethod.Jyy[i,:]

        # Retrieving Mean Flow and properties
        local = False
        for (m,n) in self.Param['modes'][i]:
            if self.Param[(m,n)]['local'][i]:
                local = True
        if local:
          Re = self.Param['prop']['Rex'][i]
        else:
          Re = self.Param['prop']['Rex'][i0]

        # Array Offset (Global system)
        u = Ny*1
        v = Ny*2
        w = Ny*3
        p = Ny*4
        T = Ny*5

        # Initializing memory
        NLT = np.zeros((M,N,5*Ny), dtype=complex)

        Um  = np.zeros((M,N,Ny), dtype=complex)
        Vm  = np.zeros((M,N,Ny), dtype=complex)
        Wm  = np.zeros((M,N,Ny), dtype=complex)
        Pm  = np.ones((M,N,Ny), dtype=complex)
        Tm  = np.ones((M,N,Ny), dtype=complex)

        Umx  = np.zeros((M,N,Ny), dtype=complex)
        Vmx  = np.zeros((M,N,Ny), dtype=complex)
        Wmx  = np.zeros((M,N,Ny), dtype=complex)
        Pmx  = np.zeros((M,N,Ny), dtype=complex)
        Tmx  = np.zeros((M,N,Ny), dtype=complex)

        Umy  = np.zeros((M,N,Ny), dtype=complex)
        Vmy  = np.zeros((M,N,Ny), dtype=complex)
        Wmy  = np.zeros((M,N,Ny), dtype=complex)
        Pmy  = np.zeros((M,N,Ny), dtype=complex)
        Tmy  = np.zeros((M,N,Ny), dtype=complex)

        Umz  = 0. 
        Vmz  = 0.
        Wmz  = 0.
        Pmz  = 0.
        Tmz  = 0.

        Upy  = np.zeros((M,N,Ny), dtype=complex)
        Vpy  = np.zeros((M,N,Ny), dtype=complex)
        Wpy  = np.zeros((M,N,Ny), dtype=complex)
        Ppy  = np.zeros((M,N,Ny), dtype=complex)
        Tpy  = np.zeros((M,N,Ny), dtype=complex)

        Upxy = np.zeros((M,N,Ny), dtype=complex)
        Vpxy = np.zeros((M,N,Ny), dtype=complex)
        Wpxy = np.zeros((M,N,Ny), dtype=complex)
        Ppxy = np.zeros((M,N,Ny), dtype=complex)
        Tpxy = np.zeros((M,N,Ny), dtype=complex)

        Upyy = np.zeros((M,N,Ny), dtype=complex)
        Vpyy = np.zeros((M,N,Ny), dtype=complex)
        Wpyy = np.zeros((M,N,Ny), dtype=complex)
        Ppyy = np.zeros((M,N,Ny), dtype=complex)
        Tpyy = np.zeros((M,N,Ny), dtype=complex)

        Upyz = np.zeros((M,N,Ny), dtype=complex)
        Vpyz = np.zeros((M,N,Ny), dtype=complex)
        Wpyz = np.zeros((M,N,Ny), dtype=complex)
        Ppyz = np.zeros((M,N,Ny), dtype=complex)
        Tpyz = np.zeros((M,N,Ny), dtype=complex)

        phi   = np.zeros((self.Param['(M,N)max'][0], self.Param['(M,N)max'][1], 5*self.NumMethod.Ny), dtype=complex)
        phix  = np.zeros((self.Param['(M,N)max'][0], self.Param['(M,N)max'][1], 5*self.NumMethod.Ny), dtype=complex)
        phiz  = np.zeros((self.Param['(M,N)max'][0], self.Param['(M,N)max'][1], 5*self.NumMethod.Ny), dtype=complex)
        phit  = np.zeros((self.Param['(M,N)max'][0], self.Param['(M,N)max'][1], 5*self.NumMethod.Ny), dtype=complex)
        phixx = np.zeros((self.Param['(M,N)max'][0], self.Param['(M,N)max'][1], 5*self.NumMethod.Ny), dtype=complex)
        phixz = np.zeros((self.Param['(M,N)max'][0], self.Param['(M,N)max'][1], 5*self.NumMethod.Ny), dtype=complex)
        phizz = np.zeros((self.Param['(M,N)max'][0], self.Param['(M,N)max'][1], 5*self.NumMethod.Ny), dtype=complex)

        # Streamwise/spanwise derivatives
        for m,n in self.Param['modes'][i]:
            A = self.amplitude[m,n,i]
            # Mode (0,0) is not considered for forcing if MFD is updated
            if not MFDcoupling and (m,n) != (0,0):# and self.amplitude[m,n,i]*self.Param[(m,n)]['A_0'] > 1e-7:
                # Retrieving Mean Flow and properties
                if self.Param[(m,n)]['local'][i]:
                    beta   = self.Param[(m,n)]['beta'][i]
                    omega  = self.Param[(m,n)]['omega'][i]
                else:
                    beta  = self.Param[(m,n)]['beta'][self.i0]
                    omega = self.Param[(m,n)]['omega'][self.i0]

                phi [m,n,:]  = self.phi[m,n,i,:]*A
                phix[m,n,:]  = (self.phix[m,n,i,:] + self.phi[m,n,i,:]*self.alpha[m,n,i])*A
                phiz[m,n,:]  = (self.phi[m,n,i,:]*beta*1j)*A
                phit[m,n,:]  = (self.phi[m,n,i,:]*omega*1j)*A
                phixx[m,n,:] = (2.j*self.alpha[m,n,i]*self.phix[m,n,i,:] + (self.alphax[m,n,i]-self.alpha[m,n,i]**2)*self.phi[m,n,i,:])*A
                phixz[m,n,:] = (self.phix[m,n,i,:]*beta*1j - self.phi[m,n,i,:]*self.alpha[m,n,i]*beta)*A
                phizz[m,n,:] = (-self.phi[m,n,i,:]*beta**2)*A


        # Spectral space -> physical space
        # Each column m and n correspond to a different
        # location on Z and t.
        # 1D FFT
        ifftPHI   = ifft2(phi  , axes=(0,1),norm='ortho',s=(M+padM, N+padN))[0:self.Param['(M,N)max'][0], 0:self.Param['(M,N)max'][1],:]
        ifftPHIX  = ifft2(phix , axes=(0,1),norm='ortho',s=(M+padM, N+padN))[0:self.Param['(M,N)max'][0], 0:self.Param['(M,N)max'][1],:]
        ifftPHIZ  = ifft2(phiz , axes=(0,1),norm='ortho',s=(M+padM, N+padN))[0:self.Param['(M,N)max'][0], 0:self.Param['(M,N)max'][1],:]
        ifftPHIT  = ifft2(phit , axes=(0,1),norm='ortho',s=(M+padM, N+padN))[0:self.Param['(M,N)max'][0], 0:self.Param['(M,N)max'][1],:]
        ifftPHIXX = ifft2(phixx, axes=(0,1),norm='ortho',s=(M+padM, N+padN))[0:self.Param['(M,N)max'][0], 0:self.Param['(M,N)max'][1],:]
        ifftPHIXZ = ifft2(phixz, axes=(0,1),norm='ortho',s=(M+padM, N+padN))[0:self.Param['(M,N)max'][0], 0:self.Param['(M,N)max'][1],:]
        ifftPHIZZ = ifft2(phizz, axes=(0,1),norm='ortho',s=(M+padM, N+padN))[0:self.Param['(M,N)max'][0], 0:self.Param['(M,N)max'][1],:]


        # Mean flow =  Base flow (laminar) 
        #            + Correction Mode (0,0)
        for m,n in self.Param['modes'][i]:
            if (m,n) != (0,0) and MFDcoupling:
                mfd = 1
            else:
                mfd = 0

            Um[m,n] = self.Flow.U[i] + mfd*self.MFD.U[i]
            Vm[m,n] = self.Flow.V[i] + mfd*self.MFD.V[i]
            Wm[m,n] = self.Flow.W[i] + mfd*self.MFD.W[i]
            Pm[m,n] = self.Flow.P[i] + mfd*self.MFD.P[i]
            Tm[m,n] = self.Flow.T[i] + mfd*self.MFD.T[i]
            
            Umx[m,n] = self.Flow.Ux[i] + mfd*self.MFD.Ux[i]
            Vmx[m,n] = self.Flow.Vx[i] + mfd*self.MFD.Vx[i]
            Wmx[m,n] = self.Flow.Wx[i] + mfd*self.MFD.Wx[i]
            Pmx[m,n] = self.Flow.Px[i] + mfd*self.MFD.Px[i]
            Tmx[m,n] = self.Flow.Tx[i] + mfd*self.MFD.Tx[i]
            
            Umy[m,n] = self.Flow.Uy[i] + mfd*self.MFD.Uy[i]
            Vmy[m,n] = self.Flow.Vy[i] + mfd*self.MFD.Vy[i]
            Wmy[m,n] = self.Flow.Wy[i] + mfd*self.MFD.Wy[i]
            Pmy[m,n] = self.Flow.Py[i] + mfd*self.MFD.Py[i]
            Tmy[m,n] = self.Flow.Ty[i] + mfd*self.MFD.Ty[i]

        # Physical fluctuation
        # (u',v',w',p',T')
        Up = ifftPHI[:,:,0:u]
        Vp = ifftPHI[:,:,u:v]
        Wp = ifftPHI[:,:,v:w]
        Pp = ifftPHI[:,:,w:p]
        Tp = ifftPHI[:,:,p:T]

        # x and z Derivatives
        Upx = ifftPHIX[:,:,0:u]
        Vpx = ifftPHIX[:,:,u:v]
        Wpx = ifftPHIX[:,:,v:w]
        Ppx = ifftPHIX[:,:,w:p]
        Tpx = ifftPHIX[:,:,p:T]

        Upz = ifftPHIZ[:,:,0:u]
        Vpz = ifftPHIZ[:,:,u:v]
        Wpz = ifftPHIZ[:,:,v:w]
        Ppz = ifftPHIZ[:,:,w:p]
        Tpz = ifftPHIZ[:,:,p:T]

        Upxx = ifftPHIXX[:,:,0:u]
        Vpxx = ifftPHIXX[:,:,u:v]
        Tpxx = ifftPHIXX[:,:,p:T]

        Upxz = ifftPHIXZ[:,:,0:u]
        Vpxz = ifftPHIXZ[:,:,u:v]
        Wpxz = ifftPHIXZ[:,:,v:w]

        Wpzz = ifftPHIZZ[:,:,v:w]
        Tpzz = ifftPHIZZ[:,:,p:T]
        
        # Time derivative
        Upt = ifftPHIT[:,:,0:u]
        Vpt = ifftPHIT[:,:,u:v]
        Wpt = ifftPHIT[:,:,v:w]
        Ppt = ifftPHIT[:,:,w:p]
        Tpt = ifftPHIT[:,:,p:T]

        # y-derivatives (evaluated numerically)
        for m,n in self.Param['modes'][i]:
            Upy[m,n,:] = D.dot(Up[m,n,:])
            Vpy[m,n,:] = D.dot(Vp[m,n,:])
            Wpy[m,n,:] = D.dot(Wp[m,n,:])
            Ppy[m,n,:] = D.dot(Pp[m,n,:])
            Tpy[m,n,:] = D.dot(Tp[m,n,:])
            
            Upyy[m,n,:] = D2.dot(Up[m,n,:])
            Vpyy[m,n,:] = D2.dot(Vp[m,n,:])
            Wpyy[m,n,:] = D2.dot(Wp[m,n,:])
            Ppyy[m,n,:] = D2.dot(Pp[m,n,:])
            Tpyy[m,n,:] = D2.dot(Tp[m,n,:])
            
            Upxy[m,n,:] = D.dot(Upx[m,n,:])
            Vpxy[m,n,:] = D.dot(Vpx[m,n,:])
            Wpxy[m,n,:] = D.dot(Wpx[m,n,:])
            Ppxy[m,n,:] = D.dot(Ppx[m,n,:])
            Tpxy[m,n,:] = D.dot(Tpx[m,n,:])
            
            Upyz[m,n,:] = D.dot(Upz[m,n,:])
            Vpyz[m,n,:] = D.dot(Vpz[m,n,:])
            Wpyz[m,n,:] = D.dot(Wpz[m,n,:])
            Ppyz[m,n,:] = D.dot(Ppz[m,n,:])
            Tpyz[m,n,:] = D.dot(Tpz[m,n,:])

        # Flow properties
        rho    = self.Flow.rho[i,:]
        rhop   = gamma*Ma**2*(Pp*rho/Pm - rho/Tm*Tp)

        mu     = self.Flow.mu[i,:]
        mup    = self.Flow.muT[i,:]*Tp
        mupx   = self.Flow.muTT[i,:]*Tmx*Tp + self.Flow.muT[i,:]*Tpx
        mupy   = self.Flow.muTT[i,:]*Tmy*Tp + self.Flow.muT[i,:]*Tpy
        mupz   = self.Flow.muTT[i,:]*Tmz*Tp + self.Flow.muT[i,:]*Tpz

        lambp  = self.Flow.lambT[i,:]*Tp
        lambpx = self.Flow.lambTT[i,:]*Tmx*Tp + self.Flow.lamb[i,:]*Tpx
        lambpy = self.Flow.lambTT[i,:]*Tmy*Tp + self.Flow.lamb[i,:]*Tpy
        lambpz = self.Flow.lambTT[i,:]*Tmz*Tp + self.Flow.lamb[i,:]*Tpz

        # Mom-x
        NLT[:,:,0:u] =   mup/Re * ( 4/3*(Upxx*Jxx**2  + 2*Upxy*Jyx*Jxx + Upyy*Jyx**2)          \
                                      + (Vpxx*Jxy**2  + 2*Vpxy*Jxy*Jyy + Vpyy*Jyy**2)          \
                                      + (Vpxx*Jxx*Jxy + Vpxy*(Jyy*Jxx+Jyx*Jxy) + Vpyy*Jxy*Jyx) \
                                      + (Wpxz*Jxx + Wpyz*Jyx) +  Wpzz )                        \
                       + 4/3/Re * (mupx*Jxx + mupy*Jyx) * (  Upx*Jxx + Upy*Jyx  )              \
                       +   1/Re * (mupx*Jxy + mupy*Jyy) * ( (Upx*Jxx + Upy*Jyy)                \
                                                           +(Vpx*Jxx + Vpy*Jyx) )              \
                       + 1/Re*mupz*(Upz + Wpx*Jxx + Wpy*Jyx)                                   \
                       - rhop*(Upt + (Um + Up) * (Upx*Jxx + Upy*Jyx)                           \
                                   + (Vm + Vp) * (Upx*Jxy + Upy*Jyy)                           \
                                   + (Wm + Wp) * Upz )                                         \
                       -  rho*(Up*(Upx*Jxx + Upy*Jyx) + Vp*(Upx*Jxy + Upy*Jyy) + Wp*Upz)       \
                       - rhop*(Up*(Umx*Jxx + Umy*Jyx) + Vp*(Umx*Jxy + Umy*Jyy) + Wp*Umz)
        # Mom-y
        NLT[:,:,u:v] =   mup/Re * (     (Upxx*Jxx**2 + 2*Upxy*Jyx*Jxx + Upyy*Jyx**2)           \
                                  + 4/3*(Vpxx*Jxy**2 + 2*Vpxy*Jxy*Jyy + Vpyy*Jyy**2)           \
                                      + (Upxx*Jxx*Jxy + Upxy*(Jyy*Jxx+Jyx*Jxy) + Upyy*Jyx*Jyy) \
                                      + (Wpx*Jxy + Wpyz*Jyy) + Wpzz )                          \
                       +   1/Re * (mupx*Jxx + mupy*Jyx) * ( (Upx*Jyx + Upy*Jyy)                \
                                                           +(Vpx*Jxx + Vpy*Jxy) )              \
                       + 4/3*Re * (mupx*Jxx + mupy*Jyy) * (  Vpx*Jxy + Vpy*Jyy  )              \
                       +   1/Re * mupz*(Vpz + Wpx*Jyx + Wpy*Jyy)                               \
                       - rhop*(Vpt + (Um + Up) * (Vpx*Jxx + Vpy*Jyx)                           \
                                   + (Vm + Vp) * (Vpx*Jxy + Vpy*Jyy)                           \
                                   + (Wm + Wp) * Vpz )                                         \
                       -  rho*(Up * (Vpx*Jxx + Vpy*Jyx) + Vp * (Vpx*Jxy + Vpy*Jyy) + Wp*Vpz)   \
                       - rhop*(Up * (Vmx*Jxx + Vmy*Jyx) + Vp * (Vmx*Jxy + Vmy*Jyy) + Wp*Vmz)
        # Mom-z
        NLT[:,:,v:w] =   mup/Re * (  (Upxx*Jxx**2 + 2*Upxy*Jyx*Jxx + Upyy*Jyx**2)              \
                                   + (Vpxx*Jxy**2 + 2*Vpxy*Jxy*Jyy + Vpyy*Jyy**2)              \
                                   + 4/3*Wpzz + (Upxz*Jxx + Upyz*Jyx) + (Vpxz*Jxx + Vpyz*Jyx) )\
                       +   1/Re * (mupx*Jxx + mupy*Jxy) * ( (Upx*Jxy + Upy*Jyy)                \
                                                           +(Vpx*Jxx + Vpy*Jyx) )              \
                       +   1/Re * (mupx*Jxy + mupy*Jyy) * (Vpz + Wpx*Jxy + Wpy*Jyy)            \
                       + 4/3*Re * mupz * Wpz                                                   \
                       - rhop*(Wpt + (Um + Up) * (Wpx*Jxx + Wpy*Jyx)                           \
                                   + (Vm + Vp) * (Wpx*Jxy + Wpy*Jyy)                           \
                                   + (Wm + Wp) * Wpz)                                          \
                       -  rho*(Up * (Wpx*Jxx + Wpy*Jyx) + Vp * (Wpx*Jxy + Wpy*Jyy) + Wp*Wpz)   \
                       - rhop*(Up * (Wmx*Jxx + Wmy*Jyx) + Vp * (Wpx*Jxy + Wmy*Jyy) + Wp*Wmz)
        # Continuity
        NLT[:,:,w:p] =   Pp*Tp*(Umx*Jxx + Umy*Jyx + Vmx*Jxy + Vmy*Jyy + Wmz)                   \
                       + (Pp*Tm + Pm*Tp)*(Upx*Jxx + Upy*Jyx + Vpx*Jxy + Vpy*Jyy + Wpz)         \
                       + Tp*(Ppt + (Um + Up) * (Ppx*Jxx + Ppy*Jyx)                             \
                                 + (Vm + Vp) * (Ppx*Jxy + Ppy*Jyy) + (Wm + Wp) * Ppz )         \
                       - Pp*(Tpt + (Um + Up) * (Tpx*Jxx + Tpy*Jyx)                             \
                                 + (Vm + Vp) * (Tpx*Jxy + Tpy*Jyy) + (Wm + Wp) * Tpz )
        # Energy
        Qpp =  4/3 * (Upx*Jxx + Upy*Jyx)**2                                                    \
             + (Upx*Jxy + Upy*Jyy + Vpx*Jxx + Vpy*Jyx) * (Vpx*Jxy + Vpy*Jyy)                   \
             + (Upz + Wpx*Jxx + Wpy*Jyx) * Wpz                                                 \
             + (Upx*Jxy + Upy*Jyy + Vpx*Jxx + Vpy*Jyx) * (Upx*Jxx + Upy*Jyx)                   \
             + 4/3 * (Vpx*Jxy + Vpy*Jyy)**2                                                    \
             + (Vpz + Wpx*Jxy + Wpy*Jyy) * Wpz                                                 \
             + (Upz + Wpx*Jxx + Wpy*Jyx) * (Upx*Jxx + Upy*Jyy)                                 \
             + (Wpx*Jxy + Wpy*Jyy + Vpz) * (Vpx*Jxy + Vpy*Jyy)                                 \
             + 4/3*Wpz**2

        Qpm =  4/3 * (Upx*Jxx + Upy*Jyx) * (Umx*Jxx + Umy*Jyx)                                \
             + (Upx*Jxy + Upy*Jyy + Vpx*Jxx + Vpy*Jyx) * (Vmx*Jxy + Vmy*Jyy)                  \
             + (Upz + Wpx*Jxx + Wpy*Jyx) * Wmz                                                \
             + (Upx*Jxy + Upy*Jyy + Vpx*Jxx + Vpy*Jyx) * (Umx*Jxx + Umy*Jyx)                  \
             + 4/3 * (Vpx*Jxy + Vpy*Jyy) * (Vmx*Jxy + Vmy*Jyy)                                \
             + (Vpz + Wpx*Jxy + Wpy*Jyy) * Wmz                                                \
             + (Upz + Wpx*Jxx + Wpy*Jyx) * (Umx*Jxx + Umy*Jyx)                                \
             + (Wpx*Jxy + Wpy*Jyy + Vpz) * (Vmx*Jxy + Vmy*Jyy)                                \
             + 4/3*Wpz * Wmz

        Qmp =  4/3 * (Umx*Jxx + Umy*Jyx) * (Upx*Jxx + Upy*Jyx)                                \
             + (Umx*Jxy + Umy*Jyy + Vmx*Jxx + Vmy*Jyx) * (Vpx*Jxy + Vpy*Jyy)                  \
             + (Umz + Wmx*Jxx + Wmy*Jyx) * Wpz                                                \
             + (Umx*Jxy + Umy*Jyy + Vmx*Jxx + Vmy*Jyx) * (Upx*Jxx + Upy*Jyx)                  \
             + 4/3 * (Vmx*Jxy + Vmy*Jyy) * (Vpx*Jxy + Vpy*Jyy)                                \
             + (Vmz + Wmx*Jxy + Wmy*Jyy) * Wpz                                                \
             + (Umz + Wmx*Jxx + Wmy*Jyx) * (Upx*Jxx + Upy*Jyx)                                \
             + (Wmx*Jxy + Wmy*Jyy + Vmz) * (Vpx*Jxy + Vpy*Jyy)                                \
             + 4/3 * Wmz * Wpz


        NLT[:,:,p:T] =  lambp/Re/Pr * ( (Tpxx*Jxx**2 + 2*Tpxy*Jyx*Jxx + Tpyy*Jyx**2)          \
                                       +(Tpxx*Jxy**2 + 2*Tpxy*Jxy*Jyy + Tpyy*Jyy**2)          \
                                       + Tpzz )                                               \
                      + 1/Re/Pr * ( (lambpx*Jxx + lambpy*Jyx) * (Tpx*Jxx + Tpy*Jyx)           \
                                   +(lambpx*Jxy + lambpy*Jyy) * (Tpx*Jxy + Tpy*Jyy)           \
                                   + lambpz * Tpz)                                            \
                      + (gamma - 1)*Ma**2*(  Up * (Ppx*Jxx + Ppy*Jyx)                         \
                                           + Vp * (Ppx*Jxy + Ppy*Jyy) + Wp*Ppz)               \
                      + (gamma - 1)*Ma**2/Re*((mu + mup)*Qpp + mup*(Qmp + Qpm))               \
                      - rhop * Cp * (Tpt + (Um + Up) * (Tpx*Jxx + Tpy*Jyx)                    \
                                         + (Vm + Vp) * (Tpx*Jxy + Tpy*Jyy)                    \
                                         + (Wm + Wp) * Tpz                                    \
                                         + Up * (Tmx*Jxx + Tmy*Jyx)                           \
                                         + Vp * (Tmx*Jxy + Tmy*Jyy)                           \
                                         + Wp * Tmz)                                          \
                      - rho*Cp*(Up * (Tpx*Jxx + Tpy*Jyx) + Vp * (Tpx*Jxy + Tpy*Jyy) + Wp*Tpz)

        # NLT Physical space -> Spectral space
        Forcing = fft2(NLT.real, axes=(0,1),s=(M+padM,N+padN),norm='ortho')

        return Forcing

    def stabCoeff(self, m, n, dx, FDO, k=1):
        if (m,n) == (0,0):
            return 0#.1*k*dx**(FDO+1)/np.math.factorial(FDO)
        else:
            alpha_real = abs(self.alpha[m,n,max(self.i,self.i0)].real)
            if self.i<self.i0+2:
                k *= 10
            return k*max(0.5/abs(alpha_real)-dx, 0.5*dx**(FDO+1)/np.math.factorial(FDO))


    def formMatrix(self, m, n, LHS, RHS, itr, BDF=0, MFDcoupling=False):
        #-----------------------------------------------------------
        # Description
        # Construct the linear operator for the LST/LPSE/NPSE      
        #
        # LST
        # System solved : 
        #      (L + N*D + P*D^2)*phi = 0
        #
        # LPSE
        # System solved : 
        #      (L + N*D + P*D^2 + Q*da/dx)*phi = -M*dphi/dx
        #
        # NPSE
        # System solved : 
        #      (L + N*D + P*D^2 + Q*da/dx)*phi = -M*dphi/dx + NLT
        #
        #-----------------------------------------------------------

        # Station number
        i = self.i
        i0 = self.i0
        i0 = self.i0

        # Numerical methods
        Ny  = self.NumMethod.Ny
        D     = self.Dy
        D2    = self.D2y

        # Geometric Jacobian
        Jxx = self.NumMethod.Jxx[i,:]
        Jxy = self.NumMethod.Jxy[i,:]
        Jyx = self.NumMethod.Jyx[i,:]
        Jyy = self.NumMethod.Jyy[i,:]

        # Fluctuation parameter
        Pr     = self.Param['prop']['Pr']
        Cp     = self.Param['prop']['Cp']
        gamma  = self.Param['prop']['gamma']
        Ma     = self.Param['prop']['Ma']
        Ps     = self.Param['prop']['Ps']
        alpha  = self.alpha[m,n,i]
        alphax = self.alphax[m,n,i]

        # Laminar Flow
        Flow = self.Flow

        # Mean Flow Distorsion
        MFD = self.MFD
        if not self.Param[(m,n)]['linear'][i] and (m,n) != (0,0) and MFDcoupling:
            mfd = 1
        else:
            mfd = 0

        # Mode (0,0)
        if (m,n) == (0,0):
            Phatx = 0
        else:
            Phatx = 1

        # Properties
        # Density
        rho   = Flow.rho[i]

        # Viscosity
        mu    = Flow.mu[i]
        muT   = Flow.muT[i]
        muTT  = Flow.muTT[i]
        muy   = Flow.muy[i]
        if not self.Param[(m,n)]['parallel'][i]:
            mux   = Flow.mux[i]
        else:
            mux = 0
    
        # Cp and conductivity
        lamb   = Flow.lamb[i]
        lambT  = Flow.lambT[i]
        lambTT = Flow.lambTT[i]
        lamby  = Flow.lamby[i]
        if not self.Param[(m,n)]['parallel'][i]:
            lambx = Flow.lambx[i]
        else:
            lambx = 0

        # Base flow and derivative
        # Streamwise
        U   = Flow.U[i]   + mfd*MFD.U[i]
        Uyy = Flow.Uyy[i] + mfd*MFD.Uyy[i]
        Uy  = Flow.Uy[i]  + mfd*MFD.Uy[i]

        if not self.Param[(m,n)]['parallel'][i]:
            Ux  = Flow.Ux[i]  + mfd*MFD.Ux[i]
            Uxx = Flow.Uxx[i] + mfd*MFD.Uxx[i] 
            Uxy = Flow.Uxy[i] + mfd*MFD.Uxy[i]
        else:
            Ux  = mfd*MFD.Ux[i] 
            Uxx = mfd*MFD.Uxx[i]
            Uxy = mfd*MFD.Uxy[i]
    
        # Normal
        V   = Flow.V[i]   + mfd*MFD.V[i]
        Vyy = Flow.Vyy[i] + mfd*MFD.Vyy[i]
        Vy  = Flow.Vy[i]  + mfd*MFD.Vy[i]
    
        if not self.Param[(m,n)]['parallel'][i]:
            Vx  = Flow.Vx[i]  + mfd*MFD.Vx[i]
            Vxx = Flow.Vxx[i] + mfd*MFD.Vxx[i] 
            Vxy = Flow.Vxy[i] + mfd*MFD.Vxy[i]
        else:
            Vx  = mfd*MFD.Vx[i] 
            Vxx = mfd*MFD.Vxx[i]
            Vxy = mfd*MFD.Vxy[i]
    
        # Crossflow
        W   = Flow.W[i]   + mfd*MFD.W[i]
        Wyy = Flow.Wyy[i] + mfd*MFD.Wyy[i]
        Wy  = Flow.Wy[i]  + mfd*MFD.Wy[i]
    
        if not self.Param[(m,n)]['parallel'][i]:
            Wx  = Flow.Wx[i]  + mfd*MFD.Wx[i]
            Wxx = Flow.Wxx[i] + mfd*MFD.Wxx[i] 
            Wxy = Flow.Wxy[i] + mfd*MFD.Wxy[i]
        else:
            Wx  =  mfd*MFD.Wx[i] 
            Wxx =  mfd*MFD.Wxx[i]
            Wxy =  mfd*MFD.Wxy[i]
      
        # Pressure
        P     = Flow.P[i]  + mfd*MFD.P[i]
        Py    = Flow.Py[i] + mfd*MFD.Py[i]
    
        if not self.Param[(m,n)]['parallel'][i]:
            Px = Flow.Px[i] + mfd*MFD.Px[i]
        else:
            Px = mfd*MFD.Px[i]
      
        # Temperature
        T     = Flow.T[i]   + mfd*MFD.T[i] 
        Tyy   = Flow.Tyy[i] + mfd*MFD.Tyy[i]
        Ty    = Flow.Ty[i]  + mfd*MFD.Ty[i]
    
        if not self.Param[(m,n)]['parallel'][i]:
            Tx    = Flow.Tx[i]  + mfd*MFD.Tx[i]
            Txx   = Flow.Txx[i] + mfd*MFD.Txx[i] 
            Txy   = Flow.Txy[i] + mfd*MFD.Txy[i]
        else:
            Tx  = mfd*MFD.Tx[i]
            Txx = mfd*MFD.Txx[i] 
            Txy = mfd*MFD.Txy[i]

        # Scaling Mean flow
        if self.Param[(m,n)]['parallel'][i]:
            # Non-dim is local not at inlet
            Re = self.Param['prop']['Rex'][i]
            Re0    = self.Param['prop']['Re']
            beta   = self.Param[(m,n)]['beta'][i]
            omega  = self.Param[(m,n)]['omega'][i]

            scale = Re0/Re
            if ((m,n) == (0,0) or (m,n) == (0,1)) and itr==0:
                rho = self.NumMethod.MethodY.scaleFunction(scale, rho, updateL=True)
            else:
                rho = self.NumMethod.MethodY.scaleFunction(scale, rho)
            # Viscosity
            mu    = self.NumMethod.MethodY.scaleFunction(scale, mu)
            muT   = self.NumMethod.MethodY.scaleFunction(scale, muT)
            muTT  = self.NumMethod.MethodY.scaleFunction(scale, muTT)
            muy   = D @ mu
            if not self.Param[(m,n)]['parallel'][i]:
                mux = self.NumMethod.MethodY.scaleFunction(scale, mux)

            # Cp and conductivity
            lamb   = self.NumMethod.MethodY.scaleFunction(scale, lamb)
            lambT  = self.NumMethod.MethodY.scaleFunction(scale, lambT)
            lambTT = self.NumMethod.MethodY.scaleFunction(scale, lambTT)
            lamby  = D @ lamb
            if not self.Param[(m,n)]['parallel'][i]:
                lambx = self.NumMethod.MethodY.scaleFunction(scale, lamb)

            # Streamwise
            U = self.NumMethod.MethodY.scaleFunction(scale, U)
            Uy = D @ U
            Uyy = D2 @ U
            if not self.Param[(m,n)]['parallel'][i]:
                Ux = self.NumMethod.MethodY.scaleFunction(scale, Ux)
                Uxx = self.NumMethod.MethodY.scaleFunction(scale,Uxx)
                Uxy = self.NumMethod.MethodY.scaleFunction(scale,Uxy)
            # Normal
            V = self.NumMethod.MethodY.scaleFunction(scale, V)
            Vy = D @ V
            Vyy = D2 @ V
            if not self.Param[(m,n)]['parallel'][i]:
                Vx = self.NumMethod.MethodY.scaleFunction(scale, Vx)
                Vxx = self.NumMethod.MethodY.scaleFunction(scale,Vxx)
                Vxy = self.NumMethod.MethodY.scaleFunction(scale,Vxy)
            # Crossflow
            W = self.NumMethod.MethodY.scaleFunction(scale, W)
            Wy = D @ W
            Wyy = D2 @ W
            if not self.Param[(m,n)]['parallel'][i]:
                Wx = self.NumMethod.MethodY.scaleFunction(scale, Wx)
                Wxx = self.NumMethod.MethodY.scaleFunction(scale,Wxx)
                Wxy = self.NumMethod.MethodY.scaleFunction(scale,Wxy)
            # Pressure
            P = self.NumMethod.MethodY.scaleFunction(scale, P)
            Py = D @ P
            # Tempereature
            T = self.NumMethod.MethodY.scaleFunction(scale, T)
            Ty = D @ T
            Tyy = D2 @ T
            if not self.Param[(m,n)]['parallel'][i]:
                Tx = self.NumMethod.MethodY.scaleFunction(scale, Tx)
                Txx = self.NumMethod.MethodY.scaleFunction(scale,Txx)
                Txy = self.NumMethod.MethodY.scaleFunction(scale,Txy)

        else:
            Re     = self.Param['prop']['Rex'][i0]
            beta   = self.Param[(m,n)]['beta'][i0]
            omega  = self.Param[(m,n)]['omega'][i0]

        # Matrix L (L*phi)
        # x-momentum equation
        L11 = rho*((-omega + W*beta + (U*Jxx + V*Jxy)*alpha)*1j + (Jxx*Ux + Jxy*Uy)) + mu/Re*(alpha**2*(4/3*Jxx**2 + Jxy**2) + beta**2) - 4/3/Re*(Jxx*mux + Jyx*muy)*alpha*1j*Jxx
        L12 = rho*(Jxy*Ux + Jyy*Uy) + 1/3*alpha**2*mu/Re*(Jxx**2 + 2*Jxx*Jxy + Jxy**2) + 2/3/Re*(Jxx*mux + Jyx*muy)*Jxy*alpha*1j - 1/Re*(Jxy*mux + Jyy*muy)*alpha*1j*Jxx
        L13 = mu*beta*alpha*Jxx/3/Re - (Jxx*mux + Jyx*muy)/Re*2/3*beta*1j
        L14 = rho/P*(U*(Jxx*Ux + Jyx*Uy) + V*(Jxy*Ux + Jyy*Uy)) + alpha*1j*Jxx
        L15 = -rho/T*(U*(Jxx*Ux + Jyx*Uy) + V*(Jxy*Ux + Jyy*Uy)) - 1/Re*(muTT*(Jxx*Tx + Jyx*Ty) + muT*alpha*1j*Jxx)*(4/3*(Jxx*Ux + Jxy*Uy)-2/3*(Jyy*Vy + Jxy*Vx)) - 1/Re*(muTT*(Jxy*Tx + Jyy*Ty)+ muT*alpha*1j*Jxy)*(Jyy*Uy + Jxy*Ux + Jxx*Vx + Jxy*Vy) - 1/Re*muT*(4/3*(Jxx**2*Uxx + 2*Jxx*Jxy*Uxy + Jyx**2*Uy) + Jyy**2*Uyy + 2*Jyy*Jxy*Uxy + Jxy**2*Uxx + 1/3*(Jxx*Jxy*Vxx + Jyy*Jyx*Vyy + (Jxx*Jyy + Jxy**2)*Vxy))

        # y-momentum equation
        L21 = rho*(Jxx*Vx + Jyx*Vy) + 1/3*mu/Re*alpha**2*(Jxx**2+2*Jxx*Jxy+Jxy**2) - 1/Re*(Jxx*mux + Jxy*muy)*alpha*1j*Jxy - 2/3/Re*(Jxy*mux + Jyy*muy)*alpha*1j*Jxx
        L22 = rho*((-omega + W*beta + (U*Jxx + V*Jxy)*alpha)*1j  + Jxy*Vx + Jyy*Vy) + mu/Re*(alpha**2*(4/3*Jxy**2 + Jxx**2) + beta**2) - 1/Re*(Jxx*mux + Jxy*muy)*alpha*1j*Jxx - 4/3/Re*(Jxy*mux + Jyy*muy)*alpha*1j*Jxy
        L23 = mu/3/Re*alpha*beta*Jxy + 2*beta*1j/3/Re*(Jxy*mux + Jyy*muy)
        L24 = rho/P*(U*(Jxx*Vx + Jxy*Vy) + V*(Jxy*Vx + Jyy*Vy)) + alpha*1j*Jxy
        L25 = -rho/T*(U*(Jxx*Vx + Jxy*Vy) + V*(Jxy*Vx + Jyy*Vy)) - 1/Re*(muTT*(Jyx*Tx + Jyy*Ty) + muT*alpha*1j*Jxx)*(Jxx*Vx + Jxy*Vy + Jyy*Uy + Jxy*Ux) - 1/Re*(muTT*(Jxy*Tx + Jyy*Ty) + muT*alpha*1j*Jxy)*(4/3*(Jyy*Vy + Jxy*Vx) -2/3*( Jxx*Ux + Jxy*Uy)) - 1/Re*(4/3*(Jyy**2*Vyy + 2*Jyy*Jxy*Vxy + Jyx**2*Vxx) + Jxx**2*Vxx + 2*Jxx*Jxy*Vxy + Jxy**2*Vyy + 1/3*(Jxx*Jxy*Uxx + Jyy*Jyx*Uyy + (Jxx*Jyy + Jxy**2)*Uxy))

        # z-momentum equation
        L31 = rho*(Jxx*Wx + Jxy*Wy + 1j*beta*W) + mu*beta*alpha/3/Re*Jxx - beta*1j/Re*(Jxx*mux + Jxy*muy)
        L32 = rho*(Jxy*Wx + Jyy*Wy) + mu*beta*alpha*Jxy - beta*1j*(Jxy*mux + Jyy*muy)/Re
        L33 = rho*((-omega + W*beta + (U*Jxx + V*Jyx)*alpha)*1j) + mu/Re*(alpha**2*(Jxx**2 + Jxy**2) + 4/3*beta**2) - 1/Re*(Jxx*mux + Jxy*muy)*alpha*1j*Jxx - 1/Re*(Jxy*mux + Jyy*muy)*alpha*1j*Jxy
        L34 =  rho/P*(U*(Jxx*Wx + Jxy*Wy) + V*(Jxy*Wx + Jyy*Wy)) + beta*1j
        L35 = -rho/T*(U*(Jxx*Wx + Jxy*Wy) + V*(Jxy*Wx + Jyy*Wy)) - 1/Re*muT*alpha*1j*Jxx*(Jxx*Wx + Jxy*Wy) - 1/Re*muy*alpha*1j*Jxy*(Jxy*Wx + Jyy*Wy) - 1/Re*muT*(Jyy**2*Wyy + 2*Jyy*Jxy*Wxy + Jxy**2*Wxx + Jxx**2*Wxx + 2*Jxx*Jxy*Wxy + Jxy**2*Wyy)

        # continuity equation
        L41 =  P*T*alpha*1j*Jxx + T*(Jxx*Px + Jxy*Py) - P*(Jxx*Tx + Jxy*Ty)
        L42 =  T*(Jxy*Px + Jyy*Py) - P*(Jxy*Tx + Jyy*Ty)
        L43 =  P*T*beta*1j
        L44 =  T*(Jxx*Ux +Jyx*Uy + Jxy*Vx + Jyy*Vy + (-omega + (U*Jxx + V*Jxy)*alpha + W*beta)*1j) - U*(Jxx*Tx + Jyx*Ty) - V*(Jxy*Tx + Jyy*Ty)
        L45 = -P*(Jxx*Ux +Jyx*Uy + Jxy*Vx + Jyy*Vy + (-omega + (U*Jxx + V*Jxy)*alpha + W*beta)*1j) - U*(Jxx*Px + Jyx*Py) - V*(Jxy*Px + Jyy*Py)
        
        # energy equation
        L51 = rho*Cp*(Jxx*Tx + Jyx*Ty) - (gamma-1)*Ma**2*(Jxx*Px + Jyx*Py) - 2.*mu*(gamma-1)*Ma**2/Re*(((4/3*Jxx**2 + Jxy**2)*Ux + (4/3*Jxx*Jxy + Jyy*Jxy)*Uy +1/3*Jxx*Jxy*Vx + (Jxy**2-2/3*Jxx*Jyy)*Vy)*alpha*1j + (Jxx*Wx + Jxy*Wy)*beta*1j)
        L52 = rho*Cp*(Jxy*Tx + Jyy*Ty) - (gamma-1)*Ma**2*(Jxy*Px + Jyy*Py) - 2.*mu*(gamma-1)*Ma**2/Re*((1/3*Jxx*Jxy*Ux + (Jxx*Jyy-2/3*Jxy**2)*Uy + (Jxx**2 + 4/3*Jxy**2)*Vx + (Jxx*Jxy + 4/3*Jyy*Jxy)*Vy)*alpha*1j + (Jxy*Wx + Jyy*Wy)*beta*1j)
        L53 = 2.*mu*(gamma-1)/Re*(((Jxx**2 + Jxy**2)*Wx + (Jxx*Jyx + Jyy*Jxy)*Wy)*alpha*1j - 2/3*(Jxx*Ux + Jyx*Uy + Jxy*Vx + Jyy*Vy)*beta*1j)
        L54 =  Cp*rho/P*(U*(Jxx*Tx+Jyx*Ty) + V*(Jxy*Tx + Jyy*Ty)) - (gamma-1)*Ma**2*((U*Jxx + V*Jxy)*alpha + W*beta)*1j
        L55 = -Cp*rho/T*(U*(Jxx*Tx + Jyx*Ty) + V*(Jxy*Tx + Jyy*Ty)) + rho*Cp*((U*Jxx + V*Jxy)*alpha + W*beta - omega)*1j + lamb/Re/Pr*((Jxx**2 + Jxy**2)*alpha**2 + beta**2) - 2/Re/Pr*lambT*((Jxx**2 + Jxy**2)*Txx + 2*(Jxx*Jyx + Jyy*Jxy)*Txy + (Jyy**2+Jyx**2)*Tyy) + (gamma-1)*Ma**2/Re*muT*((4/3*(Jxx*Ux + Jyx*Uy) -2/3*(Jxy*Vx + Jyy*Vy))*(Jxx*Ux + Jyx*Uy) + (Jxy*Ux + Jyy*Uy + Jxx*Vx + Jyx*Vy)*(Jxy*Ux + Jyy*Uy) + (Jxx*Vx + Jyx*Vy + Jxy*Ux + Jyy*Uy)*(Jxx*Vx + Jyx*Vy) + (4/3*(Jxy*Vx + Jyy*Vy) - 2/3*(Jxx*Ux + Jyx*Uy))*(Jxy*Vx + Jyy*Vy) + (Jxx*Wx + Jxy*Wy)**2 + (Jxy*Wx + Jyy*Wy)**2) - 1/Re/Pr*lambTT*((Jxx*Tx + Jyx*Ty)**2 + (Jxy*Tx + Jyy*Ty)**2)

        # Matrix Q (Q*alpha_x)
        # x-momentum equation
        Q11 = -mu/Re*(4/3*Jxx + Jxy**2)*1j
        Q12 = -mu/Re*1/3*(Jxx**2+2*Jxx*Jxy*Jxy**2)*1j

        # y-momentum equation
        Q21 = -mu/Re*1/3*(Jxx**2 + 2*Jxx*Jxy + Jxy**2)*1j
        Q22 = -mu/Re*(4/3*Jxy**2 + Jxx**2)*1j

        # z-momentum equation
        Q33 = -mu/Re*(Jxx**2 + Jxy**2)*1j

        # energy equation
        Q55 = -lamb/Re/Pr*(Jxx**2 + Jxy**2)*1j

        # Matrix N (N*d /dy phi)
        # x-momentum equation
        N11 = diagflat(rho*(Jxy*U + Jyy*V) - 2/Re*alpha*1j*(4/3*Jxx*Jxy + Jxy**2) - 1/Re*(Jxx*mux + Jxy*muy)*4/3*Jyx - 1/Re*(Jxy*mux + Jyy*muy)*Jyy).dot(D)
        N12 = diagflat(-2/3*mu/Re*alpha*1j*(Jxx*Jyy + Jxy*(Jxx+Jyy)+Jxy**2) + 2/3/Re*(Jxx*mux + Jxy*muy)*Jyy - 2/3/Re*(Jxy*mux + Jyy*muy)*Jxy).dot(D)
        N14 = diagflat(Jxy).dot(D)
        N15 = diagflat(-1/Re*muT*(4/3*(Jxx*Ux + Jyx*Uy) - 2/3*(Jxy*Vx + Jyy*Vy))*Jyx - 1/Re*muT*(Jyy*Uy + Jxy*Ux + Jxx*Vx + Jyx*Vy)*Jyy).dot(D)

        # y-momentum equation
        N21 = diagflat(-2/3*mu/Re*alpha*1j*(Jxx*Jyy + Jxy*(Jxx+Jyy) + Jxy**2) - 1/Re*(Jxx*mux + Jyx*muy)*Jyy - 2/3/Re*(Jxy*mux + Jyy*muy)*Jyx).dot(D)
        N22 = diagflat(rho*(U*Jyx + V*Jyy) - 2*mu/Re*alpha*1j*(4/3*Jxy*Jyy + Jxx*Jyx) -1/Re*(Jxx*mux + Jyx*muy)*Jyx - 4/3/Re*(Jxy*mux + Jyy*muy)*Jyy).dot(D)
        N23 = diagflat(-mu/Re*beta*1j/3*Jyy).dot(D)
        N24 = diagflat(Jyy).dot(D)
        N25 = diagflat(-1/Re*muT*(Jxx*Vx + Jxy*Vy + Jyy*Uy + Jxy*Ux)*Jyx - 1/Re*muT*(4/3*(Jyy*Vy + Jxy*Vx) - 2/3*(Jxx*Ux + Jyx*Uy))).dot(D)

        # z-momentum equation
        N31 = diagflat(-1/3*mu/Re*beta*1j*Jyx).dot(D)
        N32 = diagflat(-1/3*mu/Re*beta*1j*Jyy).dot(D)
        N33 = diagflat(rho*(Jyx*U + Jyy*V) -2*mu/Re*alpha*1j*Jxx*Jyx -1/Re*(Jxx*mux + Jyx*muy)*Jyx - 1/Re*(Jxy*mux + Jyy*muy)*Jyy).dot(D)
        N35 = diagflat(-1/Re*muT*(Jxx*Wx + Jxy*Wy)*Jyx - 1/Re*muT*(Jyy*Wy + Jxy*Wx)*Jyy).dot(D)

        # continuity equation
        N41 = diagflat(P*T*Jyx).dot(D)
        N42 = diagflat(P*T*Jyy).dot(D)
        N44 = diagflat( T*(U*Jyx + V*Jyy)).dot(D)
        N45 = diagflat(-P*(U*Jyx + V*Jyy)).dot(D)

        # energy equation
        N51 = diagflat(2.*mu*(gamma-1)/Re*((4/3*Jxx*Jyx + Jyy*Jxy)*Ux + (4/3*Jyx**2 + Jyy**2)*Uy + (Jxx*Jyy - 2/3*Jxy**2)*Vx + 1/3*Jyy*Jxy*Vy)).dot(D)
        N52 = diagflat(2.*mu*(gamma-1)/Re*((Jyx**2 - 2/3*Jyy*Jxx)*Ux + 1/3*Jyy*Jyx*Uy + (Jxx*Jyx + 4/3*Jxy*Jyy)*Vx + (Jxy**2 + 4/3*Jyy**2)*Vy)).dot(D)
        N53 = diagflat(2.*mu*(gamma-1)/Re*((Jxx*Jyx + Jyy*Jxy)*Wx + (Jyx**2 + Jyy**2)*Wy)).dot(D)
        N54 = diagflat(-(gamma-1)*Ma**2*(U*Jyx + V*Jyy)).dot(D)
        N55 = diagflat(rho*Cp*(U*Jyx + V*Jyy) - 2*lamb/Re/Pr*alpha*1j*(Jxx*Jyx + Jyy*Jxy) - 2/Re/Pr*(Jxx*lambx + Jyx*lamby)*Jyx - 2/Re/Pr*(Jxy*lambx + Jyy*lamby)*Jyy).dot(D)

        # Matrix P (P*d2/dy2 phi)
        # x-momentum equation
        P11 = diagflat(-mu/Re*(4/3*Jyx**2 + Jyy**2)).dot(D2)
        P12 = diagflat(-mu/Re*2/3*Jyy*Jyx).dot(D2)

        # y-momentum equation
        P21 = diagflat(-mu/Re*2/3*Jyy*Jyx).dot(D2)
        P22 = diagflat(-mu/Re*(4/3*Jyy**2 + Jyx**2)).dot(D2)

        # z-momentum equation
        P33 = diagflat(-mu/Re*(Jyy**2 + Jyx**2)).dot(D2)

        # energy equation
        P55 = diagflat(-lamb/Re/Pr*(Jyy**2 + Jyx**2)).dot(D2)

        # Matrix M (M*d/dx phi)
        # x-momentum
        M11 = diagflat(rho*(U*Jxx + V*Jxy) - mu/Re*2*alpha*1j*(4/3*Jxx**2 + Jxy**2) -1/Re*(Jxx*mux + Jyx*muy)*4/3*Jxx - 1/Re*(Jxy*mux + Jyy*muy)*Jxy)
        M12 = diagflat(-2/3*mu/Re*(alpha*1j*(Jxx**2 + 2*Jxx*Jxy + Jxy**2)) + 2/3/Re*(Jxx*mux + Jyx*muy)*Jxy - 1/Re*(Jxy*mux + Jyy*muy)*Jxx)
        M13 = diagflat(-1/3*mu/Re*beta*1j*Jxx)
        M14 = diagflat(Jxx) * Phatx
        M15 = diagflat(-1/Re*muT*(4/3*(Jxx*Ux + Jyx*Uy) - 2/3*(Jxy*Vx + Jyy*Vy))*Jxx -1/Re*muT*(Jxy*Ux + Jyy*Uy + Jxx*Vx + Jyx*Vy)*Jxy)
                    
        # y-momentum
        M21 = diagflat(-2/3*mu/Re*alpha*1j*(Jxx**2 + 2*Jxx*Jxy + Jxy**2) - 1/Re*(Jxx*mux + Jyx*muy)*Jxy - 1/Re*(Jxy*mux + Jyy*muy)*Jxx)
        M22 = diagflat(rho*(Jxx*U + Jxy*V) - 2*mu/Re*alpha*1j*(4/3*Jxy**2 + Jxx**2) -1/Re*(Jxx*mux + Jyx*muy)*Jxx - 1/Re*(Jxy*mux + Jyy*muy)*4/3*Jxy)
        M23 = diagflat(-1/3*mu/Re*beta*1j*Jxy)
        M24 = diagflat(Jxy) * Phatx
        M25 = diagflat(-1/Re*muT*(4/3*(Jxy*Vx + Jyy*Vy) - 2/3*(Jxx*Ux + Jyx*Uy))*Jxy -1/Re*muT*(Jxx*Vx + Jyx*Vy + Jxy*Ux + Jyy*Uy)*Jxx)
                 
        # z-momentum
        M31 = diagflat(-1/3*mu/Re*beta*1j*Jxx)
        M32 = diagflat(-1/3*mu/Re*beta*1j*Jxy)
        M33 = diagflat(rho*(Jxx*U + Jxy*V) - 2*mu/Re*alpha*1j*Jxx**2 - 1/Re*(Jxx*mux + Jyx*muy)*Jxx - 1/Re*(Jxy*mux + Jyy*muy)*Jxy)
        M34 = diagflat([0.]*Ny)
        M35 = diagflat(-1/Re*muT*(Jxx*Wx + Jxy*Wy)*Jxx - 1/Re*muT*(Jxy*Wx + Jyy*Wy)*Jxy)
                    
        # continuity
        M41 = diagflat(P*T*Jxx)
        M42 = diagflat(P*T*Jxy)
        M43 = diagflat([0.]*Ny)
        M44 = diagflat( T*(U*Jxx + V*Jxy)) * Phatx
        M45 = diagflat(-P*(U*Jxx + V*Jxy))
                    
        # energy equation
        M51 = diagflat(2.*mu*(gamma-1)*Ma**2/Re*((4/3*Jxx**2 + Jxy**2)*Ux + (4/3*Jxx*Jxy + Jyy*Jxy)*Uy + 1/3*Jxx*Jxy*Vx + (Jxy**2 - 2/3*Jxx*Jyy)*Vy))
        M52 = diagflat(2.*mu*(gamma-1)*Ma**2/Re*(1/3*Jxx*Jxy*Ux + (Jxx*Jyy - 2/3*Jxy**2)*Uy + (Jxx**2 + 4/3*Jxy**2)*Vx + (Jxy**2 + 4/3*Jyy**2)*Vy))
        M53 = diagflat(2.*mu*(gamma-1)*Ma**2/Re*((Jxx**2 + Jxy**2)*Wx + (Jxx*Jyx + Jyy*Jxy)*Wy))
        M54 = diagflat(-(gamma-1)*Ma**2*(Jxx*U + Jxy*V)) * Phatx
        M55 = diagflat(rho*Cp*(Jxx*U + Jxy*V) - 2*lamb*Re/Pr*alpha*1j*(Jxx**2 + Jxy**2) - 2/Re/Pr*(Jxx*lambx + Jyx*lamby)*Jxx - 2/Re/Pr*(Jxy*lambx + Jyy*lamby)*Jxy)

        # H (d2/dxdy phi)
        # x-momentum
        H11 = diagflat(-2*mu/Re*(4/3*Jxx*Jyx + Jyy*Jxy)).dot(D)
        H12 = diagflat(-2/3*mu/Re*(Jxx*Jyy + Jxy**2)).dot(D)

        # y-momentum
        H21 = diagflat(-2/3*mu/Re*(Jxx*Jyy + Jxy**2)).dot(D)
        H22 = diagflat(-2*mu/Re*(4/3*Jyy*Jxy + Jxx*Jyx)).dot(D)

        # z-momentum
        H33 = diagflat(-2*mu/Re*(Jyy*Jxy + Jxx*Jyx)).dot(D)

        # Energy
        H55 = diagflat(-2*lamb/Re/Pr*(Jxx*Jyx + Jyy*Jxy)).dot(D)


        # Assembling L matrix
        L11 = diagflat(L11 + Q11*alphax)
        L12 = diagflat(L12 + Q12*alphax)
        L13 = diagflat(L13)
        L14 = diagflat(L14)
        L15 = diagflat(L15)

        L21 = diagflat(L21 + Q21*alphax)
        L22 = diagflat(L22 + Q22*alphax)
        L23 = diagflat(L23)
        L24 = diagflat(L24)
        L25 = diagflat(L25)

        L31 = diagflat(L31)
        L32 = diagflat(L32)
        L33 = diagflat(L33 + Q33*alphax)
        L34 = diagflat(L34)
        L35 = diagflat(L35)

        L41 = diagflat(L41)
        L42 = diagflat(L42)
        L43 = diagflat(L43)
        L44 = diagflat(L44)
        L45 = diagflat(L45)

        L51 = diagflat(L51)
        L52 = diagflat(L52)
        L53 = diagflat(L53)
        L54 = diagflat(L54)
        L55 = diagflat(L55 + Q55*alphax)

        # Assembling Global LHS
        MatHM  = np.vstack([ np.hstack([M11 + H11, M12 + H12, M13      , M14, M15      ]), \
                             np.hstack([M21 + H21, M22 + H22, M23      , M24, M25      ]), \
                             np.hstack([M31      , M32      , M33 + H33, M34, M35      ]), \
                             np.hstack([M41      , M42      , M43      , M44, M45      ]), \
                             np.hstack([M51      , M52      , M53      , M54, M55 + H55]), \
                          ])

        # Assembling Global LHS
        LHS  = np.vstack([ np.hstack([L11 + N11 + P11, L12 + N12 + P12, L13            , L14 + N14, L15 + N15      ]), \
                                np.hstack([L21 + N21 + P21, L22 + N22 + P22, L23 + N23      , L24 + N24, L25 + N25      ]), \
                                np.hstack([L31            , L32 + N32      , L33 + N33 + P33, L34      , L35 + N35      ]), \
                                np.hstack([L41 + N41      , L42 + N42      , L43            , L44 + N44, L45 + N45      ]), \
                                np.hstack([L51 + N51      , L52 + N52      , L53 + N53      , L54 + N54, L55 + N55 + P55]), \
                              ])

        # Assembling Global RHS
        self.L[m,n] = LHS.copy()

        # RHS term, when LPSE
        RHS += -MatHM.dot(self.phix_0[m,n,i,:])
        LHS += MatHM*BDF

        return LHS, RHS

    @staticmethod
    def fit(x, f, m=1, itr=0):
        if x[1]-x[0] ==0:
            if itr==1:
                fp = (f[1] - f[0])/(x[1]-x[0])
            else:
                fp = 1 ; m=0
        else:
            fp = (f[1] - f[0])/(x[1]-x[0])
        return x[1] - m*f[1]/fp

    def updateAlpha(self, m, n, itr, s=0, relax=1, alphatol=1e-5):
        #-----------------------------------------------------------
        # Description
        # Find alpha to satisfy the normalization condition.
        # The normalization condition is enforced using a simple 
        # Newton's method.
        #
        #  For the LST:
        #  Norm. Cond.:
        #      Simply v(0) = 0 (for every mode)
        #
        #
        #  For the LPSE/NPSE :
        #  Norm. Cond. : _cc means complex conjugate
        #                _i meands a summation over velocity components
        #
        #      /  
        #      |   
        #      |  u_i(m,n)_cc * du_i/dx(m,n) dy = 0
        #      | 
        #      / 
        #               
        #-----------------------------------------------------------

        # Numerical Methods
        Ny = self.NumMethod.Ny
        IntW = self.w

        # Station number
        i  = self.i
        i0  = self.i0

        # Vector offset
        u = 0
        v = 1
        w = 2
        p = 3
        T = 4

        # Normalization condition
        Norm  = 0.j

        if self.Param[(m,n)]['local'][i]:
           # Offset
            y0 = v * Ny
            # eigen BC
            Norm = self.phi[m,n,i,y0]

        elif not self.Param[(m,n)]['local'][i]:
            if itr > 1 and s>0:
                #phix = self.LHS[m,n,:,:].dot(self.phi[m,n,i]+s*self.phix[m,n])# + s*self.phixx[m,n] #self.L[m,n,:,:].dot(self.phix[m,n,i])
                phix = self.phix[m,n,i]# + s*self.phi_st[m,n] #self.L[m,n,:,:].dot(self.phix[m,n,i])
            else:
                phix = self.phix[m,n,i]
            for j in [u,v,w,p,T]:
                # Offset
                y0 = j * Ny
                ym = y0 + Ny
                
                # Normalization condition
                q    = self.phi[m,n,i,y0:ym]
                qc   = np.conj(self.phi[m,n,i,y0:ym])
                qx   = phix[y0:ym]
                Norm += (qc*qx).dot(IntW)


        self.NormCond[m,n,i,itr,1] = self.alpha[m,n,i]
        self.NormCond[m,n,i,itr,0] = Norm

        # Updating alpha
        if itr < 1:
            self.alpha[m,n,i] *= (1+1e-3)
        else:
            if (m,n) == (0,0):
                self.alpha[m,n,i] = 0j
            else:
                self.alpha[m,n,i] = self.fit(self.NormCond[m,n,i,itr-1:itr+1,1], self.NormCond[m,n,i,itr-1:itr+1,0],itr=itr, m=relax)

#        #print(self.alpha[m,n,i].imag, self.alphax[m,n,i-1].imag)
#        if self.alpha[m,n,i].imag > 1.5e-2 and i-i0 > 5:
#            print('Mode '+str((m,n))+' removed. Reason: too stable')
#            self.Param['trash'].add((m,n))
#            self.emptyTrash()

    def setBC(self, m, n, LHS, RHS, itr):
        #-----------------------------------------------------------
        # Description
        # Apply the boundary conditions (BC) to the system)
        # 
        # BC:
        #    LST
        #        At the wall
        #               u(0) = w(0) = T(0) = 0
        #               p(0) = Dirichlet
        #        
        #        Free-stream
        #               u(oo)  = w(oo) = p(oo) = T(oo) = 0
        #               vy(oo) = 0
        #
        #               
        #    LPSE/NPSE 
        #        At the wall
        #               u(0) = v(0) = w(0) = T(0) = 0
        #               
        #        Free-stream
        #               uy(oo)  = wy(oo) = py(oo) = Ty(00) = 0
        #               vy(oo) = 0
        #
        #    Note: - BC on p is relaxed for the LPSE/NPSE
        #          - BC on v is different at y -> oo to account
        #            for V(oo) /= 0
        #-----------------------------------------------------------

        # Set the boundary condition
        # Station number
        i  = self.i

        # Differentiation Matrix
        D  = np.kron(np.eye(5),  self.Dy)

        # Length
        Ny = self.NumMethod.Ny

        # Vector offset
        u = 0
        v = 1
        w = 2
        p = 3
        T = 4

        # Boundary conditions
        # Loop on the variables
        for j in [u,v,w,p,T]:
            # offset (y0 is wall, ym is free-stream)
            y0 = j * Ny
            ym = y0 + Ny - 1

            if j == p:
                if self.Param[(m,n)]['local'][i]:# and (m,n) != (0,0):
                    # Wall
                    LHS[y0,:]  = 0
                    LHS[y0,y0] = 1
                    RHS[y0]    = self.amplitude[m,n,i]
                else:
                    LHS[y0,:]  = D[y0,:]
                    RHS[y0]    = 0
                # Free-Stream
                if (m,n) == (0,0):
                    LHS[ym,:]  = D[ym,:]
                    RHS[ym]    = 0
                else:
                    LHS[ym,:]  = 0
                    LHS[ym,ym] = 1
                    RHS[ym]    = 0

            elif j == v:
                # Wall
                if not self.Param[(m,n)]['local'][i]:# or (m,n) == (0,0):
                    LHS[y0,:]  = 0
                    LHS[y0,y0] = 1
                    RHS[y0]    = 0
                # Free-Stream
                if (m,n) == (0,0):
                    LHS[ym,:]  = D[ym,:]
                    RHS[ym]    = 0
                else:
                    LHS[ym,:]  = 0
                    LHS[ym,ym] = 1
                    RHS[ym]    = 0

            else:
                # Wall
                LHS[y0,:]  = 0
                LHS[y0,y0] = 1
                RHS[y0]    = 0
                # Free-Stream
                if (m,n) == (0,0):
                    LHS[ym,:]  = D[ym,:]
                    RHS[ym]    = 0
                else:
                    LHS[ym,:]  = 0
                    LHS[ym,ym] = 1
                    RHS[ym]    = 0

        return LHS, RHS

    def spectralBroadening(self, NLT, itr):
        # Add/remove modes based on forcing term
    
        # Number of points (x,y)
        Nx = self.NumMethod.Nx
        Ny = self.NumMethod.Ny
        p0 = Ny*3

        # Maximum number of modes
        (M,N) = self.Param['(M,N)max']

        # Current Max
        Ncurr = 0 ; Mcurr = 0
        for (m,n) in self.Param['modes'][self.i]:
            if m > Mcurr:
                Mcurr = m
            if n > Ncurr:
                Ncurr = n

        # Checking forcing amplitude
        for m,n in [(mm,nn) for nn in range(N) for mm in range(M)]:
            if (m,n) not in self.Param['modes'][self.i+1]:
                thresh = self.Param[(m,n)]['A_0']
                if np.max(abs(NLT[m,n,:])) > thresh:
                    A_0 = interp1d(np.arange(Ncurr+1), self.amplitude[m,:Ncurr+1,self.i].real, bounds_error=False, fill_value='extrapolate')(n)
                    A_h = 1e-2*interp1d(np.arange(Ncurr+1), self.phi[m,:Ncurr+1,self.i,p0].real, bounds_error=False, fill_value='extrapolate')(n)
                    print('Adding mode '+str((m,n))+': '+str(A_0)+','+str(A_h))
                    self.Param[(m,n)]['A_0'] = A_0
                    self.Param['modes'][self.i+1].append((m,n))
                    self.Param[(m,n)]['local'][self.i+1] = True
                    self.Param[(m,n)]['linear'][self.i+1] = False
                    self.amplitude[m,n,self.i+1] = A_h
                    self.alpha[m,n,self.i] = self.Param[(m,n)]['alpha0'][0]
                    for i in range(self.i+2,Nx):
                        self.Param['modes'][i].append((m,n))
                        self.Param[(m,n)]['local'][i] = False
                        self.Param[(m,n)]['linear'][i] = False

    def emptyTrash(self):
        # Remove modes that are too stable and may lead to unstable system

        # Number of points (x,y)
        Nx = self.NumMethod.Nx

        # Checking if something in trash can
        for m,n in self.Param['trash']:
            if (m,n) in self.Param['modes'][self.i]:
                for i in range(self.i, Nx):
                    self.Param['modes'][i].remove((m,n))
                    self.Param['modes'][i].sort(key=lambda mode: mode[1])

        # Emptying trash can
        self.Param['trash'] = set()

    def computeAmplitude(self, itr, relax=1):
        # Compute fluctuation amplitude
        for m,n in self.Param['modes'][self.i]:
            if (m,n) == (0,0):
                self.amplitude[m,n,self.i] = 1
            else:
                if self.i > self.i0:
                    if itr>0:
                        self.amplitude[m,n,self.i] = (1-relax)*self.amplitude[m,n,self.i]+relax*np.exp(simps(-self.alpha[m,n,self.i0:self.i+1].imag, self.NumMethod.X1D[self.i0:self.i+1])).real
                    else:
                        self.amplitude[m,n,self.i] = self.amplitude[m,n,self.i-1]
                else:
                    self.amplitude[m,n,self.i] = 1

    def dealiasing(self,m,n,f):
        Ny     = self.NumMethod.Ny
        Neq    = self.NumMethod.Neq
        icr    = Ny * Neq
        #from matplotlib import pyplot as plt
        for e in range(Neq):
            v0 = e*Ny
            vf = v0 + Ny
            f[v0+1:vf:2] = self.NumMethod.MethodY.evalFunction(self.y[1::2], f[v0:vf],Dealiasing=True)
        return f

    def Solve(self):
        #-----------------------------------------------------------
        # Description
        # Call the relevant function to build, apply BC, compute 
        # NLT terms and solve ths system. The marching procedure
        # performed using backward differentiation formula (BDF)
        # The Normalization condition is enforced locally, at every
        # station.
        #
        # Summary
        #
        # LST:
        #
        # Loop on modes (m,n):
        #   While Norm. Cond. > epsilon:
        #        solve Ay = b
        #        update alpha
        #
        # LPSE:
        #
        # Loop on modes (m,n):
        #   While Norm. Cond. > epsilon:
        #        solve Ay = b + dy/dx
        #        update alpha
        #
        # NPSE:
        # 
        # Loop on modes (m,n):
        #   While Norm. Cond. > epsilon:
        #        While NLres > tol:
        #           compute NLT (function of y)
        #           solve Ay = b + dy/dx + NLT
        #        update alpha
        #
        #-----------------------------------------------------------
        # Dealiasing
        Dealiasing = False
        MFDcoupling = False

        # Current station number
        i  = self.i
        i0 = self.i0

        # Numerical Method
        Nx = self.NumMethod.Nx
        Ny = self.NumMethod.Ny
        Neq = self.NumMethod.Neq
        FDOX = self.NumMethod.FDOX
        BDF = 0 ; b1 = i0 - 1

        # dx for stabilization
        ii = min(i+1, Nx-1)
        dx = 0 #self.x[ii] - self.x[ii-1]

        # Tolerance
        Converged = False
        tol       = 1e-6
        alphatol  = 1e-4
        NLtol     = 1e-3
        relax     = 1.0

        # Iteration number
        itr       = 0
        minItr    = 3

        # Initial Guess for alpha_(m,n) and phi
        for m,n in self.Param['modes'][i]:
            if i == i0 or self.Param[(m,n)]['local'][i]:
                self.alpha[m,n,i] = self.Param[(m,n)]['alpha0'][i]
            else:
                if True: #i == i0+1:
                    self.alpha[m,n,i] = self.alpha[m,n,i-1]
                else:
                    self.alpha[m,n,i]  = interp1d(self.x[i-2:i], self.alpha[m,n,i-2:i].imag, bounds_error=False, fill_value='extrapolate')(self.x[i])*1j
                    self.alpha[m,n,i] += interp1d(self.x[i-2:i], self.alpha[m,n,i-2:i].real, bounds_error=False, fill_value='extrapolate')(self.x[i])

        # Initial amplitude
        self.computeAmplitude(0)

        # Initialization
        dalpha = np.zeros_like(self.alpha_old)
        res    = np.zeros_like(self.alpha_old)

        # System solve
        while not Converged:
            SubItr = 0
            NLConverged = False

            # Loop on Nonlinear iterations
            while not NLConverged:
                # Updating Mean Flow Distorsion (MFD)
                if MFDcoupling:
                    self.updateMFD()

                # Updating amplitude
                self.computeAmplitude(itr)
                self.computeDKE()

                # Computing Nonlinear forcing terms
                for m,n in self.Param['modes'][i]:
                    if not self.Param[(m,n)]['linear'][i]:
                        NLT = self.forcingTerm(itr, MFDcoupling=MFDcoupling)
                        break

                # Loop on spanwise and temporal modes
                for m,n in self.Param['modes'][i]:
                    # Check if mode is already converged
                    if self.Param[(m,n)]['linear'][i] and itr>minItr and res[m,n] < tol and dalpha[m,n] < alphatol:
                        continue
                    # Mode initial amplitude and history
                    A_0 = self.Param[(m,n)]['A_0']
                    if (m,n) != (0,0):
                        A_h = self.amplitude[m,n,i]
                    else:
                        A_h = 1

                    # initializing operator
                    LHS = np.zeros((Neq*Ny, Neq*Ny), dtype=complex)
                    RHS = np.zeros(Neq*Ny, dtype=complex)

                    # Adding forcing term
                    if not self.Param[(m,n)]['linear'][i]:
                        # Dealiasing
                        if Dealiasing:# or (m,n) == (0,0):
                            NLT[m,n,:] = self.dealiasing(m,n,NLT[m,n,:])
                        RHS = NLT[m,n,:] / (A_h * A_0)
                    # PSE
                    if not self.Param[(m,n)]['local'][i]:
                        # Euler Coefficient for the previous solutions
                        self.phix_0[m,n,i] = 0
                        b1 = max(i-FDOX, i0)

                        # Lagrange Coeff
                        dCoeff = dLagrange(self.x[b1:i+1], self.x[i])
                        dx = 1/np.mean(abs(dCoeff))

                        for k in range(len(dCoeff)-1):
                            self.phix_0[m,n,i] += dCoeff[k]*self.phi[m,n,b1+k]/A_0

                        # Backward differentiation formula
                        BDF = dCoeff[-1]

                    # Assembling system
                    LHS, RHS = self.formMatrix(m, n, LHS, RHS, itr, BDF=BDF, MFDcoupling=MFDcoupling)

                    # Stabilization
                    s = self.stabCoeff(m,n,dx, FDOX, k=self.NumMethod.stabCoeff)

                    if not self.Param[(m,n)]['local'][i]:
                        LHS +=  s*self.L[m,n]*BDF
                        RHS += -s*self.L[m,n].dot(self.phix_0[m,n,i])

                    # Boundary conditions
                    LHS, RHS = self.setBC(m, n, LHS, RHS, itr)

                    # Backing up last solution
                    self.phi_old[m,n] = self.phi[m,n,i]

                    try:
                        # Solving system
                        self.phi[m,n,i] = A_0*np.linalg.solve(LHS, RHS)

                        if Dealiasing:#or (m,n) == (0,0):
                            self.phi[m,n,i,:]  = self.dealiasing(m,n, self.phi[m,n,i,:])
                        
                        # Relaxation (NL problem, default is no relaxation)
                        if not self.Param[(m,n)]['linear'][i] and relax != 1:
                            self.phi[m,n,i,:] = relax*self.phi[m,n,i,:] + (1-relax)*self.phi_old[m,n,:]
                        
                        # Updating Stream derivative
                        if not self.Param[(m,n)]['local'][i]:
                            # Euler Coefficient for the previous solutions
                            self.alphax[m,n,i] = 0
                            self.phix[m,n,i] = 0
                            self.phi_st[m,n,i] = 0
                            b1 = max(i-FDOX, i0)
                            b2 = max(i-max(FDOX,2), i0)
                            dCoeff = dLagrange(self.x[b1:i+1], self.x[i])
                            for k in range(len(dCoeff)):
                                self.phix[m,n,i] += dCoeff[k]*self.phi[m,n,b1+k]/A_0
                                self.alphax[m,n,i] += dCoeff[k]*self.alpha[m,n,b1+k]

                            if FDOX == 1:
                                Coeff = d2Lagrange(self.x[b2:i+1], self.x[i])
                            elif FDOX == 2:
                                Coeff = d3Lagrange(self.x[b2:i+1], self.x[i])
                            elif FDOX == 3:
                                Coeff = d4Lagrange(self.x[b2:i+1], self.x[i])

                            for k in range(len(dCoeff)):
                                self.phi_st[m,n,i] += Coeff[k]*self.phi[m,n,b1+k]/A_0

                            if Dealiasing:# or (m,n) == (0,0):
                                self.phix[m,n,i]  = self.dealiasing(m,n, self.phix[m,n,i])

                        if Dealiasing:#or (m,n) == (0,0):
                            self.phi[m,n,i]  = self.dealiasing(m,n, self.phi[m,n,i])
                        
                        # Relaxation (NL problem, default is no relaxation)
                        if not self.Param[(m,n)]['linear'][i] and relax != 1:
                            self.phi[m,n,i,:] = relax*self.phi[m,n,i,:] + (1-relax)*self.phi_old[m,n,:]
                        
                    except np.linalg.LinAlgError:
                        os.system('touch break.it')
                        print('Something went wrong')
                        NLConverged = True
                        Converged = True
                        break

                # Evaluating if NL problem converged
                if not self.Param[(m,n)]['linear'][i]:
                    L2Norm = np.linalg.norm(self.phi[:,:,i,:]-self.phi_old[:,:,:])/np.linalg.norm(self.phi[:,:,i,:])
                    if L2Norm < NLtol and SubItr > 1:
                        NLConverged = True
                    else:
                        SubItr += 1
                        if SubItr > 10:
                            print('NLT not converged')
                            NLConverged = True
                else:
                    NLConverged = True
               

            # Updating alpha
            if itr>0:
                resMax = np.max(res)
            else:
                resMax = -np.inf

            for m,n in self.Param['modes'][i]:
                if self.Param[(m,n)]['linear'][i] and itr>minItr and res[m,n] < tol and dalpha[m,n] < alphatol:
                    continue
                # Updating alpha and its derivative
                self.alpha_old[m,n] = self.alpha[m,n,i]
                if self.Param[(m,n)]['local']:
                    self.updateAlpha(m, n, itr, relax=0.8, alphatol=alphatol)
                else:
                    self.updateAlpha(m, n, itr, s=s, relax=0.8, alphatol=alphatol)
                dalpha[m,n] = abs((self.alpha_old[m,n] - self.alpha[m,n,i]))
                if np.isnan(dalpha[m,n]):
                    os.system('touch plot.it')
                    break

                # Normalization condition
                if (m,n) != (0,0):
                    res[m,n] = abs(self.NormCond[m,n,i,itr,0])
                resMax = max(resMax,res[m,n])
            dalphaMax = np.amax(abs(dalpha))

            # Checking if normalization condition is satisfied
            if (resMax < tol and dalphaMax < alphatol and itr > minItr) or dalphaMax < 1e-9:
                Converged = True

                # Spectral Broadening
                if not self.Param[(0,0)]['linear'][i]:
                    if i<Nx-3:
                        self.spectralBroadening(NLT,itr)

                itr += 1

            elif itr > 100:
                Converged = True
                itr += 1

            elif os.path.isfile('stop.now'):
                Converged = True
                itr +=1
                print('Stopping simulation now (requested by user)')
                os.remove('stop.now')

            elif itr >self.NormCond.shape[3]-1:
                print('-----------------------------------------')
                print('STATUS : Diverged')
                print('-----------------------------------------')
                break
            else:
                itr += 1

            # Output iteration information
            self.printResidual(itr, resMax.real, dalphaMax.real)

        # Updating amplitude
        self.computeAmplitude(itr)
        self.computeDKE()

        if os.path.isfile('plot.it'):
            print('Alright, I show you..')
            os.remove('plot.it')
            from matplotlib import pyplot as plt
            plt.subplot(311)
            for m,n in self.Param['modes'][i]:
                plt.plot(self.x[i0:i+1], self.amplitude[m,n,i0:i+1], label=str((m,n)))
            plt.legend()
            plt.xlabel(r'$x/\delta$')
            plt.ylabel(r'$A/A_0$')
            plt.subplot(312)
            for m,n in self.Param['modes'][i]:
                plt.semilogy(self.x[i0:i+1], self.DKE[m,n,i0:i+1], label=str((m,n)))
            plt.ylim([1e-14, 1e-4])
            plt.legend()
            plt.xlabel(r'$x/\delta$')
            plt.ylabel(r'$DKE$')
            plt.subplot(313)
            for m,n in self.Param['modes'][i]:
                plt.semilogy(self.x[i0:i+1], self.Param[(m,n)]['A_0']*self.amplitude[m,n,i0:i+1], label=str((m,n)))
            plt.grid()
            plt.xlabel(r'$x/\delta$')
            plt.ylabel(r'$A$')
            plt.ylim([1e-12, 1])
            plt.legend()
            plt.show()
            plt.plot
            plt.plot(self.Flow.U[i], self.y, label=r'U')
            plt.plot(self.Flow.V[i], self.y, label=r'V')
            plt.plot(self.Flow.W[i], self.y, label=r'W')
            plt.plot(self.Flow.P[i], self.y, label=r'P')
            plt.plot(self.Flow.T[i], self.y, label=r'T')
            plt.show()
            var = ['u', 'v', 'w', 'p', 'T']
            for eq in range(Neq):
                for m,n in self.Param['modes'][i]:
                    variable = r'$\hat{%s}$'%(var[eq])
                    plt.plot(self.phi[m,n,i,eq*Ny:(eq+1)*Ny], self.y,  '-', label=variable+str((m,n)))
                    try:
                        plt.plot(NLT[m,n,eq*Ny:(eq+1)*Ny], self.y, label='Forcing term '+var[eq]+' '+str((m,n)))
                    except:
                        pass
                xlabel = r'$\hat{%s}/%s_r$'%(var[eq], var[eq])
                plt.xlabel(xlabel)
                plt.ylabel(r'$y/\delta$')
                plt.legend()
                plt.show()

        return self.Param['modes'][self.i], self.alpha[:,:,self.i], self.amplitude[:,:,self.i], Converged

    def printResidual(self, itr, res, dalpha):
        #------------------------------------------
        # Arguments
        # itr   : iteration number
        # res   : normalization condition residual
        # NLres : Nonlinear residual
        #
        # Description
        # Output iteration details
        # Only prints details for the first 3 modes
        #------------------------------------------

        maxModeDisp = 3
        modeDisp = 0
        (M,N) = self.Param['(M,N)max']
        if itr == 1:
            Header = ' #       Res     relda'
            for m,n in self.Param['modes'][self.i]:
                Header += '      (%2i,%2i)'%(m,n)
                modeDisp += 1
                if modeDisp == maxModeDisp:
                    Header += '     ...     '
                    break
            Header += '      (%2i,%2i)'%(M-1,N-1)
            print(len(Header)*'-')
            print('                                                   ')
            print('        * * *   S T A T I O N   # '+str(self.i)+'   * * *')
            print('                                                   ')
            print(Header)
            print(len(Header)*'-')

        modeDisp = 0
        Format = '%2d %9.2E %9.2E'
        A = []
        for m,n in self.Param['modes'][self.i]:
            Format += ' %11.4Ej'
            A.append(self.alpha[m,n,self.i].imag)
            modeDisp += 1
            if modeDisp == maxModeDisp:
                Format += '     ...     '
                break
        Format +=' %11.3Ej'
        A.append(self.alpha[M-1,N-1,self.i].imag)

        # Output residual, im(alpha(m,n)) of the first 3 modes
        print(Format%(itr, res, dalpha, *A))

    def extractFluc(self):
        # first station
        i0 = self.i0

        # Number of points
        Ny = self.NumMethod.Ny
        Nx = self.NumMethod.Nx

        # Modal fluctuation field
        (M,N) = self.Param['(M,N)max']
        Fluc    = Field(Nx, Ny, typ=complex, M=self.Param['(M,N)max'][0], N=self.Param['(M,N)max'][1])

        # Vector offset
        u = 0*Ny
        v = 1*Ny
        w = 2*Ny
        p = 3*Ny
        T = 4*Ny
        E = 5*Ny

        # Computing growth rate
        self.Growth()

        for i in range(self.NumMethod.Nx):
            for m,n in [(i,j) for j in range(N) for i in range(M)]:
                # Storing Results
                Fluc.xc[m,n,i,:]  = self.Flow.xc[i,:]
                Fluc.yc[m,n,i,:]  = self.Flow.yc[i,:]
                Fluc.x[m,n,i,:]   = self.Flow.x[i,:]
                Fluc.y[m,n,i,:]   = self.Flow.y[i,:]
                Fluc.U[m,n,i,:]   = self.phi[m,n,i,u:v]
                Fluc.V[m,n,i,:]   = self.phi[m,n,i,v:w]
                Fluc.W[m,n,i,:]   = self.phi[m,n,i,w:p]
                Fluc.P[m,n,i,:]   = self.phi[m,n,i,p:T]
                Fluc.T[m,n,i,:]   = self.phi[m,n,i,T:E]

                # Streamwise wave number
                Fluc.alpha[m,n,i]  = self.alpha[m,n,i]
                Fluc.alphax[m,n,i] = self.alphax[m,n,i]
    
                # Disturbance Kinetic Energy (DKE)
                Fluc.DKE[m,n,i]  = self.DKE[m,n,i]
                Fluc.DKEx[m,n,i] = self.DKEx[m,n,i]
                
                # Amplitude
                Fluc.amplitude[m,n,i] = self.amplitude[m,n,i]*self.Param[(m,n)]['A_0']

                # Growth rate
                Fluc.sigma[m,n,i] = self.sigma[m,n,i]
                Fluc.F[m,n,i]     = self.Param[(m,n)]['omega'][self.i0]/self.Param['prop']['Rex'][i0]*10**6
                Fluc.omega[m,n,i] = self.Param[(m,n)]['omega'][self.i0]
                Fluc.Rex[m,n,i]   = self.Param['prop']['Rex'][i]
    
                # Drivatives
                Fluc.Ux[m,n,i,:] = self.phix[m,n,i,u:v]
                Fluc.Vx[m,n,i,:] = self.phix[m,n,i,v:w]
                Fluc.Wx[m,n,i,:] = self.phix[m,n,i,w:p]
                Fluc.Px[m,n,i,:] = self.phix[m,n,i,p:T]
                Fluc.Tx[m,n,i,:] = self.phix[m,n,i,T:E]
    
                Fluc.Uy[m,n,i,:]  = self.Dy.dot(self.phi[m,n,i,u:v])
                Fluc.Vy[m,n,i,:]  = self.Dy.dot(self.phi[m,n,i,v:w])
                Fluc.Wy[m,n,i,:]  = self.Dy.dot(self.phi[m,n,i,w:p])
                Fluc.Py[m,n,i,:]  = self.Dy.dot(self.phi[m,n,i,p:T])
                Fluc.Ty[m,n,i,:]  = self.Dy.dot(self.phi[m,n,i,T:E])

                # conservatives quantities
                Fluc.rho[m,n,i,:] = self.Flow.rho[i,:]/self.Flow.P[i,:]*Fluc.P[m,n,i,:]    \
                                  - self.Flow.rho[i,:]/self.Flow.T[i,:]*Fluc.T[m,n,i,:]

                Fluc.rhoU[m,n,i,:] = self.Flow.rho[i,:]*Fluc.U[m,n,i,:]                    \
                                   + self.Flow.U[i,:]*Fluc.rho[m,n,i,:]

                Fluc.rhoV[m,n,i,:] = self.Flow.rho[i,:]*Fluc.V[m,n,i,:]                    \
                                   + self.Flow.V[i,:]*Fluc.rho[m,n,i,:]

                Fluc.rhoW[m,n,i,:] = self.Flow.rho[i,:]*Fluc.W[m,n,i,:]                    \
                                   + self.Flow.W[i,:]*Fluc.rho[m,n,i,:]

                Fluc.E[m,n,i,:]   = 1/(self.Param['prop']['gamma']-1)*Fluc.P[m,n,i,:]      \
                                  + 0.5*( self.Flow.U[i,:]**2                              \
                                        + self.Flow.V[i,:]**2                              \
                                        + self.Flow.W[i,:]**2)*Fluc.rho[m,n,i,:]           \
                                  + self.Flow.rho[i,:]*(  self.Flow.U[i,:]*Fluc.U[m,n,i,:] \
                                                        + self.Flow.V[i,:]*Fluc.V[m,n,i,:] \
                                                        + self.Flow.W[i,:]*Fluc.W[m,n,i,:] )
        # n-factor & N-factor
        for m,n in [(i,j) for j in range(N) for i in range(M)]:
            fz = np.argmax(Fluc.sigma[m,n,:]>0)
            Fluc.nfactor[m,n,fz+1:] = cumtrapz(Fluc.sigma[m,n,fz:],self.Flow.x[fz:,0])
        Fluc.Nfactor = np.amax(np.amax(Fluc.nfactor, axis=0), axis=0)

        return Fluc

    def computeDKE(self):
        #-----------------------------------------------------------
        # Description
        # Compute the Disturbance Kinetic Energy
        #-----------------------------------------------------------

        # Numerical Methods
        Ny = self.NumMethod.Ny
        IntW = self.w

        # Station number
        i  = self.i
        i0  = self.i0

        # Vector offset
        u = 0
        v = 1
        w = 2

        # Computing mode-specific disturbance kinetic energy
        for m,n in self.Param['modes'][i]:
            self.DKE[m,n,i] = 0
            for j in [u, v, w]:
                # Offset
                y0 = j * Ny
                ym = y0 + Ny
            
                # Disturbance Kinetic Energy (DKE)
                self.DKE[m,n,i]  = self.DKE[m,n,i] + IntW.dot(abs(self.phi[m,n,i,y0:ym])**2)

    def Growth(self):
        #-----------------------------------------------------------
        # Description
        # Compute the growth rate of the disturbance for every mode
        # The growth rate is based on Juniper's definition.
        #
        #         sigma(m,n) = -alpha_i(m,n) + dE/dx/(2.*E)
        #
        # Where E is the disturbance kinetic energy (DKE)
        #
        #-----------------------------------------------------------

        # Numerical Methods
        Ny = self.NumMethod.Ny
        IntW = self.w
        
        # Modes to compute
        (M,N) = self.Param['(M,N)max']

        # Station number
        i  = self.i
        i0  = self.i0

        # Vector offset
        u = 0
        v = 1
        w = 2

        # Weno, to compute derivatives
        from weno import WENO
        weno = WENO(self.x, 3)

        # Computing mode-specific disturbance kinetic energy
        for m,n in [(i,j) for j in range(N) for i in range(M)]:
            for i in range(i0, self.NumMethod.Nx):
               for j in [u, v, w]:
                   # Offset
                   y0 = j * Ny
                   ym = y0 + Ny
            
                   # Disturbance Kinetic Energy (DKE)
                   self.DKE[m,n,i]  = self.DKE[m,n,i] + abs(IntW.dot(self.Flow.rho[i,:] * abs(self.phi[m,n,i,y0:ym])**2))

            # DKE derivative
            Dx = weno.getDiffMatrix(updateWENO=True, phi = self.DKE[m,n,:])
            self.DKEx[m,n,:] = Dx @ self.DKE[m,n,:] #mBDF.dfdx(self.DKE[m,n,:], self.NumMethod.X1D)

            for i in range(i0, self.NumMethod.Nx):
               # Growth rate
               self.sigma[m,n,i] = -self.alpha[m,n,i].imag #+ self.DKEx[m,n,i]/(2.*self.DKE[m,n,i])

