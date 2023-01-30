import numpy as np
import os
import scipy.io as sio
import scipy    as sc
import pypardiso
pypardiso.ps.set_iparm(60,2)
# Iterative solver
pypardiso.ps.set_iparm(4,31)

from scipy.sparse         import diags as diagflat
from pypardiso import spsolve
from numpy                import vstack
from scipy.interpolate    import interp1d
from scipy.interpolate    import interp2d
from scipy.integrate      import trapz
from Field                import Field

class NS(object):
    # Description
    # Handle the NS calculation. All the subroutines relevant
    # to build the operator,  and solve the system.
    def __init__(self, NumMethod, Param):
        self.sparseFormatLHS = 'lil'
        self.sparseFormatDia = 'csr'
        # Station Number
        self.i            = 0
        # Spectral methods routines
        self.NumMethod = NumMethod
        Nx = NumMethod.Nx
        Ny = NumMethod.Ny
        # Flow parameters
        self.Param = Param
        # Discretization
        self.x = NumMethod.MethodX.getPoints()
        self.y = NumMethod.MethodY.getPoints()
        X = NumMethod.getX()
        Y = NumMethod.getY()
        # Useful matrix
        self.one  = np.ones(Ny)
        self.zero = np.zeros(Ny)

        # BC top and inlet
        self.customBC   = False
        if Param['io']['LamFlow'] == 'compute' or Param['io']['LamFlow'] == 'interp':
            # Differentiation/Integration
            # Global Matrix
            self.Dx, self.D2x = self.NumMethod.getDX_AND_D2X()
            self.Dy, self.D2y = self.NumMethod.getDY_AND_D2Y()
            # Convective terms
            self.Dy_conv = self.NumMethod.MethodY.getDiffMatrix()
            # Solution at the current iteration (2D)
            self.phi       = np.ones(5*Ny*Nx)
            self.phix      = np.zeros(5*Ny*Nx)
            self.phiy      = np.zeros(5*Ny*Nx)
            self.phi_old   = np.ones(5*Ny*Nx)
            # PDE operator (2D)
            self.LHS  = sc.sparse.eye(5*Ny*Nx, format=self.sparseFormatLHS)
            self.RHS  = np.zeros(5*Ny*Nx)

    def Deriv(self, Second=False):
        # Compute the X-Y derivatives
        self.phix = self.Dx @ (self.phi)
        self.phiy = self.Dy @ (self.phi)

        # Second derivatives
        if Second:
            self.phixx = self.D2x @ (self.phi)
            self.phiyy = self.D2y @ (self.phi)
            self.phixy = self.Dy @ (self.phix)

    def updateWENO(self):
        # Update the weno coefficients
        if self.NumMethod.MethodNameX == 'WENO':
            self.Dx, self.D2x = self.NumMethod.getDX_AND_D2X(updateWENO=True,phi=self.phi)
        if self.NumMethod.MethodNameY == 'WENO':
            self.Dy, self.D2y = self.NumMethod.getDY_AND_D2Y(updateWENO=True,phi=self.phi)


    def formMatrixPicard(self, coupled=False):
        #-----------------------------------------------------------
        # Description
        # Construct the linear operator for the N-S equations
        #-----------------------------------------------------------
        # Numerical methods
        Ny = self.NumMethod.Ny
        Nx = self.NumMethod.Nx
        self.RHS  = np.zeros(5*Ny*Nx)

        # Loop on x-stations
        for i in range(0, Nx):
            # Finding non-zeros coefficients
            JIndices = np.array([i])
            for n in range(self.NumMethod.Neq):
                I = (n+self.NumMethod.Neq*i)*self.NumMethod.Ny
                JIndices = np.append(JIndices, self.D2x[I,:].nonzero()[1]//self.NumMethod.Neq//self.NumMethod.Ny)
                JIndices = np.append(JIndices,  self.Dx[I,:].nonzero()[1]//self.NumMethod.Neq//self.NumMethod.Ny)
            JIndices = np.unique(JIndices)

            # Loop over elements
            for j in JIndices:
                self.Operator(i,j, coupled=coupled)



    def computeJac(self):
        #-----------------------------------------------------------
        # Description
        # Compute the numerical jacobian for the N-S equations
        #-----------------------------------------------------------
        # Computing unperturbed Operator
        self.formMatrixPicard()
        LHS  = self.LHS.copy()
        RHS  = self.LHS.copy()

        # Adding perturbation to phi
        deltaPhi = 1e-3
        self.phi = self.phi + deltaPhi
        self.Deriv()

        # Computing perturbed Operator
        self.formMatrixPicard()

        # Computing numerical Jacobian
        self.LHS += -LHS
        self.RHS += -RHS


    def formMatrixNewton(self):
        #-----------------------------------------------------------
        # Description
        # Construct the linear operator for the N-S equations
        #-----------------------------------------------------------
        # Numerical methods
        Ny = self.NumMethod.Ny
        Nx = self.NumMethod.Nx

        # compute derivative
        self.Deriv()

        # Loop on x-stationa
        bnd = self.NumMethod.FDOX//2
        for i in range(0, Nx):
            if i < bnd or i >= Nx-bnd:
                off = self.NumMethod.FDOX + 2 + 2
            else:
                off = bnd + 2
            for j in range(max(i-off,0), min(i+off+1,Nx)):
                self.Operator(i,j)
                  
    def Operator(self,i,j, coupled=False):
        # Numerical methods
        Ny = self.NumMethod.Ny
        Nx = self.NumMethod.Nx

        # Semi-coupled parameter
        relax = 1

        # Flow parameters
        Pr    = self.Param['prop']['Pr']
        Ps    = self.Param['prop']['Ps']
        Cp    = self.Param['prop']['Cp']
        S     = self.Param['prop']['S']
        Tref  = self.Param['prop']['Tref']
        gamma = self.Param['prop']['gamma']
        Ma    = self.Param['prop']['Ma']
        Re    = self.Param['prop']['Re']

        # Geometric Jacobian
        Jxx = self.NumMethod.Jxx[i,:]
        Jxy = self.NumMethod.Jxy[i,:]
        Jyx = self.NumMethod.Jyx[i,:]
        Jyy = self.NumMethod.Jyy[i,:]
       
        ui = (0+5*i)*Ny
        vi = (1+5*i)*Ny
        wi = (2+5*i)*Ny
        pi = (3+5*i)*Ny
        ti = (4+5*i)*Ny

        uif = (0+5*i)*Ny+Ny
        vif = (1+5*i)*Ny+Ny
        wif = (2+5*i)*Ny+Ny
        pif = (3+5*i)*Ny+Ny
        tif = (4+5*i)*Ny+Ny

        uj = (0+j*5)*Ny
        vj = (1+j*5)*Ny
        wj = (2+j*5)*Ny
        pj = (3+j*5)*Ny
        tj = (4+j*5)*Ny

        ujf = (0+j*5)*Ny+Ny
        vjf = (1+j*5)*Ny+Ny
        wjf = (2+j*5)*Ny+Ny
        pjf = (3+j*5)*Ny+Ny
        tjf = (4+j*5)*Ny+Ny

        # Differentiation Matrix for
        # the diffusive terms
        Dy_diff  = self.Dy[ui:uif,ui:uif].todense()
        D2y_diff = self.D2y[ui:uif,ui:uif].todense()

        # Differentiation Matrix for 
        # the convective terms
        Dy_conv = self.Dy_conv

        # solution (last iteration)
        U = self.phi[ui:uif]
        V = self.phi[vi:vif]
        W = self.phi[wi:wif]
        P = self.phi[pi:pif]
        T = self.phi[ti:tif]

        Ux = self.phix[ui:uif]
        Vx = self.phix[vi:vif]
        Wx = self.phix[wi:wif]
        Px = self.phix[pi:pif]
        Tx = self.phix[ti:tif]

        Uy = self.phiy[ui:uif]
        Vy = self.phiy[vi:vif]
        Wy = self.phiy[wi:wif]
        Py = self.phiy[pi:pif]
        Ty = self.phiy[ti:tif]

        # Flow properties
        rho = gamma*Ma**2*P/T
        
        # Viscosity
        if self.Param['prop']['viscosity'] == 'Sutherland': 
            # Sutherland's law (dimensionless)
            mu  = T**1.5*(Tref+S)/(T+S)
            muT = T**0.5*(3*S+T)*(Tref+S)/(2*(S+T)**2)
            mux = muT*Tx
            muy = muT*Ty
        else:
            # Constant viscosity
            mu  = self.one
            muT = self.zero
            mux = self.zero
            muy = self.zero
        
        lamb  = mu#*Cp/Pr
        lambx = mux#*Cp/Pr
        lamby = muy#*Cp/Pr
        
        # x-derivatives
        # x-momemtum equation
        if coupled:
            Conv11 = np.diagflat(rho*(U*Jxx + V*Jxy)) * self.Dx[ui:uif,uj:ujf]
        else:
            Conv11 = relax * np.diagflat(rho*(U*Jxx + V*Jxy)) * self.Dx[ui:uif,uj:ujf]

        Diff11 = np.diagflat(-1/Re*((4/3*Jxx**2+Jxy**2)*mux+(4/3*Jxx*Jyx+Jyy*Jxy)*muy)              \
                             + Dy_diff @ (-2*mu/Re*(4/3*Jxx*Jyx + Jxy*Jyy)))*self.Dx[ui:uif,uj:ujf] \
                             - np.diagflat(mu/Re*(4/3*Jxx**2 + Jxy**2))*self.D2x[ui:uif,uj:ujf]

        Diff12 = np.diagflat(-1/Re*(mux*Jxy*Jxx + muy*Jxx*Jyy) + Dy_diff @ (-mu/Re*(Jxx*Jyy + Jxy*Jyx)) ) * self.Dx[vi:vif,vj:vjf]  \
               - np.diagflat(mu/Re*(Jxx*Jxy + Jxy**2)) * self.D2x[vi:vif,vj:vjf]

        dPdx = np.diagflat(Jxx*self.one) * self.Dx[pi:pif,pj:pjf]

        # y-momemtum equation
        Diff21 = np.diagflat(-1/Re*(mux*Jxy*Jxx + muy*Jxy*Jyx) + Dy_diff @ (-2*mu/Re*(4/3*Jxx*Jyx + Jxy*Jyy)) ) * self.Dx[ui:uif,uj:ujf] \
               - np.diagflat(mu/Re*(Jxx*Jxy)) * self.D2x[ui:uif,uj:ujf]

        if coupled:
            Conv22 = np.diagflat(rho*(U*Jxx + V*Jxy)) * self.Dx[vi:vif,vj:vjf]
        else:
            Conv22 = relax * np.diagflat(rho*(U*Jxx + V*Jxy)) * self.Dx[vi:vif,vj:vjf]

        Diff22 = np.diagflat(-1/Re*((4/3*Jyx**2+Jxx**2)*mux+(4/3*Jxx*Jyx+Jyy*Jxy)*muy) \
                + Dy_diff @ (-mu/Re*(Jxx*Jyy + Jxy*Jyx)) ) * self.Dx[vi:vif,vj:vjf]    \
                - np.diagflat(mu/Re*(Jxx**2+4/3*Jxy**2)) * self.D2x[vi:vif,vj:vjf]

        dPdy = np.diagflat(Jxy*self.one) * self.Dx[pi:pif,pj:pjf]

        # z-momemtum equation
        if coupled:
            Conv33 = np.diagflat(rho*(U*Jxx + V*Jxy)) * self.Dx[wi:wif,wj:wjf]
        else:
            Conv33 = relax * np.diagflat(rho*(U*Jxx + V*Jxy)) * self.Dx[wi:wif,wj:wjf]

        Diff33 = np.diagflat(Dy_diff @ (-2*mu/Re*(Jyx*Jxx+Jxy*Jyy))) * self.Dx[wi:wif,wj:wjf] \
                -np.diagflat(mu/Re*(Jxx**2+Jxy**2)) * self.D2x[wi:wif,wj:wjf]

        # Continuity equation
        Cont41 = np.diagflat(Jxx*self.one) * self.Dx[ui:uif,uj:ujf]
        Cont42 = np.diagflat(Jxy*self.one) * self.Dx[vi:vif,vj:vjf]
        Cont44 = np.diagflat(1/P*(U*Jxx+V*Jxy)) * self.Dx[pi:pif,pj:pjf]
        Cont45 = np.diagflat(-1/T*(U*Jxx+V*Jxy)) * self.Dx[ti:tif,tj:tjf]

        # Energy equation
        Diss51 = np.diagflat(-mu*(gamma-1)*Ma**2/Re*((4/3*Jxx**2+Jxy**2)*Ux + 2*Jxx*Jyx*Vx)) * self.Dx[ui:uif,uj:ujf]
        Diss52 = np.diagflat(-mu*(gamma-1)*Ma**2/Re*(4/3*Jxy**2+Jxx**2)*Vx) * self.Dx[vi:vif,vj:vjf]
        Diss53 = np.diagflat(-mu*(gamma-1)*Ma**2/Re*(Jxy**2+Jxx**2)*Wx) * self.Dx[wi:wif,wj:wjf]
        Comp54 = np.diagflat(-(gamma-1)*Ma**2*(U*Jxx+V*Jxy)) * self.Dx[pi:pif,pj:pjf]

        if coupled:
            Conv55 = np.diagflat(rho*Cp*(U*Jxx + V*Jxy)) * self.Dx[ti:tif,tj:tjf]
        else:
            Conv55 = relax * np.diagflat(rho*Cp*(U*Jxx + V*Jxy)) * self.Dx[ti:tif,tj:tjf]

        Diff55 = np.diagflat(-1/(Re*Pr)*((Jxx**2+Jxy**2)*lambx + (Jxx*Jyx + Jxy*Jyy)*lamby)        \
                          - Dy_diff @ (2*lamb/(Re*Pr)*(Jyx*Jxx+Jyy*Jxy))) * self.Dx[ti:tif,tj:tjf] \
               - np.diagflat( lamb/(Re*Pr)*(Jxx**2+Jxy**2)) * self.D2x[ti:tif,tj:tjf]

        # y-derivatives
        if i == j:
            # x-momentum equation
            if coupled:
                Conv11 += np.diagflat(rho*(U*Jyx + V*Jyy)) @ Dy_conv
            else:
                Conv11 += relax * np.diagflat(rho*(U*Jyx + V*Jyy)) @ Dy_conv
            Diff11 += np.diagflat(-1/Re*((4/3*Jxx*Jyx+Jyy*Jxy)*mux + (4/3*Jyx**2+Jyy**2)*muy)) @ Dy_diff + np.diagflat(-mu/Re*(4/3*Jyx**2+Jyy**2)) @ D2y_diff
            Diff12 += np.diagflat(-1/Re*(Jxy*Jyx*mux + Jyy*Jyx*muy)) @ Dy_diff + np.diagflat(-mu/Re*Jyx*Jxy) @ D2y_diff
            dPdx   += np.diagflat(Jyx*self.one) @ Dy_conv
            
            # y-momentum equation
            Diff21 += np.diagflat(-1/Re*(Jxx*Jyy*mux + Jyx*Jyy*muy)) @ Dy_diff + np.diagflat(-mu/Re*Jyx*Jxy) @ D2y_diff
            if coupled:
                Conv22 += np.diagflat(rho*(U*Jyx + V*Jyy)) @ Dy_conv
            else:
                Conv22 += relax * np.diagflat(rho*(U*Jyx + V*Jyy)) @ Dy_conv
            Diff22 += np.diagflat( - 1/Re*((4/3*Jxy*Jyy+Jxx*Jyx)*mux + (4/3*Jyy**2+Jyx**2)*muy)) @ Dy_diff + np.diagflat(-mu/Re*(4/3*Jyy**2+Jyx**2)) @ D2y_diff
            dPdy   += np.diagflat(Jyy*self.one) @ Dy_diff
            
            # z-momentum equation
            if coupled:
                Conv33 += np.diagflat(rho*(U*Jyx+V*Jyy)) @ Dy_conv
            else:
                Conv33 += relax * np.diagflat(rho*(U*Jyx+V*Jyy)) @ Dy_conv
            Diff33 += np.diagflat(-1/Re*((Jxx*Jyx+Jxy*Jyy)*mux + (Jyx**2+Jyy**2)*muy)) @ Dy_diff + np.diagflat(-mu/Re*(Jyx**2+Jyy**2)) @ D2y_diff
            
            # Continuity equation
            Cont41 += np.diagflat(Jyx*self.one) @ Dy_conv
            Cont42 += np.diagflat(Jyy*self.one) @ Dy_conv
            Cont44 += np.diagflat( 1/P*(U*Jyx+V*Jyy)) @ Dy_conv
            Cont45 += np.diagflat(-1/T*(U*Jyx+V*Jyy)) @ Dy_conv
            
            # Energy equation
            Diss51 += np.diagflat(-mu*(gamma-1)*Ma**2/Re*(2*(4/3*Jxx*Jyx+Jxy*Jyy)*Ux + (4/3*Jyx**2+Jyy**2)*Uy + 2*Jxx*Jyy*Vx + 2*Jyx*Jyy*Vy)) @ Dy_conv
            Diss52 += np.diagflat(-mu*(gamma-1)*Ma**2/Re*(2*(4/3*Jxy*Jyy+Jxx*Jyx)*Vx + (4/3*Jyy**2+Jyx**2)*Vy + 2*Jyx*Jxy*Ux)) @ Dy_conv
            Diss53 += np.diagflat(-mu*(gamma-1)*Ma**2/Re*(2*(Jxx*Jyx+Jxy*Jyy)*Wx + (Jyx**2+Jyy**2)*Wy)) @ Dy_conv
            Comp54 += np.diagflat(-(gamma-1)*Ma**2*(U*Jyx+V*Jyy)) @ Dy_conv
            if coupled:
                Conv55 += np.diagflat(rho*Cp*(U*Jyx+V*Jyy)) @ Dy_conv
            else:
                Conv55 += relax * np.diagflat(rho*Cp*(U*Jyx+V*Jyy)) @ Dy_conv
            Diff55 += np.diagflat(-1/(Re*Pr)*((Jxx*Jyx+Jxy*Jyy)*lambx + (Jyy**2+Jyx**2)*lamby)) @ Dy_diff + np.diagflat(-lamb/(Re*Pr)*(Jyx**2+Jyy**2)) @ D2y_diff

            if not coupled:
                # For the decoupled algorithm, the convective terms
                # we use the convective term from last iteration
                self.RHS[ui:uif] = -(1-relax) * (rho * (U*Jyx + V*Jyy) * Uy + rho * (U*Jxx + V*Jxy) * Ux)
                self.RHS[vi:vif] = -(1-relax) * (rho * (U*Jyx + V*Jyy) * Vy + rho * (U*Jxx + V*Jxy) * Vx)
                self.RHS[wi:wif] = -(1-relax) * (rho * (U*Jyx + V*Jyy) * Wy + rho * (U*Jxx + V*Jxy) * Wx)
                self.RHS[ti:tif] = -(1-relax) * (Cp * rho * (U*Jyx + V*Jyy) * Ty + rho * (U*Jxx + V*Jxy) * Tx)
            
        # Assembling Operator (LHS)
        # x-momentum equation
        self.LHS[ui:uif,uj:ujf] = Conv11 + Diff11
        self.LHS[ui:uif,vj:vjf] = Diff12
        self.LHS[ui:uif,pj:pjf] = dPdx  
        
        # y-momentum equation
        self.LHS[vi:vif,uj:ujf] = Diff21
        self.LHS[vi:vif,vj:vjf] = Conv22 + Diff22
        self.LHS[vi:vif,pj:pjf] = dPdy
        
        # z-momentum equation
        self.LHS[wi:wif,wj:wjf] = Diff33
        
        # Continuity equation
        self.LHS[pi:pif,uj:ujf] = Cont41  
        self.LHS[pi:pif,vj:vjf] = Cont42 
        self.LHS[pi:pif,pj:pjf] = Cont44 
        self.LHS[pi:pif,tj:tjf] = Cont45 
        
        # Energy equation
        self.LHS[ti:tif,uj:ujf] = Diss51  
        self.LHS[ti:tif,vj:vjf] = Diss52 
        self.LHS[ti:tif,wj:wjf] = Diss53 
        self.LHS[ti:tif,pj:pjf] = Comp54 
        self.LHS[ti:tif,tj:tjf] = Conv55 + Diff55

    def extrapolate(self, var, bound, i):
        Ny = self.NumMethod.Ny
        # variable name
        if var == 'U':
            off = 0
        elif var =='V':
            off = 1
        elif var == 'W':
            off = 2
        elif var == 'P':
            off = 3
        elif var == 'T':
            off = 4

        if bound == 'inlet':
            v01 = (off+5*(i+1))*Ny
            v02 = (off+5*(i+2))*Ny
        elif bound == 'outlet':
            v01 = (off+5*(i-1))*Ny
            v02 = (off+5*(i-2))*Ny
        elif bound == 'free-stream':
            v01 = (off+5*i)*Ny + Ny - 2
            v02 = (off+5*i)*Ny + Ny - 1
        
        if bound == 'inlet':
            Val = np.zeros(Ny)
            for jj in range(0,Ny):
                Valtable = np.array([self.phi[v01+jj], self.phi[v02+jj]])
                Val[jj]   = interp1d(self.x[i+1:i+3], Valtable, fill_value='extrapolate')(self.x[i])
        elif bound == 'outlet':
            Val = np.zeros(Ny)
            for jj in range(0,Ny):
                Valtable = np.array([self.phi[v01+jj], self.phi[v02+jj]])
                Val[jj]   = interp1d(self.x[i-2:i], Valtable, fill_value='extrapolate')(self.x[i])
        elif bound == 'free-stream':
            Valtable = np.array([self.phi[v01], self.phi[v02]])
            Val = interp1d(self.y[Ny-2:Ny], Valtable, fill_value='extrapolate')(self.y[-1])
        return Val

    def setBC(self, LHS, RHS):
        #-----------------------------------------------------------
        # Description
        # Apply the boundary conditions (BC)
        #-----------------------------------------------------------

        # Length
        Ny = self.NumMethod.Ny
        Nx = self.NumMethod.Nx

        gamma = self.Param['prop']['gamma']
        Ma    = self.Param['prop']['Ma']
        Re    = self.Param['prop']['Re']
        Cp    = self.Param['prop']['Cp']
        Pr    = self.Param['prop']['Pr']
        Twall = self.Param['prop']['Twall']
        Tref  = self.Param['prop']['Tref']
        S     = self.Param['prop']['S']
        Uref  = self.Param['prop']['Uref']
        CF    = self.Param['prop']['CF']
        Ps    = self.Param['prop']['Ps']
        Pt    = self.Param['prop']['Pt']
        Qwall = self.Param['prop']['Qwall']

        # Boundary conditions
        for i in range(Nx):
            # Global indices
            u0 = (0+5*i)*Ny
            uf = (0+5*i)*Ny + Ny - 1
            v0 = (1+5*i)*Ny
            vf = (1+5*i)*Ny + Ny - 1
            w0 = (2+5*i)*Ny
            wf = (2+5*i)*Ny + Ny - 1
            p0 = (3+5*i)*Ny
            pf = (3+5*i)*Ny + Ny - 1
            t0 = (4+5*i)*Ny
            tf = (4+5*i)*Ny + Ny - 1
                            
            Jxx = self.NumMethod.Jxx[i,:]
            Jxy = self.NumMethod.Jxy[i,:]
            Jyx = self.NumMethod.Jyx[i,:]
            Jyy = self.NumMethod.Jyy[i,:]
            theta = self.NumMethod.theta[i,:]

            # Equation of state
            rho = gamma*Ma**2*self.phi[p0:pf+1]/self.phi[t0:tf+1]

            # Viscosity & Conduction
            if self.Param['prop']['viscosity'] == 'Sutherland': 
                T = self.phi[t0:tf+1]
                mu  = abs(T)**(3/2)*(Tref+S)/(T+S)
            else:
                mu = self.one
            lamb  = mu#*Cp/Pr

            # Wall
            if self.x[i] >= 0:
                # U = 0
                row = self.LHS[u0,:].nonzero()[1]
                self.LHS[u0,row]= 0
                self.LHS[u0,u0] = 1
                self.RHS[u0]    = 0

                # Wall temperature
                if self.Param['prop']['Twall type'] == 'variable':
                    row = self.LHS[t0,:].nonzero()[1]
                    self.LHS[t0,row] = 0
                    self.LHS[t0,t0]  = 1
                    self.RHS[t0] = self.Param['prop']['T(x)'](self.x[i]) / Tref
                elif self.Param['prop']['Twall type'] == 'adiabatic':
                    row = self.LHS[t0,:].nonzero()[1]
                    self.LHS[t0,row] = 0
                    self.LHS[t0,t0]  = 1
                    self.RHS[t0] = 1 + (gamma-1)/2 * Pr**0.5 * Ma**2
                elif self.Param['prop']['Twall type'] == 'constant':
                    row = self.LHS[t0,:].nonzero()[1]
                    self.LHS[t0,row] = 0
                    self.LHS[t0,t0]  = 1
                    self.RHS[t0] = Twall/Tref
                elif self.Param['prop']['Twall type'] == 'flux':
                    # Qwall = lambda*dTdy
                    self.LHS[t0,:] = self.Dy[t0,:]
                    self.RHS[t0]   = Qwall/lamb[-1]
            else:
#                # U
#                row = self.LHS[u0,:].nonzero()[1]
#                self.LHS[u0,row]= 0
#                self.LHS[u0,u0] = 1
#                self.RHS[u0]    = 1
                # Uy = 0
                self.LHS[u0,:] = self.Dy[u0,:]
                self.RHS[u0]   = 0
                # Ty = 0
                self.LHS[t0,:] = self.Dy[t0,:]
                self.RHS[t0]   = 0

            # U = 0
            row = self.LHS[u0,:].nonzero()[1]
            self.LHS[u0,row]= 0
            self.LHS[u0,u0] = 1
            self.RHS[u0]    = (1-np.tanh(self.x[i]/20))/2

            # T = 0
            row = self.LHS[t0,:].nonzero()[1]
            self.LHS[t0,row]= 0
            self.LHS[t0,t0] = 1
            self.RHS[t0]    = 1#-(1-np.tanh(self.x[i]))/2

            # V = 0
            row = self.LHS[v0,:].nonzero()[1]
            self.LHS[v0,row]= 0
            self.LHS[v0,v0] = 1
            self.RHS[v0]    = 0
            # W = 0
            row = self.LHS[w0,:].nonzero()[1]
            self.LHS[w0,row]= 0
            self.LHS[w0,w0] = 1
            self.RHS[w0]    = 0
            # Py = 0
            self.LHS[p0,:] = self.Dy[p0,:]
            self.RHS[p0]   = 0
                
            # Free stream
            # U = 1
#            row = self.LHS[uf,:].nonzero()[1]
#            self.LHS[uf,row] = 0
#            self.LHS[uf,uf]  = 1
#            self.RHS[uf]     = 1
#            # V = 1
#            row = self.LHS[vf,:].nonzero()[1]
#            self.LHS[vf,row] = 0
#            self.LHS[vf,vf]  = 1
#            self.RHS[vf]     = 0
            # Uy = 0
            self.LHS[uf,:] = self.Dy[uf,:]
            self.RHS[uf]   = 0
            # Vy = 0
            self.LHS[vf,:] = self.Dy[vf,:]
            self.RHS[vf]   = 0
            # Wy = 0
            self.LHS[wf,:] = self.Dy[wf,:]
            self.RHS[wf]   = 0
            # P = Ps
            row = self.LHS[pf,:].nonzero()[1]
            self.LHS[pf,row] = 0
            self.LHS[pf,pf]  = 1
            self.RHS[pf]     = Ps
 
#            if self.Param['prop']['Twall type'] == 'flux':
            # Ty = 0
            self.LHS[tf,:] = self.Dy[tf,:]
            self.RHS[tf]   = 0
#            else:
#            row = self.LHS[tf,:].nonzero()[1]
#            self.LHS[tf,row] = 0
#            self.LHS[tf,tf]  = 1
#            self.RHS[tf]     = 1

            # Inlet
            if i == 0:
                # U
                row = self.LHS[u0:uf+1,:].nonzero()[1]
                self.LHS[u0:uf+1,row]      = 0
                self.LHS[u0:uf+1, u0:uf+1] = sc.sparse.eye(Ny, format=self.sparseFormatDia)
                self.RHS[u0:uf+1]          = 1
                # V
                row = self.LHS[v0:vf+1,:].nonzero()[1]
                self.LHS[v0:vf+1,row]      = 0
                self.LHS[v0:vf+1, v0:vf+1] = sc.sparse.eye(Ny, format=self.sparseFormatDia)
                self.RHS[v0:vf+1]          = 0
                # W
                row = self.LHS[w0:wf+1,:].nonzero()[1]
                self.LHS[w0:wf+1,row]      = 0
                self.LHS[w0:wf+1, w0:wf+1] = sc.sparse.eye(Ny, format=self.sparseFormatDia)
                self.RHS[w0:wf+1]          = 0
                # Px = 0
                self.LHS[p0:pf+1,:] = self.D2x[p0:pf+1,:]
                self.RHS[p0:pf+1]   = np.zeros(Ny)
                # T
                row = self.LHS[t0:tf+1,:].nonzero()[1]
                self.LHS[t0:tf+1,row]      = 0
                self.LHS[t0:tf+1, t0:tf+1] = sc.sparse.eye(Ny, format=self.sparseFormatDia)
                self.RHS[t0:tf+1]          = 1

            # outlet
            if i == Nx-1:
               # P
               row = self.LHS[p0:pf+1,:].nonzero()[1]
               self.LHS[p0:pf+1,row]      = 0
               self.LHS[p0:pf+1, p0:pf+1] = sc.sparse.eye(Ny, format=self.sparseFormatDia)
               self.RHS[p0:pf+1]          = Ps



    def InitSol(self):
        # Numerical methods
        Ny = self.NumMethod.Ny
        Nx = self.NumMethod.Nx

        # Param
        Ps = self.Param['prop']['Ps']
        Pt = self.Param['prop']['Pt']
        Uref = self.Param['prop']['Uref']
        Re = self.Param['prop']['Re']

        # Initial conditions
        for i in range(Nx):
            Re = self.Param['prop']['Re']
            # Global indices
            u0 = (0+5*i)*Ny
            uf = (0+5*i)*Ny + Ny
            v0 = (1+5*i)*Ny
            vf = (1+5*i)*Ny + Ny
            w0 = (2+5*i)*Ny
            wf = (2+5*i)*Ny + Ny
            p0 = (3+5*i)*Ny
            pf = (3+5*i)*Ny + Ny
            t0 = (4+5*i)*Ny
            tf = (4+5*i)*Ny + Ny
            theta = self.NumMethod.theta[i,:]

            self.phi[u0:uf] = 1 #self.U0 #np.tanh(self.y/3) #np.cos(theta)
            self.phi[v0:vf] = 0 #self.V0*0.86*Uref/Re #np.sin(theta) * 0.1/Re
            self.phi[w0:wf] = 0
            self.phi[p0:pf] = Ps
            self.phi[t0:tf] = 1

    def dealiasing(self, updateWENO=False):
        Nx     = self.NumMethod.Nx
        Ny     = self.NumMethod.Ny
        Neq    = self.NumMethod.Neq
        icr    = Ny * Neq
#        from matplotlib import pyplot as plt
        for j in range(1,Ny):
            for n in range(Neq):
                v0 = n * Ny + j
#                plt.plot(self.x, self.phi[v0::icr])
                self.phi[v0+icr::2*icr]  = self.NumMethod.MethodX.evalFunction(self.x[1::2],  self.phi[v0::icr], updateWENO=updateWENO, Dealiasing=True)
#                plt.plot(self.x, self.phi[v0::icr])
#                plt.show()
                self.phix[v0+icr::2*icr] = self.NumMethod.MethodX.evalFunction(self.x[1::2], self.phix[v0::icr], updateWENO=False, Dealiasing=True)
                self.phiy[v0+icr::2*icr] = self.NumMethod.MethodX.evalFunction(self.x[1::2], self.phiy[v0::icr], updateWENO=False, Dealiasing=True)
#                else:
#                    self.phi[v0+icr::2*icr]  = self.NumMethod.MethodX.evalFunction(self.x[1::2],  self.phi[v0::icr], updateWENO=False, Dealiasing=True)
#                    self.phix[v0+icr::2*icr] = self.NumMethod.MethodX.evalFunction(self.x[1::2], self.phix[v0::icr], updateWENO=False, Dealiasing=True)
#                    self.phiy[v0+icr::2*icr] = self.NumMethod.MethodX.evalFunction(self.x[1::2], self.phiy[v0::icr], updateWENO=False, Dealiasing=True)
        for i in range(Nx):
            for n in range(Neq):
                v0 = (n+5*i)*Ny
                vf = v0 + Ny
#                plt.plot(self.phi[v0:vf], self.y)
                self.phi[v0+1:vf:2]  = self.NumMethod.MethodY.evalFunction(self.y[1::2], self.phi[v0:vf],updateWENO=updateWENO,Dealiasing=True)
#                plt.plot(self.phi[v0:vf], self.y)
#                plt.show()
                self.phix[v0+1:vf:2] = self.NumMethod.MethodY.evalFunction(self.y[1::2],self.phix[v0:vf],Dealiasing=True)
                self.phiy[v0+1:vf:2] = self.NumMethod.MethodY.evalFunction(self.y[1::2],self.phiy[v0:vf],Dealiasing=True)

    def relaxUpdate(self, itr):
        from Geom import Geom
        Smooth = Geom.SmoothConv
        Nx  = self.NumMethod.Nx
        Ny  = self.NumMethod.Ny
        Neq = self.NumMethod.Neq
        relax = self.NumMethod.relaxU
        icr = Ny * Neq
        # Relaxation update
        for i in range(0,Nx):
            for n in range(Neq):
                v0 = (n+5*i)*Ny
                vf = (n+5*i)*Ny + Ny

                if relax != 1:
                    self.phi[v0:vf] = relax*self.phi[v0:vf] + (1-relax)*self.phi_old[v0:vf]

    def Solve(self):
        #-----------------------------------------------------------
        # Description
        # Call the relevant function to build, apply BC, and solve 
        # the system.
        #-----------------------------------------------------------
        # Dealiasing
        Dealiasing = True

        # Tolerance
        Converged = False
        tol = self.NumMethod.tol

        # Iteration number
        itr = 0

        # Residual norm
        L2Norm = 999

        # System solve
        while not Converged:
            # Assembling system
            self.formMatrixPicard(coupled=False)

            # Boundary conditions
            self.setBC(self.LHS, self.RHS)

            # Backing up last solution
            self.phi_old = self.phi

            # Solving system
            self.phi = spsolve(self.LHS.tocsr(), self.RHS)

            # Dealiasing
            if Dealiasing:
                self.dealiasing(updateWENO=True)

            # Convergence criteria
            if itr > 0:
                L2Norm = np.linalg.norm(self.phi - self.phi_old)
            else:
                L2Norm = 1 

            SolNorm = np.linalg.norm(self.phi)
            RelNorm = L2Norm/SolNorm

            # Relaxation (default is no relax)
            self.relaxUpdate(itr)

#            # Updating WENO coefficients
#            self.updateWENO()

            # Compute derivatives
            self.Deriv()

            if RelNorm < tol and itr > 1:
                Converged = True
                itr += 1
                self.save()
            elif os.path.isfile('stop.now'):
                Converged = True
                itr +=1
                self.save()
                print('Stopping simulation now (requested by user)')
                os.remove('stop.now')
            elif RelNorm > 1:
                print('-----------------------------------------')
                print('STATUS : DIVERGED')
                print('-----------------------------------------')
                break

            else:
                itr += 1

            # Output iteration information
            self.printResidual(itr, RelNorm, SolNorm)

        return self.phi, Converged

    def save(self):
        import hickle
        # Retriving folder
        Name = os.path.join(os.path.dirname(__file__), self.Param['io']['SaveName'])
        path = Name+'_phi.bin'
        hickle.dump({'phi':self.phi, 'eta':self.y}, path)

    def load(self):
        import hickle
        # Retriving folder
        Name = self.Param['io']['LoadName']
        path = Name+'_phi.bin'
        # Load solution
        sol = hickle.load(path)
        # Checking dimension
        Nx = self.NumMethod.Nx
        Ny = self.NumMethod.Ny
        try:
            Err = not (sol['phi'][0,:].shape == (1,5*Nx*Ny,))
            if Err:
                raise IndexError('Unable to load solution: Dimension mismatch')
        except:
            pass

        # Read solution
        self.phi = sol['phi']
        self.Deriv(Second=True)


    def printResidual(self, itr, RelNorm, SolNorm):
        #------------------------------------------
        # Arguments
        # itr   : iteration number
        # res   : residual
        #
        # Description
        # Output iteration details
        # Only prints details for the first 3 modes
        #------------------------------------------
        if itr == 1:
            print('___________________________________________________')
            print('                                                   ')
            print(' #                 Residual            Solution    ')
            print('___________________________________________________')
        Format = '%2d %24.4E %19.4E'
        print(Format%(itr, RelNorm, SolNorm))

    def evalSol(self, Var, x, y):
        # Evaluate the solution on the new grid (xx, yy)
        # We use the collocation spectral basis functions
        # to compute, so the field sol is evaluated in (xx,yy)
        # rather than interpolated.

        # Number of points
        Ny = self.NumMethod.Ny
        Nx = self.NumMethod.Nx

        # Allocating
        F   = np.zeros((Nx,Ny))
        Fx  = np.zeros((Nx,Ny))
        Fxx = np.zeros((Nx,Ny))
        Fy  = np.zeros((Nx,Ny))
        Fyy = np.zeros((Nx,Ny))
        Fxy = np.zeros((Nx,Ny))

        # Vector Offset
        if Var == 'U':
            off = 0
        elif Var == 'V':
            off = 1
        elif Var == 'W':
            off = 2
        elif Var == 'P':
            off = 3
        elif Var == 'T':
            off = 4
        # Vector -> Table
        for i in range(0, Nx):
            f0 = (off+5*i)*Ny
            ff = (off+5*i)*Ny + Ny

            F[i,:]   = self.phi[f0:ff]
            Fx[i,:]  = self.phix[f0:ff]
            Fxx[i,:] = self.phixx[f0:ff]
            Fy[i,:]  = self.phiy[f0:ff]
            Fyy[i,:] = self.phiyy[f0:ff]
            Fxy[i,:] = self.phixy[f0:ff]

        return F, Fx, Fxx, Fy, Fyy, Fxy

    def extractSol(self, NumMethodPSE):
        # Computing derivatives
        self.Deriv(Second=True)

        # Number of points
        Ny = NumMethodPSE.Ny
        Nx = NumMethodPSE.Nx

        # Modal fluctuation field
        Flow    = Field(Nx, Ny)
        
        # Flow properties
        Re    = self.Param['prop']['Re']
        gamma = self.Param['prop']['gamma']
        Ma    = self.Param['prop']['Ma']
        Tref  = self.Param['prop']['Tref']
        S     = self.Param['prop']['S']
        Cp    = self.Param['prop']['Cp']
        Pr    = self.Param['prop']['Pr']
        Ps    = self.Param['prop']['Ps']
        CF    = self.Param['prop']['CF']

        # Mesh
        theta = NumMethodPSE.theta
        Flow.theta = theta
        dy = np.zeros_like(NumMethodPSE.Y1D)
        dy[1:] = np.diff(NumMethodPSE.Y1D)
        
        # Coordinates
        # Curvilinear
        Flow.x = NumMethodPSE.X
        Flow.y = NumMethodPSE.Y

        # Cartesian
        Flow.xc[0,:]  = NumMethodPSE.X[0,:] - NumMethodPSE.Y[0,:]*np.sin(theta[0,:])
        Flow.yc[0,:]  = NumMethodPSE.Y[0,:] * np.cos(theta[0,:])
        Flow.Rex[0,:] =  Re*np.sign(NumMethodPSE.X1D[0])*((abs(NumMethodPSE.X1D[0])/Re))**0.5

        for i in range(1, Nx):
            if i == Nx-1:
                dx = NumMethodPSE.X[i] - NumMethodPSE.X[i-1]
            else:
                dx = (NumMethodPSE.X[i+1] - NumMethodPSE.X[i-1])/2.
            Flow.Rex[i,:] = Re*np.sign(NumMethodPSE.X1D[i])*((abs(NumMethodPSE.X1D[i])/Re))**0.5 
            Flow.xc[i,:] = Flow.xc[i-1,0] + dx*np.cos(theta[i,:]) - NumMethodPSE.Y1D*np.sin(theta[i,:])
            Flow.yc[i,:] = Flow.yc[i-1,0] + dx*np.sin(theta[i,:]) + NumMethodPSE.Y1D*np.cos(theta[i,:])

        for Var in ['U', 'V', 'W', 'P', 'T']:
            F, Fx, Fxx, Fy, Fyy, Fxy = self.evalSol(Var, self.x, self.y)

            if Var == 'P':
                exec('Flow.'+Var+'   = self.NumMethod.eval2DFunc(NumMethodPSE.X1D, NumMethodPSE.Y1D,   F)')
                exec('Flow.'+Var+'x  = self.NumMethod.eval2DFunc(NumMethodPSE.X1D, NumMethodPSE.Y1D,  Fx)')
                exec('Flow.'+Var+'y  = self.NumMethod.eval2DFunc(NumMethodPSE.X1D, NumMethodPSE.Y1D,  Fy)')
            else:
                exec('Flow.'+Var+'   = self.NumMethod.eval2DFunc(NumMethodPSE.X1D, NumMethodPSE.Y1D,   F)')
                exec('Flow.'+Var+'y  = self.NumMethod.eval2DFunc(NumMethodPSE.X1D, NumMethodPSE.Y1D,  Fy)')
                exec('Flow.'+Var+'yy = self.NumMethod.eval2DFunc(NumMethodPSE.X1D, NumMethodPSE.Y1D, Fyy)')
                exec('Flow.'+Var+'x  = self.NumMethod.eval2DFunc(NumMethodPSE.X1D, NumMethodPSE.Y1D,  Fx)')
                exec('Flow.'+Var+'xx = self.NumMethod.eval2DFunc(NumMethodPSE.X1D, NumMethodPSE.Y1D, Fxx)')
                exec('Flow.'+Var+'xy = self.NumMethod.eval2DFunc(NumMethodPSE.X1D, NumMethodPSE.Y1D, Fxy)')

        # Crossflow
        theta = np.arctan(np.sin(CF*np.pi/180.))
        UU = Flow.U
        WW = Flow.W

        Flow.U = UU*np.cos(theta) + WW*np.sin(theta)
        Flow.W = WW*np.cos(theta) - UU*np.sin(theta)

        # Properties
        Flow.rho  = gamma*Ma**2*Flow.P/Flow.T

        # viscosity
        if self.Param['prop']['viscosity'] == 'Sutherland': 
            Flow.mu   = Flow.T**(3/2)*(Tref+S)/(Flow.T+S)
            Flow.muT  = Flow.T**0.5*(3*S+Flow.T)*(Tref+S)/(2*(S+Flow.T)**2)
            Flow.muTT = (S+1)*(3*S**2 - 6*S*Flow.T - Flow.T**2)/(4*Flow.T**0.5*(S+Flow.T)**3)
        else:
            Flow.mu   =  np.ones_like(Flow.T)
            Flow.muT  = np.zeros_like(Flow.T)
            Flow.muTT = np.zeros_like(Flow.T)

        Flow.mux  = Flow.muT*Flow.Tx
        Flow.muxx = Flow.muT*Flow.Txx
        Flow.muxy = Flow.muT*Flow.Txy
        Flow.muyy = Flow.muT*Flow.Tyy
        Flow.muy  = Flow.muT*Flow.Ty

        Flow.lamb   = Flow.mu#*Cp/Pr
        Flow.lambT  = Flow.muT#*Cp/Pr
        Flow.lambTT = Flow.muTT#*Cp/Pr
        Flow.lambx  = Flow.mux#*Cp/Pr
        Flow.lambxx = Flow.muxx#*Cp/Pr
        Flow.lambxy = Flow.muxy#*Cp/Pr
        Flow.lambyy = Flow.muyy#*Cp/Pr
        Flow.lamby  = Flow.muy#*Cp/Pr

        # Nektar++ format
        Flow.E    = Flow.P/(gamma-1)+0.5*Flow.rho*(Flow.U**2+Flow.V**2+Flow.W**2)
        Flow.rhoU = Flow.rho*Flow.U
        Flow.rhoV = Flow.rho*Flow.V
        Flow.rhoW = Flow.rho*Flow.W

        return Flow
