import numpy as np
class Parameter:
    """ Class Flow Parameters"""
    #   Mean Flow
    """
        Uref    : Reference velocity
        Tref    : Reference Temperature
        rho0  : Reference density
        mu0   : Reference viscosity
        nu0   : mu0/rho0
        gamma : cp/cv
        Pr    : Prandtl Number
        Re    : Reynold Number
        Re0   : Reynold Number at first station
        Ma    : Mach Number
        L     : Reference Length scale
        m     : Pressure gradient parameter
                (Falkner-Skan)
        betap : beta = 2m/(m+1)
        x     : Streamwise position (PSE)
        S     : Sutherland's Law S parameter
        yInfi : upper bound for the fluctuating flow
    """
    #   Fluctuating Flow
    """
        u' = u_h * exp[ mode*( theta + i*(beta*z - omega*t) ) ]
        mode      : Mode number (1, 2, 3, ...)
        beta      : Spatial mode
        omega     : Temporal mode
        Amplitude : Initial amplitude for the initial condition
    """
    def __init__(self, M, N, X1D, maxM=1,maxN=25, MFD=False):
        # Default parameter values
        Nx = len(X1D)
        self.dic = {}
        self.dic['prop'] = {}
        self.dic['prop']['X1D']    = X1D
        self.dic['prop']['Uref']   = 1.
        self.dic['prop']['Lref']   = 1/200.
        self.dic['prop']['CF']     = 0.
        self.dic['prop']['rhoref'] = 1.
        self.dic['prop']['Tref']   = 1.
        self.dic['prop']['Twall']  = 1.
        self.dic['prop']['Twall type']  = 'constant'
        self.dic['prop']['Qwall']  = -999
        self.dic['prop']['viscosity']  = 'Sutherland'
        self.dic['prop']['gamma']  = 1.4
        self.dic['prop']['Cp']     = 3.5
        self.dic['prop']['Pr']     = 1.4
        self.dic['prop']['Re']     = 200
        self.dic['prop']['Rex']    = np.linspace(200,1000)
        self.dic['prop']['Ma']     = 0.1
        self.dic['prop']['S']      = 110.4/273.15
        self.dic['prop']['F']      = 100.
        self.dic['prop']['dF']     = 10.
        self.dic['prop']['B']      = 0.
        self.dic['prop']['dB']     = 0.

        self.dic['io'] = {}
        self.dic['io']['save']     = False
        self.dic['modes'] = []
        self.dic['MFD'] = MFD

        self.dic['trash'] = set()
        if (maxN, maxN) == (None,None):
            self.dic['(M,N)max'] = (M,N)
            (maxM,maxN) = self.dic['(M,N)max']
        else:
            self.dic['(M,N)max'] = (maxM,maxN)
            (maxM,maxN) = self.dic['(M,N)max']

        for i in range(Nx):
            self.dic['modes'].append([(i,j) for j in range(N) for i in range(M)])
            if not MFD:
                self.dic['modes'][i].remove((0,0))

            for m,n in [(i,j) for j in range(maxN) for i in range(maxM)]:
                self.dic[(m,n)]={}
                self.dic[(m,n)]['local']    = [True]*Nx
                self.dic[(m,n)]['linear']   = [True]*Nx
                self.dic[(m,n)]['omega']    = np.zeros(Nx,dtype=complex)
                self.dic[(m,n)]['alpha0']   = np.zeros(Nx,dtype=complex)
                self.dic[(m,n)]['beta']     = np.zeros(Nx,dtype=complex)
                self.dic[(m,n)]['parallel'] = [False]*Nx
                self.dic[(m,n)]['A_0']      = 1e-12
#        if MFD:
#            self.dic['modes'][0].remove((0,0))

    def getParam(self):
        self.dic['prop']['Cp']     = 1/((self.dic['prop']['gamma']-1)*self.dic['prop']['Ma']**2)
        self.dic['prop']['nuref']  = self.dic['prop']['Uref']/self.dic['prop']['Re']**2/self.dic['prop']['Lref']
        self.dic['prop']['muref']  = self.dic['prop']['rhoref']*self.dic['prop']['nuref']
        self.dic['prop']['Ps']     = self.dic['prop']['Tref']/(self.dic['prop']['rhoref']*self.dic['prop']['gamma']*self.dic['prop']['Ma']**2)
        self.dic['prop']['Pt']     = self.dic['prop']['Ps']*(1+(self.dic['prop']['gamma']-1)/2*self.dic['prop']['Ma']**2)**(self.dic['prop']['gamma']/(self.dic['prop']['gamma']-1))
        self.dic['prop']['S']      = self.dic['prop']['S']/self.dic['prop']['Tref']
        self.dic['prop']['Rex']    = self.dic['prop']['Re']*np.sign(self.dic['prop']['X1D'])*((abs(self.dic['prop']['X1D'])*self.dic['prop']['Lref']))**0.5
        (maxM,maxN) = self.dic['(M,N)max']
        for i in range(0, len(self.dic['modes'])):
            for m,n in [(i,j) for j in range(maxN) for i in range(maxM)]:
                self.dic[(m,n)]['omega'][i]  = (self.dic['prop']['F']+self.dic['prop']['dF']*n)*self.dic['prop']['Rex'][i]/1e6
                self.dic[(m,n)]['beta'][i]   = (self.dic['prop']['B']+self.dic['prop']['dB']*m)*self.dic['prop']['Rex'][i]/1e6
                self.dic[(m,n)]['alpha0'][i] = self.dic[(m,n)]['omega'][i]/0.35
        return self.dic
