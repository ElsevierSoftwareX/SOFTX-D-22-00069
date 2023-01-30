import sys
import os

#import pyximport; pyximport.install()
from NS             import NS
from Field          import Field
import scipy.io as sio
import numpy    as np

def LaminarFlow(NumMethod, Param):
    if Param['io']['LamFlow'] == 'interp':
        # PDE definition
        PDE = NS(NumMethod.NSE, Param)
        PDE.load()
        Converged = True

        # Interpolate NSE to PSE grid
        Flow = PDE.extractSol(NumMethod.PSE)

        # Dumping results
        path_results = os.path.join(os.path.dirname(__file__), Param['io']['SaveName'])
        Flow.dump(path_results+'/Flow')

    elif Param['io']['LamFlow'] == 'guess':
        # PDE definition
        PDE = NS(NumMethod.NSE, Param)
        PDE.load()

        # Solving system
        PHI, Converged = PDE.Solve()

        if Converged:
            # Computing growth rate
            Flow = PDE.extractSol(NumMethod.PSE)

    elif Param['io']['LamFlow'] == 'load':
        Flow = Field(0,0,path=Param['io']['LoadName']+'/Flow')
        Converged = True

    elif Param['io']['LamFlow'] == 'compute':
        # PDE definition
        PDE = NS(NumMethod.NSE, Param)
        # Initialization
        PDE.InitSol()
        # Solving system
        PHI, Converged = PDE.Solve()

        if Converged:
           # Computing growth rate
           Flow = PDE.extractSol(NumMethod.PSE)

    elif Param['io']['LamFlow'] == 'outgrid':
        # Number of points
        Ny = NumMethod.NSE.Ny
        Nx = NumMethod.NSE.Nx

        # Modal fluctuation field
        Flow    = Field(Nx, Ny)

        # Mesh
        theta = NumMethod.NSE.theta
        Flow.theta = theta

        # Coordinates
        # Curvilinear
        Flow.x = NumMethod.NSE.X
        Flow.y = NumMethod.NSE.Y

        # Cartesian
        Flow.xc[0,:] = NumMethod.NSE.X[0,0] - NumMethod.NSE.Y[0,:]*np.sin(theta[0,:])
        Flow.yc[0,:] = NumMethod.NSE.Y[0,0] + NumMethod.NSE.Y[0,:]*np.cos(theta[0,:])

        for i in range(1, Nx):
            dx = NumMethod.NSE.X1D[i] - NumMethod.NSE.X1D[i-1]
            avth = (theta[i,0] + theta[i-1,0])/2
            Flow.xc[i,:] = Flow.xc[i-1,0] + dx*np.cos(avth) - NumMethod.NSE.Y[i,:]*np.sin(avth)
            Flow.yc[i,:] = Flow.yc[i-1,0] + dx*np.sin(avth) + NumMethod.NSE.Y[i,:]*np.cos(avth)

        # save grid
        if Param['io']['save']:
            path_results = os.path.join(os.path.dirname(__file__), Param['io']['SaveName'])
            Flow.dump(path_results+'/Flow')
        Converged = False

    # Extracting solution
    if Converged:
        print('___________________________________________________')
        print('                                                   ')
        print('                C O N V E R G E D                  ')
        print('                                                   ')
        print('          * * *   S U M M A R Y   * * *            ')
        print('                                                   ')
        print('                                                   ')
        if Param['io']['LamFlow'] != 'load':
            path_results = os.path.join(os.path.dirname(__file__), Param['io']['SaveName'])

            # Copy input file as it might be useful
            if Param['io']['save']:
                Flow.dump(path_results+'/Flow')

            if Param['io']['vtk']: # and not Param['io']['LamFlow'] == 'load':
                Flow.writeVTK(path_results, name='Flow', outputDeriv=True)

    return Flow
