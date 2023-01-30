print(        '        _  __                 _                     '        )
print(        '       | |/ /                | |                    '        )
print(        '       | " / _ __ _   _ _ __ | |_ ___  _ __         '        )
print(        '       |  ( | "__| | | | "_ \| __/ _ \| "_ \        '        )
print(        '       | . \| |  | |_| | |_) | || (_) | | | |       '        )
print(        '       |_|\_\_|   \__, | .__/ \__\___/|_| |_|       '        )
print(        '                   __/ | |                          '        )
print(        '                  |___/|_|                          '        )
print(        '                                                    '        )
#---------------------------------------------------------------------------
# Date    : 2022-03
# Author  : Francis Lacombe, PhD Student
# Contact : flacombe@uwaterloo.ca
# Version : 1.0
#
# Description:
# Krypton   is a python-based modal stability solver for compressible 
# quasi-3D boundary-layer flow. It is based on the Parabolized Stability
# Equations (PSE). It can solve the linear and nonlinear version of the PSE
# (LPSE and NPSE). The base flow is computed directly from the compressible
# laminar navier-stokes equations in curvilinear coordinates. The initial
# conditions are computed through the local stability theory (LST).
#
# LST
# System solved : (L + N*D + P*D^2)*phi = 0
#
# LPSE
# System solved : (L + N*D + P*D^2 + Q*da/dx)*phi = -M*dphi/dx
#
# NPSE
# System solved : (L + N*D + P*D^2 + Q*da/dx)*phi = -M*dphi/dx + NLT
#
#
# The code uses a combination of numerical methodis in X and Y. For the normal
# direction (Y), it uses a high-order finite difference scheme and for the 
# streamwise direction (X), it uses Stabilized Euler backward differentiation
# Formulas (BDF).
# --------------------------------------------------------------------------
import sys
sys.path.insert(1, '../src/')

from LaminarFlow     import LaminarFlow
from Param           import Parameter
from FlucFlow        import FlucFlow
from NumMethod       import NumericalMethod
from Geom            import Geom
import numpy as np
import plotStuff
import time
# *****************************************************************************
#
#                               G E O M E T R Y
#
# *****************************************************************************
# You can either enter the points directly if the geometry is simple, or use a
# separate file containing the boundary points
#
# Example 1 (Flat plate compared with Juniper's):

FDOX   = 1            # Order of approximation (Numerical scheme)
FDOY   = 60           # Order of approximation (Numerical scheme)
Nx     = 400          # Number of points in x-direction
Ny     = 240          # Number of points in y-direction
Re0    = 400          # Reference Reynolds (PSE and N-S)
Xf     = 6.25         # Last station (N-S and PSE)
Lref   = 1/Re0        # Reference length (Blasius scale)
X0     = 1            # First station (PSE)
Yf     = 400          # Last point in the normal direction 
Twall  = 1            # Temperature at the wall

# Geometric description
# Krypton will automatically create an orthonormal grid from
# the geometry defined by x and y
x = np.linspace(-5*X0, Xf, Nx) ; y = np.zeros_like(x)
coords = np.vstack([x.reshape((1,Nx)), y.reshape(1,Nx)]).T

# Domain for the N-S computations
GeomNS  = Geom(coords, Lref, Nx, Ny, Y0=0, Yf=Yf, dywall=0.1)

# For the PSE, the solution is interpolated from the NS results
# Start at the reference Reynolds (X=1)
x = np.linspace(1, Xf, Nx) ; y = np.zeros_like(x)
coords = np.vstack([x.reshape((1,Nx)), y.reshape(1,Nx)]).T

# Domain for the PSE computations
GeomPSE = Geom(coords, Lref, 400, 6, dywall=20, Yf=400)
NumMethod = NumericalMethod(GeomNS, GeomPSE, FDOX=FDOX, FDOY=FDOY, stabCoeff=0.5)
NumMethod.SetUp()

# Number of modes (M,N)
# Krypton can be used for crossflow instabilities, but
# this option was never tested M is the number of modes
# in the span-wise direction and N in the temporal dimension
# Here, no cross-flow
M  = 1
dF = 25
# First mode
F0 = 75
# Number of modes
N  = 2

# Initializing Parameter
setParam = Parameter(M,N, NumMethod.PSE.X1D, maxN=N, MFD=False)

#  * * *  Flow properties  * * *
# Reference length
setParam.dic['prop']['Lref']  = Lref
# Reference Mach Number
setParam.dic['prop']['Ma']    = 0.1
# Local/reference Reynolds Number
setParam.dic['prop']['Re']    = Re0
# Non-dimensional frequencies
setParam.dic['prop']['F']     = F0
setParam.dic['prop']['dF']    = dF

# I/O
setParam.dic['io']['LoadName']   = '../_results/example1'
setParam.dic['io']['SaveName']   = '../_results/example1'
setParam.dic['io']['LamFlow']    = 'compute'
setParam.dic['io']['save']       = True
setParam.dic['io']['vtk']        = True

#----------------------------------------
# Note: Even if save == False, a table containing the raw laminar
#       flow vector (phi) is saved in _tables/<SaveName>_lamFlow.bin
#       which can be loaded using hickle.load(<path>)
# ---------------------------------------
# Where is the Laminar flow coming from ?
# ---------------------------------------
# load   : You simply load the results from a previous solution.
#          in that case, make sure the dimensions (Nx, Nelmy, Npoly)
#          are the same than in the saved solution. Krypton
#          assumes the results are located in _results/XXXX
#          where XXXX is the name given in : 
#
#          setParam.dic['io']['LoadName']   = 'XXXX'
#
# interp : This option is tricky, it load the PHI matrix and
#          and interpolate the solution from it. It is meant
#          to be used as initial conditions for more complex
#          laminar flow computation. For instance, if the
#          laminar solver does not converge for a particular 
#          case. Krypton assumes the results are located in 
#          _tables/LamFlow.mat
#
#          setParam.dic['io']['LoadName']   = 'XXXX'
#
# compute: The laminar flow is simply computed from the input
#          in this file.
#
#------------------------------------------------------------
# Modal stability 101:
#
# NonParallel baseflow -> dUdx, dVdx, dWdx, dPdx, dTdx /= 0
#
# Local    : Means that the history effect ARE NOT taken into
#            account -> d phi/dx = 0
#
# Nonlocal : Means that the history effect ARE taken into
#            account -> d phi/dx /= 0
#
# Model choices :
# Linear stability theory : No modes interaction
#       - LST  (linear stability theory)
#       - LPSE (linear Parabolized Stability Equations)
#
# Nonlinear Stability theory:
#       - NPSE (Nonlinear Parabolized Stability Equations)
#------------------------------------------------------------
# Setting up the initial conditions
for m, n in setParam.dic['modes'][0]:
    setParam.dic[(m,n)]['local'][0]    = True
    setParam.dic[(m,n)]['linear'][0]   = True
    setParam.dic[(m,n)]['parallel'][0] = False
    setParam.dic[(m,n)]['A_0'] = 1

# And then the other stations
for i in range(1, NumMethod.PSE.Nx):
    for m,n in [(i,j) for j in range(N) for i in range(M)]:
        setParam.dic[(m,n)]['local'][i]    = False
        setParam.dic[(m,n)]['linear'][i]   = True
        setParam.dic[(m,n)]['parallel'][i] = False

Param = setParam.getParam()

print(        '___________________________________________________'        )
print(        '                                                   '        )
print(        '              L a m i n a r   F l o w              '        )
print(        '___________________________________________________'        )
print(        'System to solve:'                                           )
print(        '       Nx = ', NumMethod.NSE.Nx                             )
print(        '       Ny = ', NumMethod.NSE.Ny                             )
print(        '      DoF = ', 5*NumMethod.NSE.Nx*NumMethod.NSE.Ny          )
time_mf1 = time.time()
Flow = LaminarFlow(NumMethod, Param)
time_mf2 = time.time()
MFT = time_mf2 - time_mf1
print(        'Laminar flow computation time = '            , MFT          )
print(        '___________________________________________________'        )
print(        '                                                   '        )
print(        '           M o d a l     S t a b i l i t y         '        )
time_ff1 = time.time()
Fluc = FlucFlow(NumMethod.PSE, Param, Flow)
time_ff2 = time.time()
FFT = time_ff2 - time_ff1
print(        '___________________________________________________'        )
print(        '                                                   '        )
print(        '        E N D    O F   C O M P U T A T I O N       '        )
print(        '___________________________________________________'        )
print(        'Laminar flow computation time = '            , MFT          )
print(        'Stability computation time    = '            , FFT          )
print(        'Total time elapsed     = '                   , FFT + MFT    )


# Visualization
# Independant of the number of mode solved
# Just enter a list containing the modes you
# want to visualize. Default is modes 1 and 2
# at the first and last station.
Igraph = [0, NumMethod.PSE.Nx-1]

# Plotting graph
plotStuff.plotAll(Param, Flow, Fluc, Igraph, example=1)
