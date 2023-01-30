                   | |/ /                | |                    
                   | " / _ __ _   _ _ __ | |_ ___  _ __         
                   |  < | "__| | | | "_ \| __/ _ \| "_ \        
                   | . \| |  | |_| | |_) | || (_) | | | |       
                   |_|\_\_|   \__, | .__/ \__\___/|_| |_|       
                               __/ | |                          
                              |___/|_|                          
                                                                
# Date    : 2022-03
# Author  : Francis Lacombe, PhD Student
# Contact : flacombe@uwaterloo.ca
# Version : 1.0
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

# NPSE
# System solved : (L + N*D + P*D^2 + Q*da/dx)*phi = -M*dphi/dx + NLT
#
#
# The code uses a combination of numerical method in Y and X. For the normal
# direction (Y), it uses a Pseudo-spectral collocation method and for the 
# streamwise direction (X), it uses Euler backward differentiation (BDF).

Dependency:
	- Latest python version (3.6.4 or higher), I recommend installing
	  the Anaconda Distribution (way faster).
	  Link : https://anaconda.org/anaconda/python

	- Python Packages: Numpy, Scipy, Matplotlib, hickle, pypardiso, pyevtk

	-conda install -c conda-forge numpy, scipy, matplotlib, hickle, pypardiso, pyevtk


Getting Started

	The file main.py is the first script you should check. This is where you will set the problem
	parameters (Ma, Re, Pr, wall temperature, pressure gradient, etc.), this is also where you
	set the discretization, BDF order. This is basically the only file you need to edit if you want
	to run a simulation.

	command
		python main.py

	At some point, you might have to develop the code. The stability equations are located in the
	PSE.py script.

	The code is object oriented and designed so it is easy to add new features and change the equations
	solved. If you have any suggestion to improve the efficiency or make the code clearer, I am open to
	suggestion.

	At the moment, there is no documentation (other than this readme file).
