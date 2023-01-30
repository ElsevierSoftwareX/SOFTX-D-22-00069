# Contiains main tools for extracting/plotting solution
# from distributed array
import numpy as np
from Field    import Field
from matplotlib import pylab
import matplotlib.pyplot as plt
from scipy.integrate import simps
import scipy.io as sio


def plotAll(Param, Flow, Fluc, Igraph, Mgraph=[0], Ngraph=[1], i0=0, example=-1, Juniper=None, dispGrid=False, nfactor=False, baseFlow=True, stabilityMap=True, JFM=False, plotModes=False):
    # Fig number
    fig = 1
    if Param['io']['save']:
        import os

        if example==1:
            Juniper='PSE'
            baseFlow = True
            nfactor = True

        if example==2:
            baseFlow = True
            nfactor = True
            stabilityMap = True

        if example==3:
            baseFlow = True
            nfactor = True

        # Exctracting data
        if dispGrid:
            plt.figure(fig)
            plt.plot(Flow.xc[i0:], Flow.yc[i0:], 'gray', linewidth='0.25')
            plt.plot(Flow.xc[i0:].T, Flow.yc[i0:].T, 'gray', linewidth='0.25')
            plt.xlabel('x')
            plt.xlabel('y')
            fig += 1

        if nfactor:
            plt.figure(fig)
            plt.title('n-factor curves')
            for m in Mgraph:
                for n in Ngraph:
                    plt.plot(Flow.Rex[Igraph[0]:,0].real, Fluc.nfactor[m,n,Igraph[0]:].real, label='('+str(m)+','+str(n)+')')
            plt.ylabel('n-factor')
            plt.xlabel(r'$Re_x$')
            plt.legend()
            fig += 1

        if stabilityMap:
            plt.figure(fig)
            plt.title('Stability map')
            plt.contourf(Fluc.Rex[0,:,Igraph[0]:,0], Fluc.F[0,:,Igraph[0]:], Fluc.sigma[0,:,Igraph[0]:], 256, cmap='coolwarm')
            plt.clim([-0.02, 0.02])
            plt.contour( Fluc.Rex[0,:,Igraph[0]:,0], Fluc.F[0,:,Igraph[0]:], Fluc.sigma[0,:,Igraph[0]:], levels=[0.])
            plt.ylabel(r'$F=10^6\frac{\omega \nu}{u_0}$')
            plt.xlabel(r'$Re_\delta$')
            fig += 1

        if baseFlow:
            plt.figure(fig)
            plt.title('Laminar base flow')
            plt.subplot(2,2,1)
            plt.title(r'$U/U_{ref}$')
            plt.contourf(Flow.xc[i0:].real, Flow.yc[i0:].real, Flow.U[i0:].real, 96, cmap='coolwarm')
            plt.colorbar()
#            plt.ylim([-5,25])
            plt.contour(Flow.xc[i0:].real, Flow.yc[i0:].real, Flow.U[i0:].real, levels=[0.99*Flow.U[-1,-1].real])
            plt.xlabel(r'$x/\delta$')
            plt.ylabel(r'$y/\delta$')
            plt.subplot(2,2,2)
            plt.title(r'$V/U_{ref}$')
            plt.contourf(Flow.xc[i0:].real, Flow.yc[i0:].real, Flow.V[i0:].real, 96, cmap='coolwarm')
            plt.colorbar()
#            plt.ylim([-5,25])
            plt.xlabel(r'$x/\delta$')
            plt.ylabel(r'$y/\delta$')
            plt.subplot(2,2,3)
            plt.title(r'$P/P_{ref}$')
            plt.contourf(Flow.xc[i0:].real, Flow.yc[i0:].real, Flow.P[i0:].real, 96, cmap='coolwarm')
            plt.colorbar()
            plt.xlabel(r'$x/\delta$')
#            plt.ylim([-5,25])
            plt.ylabel(r'$y/\delta$')
            plt.subplot(2,2,4)
            plt.title(r'$T/T_{ref}$')
            plt.contourf(Flow.xc[i0:].real, Flow.yc[i0:].real, Flow.T[i0:].real, 96, cmap='coolwarm')
            plt.colorbar()
#            plt.ylim([-5,25])
            plt.xlabel(r'$x/\delta$')
            plt.ylabel(r'$y/\delta$')
            plt.show()
            fig += 1

        plt.show()

        if Juniper == 'LST' or Juniper == 'PSE':
            # Juniper's
            # Plotting Growth rate (compared against Juniper's incompressible results
            # Path to results
            path_Juniper = os.path.join('../_juniper/','Juniper_'+Juniper+'.mat')
            JuniperMat = sio.loadmat(path_Juniper)['Juniper']

            plt.figure(fig)
            closestMode = np.argmin(abs(Fluc.F[0,:,0] - 100)) 
            plt.plot(Flow.Rex[i0:,0].real, Fluc.sigma[0,closestMode,i0:].real, 'k-', label='Krypton', linewidth=1)
            plt.plot(JuniperMat[0,:], JuniperMat[1,:], 'k--', label='Juniper ('+Juniper+')', linewidth=1)
            plt.legend()
            plt.xlabel(r'$Re_\delta$')
            plt.ylabel(r'$Growth rate$')
            plt.show()
            fig += 1
        if JFM:
            # Path to results
            plt.figure(fig)
            closestMode = np.argmin(abs(Fluc.F[0,:,0] - 18))
            Fluc_fp = Field(0,0,path='../_results/fp_JFM6/Fluc')
            plt.plot(       Flow.Rex[i0:,0].real, Fluc.sigma[0,closestMode,i0:].real, 'k-', label='Rough', linewidth=1)
            plt.plot(Fluc_fp.Rex[0,1,i0:,0].real, Fluc_fp.sigma[0,1,i0:].real, 'k--', label='Flat plate', linewidth=1)
            plt.legend()
            plt.xlabel(r'$Re_\delta$')
            plt.ylabel(r'$Growth rate$')
            plt.show()
            fig += 1


    if plotModes:
        # Plotting Flow and Fluctuation (for all modes)
        for i in Igraph:
            title = 'Base Flow at x = '+str(Flow.x[i,0])
            plotField(Flow, fig, title, 'Flow', 0, 0, i)
            fig += fig
            for m in Mgraph:
                for n in Ngraph:
                    title = 'Mode ('+str(m)+','+str(n)+') at Re = '+str(Flow.Rex[i,0].real)
                    plotField(Fluc, fig, title, 'Fluc', m, n, i)
                    fig += 1
        pylab.show()

def plotField(Flow, figure, title, Field, m, n, i, ymin=-5, ymax=25):
    # Create figure and plot solution elegantly
    pylab.figure(figure)
    pylab.suptitle(title)
    pylab.style.use('seaborn-whitegrid')

    pylab.subplot(221)
    if Field == 'Fluc':
        pylab.plot(Flow.U[m,n,i,:].real, Flow.y[m,n,i,:].real, linewidth=1)
        pylab.xlabel(r'$\hat{u}/U_{ref}$')
    elif Field == 'Flow':
        pylab.plot(Flow.U[i,:].real, Flow.y[i,:].real, linewidth=1)
        pylab.xlabel(r'$U/U_{ref}$')
    pylab.ylabel(r'$y/\delta$')
    pylab.ylim(ymin, ymax)

    pylab.subplot(222)
    if Field == 'Fluc':
        pylab.plot(Flow.V[m,n,i,:].real, Flow.y[m,n,i,:].real, linewidth=1)
        pylab.xlabel(r'$\hat{v}/U_{ref}$')
    elif Field == 'Flow':
        pylab.plot(Flow.V[i,:].real, Flow.y[i,:].real, linewidth=1)
        pylab.xlabel(r'$V/U_{ref}$')
    pylab.ylabel(r'$y/\delta$')
    pylab.ylim(ymin, ymax)

    pylab.subplot(223)
    if Field == 'Fluc':
        pylab.plot(Flow.P[m,n,i,:].real, Flow.y[m,n,i,:].real, linewidth=1)
        pylab.xlabel(r'$\hat{p}/P_{ref}$')
    # Base Flow pressure is constant (normal direction)
    # So better show crossflow
    elif Field == 'Flow':
        pylab.plot(Flow.P[i,:].real, Flow.y[i,:].real, linewidth=1)
        pylab.xlabel(r'$P/P_{ref}$')
    pylab.ylabel(r'$y/\delta$')
    pylab.ylim(ymin, ymax)

    pylab.subplot(224)
    if Field == 'Fluc':
        pylab.plot(Flow.T[m,n,i,:].real, Flow.y[m,n,i,:].real, linewidth=1)
        pylab.xlabel(r'$\hat{T}/T_{ref}$')
    elif Field == 'Flow':
        pylab.plot(Flow.T[i,:].real, Flow.y[i,:].real, linewidth=1)
        pylab.xlabel(r'$T/T_{ref}$')
    pylab.ylabel(r'$y/\delta$')
    pylab.ylim(ymin, ymax)

def CPUstats(Num, M, N, MFT, FFT):
    f = open('_stats/CPU.time', 'a+')
    f.write(str(Num.NSE.Nx)+' '+str(Num.NSE.FDOX)+' '+str(Num.NSE.Nelmy)+' '+str(Num.NSE.FDOY)+' '+str(Num.NSE.Nelmy*Num.NSE.FDOY)+' '+str(Num.NSE.Nx)+' '+str(N)+' '+str(M)+' '+str(MFT)+' '+str(FFT)+'\n')
    f.close()
