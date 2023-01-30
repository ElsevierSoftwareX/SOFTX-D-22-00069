import os
import sys
from PSE            import PSE
from Field          import Field
import scipy.io as sio
import numpy    as np

def FlucFlow(NumMethod, Param, Flow, i0=0):
    # Discretization
    Nx = NumMethod.Nx 
    Ny = NumMethod.Ny

    # PDE definition
    PDE = PSE(NumMethod, Param, Flow, i0=i0)
    alpha   = np.zeros(Nx, dtype=complex)

    if Param['io']['LamFlow']    == 'outgrid':
        return 

    # Marching
    for i in range(i0, Nx):
        # Station number
        PDE.i  = i

        # Solving system
        modes, alpha, amplitude, Converged = PDE.Solve()

        # Extracting solution
        if Converged:
            print('_______________________________________________________________________________________')
            print('                                                                                       ')
            print('                                   C O N V E R G E D                                   ')
            print('                                                                                       ')
            print('                             * * *   S U M M A R Y   * * *                             ')
            print('                                                   ')
            print('     Reynolds - x  %36.1F' %     Param['prop']['Rex'][i])
            print('     Position - x  %36.1F' %            NumMethod.X1D[i])
            print('                                                   ')
    
            for m,n in modes:
                if alpha[m,n].imag < 0.:
                    sign_i = '-'
                else:
                    sign_i = '+'
               
                if alpha[m,n].real < 0.:
                    sign_r = '-'
                else:
                    sign_r = ' '
                Format = ' alpha ('+str(m)+','+str(n)+')         '+sign_r+' %3.4E '+sign_i+' %3.4E j'
                print(Format % (abs(alpha[m,n].real), abs(alpha[m,n].imag)))
                Format = '      amplitude             %3.4E'
                print(Format % (amplitude[m,n]*Param[(m,n)]['A_0']))

        if os.path.isfile('print.it'):
            print('I have to print')
            os.remove('print.it')
            # Computing growth rate
            Fluc = PDE.extractFluc()
#            RMS  = PDE.RMS()

        if os.path.isfile('break.it'):
            print('I have to go..')
            os.remove('break.it')
            # Computing growth rate
            Fluc = PDE.extractFluc()
#            RMS  = PDE.RMS()
            break


    if Param['io']['save']: 
        # Computing growth rate
        Fluc = PDE.extractFluc()
#        RMS  = PDE.RMS()

        from shutil import copyfile

        # Saving results
        path_results = os.path.join(os.path.dirname(__file__), Param['io']['SaveName'])
        # Copy input file as it might be useful
        if not os.path.exists(path_results):
            os.makedirs(path_results)
        Fluc.dump(path_results+'/Fluc')
#        RMS.dump(path_results+'/RMS')

    return Fluc
