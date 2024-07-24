# code to receive the data
# what we receive is the electron density vs frequency
from scipy import constants
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def read_data():
    df = pd.read_csv('Ne_z_data.csv')
    z = df['z'].values
    Ne = df['Ne'].values
    df = pd.read_csv('frequency_data.csv')
    f = df['f'].values
    return Ne,z,f
# Ne: electron density
# frq: plasma frequency

######################################################################
# data that should be the input
Ne, h, range_f = read_data()
######################################################################

if h[0]!=0:
    h  = [0]+h
    Ne = [0]+Ne

n_elements = len(h)
dif_h = h[1:len(h)] - h[0:len(h)-1]

# not sure about the use of numel vs len in matlab
n_fre = len(range_f)

# arrays where we will save the integral results per frequency
f = []
hv =  [] # virtual frequency

# physical constants
qe = constants.e
e0 = constants.epsilon_0
me = constants.electron_mass
pi = constants.pi


# plasma frequecuency ^ 2 / 1e6
# we divide it by 1e6 to get rid of the units of MHz
fp2 = Ne*(qe**2/(e0*me)/(2e6*pi)**2)
max_possible_fp = max(fp2)


for f_i in range_f:

    f_probe = f_i**2

    # f_probe should be in the range of frequencies that reflects
    # the EM for the first time
    if f_probe<max_possible_fp:

        # INTEGRATION METHOD:
        # we will be using the trapezoid method
        # for this we need to take two consecutive data points
        # as well as the values of the function evaluated at those points
        # int = (a-b)/2 * ( f(a) + f(b) )

        # we start with the first element because we have to compare it
        # with the second and we don't want to add additional logic just
        # to handle the first case
        n2l = 1 - fp2[0]/f_probe

        # variable of the contribution for the integral between those two
        # data points
        integral = 0

        # make sure the square root is positive>=0
        if n2l>0:

            sqrt_n2l = np.sqrt(n2l)
            for i in range(n_elements-1):
                n2r = 1 - fp2[i+1]/f_probe

                # make sure the square root is positive>=0
                if n2r>0:
                    sqrt_n2r = np.sqrt(n2r)

                    # i don't understand this part for the integration
                    integral += 2*dif_h[i]/(sqrt_n2l+sqrt_n2r)

                    # assign values for the next interation
                    n2l = n2r
                    sqrt_n2l=sqrt_n2r
                else:
                    integral += 2*dif_h[i]*sqrt_n2r/(n2l-n2r)
                    break
        f.append(f_i)
        hv.append(integral)

plt.ylim(0, 450) # zoom in in relevant y range
plt.ylabel('Km',rotation='horizontal')
plt.xlabel('f MHz')
plt.title('Milla Python version')
plt.plot(f,hv)
plt.show()

# save data 
data = np.column_stack((f,hv))
np.savetxt('milla_data.csv', data, delimiter=',', header='f,hv', comments='')
