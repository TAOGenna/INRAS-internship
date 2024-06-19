import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import scipy.integrate as integrate


z0=300
h0=8.1
chi=0 # zenith angle
constant_to_electron_ion_density=1 # its real valur is computed later

def equation_find_zr(x,f,cte=constant_to_electron_ion_density,z0=z0,h0=h0,chi=chi):
    """
    given a frequency f, the function solves the equation of f_p( z_r ) = f
    and finds the corresponding z_r
    """
    sec_chi = 1/np.cos(np.radians(chi))
    return [np.sqrt(80.6)*np.sqrt( cte*np.sqrt(np.exp(1-((x[0]-z0)/h0)-np.exp(-(x[0]-z0)/h0)*sec_chi)) )-f]

def plasma_frequency(z,cte=constant_to_electron_ion_density,z0=z0,h0=h0,chi=chi):
    """
    density of electrons for a given z
    """
    Z       = ( z - z0 ) / h0
    sec_chi = 1/np.cos(np.radians(chi))
    fp      = np.sqrt(80.6)*np.sqrt(cte*np.sqrt(np.exp(1-Z-np.exp(-Z)*sec_chi)))
    return fp

def chapman_function(z, z0=z0, h0=h0, chi=chi):
    Z       = (z - z0) / h0
    sec_chi = 1 / np.cos(chi)
    q_q0    = np.exp(1 - Z - np.exp(-Z)*sec_chi)
    return q_q0

def integrand(z,frecuency):
    """
    finds the ionospheric virtual reflection height
    """
    return 1.0/np.sqrt(1.0-(plasma_frequency(z)/frecuency)**2)


# Altitude range
z = np.linspace(100, 800, 500)  # km
max_electron_density = max(np.sqrt(chapman_function(z, z0, h0, chi)))
constant_to_electron_ion_density = 1e12/max_electron_density # cte = sqrt{q0/alpha}. N_e(z)=(cte)x(sqrt{q(z)})


eps=1e-5
len=1000
frequencies = np.linspace(0.02, 9-eps, len)  # MHz
virtual_heigths = [0]*len


for index,f in enumerate(frequencies):
    min_root=1e9+7
    for guess in range(z0): # here we brute force the initial guess
        root = optimize.fsolve(equation_find_zr, [guess], args=(f,)) # compute the root for our equation y(z)=fp(z)-f=0, where f is defined as a global constant
        prueba_f = plasma_frequency(root) # we check if that root actually gives the frequency previously mentioned
        if(root[0]!=guess and abs(prueba_f-f)<eps):
            min_root = min(min_root,root[0]) # there is gonna be two roots, we choose the minimum one
    if min_root!=1e9+7 and min_root>10:
        h_f , _  = integrate.quad(integrand,0,min_root,args=(f))
        virtual_heigths[index]=h_f

# plot f(MHZ) vs h(Km)
# filtering the arrays
x_range = [x for ind,x in enumerate(frequencies) if virtual_heigths[ind]>10]
y_range = [x for ind,x in enumerate(virtual_heigths) if virtual_heigths[ind]>10]
plt.ylim(0, 450) # zoom in in relevant y range
plt.ylabel('Km',rotation='horizontal')
plt.xlabel('f MHz')
plt.title('ionospheric virtual reflection height')
plt.plot(x_range,y_range)
plt.show()
