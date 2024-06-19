import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import scipy.integrate as integrate
import tkinter as tk
import operator

z0=300
h0=8.1
chi=0 # zenith angle


def chapman_function(z, z0=z0, h0=h0, chi=chi):
    """
    Gives the electron density Ne for a height z
    """
    Z       = (z - z0) / h0
    sec_chi = 1 / np.cos(chi)
    q_q0    = np.exp(1 - Z - np.exp(-Z)*sec_chi)
    return q_q0


def calc_parabola_vertex(x1, y1, x2, y2, x3, y3):
	"""
	Adapted and modifed to get the unknowns for defining a parabola:
	http://stackoverflow.com/questions/717762/how-to-calculate-the-vertex-of-a-parabola-given-three-points
	"""
	denom = (x1-x2) * (x1-x3) * (x2-x3);
	A     = (x3 * (y2-y1) + x2 * (y1-y3) + x1 * (y3-y2)) / denom;
	B     = (x3*x3 * (y1-y2) + x2*x2 * (y3-y1) + x1*x1 * (y2-y3)) / denom;
	C     = (x2 * x3 * (x2-x3) * y1+x3 * x1 * (x3-x1) * y2+x1 * x2 * (x1-x2) * y3) / denom;

	return A,B,C

def find_zr(plasma_frequency,frequency):
    """
    finds the solution of the equation fp(z_r)=f
    - since we are only interested in the first encounter of x=f with fp
    we only need to find the segment of points P1,P2 such that its x's components
    enclose, i.e. f. P1.x <= x=f <= P2.x
    - index: index of the array we've reached before
    RETURNS:
    - updated index
    - zr
    """
    frequency*=1e6
    for i in range(0,len(plasma_frequency)-1):
        if plasma_frequency[i] <= frequency and frequency <= plasma_frequency[i+1]:
            return i
    return -1 # if it doesn't find suitable solutions then it returns -1

def integrand(z,A,B,C,frequency):
    #print("frequency ",frequency)
    """
    given a cuadratic expression with coefficients A,B,C; evaluate
    the integral only in the defined range of that parabola.
    """
    frequency*=1e6
    return 1.0/np.sqrt(1.0-((A*z**2+B*z+C)/frequency)**2)


def init():
    """
    Artificial Data - we take as an example the chapman function
    - It is important to preprocess this data, only returning up to the
    highest value in Ne
    """
    steps = 20000 # probably we should need like 1e5 samples to achieve a smooth curve
    z = np.linspace(0.1,800,steps)
    q_q0 = chapman_function(z=z)
    constant_Ne = 1e12/np.sqrt(max(q_q0))
    Ne = constant_Ne*np.sqrt(q_q0)


    # Save Ne and z in a csv file
    data = np.column_stack((z, Ne))
    np.savetxt('Ne_z_data.csv', data, delimiter=',', header='z,Ne', comments='')

    # data for the frequency probes
    steps_f=1000
    f = np.linspace(0.1, 10, steps_f)
    np.savetxt('frequency_data.csv', f, delimiter=',', header='f', comments='')
    # some preprocessing
    index, value = max(enumerate(Ne), key=operator.itemgetter(1))
    return Ne[:index+1], z[:index+1], f

"""
Data processing before integration:
- we are given a Ne array with the electron density per height
- 1)first we need to get the plasma frequency out of the Ne through
    the formula f_p = sqrt{80.6}*sqrt{Ne}
- 2)for each frequency we need to find its corresponding zr
"""

# some additional settings for how the plots will appear in the monitor
window_x = 100  # X coordinate for the window
window_y = 100  # Y coordinate for the window

Ne, z, F = init() # this can be modified to receive any kind of Ne

print("electron density ", Ne[:10])
print("heights ",z[:10])
print("frequency samples ",F[:10])


plasma_frequency = np.sqrt(80.6)*np.sqrt(Ne)

f_steps = len(F)
eps=1e-5
#F = np.linspace(0.1, 10, f_steps) # frequency in MHz whose segment we'd like to know
H = []
FF = []

x = z
y = plasma_frequency
for f in F:
    index_zr = find_zr(plasma_frequency,f)
    if index_zr==-1:
        continue
    integral = 0
    for i in range(0,index_zr+1-2,2):
        A,B,C     = calc_parabola_vertex(x[i],y[i],x[i+1],y[i+1],x[i+2],y[i+2])
        result, _ = integrate.quad(integrand,x[i],x[i+2],args=(A,B,C,f))
        integral  += result
    H.append(integral)
    FF.append(f)


plt.ylim(0, 450) # zoom in in relevant y range
plt.ylabel('Km',rotation='horizontal')
plt.xlabel('f MHz')
plt.title('ionospheric virtual reflection height')
plt.plot(FF,H)
plt.show()

# save data 
data = np.column_stack((FF,H))
np.savetxt('takagui_data.csv', data, delimiter=',', header='f,hv', comments='')
