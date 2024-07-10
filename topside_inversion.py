"""
- self consistent method to compute the coefficients Ai for the inversion problem
- We concentrate on the O trace so f'k = fk and there is no iteration procedure,
  instead just compute the Chebyshev coefficients once
"""
RED = "\033[31m"
GREEN = "\033[92m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"
RESET = "\033[0m"

from scipy import constants
from scipy import integrate
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy.polynomial.chebyshev as cheb

def read_data():
        route = r"2024/INRAS internship/code/"
        df    = pd.read_csv(route+'new_Ne_z_data.csv')
        z     = df['z'].values
        Ne    = df['Ne'].values
        df    = pd.read_csv(route+'new_frequency_data.csv')
        f     = df['f'].values
        return Ne,z,f

Ne, h, range_f = read_data()

filename = 'analytic_data.csv'

if not os.path.isfile(filename):
    print("Producing artificial data")

    ########################################################################################
    ############################## ARTIFICIAL DATA #########################################
    ########################################################################################
    ########################################################################################


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
        if f_probe<max_possible_fp:
            
            final_id = len(fp2)-1
            n2l = 1 - fp2[final_id]/f_probe
            integral = 0

            if n2l>0:

                sqrt_n2l = np.sqrt(n2l)
                for i in range(final_id,0,-1):

                    n2r = 1 - fp2[i-1]/f_probe
                    if n2r>0:
                        sqrt_n2r = np.sqrt(n2r)
                        integral += 2*dif_h[i-1]/(sqrt_n2l+sqrt_n2r)
                        n2l = n2r
                        sqrt_n2l=sqrt_n2r
                    else:
                        integral += 2*dif_h[i-1]*sqrt_n2r/(n2l-n2r)
                        break
            f.append(f_i)
            hv.append(integral)
    """
    at this point we've created a topside ionogram
    f  : frequencies
    hv : virtual reflection height but downwards, remember the pulse is emitted from a satellite 
    """
    # save data
    data = np.column_stack((f,hv))
    np.savetxt('analytic_data.csv', data, delimiter=',', header='f,hv', comments='')
else:
    print(f"{GREEN}~ There is already some artificial data in the directory{RESET}")
########################################################################################
########################################################################################
################################# DATA READ AND TESTING ######################################################
########################################################################################

def read_data2():
    df        = pd.read_csv('analytic_data.csv')
    freq      = df['f'].values
    virtual_h = df['hv'].values
    return freq,virtual_h


def test_gt(f_prime,fNs,fNm):
    """
    - test if with 0 <= t <= 1 and all the f_prime data
      0 <= g <= 1 holds
    - important to add an epsilon to the range of t
    """
    eps = 1e-5
    t_vals = np.linspace(0+eps, 1-eps, num=100) 
    for fi in f_prime:
        for t in t_vals:
            fN2 = fi**2 - (t**2)*(fi**2 - fNs**2)    
            g = np.log(np.sqrt(fN2)/fNm)/np.log(fNs/fNm)
            if g<0 or g>1:
                print(f"{YELLOW}INCORRECT DATA{RESET}")
                return
    print(f"{GREEN}~ DATA for fNs:plasma freq at the satellite location, fNm: critical frequency of the O trace  SEEMS CORRECT{RESET}")

f_data,h_data = read_data2()

fig, ax = plt.subplots()
ax.xaxis.set_ticks_position('top')
ax.xaxis.set_label_position('top')
ax.spines['top'].set_position(('outward', 0))  # Move the top spine outward to make space for the ticks and label

plt.xlabel("frequency [MHz]")
plt.ylabel("range [Km]")
plt.plot(f_data,h_data) # f , hv
plt.gca().invert_yaxis()
plt.show()

################################################################################################
################################################################################################


mag_angle        = np.radians(0) # assuming we are in the magnetic equator
truncate         = 7
gyrofrequency    = 1.0 # estimated from paper1
fNs              = f_data[0] # Satellite critical frequency, estimated from paper2
foF2             = f_data[-1] # MHz, F layer critical frequency, estimated from plot above

test_gt(f_prime=f_data,fNs=fNs,fNm=foF2)
print("foF2 = ", foF2, " [MHz] | Z_m = ",h_data[-1], " [Km] | fNs or fN_satellite = ",fNs, " [MHz]")
print("some frequency values from the data (data in MHz)",f_data[:20])
print("virtual reflection height from data (data in Km) ", h_data[:20])

#####################################################################################
#####################################################################################
########################### COMPUTATION MATRIX Sik ##################################
#####################################################################################

def compute_Sik(f_prime,fNs,fNm, fH, I, theta):
    """
    f_prime : reduced frequency | we are studying the O trace then set of frequencies f'k = fk, same as DATA
    fNs     : plasma frequency at the position of the satellite
    fMm     : critical frequency of the O trace, generally denoted foF2
    fH      : gyrofrequency
    I       : truncated order of the polynomial
    theta   : dip angle of the earth's magnetic field (does it depend on z?) 
    """
    mat = np.zeros((I,len(f_prime)))
    for i in range(I):
        print(i)
        for k in range(len(f_prime)):
            
            fk = f_prime[k]

            def fN2(t):
                return  fk**2-(t**2)*(fk**2 - fNs**2)

            def g_t(t):
                return np.log(np.sqrt(fN2(t))/fNm)/np.log(fNs/fNm)

            def cheb(t,j): # shifted Chebyshev polynomial
                if j==0: return 1
                if j==1: return 2*t-1
                return 2*(2*t-1)*cheb(t,j-1)-cheb(t,j-2)

            def d_cheb(t,j): # derivative of Chebyshev polynomial
                if j==0: return 0
                if j==1: return 2
                return d_cheb(t,j-1)*(2*(2*t-1))+4*cheb(t,j-1)-d_cheb(t,j-2)

            def u_prime(t): # for the ordinary trace
                Xo    = fN2(t)/fk**2
                yo    = fH/fk
                to    = np.sqrt(1-Xo)
                gamma = (4*np.tan(theta)**2)/((yo**2)*(np.cos(theta)**2))
                deno  = (1+gamma*to**4)**(1/2)
                M     = 1+(to**2)*(2*np.tan(theta)**2)/(1+deno)
                to_no_numerator   = 1+(to**2)*(2*np.tan(theta)**2)/(1+deno) 
                to_no_denominator = 1+(2*np.tan(theta)**2)/(1+deno)
                #to_no = np.sqrt((1+(to**2)*(2*np.tan(theta)**2)/(1+deno))/((1+(2*np.tan(theta)**2)/(1+deno))))
                to_no = np.sqrt(to_no_numerator/to_no_denominator)
                Go    = (to_no)*(1 + ( (Xo*np.tan(theta)**2)/M**2 )*( (1+Xo)/deno - 2/(1+deno)  )  )
                return Go

            def integrand(t,it):
                deno = 1/((fN2(t))*(g_t(t)**(1/2)))
                ans1 = u_prime(t)*t*deno
                ans2 = cheb(g_t(t),it) + 2*g_t(t)*d_cheb(g_t(t),it)
                return deno*ans1*ans2

            coeff        = (fk**2-fNs**2)/(2*np.log(fNs/fNm))
            eps          = 1e-6
            integral, _  = integrate.quad(integrand,0+eps,1-eps,args=(i))
            mat[i][k]    = coeff*integral
    return mat

filename = 'matrix_values.csv'
Sik = None
if not os.path.isfile(filename):
    print(f"{YELLOW}~ producing matrix S_ik{RESET}")
    Sik = compute_Sik(f_prime=f_data,fNs=fNs,fNm=foF2,fH=gyrofrequency,I=truncate,theta=mag_angle)
    print(f"{GREEN}calculation of matrix Sik is complete{RESET}")
    print(Sik.shape)
    np.savetxt("matrix_values.csv", Sik, delimiter=",")
else:
    print(f"{GREEN}~ Matrix values found in current directory{RESET}")
    Sik = np.loadtxt("matrix_values.csv", delimiter=",")

Pk = h_data
Qij = np.dot(Sik,Sik.T)
LHS = np.dot(Pk,Sik.T)

"""
- where is the initial profile, where is applied? 
- the condition of A_{I+1} = Z_m should hold for what case? O or X? 
"""

#solve the linear system
A      = np.linalg.solve(Qij,LHS)
A_last = -np.sum(A)
print("A_i coefficients: ", A)
print("A_I+1 = ",A_last, "negative of the all together sum of A_i")


"""
Computation of the Ne profile using equation (6) from paper2
- A: A_i coefficients for the Chebyshev sum
"""

def function_z(g,I):

    def cheb(t,j): # shifted Chebyshev polynomial
        if j==0: return 1
        if j==1: return 2*t-1
        return 2*(2*t-1)*cheb(t,j-1)-cheb(t,j-2)
    
    cheb_sum = sum(A[i] * cheb(g, i) for i in range(I))

    return A_last + (g**(1/2))*cheb_sum

def function_g(fN,fNm,fNs):
    return np.log(fN/fNm)/np.log(fNs/fNm)

# sweep through Ne values

eps = 1e-5
values_f = np.linspace(0+eps,foF2,num=200)

temp = function_g(values_f,foF2,fNs)
maxi = np.max(temp)
mini = np.min(temp)
min_index = np.argmin(temp)
print("maximum: ",maxi, "  | minimum: ",mini)
print("frequency in MHz causing g=0 is ",values_f[min_index])

F2_range = np.array([])
Ze_range = np.array([]) 

for value in values_f:
    fN = value # in MHz units
    pre_g = function_g(fN,foF2,fNs) 
    # values fN should be lesser than foF2
    # that is the only way the condition for g is gonna hold
    if pre_g<0 or 1<pre_g:
        continue
    computed_Zne = function_z(pre_g,truncate)
    F2_range = np.append(F2_range,value**2)
    Ze_range = np.append(Ze_range,computed_Zne)



unit_MHz2 = 1e12
Ne_range = unit_MHz2 * F2_range * (1/80.6)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# ax1.set_title("data Ne vs h plot WITH log scale")
# ax1.set_xscale("log")
# ax1.plot(Ne,h)
# ax1.set_xlabel('f$N_e$')
# ax1.set_ylabel("Altitude")

# ax2.set_title("algorithm result WITH log scale")
# ax2.set_xscale("log")
# ax2.set_ylabel("Altitude [Km]")
# ax2.set_xlabel('f$N_e$')  # Using LaTeX formatting for the label
# ax2.plot(Ne_range,Ze_range)

ax1.set_title("data Ne vs h plot WITHOUT log scale")
ax1.plot(Ne,h)
ax1.set_ylabel("altitude")
ax1.set_xlabel('f$N_e$')

# ax2.xaxis.set_ticks_position('top')
# ax2.xaxis.set_label_position('top')
# ax2.spines['top'].set_position(('outward', 0))  # Move the top spine outward to make space for the ticks and label
ax2.invert_yaxis()
ax2.set_title("algorithm result WITHOUT log scale")
ax2.set_ylabel("Altitude [Km]")
ax2.set_xlabel('f$N_e$')  # Using LaTeX formatting for the label
ax2.plot(Ne_range,Ze_range)

plt.show()


"""
With A and S_ik we can build a new P'(f_k) and compare it with the original ionogram
"""


print("A coefficient values: ",A)





# cutoff = 0.5 # Km
# convergence_steps = 10
# prev_A = ""
# Pk  = "" # shape = (K)
# for i in range(convergence_steps): # 10 iterations until convergence

#     # shape = (I,K)
#     Sik = compute_Sik(f_prime=f,fNs=fN_satellite,fNm=foF2,fH=gyrofrequency,I=truncate,theta=mag_angle)
    
#     # shape = (I,I)
#     Qij = np.dot(Sik,Sik.T)


#     # form the linear equations P'S = AQ, where A are our variables
#     LHS = np.dot(Pk,Sik.T)
#     # solve the linear system
#     A = np.linalg.solve(Qij,LHS)

#     difference = np.maximum(np.absolute(prev_A-A))

#     if difference < cutoff: # remember that the units of both should be Km
#         break
