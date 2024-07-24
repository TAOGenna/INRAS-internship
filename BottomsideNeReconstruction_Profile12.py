import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy import constants
from scipy import integrate
from scipy.interpolate import splrep, BSpline
import numpy.polynomial.chebyshev as cheb
import inspect
import time
import tkinter as tk  # or use pyautogui or wx

RED     = "\033[31m"
GREEN   = "\033[92m"
YELLOW  = "\033[33m"
BLUE    = "\033[34m"
MAGENTA = "\033[35m"
CYAN    = "\033[36m"
RESET   = "\033[0m"

np.set_printoptions(threshold=np.inf)

def trace(*args):
    frame = inspect.currentframe().f_back
    var_names = {id(v): k for k, v in frame.f_locals.items()}
    
    for arg in args:
        var_id = id(arg)
        name = var_names.get(var_id, '<unknown>')
        print(f"{name}:\n {arg}")

# some configuration for the plots 
root = tk.Tk()
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
root.destroy()

# Convert screen dimensions from pixels to inches (assuming 100 dpi)
dpi = 100  # This can vary depending on your display settings
figsize = (screen_width / dpi, screen_height / dpi)

# physical constants
qe = constants.e
e0 = constants.epsilon_0
me = constants.electron_mass
pi = constants.pi

def ne_to_fp2(ne): # Ne should be in units of cm-3
    return ne*(qe**2/(e0*me)/(2*pi)**2)

def fp2_to_ne(fp2):
    return fp2*((2*pi)**2)*(e0*me)/(qe**2)

# plasma_freq = np.array([ 0, 0.5, 1, 1.5,  2, 2.3, 2.566,  2.7, 2.8,   3, 3.4,   4, 4.5,   5, 5.5,   6, 6.5,   7, 7.5,   8, 8.5,   9, 9.5,  10, 10.5,  11, 11.5,  12, 12.5,  13, 13.2])
# height      = np.array([79,  80, 83, 87, 94, 100, 107,  114, 125, 140, 155, 168, 174, 180, 185, 191, 196, 201, 206, 210, 215, 220, 224, 228,  233, 237,  241, 247,  254, 265,  275])

plasma_freq = np.array([ 0, 0.5, 1, 1.5,  2, 2.3, 2.566,  2.7, 2.8,   3, 3.4,   4, 4.5,   5, 5.5,   6, 6.5,   7, 7.5,   8, 8.5,   9, 9.5,  10, 10.5,  11, 11.5,  12, 12.5,  13, 13.2])
height      = np.array([79,  80, 83, 87, 94, 100, 107,  114, 125, 140, 155, 168, 174, 180, 185, 191, 196, 201, 206, 210, 215, 220, 224, 228,  233, 237,  241, 247,  254, 265,  275])
highest_freq = plasma_freq[-1] # MHz

plasma_freq *= 1e6 # to make it MHz
height = height.astype(np.float64)
height *= 1e3 # to make it Km

x = plasma_freq
y = height

number_points_d_layer = 20

fp_range = plasma_freq
hh_range = height
font_size = 16
######### experiment interpolating ############
tck = splrep(fp_range, hh_range, s=0)
xnew = np.arange(0,highest_freq,1/40)*1e6 # frequency in MHz
ynew = BSpline(*tck)(xnew)
def initial_plot_interpolate():
    plt.title('Plasma Frequency vs Height | PROFILE AFTER INTERPOLATION',fontsize=font_size)
    plt.xlabel('PLASMA FREQUENCY (MHZ)',fontsize=font_size)
    plt.ylabel('HEIGHT (KM)',fontsize=font_size)
    plt.xticks(fontsize = font_size)
    plt.yticks(fontsize = font_size)
    plt.plot(xnew/1e6, ynew/1e3, '-', label='s=0')
    plt.show()
initial_plot_interpolate()
###############################################

fp_range = xnew
hh_range = ynew

hh_to_add = np.linspace(0,hh_range[0],num=number_points_d_layer)
fp_to_add = np.zeros(number_points_d_layer)

fp_range = np.append(fp_to_add,fp_range)
hh_range = np.append(hh_to_add,hh_range)


def initial_plot_fp():
    plt.plot(fp_range/1e6, hh_range/1e3, 'o', color='black')
    plt.title('Plasma Frequency vs Height | w/ more data points',fontsize=font_size)
    plt.xlabel('PLASMA FREQUENCY (MHZ)',fontsize=font_size)
    plt.ylabel('HEIGHT (KM)',fontsize = font_size )
    plt.xticks(fontsize = font_size)
    plt.yticks(fontsize = font_size)
    plt.xlim((0,14))
    plt.ylim((0,440))
    plt.savefig("Figure 10 Plasma Frequency vs Height.pdf", format="pdf")
    plt.show()

initial_plot_fp()
ne_range = fp2_to_ne(fp_range**2)#(fp_range**2)/80.6

def initial_plot_ne():
    plt.plot(ne_range, hh_range/1e3, 'o', color='black')
    plt.xlabel('NE (CM-3)',fontsize=font_size)
    plt.ylabel('HEIGHT (KM)',fontsize=font_size)
    plt.title('Electron density profile',fontsize=font_size)
    plt.xticks(fontsize = font_size)
    plt.yticks(fontsize = font_size)
    plt.show()

initial_plot_ne()


def ionogram(h,Ne,range_f):
    if h[0]!=0:
        h  = [0]+h
        Ne = [0]+Ne

    n_elements = len(h)
    dif_h = h[1:len(h)] - h[0:len(h)-1]

    # arrays where we will save the integral results per frequency
    f  = np.array([])
    hv = np.array([])# virtual frequency

    # physical constants
    qe = constants.e
    e0 = constants.epsilon_0
    me = constants.electron_mass
    pi = constants.pi


    # plasma frequecuency ^ 2 / 1e6
    # we divide it by 1e6 to get rid of the units of MHz
    fp2 = ne_to_fp2(Ne)#Ne*(qe**2/(e0*me)/(2*pi)**2)
    max_possible_fp = max(fp2)

    print(os.getcwd())
    for f_i in range_f:
        f_probe = f_i**2
        if f_probe<max_possible_fp:
            n2l = 1 - fp2[0]/f_probe
            integral = 0
            if n2l>0:

                sqrt_n2l = np.sqrt(n2l)
                for i in range(n_elements-1):
                    n2r = 1 - fp2[i+1]/f_probe
                    if n2r>0:
                        sqrt_n2r = np.sqrt(n2r)
                        integral += 2*dif_h[i]/(sqrt_n2l+sqrt_n2r)
                        n2l = n2r
                        sqrt_n2l=sqrt_n2r
                    else:
                        integral += 2*dif_h[i]*sqrt_n2r/(n2l-n2r)
                        break
            f  = np.append(f,f_i)
            hv = np.append(hv,integral)
    def print_ionogram():
        plt.plot(f,hv)
        plt.title('IONOGRAM',fontsize=font_size)
        plt.xlabel('FREQUENCY (MHZ)',fontsize=font_size)
        plt.ylabel('HEIGHT (KM)',fontsize=font_size)
        plt.xticks(fontsize = font_size)
        plt.yticks(fontsize = font_size)
        plt.show()
    #print_ionogram()
    return f, hv


#range_f = np.linspace(0.1,13.2,num=300)*1e6
range_f = np.linspace(fp_range[0],fp_range[-1],num=500)
f, hv = ionogram(h=hh_range,Ne=ne_range, range_f=range_f)
print('highest altitude in ionogram ',hv[-1])
print('highest altitud in Ne profile ',ne_range[-1])
np.savetxt("f_values.csv", f, delimiter=",")
eps = 1e-6
#print("this is what comes out of function ionogram, check units: ",f)
# assume there is no valley then there will be only E layer and F layer

# E layer is defined by z_E and y_E paramenters. See Fig 9 of paper 3
# since we are omiting the data analysis of the E layer we just cheat
# and take a look at the electron density profile we have

def delta_he_prime(fk,zE,yE,fE):
    return  zE-yE+(1/2)*(yE)*(fk/fE)*np.log((fk+fE)/(fk-fE))

def delta_he(fk,zE,yE,fE):
    return  zE-yE+(1/2)*(yE)*(fk/fE)*np.log((fE+fk)/(fE-fk))

def compute_Sik2(f_prime,fE,fF, fH, I, theta):
    trace(fE,fF,fH,I,theta)
    """
    f_prime : reduced frequency | we are studying the O trace then set of frequencies f'k = fk, same as DATA
    fE      : critical frequency of the E layer
    fF      : critical frequency of the F layer
    fH      : gyrofrequency
    I       : truncated order of the polynomial
    theta   : dip angle of the earth's magnetic field (does it depend on z?)
    """
    def cheb(t,j): # shifted Chebyshev polynomial
        if j==0: return 1
        if j==1: return 2*t-1
        return 2*(2*t-1)*cheb(t,j-1)-cheb(t,j-2)

    def d_cheb(t,j): # derivative of Chebyshev polynomial
        if j==0: return 0
        if j==1: return 2
        return d_cheb(t,j-1)*(2*(2*t-1))+4*cheb(t,j-1)-d_cheb(t,j-2)

    def integrand(t,itera,theta,fk,fE,fF,fH):
        #trace(t,iterations,theta,fk,fE,fF,fH)
        # print(t)
        # print(itera)
        # print(theta)
        # print('-------------')
        fN2  = fk**2-(t**2)*(fk**2 - fE**2)
        #print(fN2)
        #print(np.log(fE/fF))
        #print(np.sqrt(fN2/fF))
        g_t  = np.log(np.sqrt(fN2)/fF)  /   np.log(fE/fF)
        #print(g_t)
        int_denominator = 1/(fN2*(g_t**(1/2) ) ) # esto es muy chiquito pero bueno
        #### U PRIME #################################################################################
        Xo    = fN2/(fk**2)
        yo    = fH/fk
        to    = np.sqrt(1-Xo)
        gamma = (4*(np.tan(theta)**2))/((yo**2)*(np.cos(theta)**2))
        deno  = (1+gamma*(to**4))**(1/2)
        M     = 1+(to**2)*(2*np.tan(theta)**2)/(1+deno)
        to_no_numerator   = 1+(to**2)*(2*(np.tan(theta)**2))/(1+deno)
        to_no_denominator = 1+(2*(np.tan(theta)**2))/(1+deno)
        to_no = np.sqrt(to_no_numerator/to_no_denominator)
        U_PRIME    = (to_no)*(1 + ( (Xo*(np.tan(theta)**2))/(M**2) )*( (1+Xo)/deno - 2/(1+deno)  )  )
        ##############################################################################################
        int_part1 = U_PRIME*t
        int_part2 = cheb(g_t,itera) + 2*g_t*d_cheb(g_t,itera)
        #trace(int_denominator,int_part1,int_part2)
        return int_denominator*int_part1*int_part2

    mat = np.zeros((I,len(f_prime)))

    for i in range(I):
        for k in range(len(f_prime)):
            fk = f_prime[k]
            coeff        = (fk**2-fE**2)/(2*np.log(fE/fF))
            integral, _  = integrate.quad(integrand,0+eps,1-eps,args=(i,theta,fk,fE,fF,fH))
            mat[i][k]    = coeff*integral

    return mat

"""
definition of values from scaling
"""
truncate = 8
magnetic_angle = np.radians(0)
fE = 2.717e6 # DATA
nE = fp2_to_ne(fE**2)
yE = 1.335e5-7.94e4 # DATA - KM --- half thickness of the parabolic profile | put by hand
zE = 1.351e5  # DATA - KM --- height of the E layer peak
fF = range_f[-1]   #MHZ --- critical frequency of the F layer
fH = 1.0e6
index = np.argmax(f > fE)# ALWAYS make sure just take f>fE

# data only considering F layer
hv_Flayer   = hv[index:]
freq_Flayer =  f[index:]

start_time = time.time()

Sik = compute_Sik2(f_prime=freq_Flayer,fE=fE,fF=fF,fH=1e-6,I=truncate,theta=magnetic_angle)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time for matrix S_ik: {elapsed_time} seconds")

"""
compute Chebyshev coefficients A_i
"""
# we are just interested in the F layer so we cut the data to get only 
# F layer frequencies and ionogram heights 

Pk = hv_Flayer - delta_he_prime(fk=freq_Flayer,zE=zE,yE=yE,fE=fE)
##############
plt.title('comparion before and after Pk-delta_h_e_prime')
plt.plot(freq_Flayer/1e6,Pk/1e3,label='reduced')
plt.plot(freq_Flayer/1e6,hv_Flayer/1e3,label='original')
plt.legend()
plt.show()
##############
def check():
    index = np.array([], dtype=int)
    for it,elem in enumerate(Pk):
        if elem>0:
            index = np.append(index,it)
    return Pk[index], Sik[:,index], freq_Flayer[index], index

Pk, Sik, freq_Flayer, index_list = check()

chi = np.prod(Pk>0)
if chi:
    print(f'{GREEN}OK All Pk elements are positive{RESET}')
    #print('maximun distance between h and Delta h_E is', np.amax(np.abs(hv_Flayer[index_list]-delta_he_prime(fk=freq_Flayer,zE=zE,yE=yE,fE=fE))))
else:
    print(f'{RED}ERROR, there are elements in Pk which are negative{RESET}')

#print(Sik)
Qij = np.dot(Sik,Sik.T)
Qij = np.column_stack((Qij,np.zeros(Qij.shape[0])))
Qij = np.vstack((Qij,np.ones(Qij.shape[1])))

LHS = np.dot(Pk,Sik.T)
LHS = np.append(LHS,zE)
A = np.linalg.solve(Qij,LHS)
A_last = A[-1]
A = np.delete(A,-1)

trace(A_last,zE)
print('A array elements: ',A)


"""
check if it reseambles the ionogram
"""
def ionogram_reconstruction():
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
    ax1.tick_params(labelsize=font_size)
    ax2.tick_params(labelsize=font_size)
    ax3.tick_params(labelsize=font_size)

    ax1.set_title("Original ionogram",fontsize=font_size)
    ax1.set_xlabel("frequency [MHz]",fontsize=font_size)
    ax1.set_ylabel("range [Km]",fontsize=font_size)
    #ax1.set_xlim(0,1.4e7)
    ax1.set_ylim(0,hv[-1]/1e3) # zoom in in relevant y range
    ax1.plot(f/1e6,hv/1e3,label='F layer')
    ax1.legend()

    freq_Elayer = f[f < fE]
    P_elayer = delta_he(fk=freq_Elayer,zE=zE,yE=yE,fE=fE)
    ax2.plot(freq_Elayer/1e6,P_elayer/1e3,label='E layer')

    P = delta_he_prime(fk=freq_Flayer,zE=zE,yE=yE,fE=fE) + np.transpose(Sik).dot(A)
    ax2.set_title("Reconstruction",fontsize=font_size)
    ax2.set_xlabel("frequency [MHz]",fontsize=font_size)
    ax2.set_ylabel("range [Km]",fontsize=font_size)
    #ax2.set_xlim(0,1.4e7)
    ax2.set_ylim(0,P[-1]/1e3) # zoom in in relevant y range
    ax2.plot(freq_Flayer/1e6,P/1e3,label='F layer')
    ax2.legend()

    ax3.set_title('Superposition',fontsize=font_size)
    ax3.plot(freq_Flayer/1e6,P/1e3       ,'o',label='reconstructed',color='red',alpha=0.5)
    ax3.plot(freq_Elayer/1e6,P_elayer/1e3,'o',label='',color='red',alpha=0.5)
    ax3.plot(f/1e6,hv/1e3,label='original',color='blue')
    ax3.legend()
    plt.xticks(fontsize = font_size)
    plt.yticks(fontsize = font_size)

    # for ax in [ax1, ax2, ax3]:
    #     ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True, nbins='auto'))
    #     ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True, nbins='auto'))

    plt.tight_layout()  # Automatically adjust subplot parameters to fit the figure area.
    plt.subplots_adjust(wspace=0.4, hspace=0.6)  # Manually adjust horizontal and vertical spacing if needed.

    plt.savefig("Figure 10 ionogram reconstruction.pdf",format="pdf")
    plt.show()


ionogram_reconstruction()

"""
compute the F layer electron density profile 
"""

def function_z(g,I):
    def cheb(t,j): # shifted Chebyshev polynomial
        if j==0: return 1
        if j==1: return 2*t-1
        return 2*(2*t-1)*cheb(t,j-1)-cheb(t,j-2)
    
    cheb_sum = sum(A[i] * cheb(g, i) for i in range(I))
    return A_last + (g**(1/2))*cheb_sum

def function_g(fN):
    return np.log(fN/fF)/np.log(fE/fF)

answer_frqx = np.array([])
answer_rngy = np.array([])
eps=1e-5

# we are only considering the frequencies for the F layer
values_f = np.linspace(freq_Flayer[0]+eps,freq_Flayer[-1]-eps,num=200)

for value in values_f:
    fN = value
    pre_g = function_g(fN) 
    if pre_g<0 or 1<pre_g:
        continue
    computed_Zne = function_z(pre_g,truncate)
    answer_frqx  = np.append(answer_frqx,value)
    answer_rngy  = np.append(answer_rngy,computed_Zne)

answer_nepx  = fp2_to_ne(answer_frqx**2)

def final_plot_fp():
    plt.xlabel('PLASMA FREQUENCY (MHZ)',fontsize=font_size)
    plt.ylabel('HEIGHT (KM)',fontsize=font_size)
    z_fE  = np.linspace(zE-yE,zE,num=20)
    ne_fE = [nE*(1-((z-zE)/yE)**2) for z in z_fE]
    fp_fE = np.sqrt([ne_to_fp2(nes) for nes in ne_fE])
    otx = np.append(fp_fE,answer_frqx)
    oty = np.append(z_fE,answer_rngy)
    plt.plot(otx/1e6,oty/1e3,label='reconstructed',color='magenta',linewidth=3.2,alpha=0.5)
    plt.plot(fp_range/1e6,hh_range/1e3,label='orginal',color='blue')
    plt.show()

def final_plot_ne():
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
    ax1.tick_params(labelsize=font_size)
    ax2.tick_params(labelsize=font_size)
    ax3.tick_params(labelsize=font_size)

    #ax1.set_xlim(0,2.5e12)
    ax1.set_ylim(0,hh_range[-1]/1e3)
    ax1.set_xlabel("Ne",fontsize=font_size)
    ax1.set_ylabel("range [Km]",fontsize=font_size)
    ax1.set_title('original profile',fontsize=font_size)
    ax1.plot(ne_range, hh_range/1e3)

    #ax2.set_xlim(0,2.5e12)
    ax2.set_ylim(0,answer_rngy[-1]/1e3)
    ax2.set_xlabel("Ne",fontsize=font_size)
    ax2.set_ylabel("range [Km]",fontsize=font_size)
    ax2.set_title('reconstruction',fontsize=font_size)
    ax2.plot(answer_nepx,answer_rngy/1e3,'o',label='F layer',color='blue')
    # PLOT the fE layer contribution
    z_fE  = np.linspace(zE-yE,zE,num=20)
    ne_fE = [nE*(1-((z-zE)/yE)**2) for z in z_fE]
    ax2.plot(ne_fE,z_fE/1e3,'o',label='E layer',color='red')
    ax2.legend()

    #ax3.set_xlim(0,2.5e12)
    ax3.set_title('Superposition')
    ax3.set_ylim(0,answer_rngy[-1]/1e3)
    ax3.plot(ne_range, hh_range/1e3,color='blue',label='original')
    otx = np.append(ne_fE,answer_nepx)
    oty = np.append(z_fE,answer_rngy)
    ax3.plot(otx,oty/1e3,color='magenta',label='reconstructed',linewidth=3.2,alpha=0.5)
    #ax3.plot(ne_fE,z_fE,color='red')
    ax3.legend()

    plt.tight_layout()  # Automatically adjust subplot parameters to fit the figure area.
    plt.subplots_adjust(wspace=0.4, hspace=0.6)  # Manually adjust horizontal and vertical spacing if needed.
    plt.savefig("figure 10 Ne reconstruction.pdf",format='pdf')
    plt.show()

final_plot_fp()
final_plot_ne()
