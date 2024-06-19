# Comments: the constant q0 is not included
# this code can plot the chapman function for ion-production in the ionosphere
# AND the distribution of electron density which is just sqrt{q(z)}x{1e12}.
# 1e12 because we want the peak of the function to be the maximun free electron density

import numpy as np
import matplotlib.pyplot as plt

def chapman_function(z, z0, H0, chi):
    Z = (z - z0) / H0
    sec_chi = 1 / np.cos(chi)
    q_q0 = np.exp(1 - Z - np.exp(-Z)*sec_chi)
    return q_q0

# Parameters
z0 = 300  # km
H0 = 8.4   # km

# Altitude range
z = np.linspace(100, 800, 1000)  # km

# Solar zenith angles in radians
chi_values = [0, np.radians(30), np.radians(60), np.radians(85)]

# Plot Chapman function for different solar zenith angles
plt.figure(figsize=(10, 10))

for chi in chi_values:
    q_q0 = chapman_function(z, z0, H0, chi)
    ne = 1e12*np.sqrt(q_q0) # electron density
    tmp = (z-z0)/H0
    plt.plot(q_q0,tmp, label=f'χ = {np.degrees(chi):.0f}°')

plt.ylim(-3, 7) # zoom in in relevant y range
plt.xlim(-0.1,1.2) # zoom in in relevant x range
plt.ylabel('Reduced height Z')
plt.xlabel('Chapman production function q(z) / q0')
plt.title('Chapman Production Function')
plt.legend()
plt.grid(True)
plt.show()
