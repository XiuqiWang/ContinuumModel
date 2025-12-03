# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 10:58:34 2025

@author: WangX3
"""

import numpy as np
import matplotlib.pyplot as plt 

# burel - c
c = np.linspace(0, 0.3)
Cref_urel = 0.0088
burel1 = 1/(1+c/Cref_urel)
burel2 = 1/np.sqrt(1+c/Cref_urel)

Mdrag_order1 = c*burel1**2
Mdrag_order2 = c*burel2**2

plt.figure()
plt.plot(c, burel2, label='burel=1/np.sqrt(1+c/Cref_urel)')
plt.xlabel('c')
plt.ylabel('b_urel')
plt.legend()

plt.figure()
plt.plot(c, Mdrag_order1, label='burel1=1/(1+c/Cref_urel)')
plt.plot(c, Mdrag_order2, label='burel2=1/np.sqrt(1+c/Cref_urel)')
plt.xlabel('c')
plt.ylabel('Mdrag_order')
plt.legend()

# b - c
Cref = 1.0
b = np.sqrt(1 - c/(c+Cref))
plt.figure()
plt.plot(c, b, label='b=np.sqrt(1-c/(c+Cref))')
plt.xlabel('c')
plt.ylabel('b')
plt.legend()


def CalUincfromU(U, Omega):
    a0, b0 = 0.92, 0.58
    # a1, b1 = 0.01, 0.99
    a1, b1 = 0.098, 0.1
    # if Omega == 0:
    #     # Uinc = 0.43*U
    #     Uinc = 0.61*U**0.44
    # else:
    #     # Uinc = 0.85*U
    #     Uinc = 0.44*U**1.36
    A = a0 + a1*Omega
    n = b0 + b1*Omega
    Uinc = A*U**n
    return Uinc

U = np.linspace(0, 10, 100)
Omega = [0, 0.01, 0.05, 0.1, 0.2]
colors = plt.cm.viridis(np.linspace(1, 0, 5))
plt.figure()
for i in range(5):
    Uinc = CalUincfromU(U, Omega[i])
    plt.plot(U, Uinc, color=colors[i])
plt.xlabel('U')
plt.ylabel('Uinc')

# tuning NE for low Uinc can help improve the increase phase of C (unfinished)
# Omega effect needs to be added for the tuning part
g = 9.81
d = 0.00025                # grain diameter [m]  
const = np.sqrt(g*d)       # √(g d)

def CalUincfromU(U, Omega):
    Cref, Cref_urel, B, p = 1.0, 0.0088, 9.52, 18.05 
    a0, b0 = 0.92, 0.58
    a1, b1 = 0.01, 0.99
    # if Omega == 0:
    #     # Uinc = 0.43*U
    #     Uinc = 0.61*U**0.44
    # else:
    #     # Uinc = 0.85*U
    #     Uinc = 0.44*U**1.36
    A = a0 - a1*Omega
    n = b0 + b1*Omega
    Uinc = A*U**n
    return Uinc

def NE_linear_from_Uinc(Uinc, Omega):
    NE_linear = (0.03-0.028*Omega**0.19) * (abs(Uinc)/const) 
    return NE_linear

def NE_from_Uinc(Uinc, Omega, Uc, dU): 
    NE_lin = (0.03-0.028*Omega**0.19) * (abs(Uinc)/const) 
    switch = 1.0 / (1.0 + np.exp(-(U - Uc)/dU))
    NE = NE_lin * switch
    return NE

U = np.linspace(0, 5, 100)
Uinc = CalUincfromU(U, 0)
NE_linear = NE_linear_from_Uinc(Uinc, 0)
NE_tuned = NE_from_Uinc(Uinc, 0, 1.0, 0.2)

plt.figure()
plt.plot(U, NE_linear)
plt.plot(U, NE_tuned, '--')
plt.xlabel('U');plt.ylabel('NE')

def calc_Pr(Uinc, Omega, params):
    ap, bp, cp = params
    Pr = ap*np.exp(-bp*np.exp(-cp*abs(Uinc)/const))
    return Pr

Uinc = np.linspace(0, 5, 100)
Pr = calc_Pr(Uinc, 0, [0.74, 4.46, 0.10])
Pr2 = calc_Pr(Uinc, 0, [0.45, 4.46, 0.10])
Pr3 = calc_Pr(Uinc, 0, [0.99, 4.46, 0.10])

plt.figure()
plt.plot(Uinc, Pr)
plt.plot(Uinc, Pr2)
plt.plot(Uinc, Pr3)

Pr4 = calc_Pr(Uinc, 0, [0.74, 1.0, 0.10])
Pr5 = calc_Pr(Uinc, 0, [0.74, 10.0, 0.10])

plt.figure()
plt.plot(Uinc, Pr)
plt.plot(Uinc, Pr4)
plt.plot(Uinc, Pr5)

Pr6 = calc_Pr(Uinc, 0, [0.74, 4.46, 0.01])
Pr7 = calc_Pr(Uinc, 0, [0.74, 4.46, 0.20])

plt.figure()
plt.plot(Uinc, Pr)
plt.plot(Uinc, Pr6)
plt.plot(Uinc, Pr7)

Pr_opt = calc_Pr(Uinc, 0, [0.99, 1.20, 0.17])
plt.figure()
plt.plot(Uinc, Pr)
plt.plot(Uinc, Pr_opt)
