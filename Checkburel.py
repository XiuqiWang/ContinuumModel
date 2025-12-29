# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 10:58:34 2025

@author: WangX3
"""

import numpy as np
import matplotlib.pyplot as plt 

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

# tuning NE for low Uinc can help improve the increase phase of C (unfinished)
# Omega effect needs to be added for the tuning part
g = 9.81
d = 0.00025                # grain diameter [m]  
const = np.sqrt(g*d)       # √(g d)

def NE_linear_from_Uinc(Uinc, Omega, params):
    a, b, c = params
    NE_linear = (a-b*Omega**c) * (abs(Uinc)/const) 
    return NE_linear

def NE_from_Uinc(Uinc, Omega, Uc, dU): 
    NE_lin = (0.03-0.025*Omega**0.19) * (abs(Uinc)/const) 
    switch = 1.0 / (1.0 + np.exp(-(U - Uc)/dU))
    NE = NE_lin * switch
    return NE

Omega_list = [0, 0.01, 0.05, 0.1, 0.2]
Uinc = np.linspace(0, 8, 100)
# NE, NE_check = [], []
# for i, omega in enumerate(Omega_list):
#     NE.append(NE_linear_from_Uinc(Uinc, omega, [0.03, 0.025, 0.21]))
#     NE_check.append(NE_linear_from_Uinc(Uinc, omega, [0.02, 0.02, 0.1]))
 
# plt.figure()
# for i in range(len(Omega_list)):
#     plt.plot(Uinc, NE[i], color=colors[i])
#     plt.plot(Uinc, NE_check[i], '--', color=colors[i])
    
# NE_tuned = NE_from_Uinc(Uinc, 0, 1.0, 0.2)
NE_linear = NE_linear_from_Uinc(Uinc, 0, [0.03, 0.025, 0.21])
NE_opt = NE_linear_from_Uinc(Uinc, 0, [0.021, 0.021, 0.15])
Hcr = 1.5
NE_Jiang = (-0.001*Hcr + 0.012)*Uinc/const
plt.figure()
plt.plot(Uinc, NE_linear, label='Original')
plt.plot(Uinc, NE_opt, label='Optimal')
plt.plot(Uinc, NE_Jiang, 'k', label='Jiang et al. (2024)')
plt.legend()
plt.xlabel(r'$U_\mathrm{inc}$ [m/s]');plt.ylabel(r'$N_\mathrm{E}$ [-]')
plt.xlim(left=0);plt.ylim(bottom=0)


def calc_Pr(Uinc, params):
    ap, bp, cp = params
    Pr = ap*np.exp(-bp*np.exp(-cp*abs(Uinc)/const))
    return Pr

Uinc = np.linspace(0, 12, 100)
Pr = calc_Pr(Uinc, [0.74, 4.46, 0.10])
# Pr2 = calc_Pr(Uinc, [0.45, 4.46, 0.10])
# Pr3 = calc_Pr(Uinc, [0.99, 4.46, 0.10])

# plt.figure()
# plt.plot(Uinc, Pr)
# plt.plot(Uinc, Pr2)
# plt.plot(Uinc, Pr3)

# Pr4 = calc_Pr(Uinc, 0, [0.74, 1.0, 0.10])
# Pr5 = calc_Pr(Uinc, 0, [0.74, 10.0, 0.10])

# plt.figure()
# plt.plot(Uinc, Pr)
# plt.plot(Uinc, Pr4)
# plt.plot(Uinc, Pr5)

# Pr6 = calc_Pr(Uinc, 0, [0.74, 4.46, 0.01])
# Pr7 = calc_Pr(Uinc, 0, [0.74, 4.46, 0.20])

# plt.figure()
# plt.plot(Uinc, Pr)
# plt.plot(Uinc, Pr6)
# plt.plot(Uinc, Pr7)

# Pr_opt = calc_Pr(Uinc, [0.95, 1.2, 0.2])
Pr_ori = 0.9143*(1-np.exp(-0.0268*Uinc/const))
# #anderson
Pr_and = 0.95*(1-np.exp(-2*Uinc))
# Jiang
Hcr = 1.5
Pr_Jiang = 0.9945*Hcr**(-0.0166)*(1-np.exp(-0.1992*Hcr**(-0.8686)*Uinc/const))
Pr_test = 0.91*(1-np.exp(-0.22*Uinc/const))
plt.figure()
plt.plot(Uinc, Pr_ori, label='Original')
# plt.plot(Uinc, Pr_opt, label='opt')
plt.plot(Uinc, Pr_test, label='Optimal')
plt.plot(Uinc, Pr_and, 'k', label='Anderson & Haff (1991)')
plt.plot(Uinc, Pr_Jiang, 'k--', label='Jiang et al. (2024)')
plt.legend()
plt.xlabel(r'$U_\mathrm{inc}$ [m/s]')
plt.ylabel('Pr [-]')
plt.xlim(left=0);plt.ylim(bottom=0)

