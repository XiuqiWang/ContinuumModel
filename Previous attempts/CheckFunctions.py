# -*- coding: utf-8 -*-
"""
Created on Mon Sep 22 11:30:51 2025

@author: WangX3
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

# -------------------- constants & inputs --------------------
g = 9.81
d = 0.00025
const = np.sqrt(g*d)
h = 0.197
u_star = 0.56

rho_a  = 1.225
nu_a   = 1.46e-5
rho_p  = 2650.0
mp = rho_p * (np.pi/6.0) * d**3

UE_mag = 4.53 * const
thetaE_deg = 40.0
cos_thetaE = np.cos(np.deg2rad(thetaE_deg))

U = np.linspace(0, 10, 100)
Uinc = np.linspace(0, 10, 100)
Uim = np.linspace(0.01, 10, 100)
# --------------------- functions -----------------------------
def Preff_of_U_param(U):
    Pmin = 0.0
    Pmax, Uc, pshape = 0.95, 0.93, 3 
    Uabs = np.abs(U)
    return np.clip(Pmin + (Pmax - Pmin)*(1.0 - np.exp(-(Uabs/max(Uc,1e-6))**pshape)), 0.0, 1.0)

def NE_from_Uinc_param(Uinc):
    A_NE = 0.018
    return A_NE * (np.abs(Uinc)/const)

def e_COR_from_Uim(Uim):
    A_e, B_e = 10, 0.62
    return A_e * (abs(Uim)/const + 1e-12)**(-B_e)

# check dependence
Pr = Preff_of_U_param(U)
NE = NE_from_Uinc_param(Uinc)
e = e_COR_from_Uim(Uim)

fig, axs = plt.subplots(1, 3, figsize=(12, 4))
axs[0].plot(U, Pr)
axs[0].set_xlabel('U')
axs[0].set_ylabel('Pr')
axs[1].plot(Uinc, NE)
axs[1].set_xlabel('Uinc')
axs[1].set_ylabel('NE')
axs[2].plot(Uim, e)
axs[2].set_xlabel('Uim')
axs[2].set_ylabel('e')
# Apply to all subplots at once
for ax in axs.flat:
    ax.set(xlim=(0, None), ylim=(0, None))  # start both axes at 0
    ax.grid(True)
plt.tight_layout()
plt.show()






