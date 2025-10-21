# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 16:25:52 2025

@author: WangX3
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
import pandas as pd
from scipy.signal import savgol_filter

h = 0.2 - 0.00025*12
D = 0.00025
g = 9.81
rho_a = 1.225
rho_sand = 2650
nu_a = 1.46e-5
Shields = np.linspace(0.02, 0.06, 5)
u_star = np.sqrt(Shields * (2650-1.225)*9.81*D/1.225)
mp = 2650 * np.pi/6 * D**3 #particle mass
const = np.sqrt(g*D)

cases = range(2, 7)  # S002..S006
C_list, Rim_list, D_list = [], [], []
U_list = []
t_dpm = np.linspace(0.01, 5, 501)
dt = 0.01
for i in cases:
    # CG data: columns -> Q, C, U (501 rows)
    cg = np.loadtxt(f"CGdata/hb=12d/Shields{i:03d}dry.txt")
    c, U = cg[:, 1], cg[:, 2]
    c_center = 0.5*(c[:-1]+c[1:])
    U_center = 0.5*(U[:-1]+U[1:])
    C_list.append(c_center)
    U_list.append(U_center)
   
    # Rim, E and D
    RimED = np.loadtxt(f"dcdt/RimED/S{i:03d}RimEandD.txt")
    Rim, E, D = RimED[:,0], RimED[:,1], RimED[:,2]
    Rim_list.append(Rim)
    D_list.append(D)

def Uim_from_U(U):
    """U_im from instantaneous saltation-layer velocity U (includes Ω effect)."""
    Uim_mag = 0.04*(abs(U)/const)**1.54 + 32.23
    return np.sign(U) * (Uim_mag*const)

def UD_from_U(U):
    """U_D from U. PDF only provides Dry (Ω=0). We use Dry law for all Ω unless you add a wet fit."""
    UD_mag = 6.66         
    return np.sign(U) * (UD_mag*const)

def Tim_from_Uim(Uim):
    return 0.04 * (abs(Uim))**0.84         

def TD_from_UD(UD):
    # A_td, p_td, Pmax, Uc, p, A_NE = params
    # return 0.06 * (abs(UD))**0.66  
    return 0.44 * (abs(UD))** 0.18

def Preff_of_U(U):
    """State-conditioned rebound fraction. PDF marks 'needs calibration'—placeholder Gompertz-like."""
    # P_min + (P_max-P_min) * (1 - exp(-(U/Uc)^p))
    Pmin, Pmax, Uc, p = 0, 0.999, 3.84, 0.76 
    U = abs(U)
    return Pmin + (Pmax - Pmin)*(1.0 - np.exp(-(U/Uc)**p)) 

def CalRimfromU(U, c_vec):
    Uim, UD = Uim_from_U(U), UD_from_U(U)
    Tim = Tim_from_Uim(Uim)
    TD = TD_from_UD(UD)
    Pr = Preff_of_U(U)
    cim = Pr*Tim/(Pr*Tim + (1-Pr)*TD) * c_vec
    Rim = cim/Tim
    return Rim
    
def CalDfromU(U, c_vec):
    Uim, UD = Uim_from_U(U), UD_from_U(U)
    Tim = Tim_from_Uim(Uim)
    TD = TD_from_UD(UD)
    Pr = Preff_of_U(U)
    cD = (1-Pr)*TD/(Pr*Tim + (1-Pr)*TD) * c_vec
    D = cD/TD
    return D

def savgol_keep_first(x, window_length=21, polyorder=2):
    x = np.asarray(x, dtype=float)

    # Handle NaNs if any
    if np.isnan(x).any():
        idx = np.arange(len(x))
        x = np.interp(idx, idx[~np.isnan(x)], x[~np.isnan(x)])

    n = len(x)
    w = int(window_length)
    if w % 2 == 0:
        w += 1
    w = min(w, n if n % 2 == 1 else n - 1)
    w = max(w, polyorder + 2 if (polyorder + 2) % 2 == 1 else polyorder + 3)

    # Smooth only from 2nd element onward
    y = np.empty_like(x)
    y[0] = x[0]
    if n > 1:
        y[1:] = savgol_filter(x[1:], window_length=w if w < n else n - 1, polyorder=polyorder, mode='interp')
    return y
    
plt.figure(figsize=(12, 6))
for i in range(5):  # 5 groups
    plt.subplot(2, 3, i+1) 
    # example
    RIM_i = savgol_keep_first(Rim_list[i], window_length=21, polyorder=2) #smooth
    U_center = 0.5*(U_list[i][:-1] + U_list[i][1:]) 
    plt.plot(U_list[i], RIM_i, '.', label='DPM')
    Rim_com = CalRimfromU(U_list[i], C_list[i])
    plt.plot(U_list[i], Rim_com, '.', label='computed')
    plt.title(f'$\Theta$=0.0{i+2}')
    plt.ylabel(r'$R_{im}$ [kg/m$^2$/s]')
    plt.xlabel(r'U [m/s]')
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
for i in range(5):  # 5 groups
    plt.subplot(2, 3, i+1)
    index = i * 5 
    # example
    D_i = savgol_keep_first(D_list[i], window_length=21, polyorder=2) #smooth
    plt.plot(U_list[i], D_i, '.', label='Deposition rate')
    D_com = CalDfromU(U_list[i], C_list[i])
    plt.plot(U_list[i], D_com, '.', label='computed')
    plt.title(f'$\Theta$=0.0{i+2}')
    plt.ylabel(r'$D$ [kg/m$^2$/s]')
    plt.xlabel(r'U [m/s]')
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()