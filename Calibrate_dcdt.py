# -*- coding: utf-8 -*-
"""
Created on Tue Sep  2 12:10:57 2025

@author: WangX3
"""

from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.optimize import lsq_linear

# -------------------- constants & inputs --------------------
g = 9.81
d = 0.00025                # grain diameter [m]  
const = np.sqrt(g*d)       # √(g d)
h = 0.1975                 # domain height [m]
u_star = 0.56              # shear velocity [m/s]

rho_a  = 1.225             # air density [kg/m^3] 
nu_a   = 1.46e-5           # air kinematic viscosity [m^2/s]
rho_p  = 2650.0            # particle density [kg/m^3]           

# Particle mass
mp = rho_p * (np.pi/6.0) * d**3

# Moisture (Ω) for U_im mapping; affects only U_im per PDF
Omega = 0.0  

# -------------------- closures from the PDF --------------------
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
    return 0.06 * (abs(UD))**0.66       

def NE_from_Uinc(Uinc):
    return 0.04 * (abs(Uinc)/const)        

def Preff_of_U(U):
    """State-conditioned rebound fraction. PDF marks 'needs calibration'—placeholder Gompertz-like."""
    # P_min + (P_max-P_min) * (1 - exp(-(U/Uc)^p))
    Pmin, Pmax, Uc, p = 0, 1, 3.84, 0.76      # tune to your binned counts
    U = abs(U)
    return Pmin + (Pmax - Pmin)*(1.0 - np.exp(-(U/max(Uc,1e-6))**p)) 

#----------- DPM data ----------
data = np.loadtxt('CGdata/Shields006dry.txt')
Q_dpm = data[:, 0]
C_dpm = data[:, 1]
U_dpm = data[:, 2]
t_dpm = np.linspace(0,5,501)
data_ua = np.loadtxt('TotalDragForce/Uair_ave-tS006Dryh02.txt', delimiter='\t')
Ua_dpm = data_ua[0:,1]    

def savgol_keep_first(x, window_length=21, polyorder=2):
    x = np.asarray(x, dtype=float)

    # handle NaNs (optional)
    if np.isnan(x).any():
        idx = np.arange(len(x))
        x = np.interp(idx, idx[~np.isnan(x)], x[~np.isnan(x)])

    # make window valid: odd, <= len(x), > polyorder
    n = len(x)
    w = int(window_length)
    if w % 2 == 0: w += 1
    w = min(w, n if n % 2 == 1 else n-1)
    w = max(w, polyorder + 2 if (polyorder + 2) % 2 == 1 else polyorder + 3)

    y = savgol_filter(x, window_length=w, polyorder=polyorder, mode='interp')
    y[0] = x[0]             # keep the first value fixed
    return y

# example
U_smooth = savgol_keep_first(U_dpm, window_length=21, polyorder=2)

dt = 0.01
C_com = np.zeros_like(C_dpm)
C_com[0] = 0.147
Tmin = 1e-6  # floor to avoid division blow-ups
# --- before the time loop ---
diag = {
    'lam': [], 'a_im': [], 'a_dep': [], 'b': [],
    'phi': [], 'Tim': [], 'TD': [],
    'NEim': [], 'NEd': [], 'Uim': [], 'UD': []
}
for i in range(len(C_com)-1):
    U  = U_smooth[i]
    Uim = Uim_from_U(U);  UD = UD_from_U(U)      # your mappings
    Tim = max(Tmin, Tim_from_Uim(Uim))
    TD  = max(Tmin, TD_from_UD(UD))
    P   = np.clip(Preff_of_U(U), 0.0, 1.0)

    phi = P * Tim / (P*Tim + (1-P)*TD)
    x1  = phi/Tim
    x2  = (1.0-phi)/TD
    NEim = max(0.0, NE_from_Uinc(Uim))           # keep nonnegative
    NEd  = max(0.0, NE_from_Uinc(UD))

    a_im  = x1 * NEim * 0.3
    a_dep = x2 * NEd * 0.1
    a = a_im + a_dep
    b = x2 * 0.2
    lam = a - b 
    
    # Exponential Euler (stable, positive)
    C_com[i+1] = C_com[i] * np.exp(np.clip(lam*dt, -0.5, 0.5))
    
    # store diagnostics
    diag['lam'].append(lam)
    diag['a_im'].append(a_im)
    diag['a_dep'].append(a_dep)
    diag['b'].append(b)
    diag['phi'].append(phi)
    diag['Tim'].append(Tim)
    diag['TD'].append(TD)
    diag['NEim'].append(NEim)
    diag['NEd'].append(NEd)
    diag['Uim'].append(Uim)
    diag['UD'].append(UD)
  
plt.close('all')
lam_emp = (C_dpm[1:] - C_dpm[:-1]) / (dt * np.maximum(C_dpm[:-1], 1e-12))
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(t_dpm, C_dpm, label='DPM')
plt.plot(t_dpm, C_com, label='computed')
plt.xlabel('t [s]')
plt.ylabel(r'C [kg/m$^2$]')
plt.legend()
plt.subplot(1,2,2)
plt.plot(lam_emp, label='λ_DPM')
plt.plot(diag['lam'][:-1], label='λ_model')
plt.xlabel('t [s]')
plt.ylabel(r'$\lambda$ [s$^{-1}$]')
plt.legend(); plt.grid(True)