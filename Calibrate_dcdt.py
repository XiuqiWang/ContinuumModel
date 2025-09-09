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
  
def Preff_of_U(U, Pmax, Uc, p):
    """State-conditioned rebound fraction. PDF marks 'needs calibration'—placeholder Gompertz-like."""
    # P_min + (P_max-P_min) * (1 - exp(-(U/Uc)^p))
    # Pmin, Pmax, Uc, p = 0, 0.6, 2, 0.5 #1 #3.84, 0.76      # tune to your binned counts
    Pmin = 0
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

    a_im  = x1 * NEim 
    a_dep = x2 * NEd 
    a = a_im + a_dep
    b = x2 
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
plt.plot(t_dpm[1:], lam_emp, label='λ_DPM')
plt.plot(t_dpm[1:], diag['lam'], label='λ_model')
plt.xlabel('t [s]')
plt.ylabel(r'$\lambda$ [s$^{-1}$]')
plt.legend(); plt.grid(True)

from scipy.optimize import least_squares

# --- helpers: simulate C(t) for given Pr parameters ---
def simulate_C(params, U_series, C0, dt):
    Pmax, Uc, p = params
    # guardrails
    Pmax = float(np.clip(Pmax, 0.0, 0.999))
    Uc   = float(max(Uc, 1e-6))
    p    = float(max(p, 1e-3))

    T = len(U_series)
    C = np.zeros(T); C[0] = C0
    lam = np.zeros(T-1)

    for i in range(T-1):
        U  = U_series[i]
        Uim = Uim_from_U(U);  UD = UD_from_U(U)
        Tim = max(Tmin, Tim_from_Uim(Uim))
        TD  = max(Tmin, TD_from_UD(UD))

        # Pr(U) with parameters to fit
        Uabs = abs(U)
        P = Pmax * (1.0 - np.exp(-(Uabs/Uc)**p))      # Gompertz/Weibull-like CDF
        P = np.clip(P, 0.0, 0.999)

        # mixture fraction and rates
        phi = P * Tim / (P*Tim + (1.0-P)*TD + 1e-12)
        x1  = phi/Tim
        x2  = (1.0-phi)/TD

        NEim = max(0.0, NE_from_Uinc(Uim))
        NEd  = max(0.0, NE_from_Uinc(UD))

        a_im  = x1 * NEim
        a_dep = x2 * NEd
        b     = x2
        lam_i = (a_im + a_dep) - b
        lam[i] = lam_i

        # Exponential Euler step
        C[i+1] = C[i] * np.exp(np.clip(lam_i*dt, -50.0, 50.0))

    return C, lam

# --- residuals for least-squares fit ---
def residuals(params):
    C_mod, lam_mod = simulate_C(params, U_smooth, C_dpm[0], dt)
    # main misfit on C(t)
    rC = (C_mod - C_dpm)

    # weights: downweight tiny C, emphasize transient (0–1.5 s)
    w_mag   = np.sqrt(np.clip(C_dpm, 1e-6, None))
    w_trans = np.ones_like(C_dpm); w_trans[:int(1.5/dt)] = 3.0
    w = w_mag * w_trans
    rC *= w

    # small steady-state constraint: mean lambda ≈ 0 (match DPM trend)
    lam_emp = np.gradient(C_dpm, dt) / np.maximum(C_dpm, 1e-12)
    # smooth a little to tame noise
    win = 21 if len(C_dpm) >= 21 else (len(C_dpm)//2*2+1)
    lam_emp = savgol_filter(lam_emp, win, 2, mode='interp')
    Ns = min(100, len(C_dpm)//5)
    ss_pen = (lam_mod[-Ns:].mean() - lam_emp[-Ns:].mean())
    return np.hstack([rC, 0.1*ss_pen])   # 0.1 sets a light weight on the penalty

# --- run the fit ---
p0 = np.array([0.6, 2.0, 0.5])          # initial guess: [Pmax, Uc, p]
lb = np.array([0.2, 0.1, 0.2])          # bounds (tune as needed)
ub = np.array([0.999, 20.0, 5.0])

res = least_squares(residuals, p0, bounds=(lb, ub), verbose=2, max_nfev=300)
Pmax_hat, Uc_hat, p_hat = res.x
print(f"Fitted: Pmax={Pmax_hat:.3f}, Uc={Uc_hat:.3f}, p={p_hat:.3f}")

# --- recompute with fitted parameters and plot ---
C_fit, lam_fit = simulate_C(res.x, U_smooth, C_dpm[0], dt)

plt.figure(figsize=(12,5))
t = t_dpm
plt.subplot(1,2,1)
plt.plot(t, C_dpm, label='DPM')
plt.plot(t, C_fit, label='computed (fitted)')
plt.xlabel('t [s]'); plt.ylabel(r'C [kg/m$^2$]'); plt.legend(); plt.grid(True)

plt.subplot(1,2,2)
lam_emp = np.gradient(C_dpm, dt) / np.maximum(C_dpm, 1e-12)
plt.plot(t[1:], lam_emp[1:], label=r'$\lambda_{\rm DPM}$')
plt.plot(t[1:], lam_fit,       label=r'$\lambda_{\rm model}$')
plt.xlabel('t [s]'); plt.ylabel(r'$\lambda$ [s$^{-1}$]'); plt.legend(); plt.grid(True)


# plt.figure()
# plt.plot(t_dpm[1:], diag['Uim']/const)
# plt.xlabel('t [s]')
# plt.ylabel(r'$U_{im}/\sqrt{gd}$ [-]')

# plt.figure()
# plt.plot(t_dpm, U_smooth)
# plt.xlabel('t [s]')
# plt.ylabel(r'$U_{sal}$ [m/s]')


U_s = np.linspace(0,max(U_smooth), 100)
Pr = Preff_of_U(U_s, 1, 3.84, 0.76)
Pr_adj = Preff_of_U(U_s, 0.999, 10.038, 0.429)
 
plt.figure()
plt.plot(U_s, Pr, label='ori')
plt.plot(U_s, Pr_adj, label='adj')
plt.xlabel(r'$U_{sal}$ [m/s]')
plt.ylabel(r'$Pr_{eff}$')
plt.legend()

# # balance in steady state
# Us = np.mean(U_smooth[200:])
# Uim = Uim_from_U(Us);  UD = UD_from_U(Us)      # your mappings
# Tim = max(Tmin, Tim_from_Uim(Uim))
# TD  = max(Tmin, TD_from_UD(UD))
# P   = np.clip(Preff_of_U(Us, 0.6, 2, 0.5), 0.0, 1.0)

# phi = P * Tim / (P*Tim + (1-P)*TD)
# x1  = phi/Tim
# x2  = (1.0-phi)/TD
# NEim = max(0.0, NE_from_Uinc(Uim))           # keep nonnegative
# NEd  = max(0.0, NE_from_Uinc(UD))

# a_im  = x1 * NEim 
# a_dep = x2 * NEd 
# a = a_im + a_dep
# b = x2 
# lams = a - b 
# print('lam_S', lams)