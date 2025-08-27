# -*- coding: utf-8 -*-
"""
Created on Thu Aug 14 17:37:15 2025

@author: WangX3
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
import pandas as pd
from scipy.signal import savgol_filter

# from FD to Ua; Calibrate h dUa/dt = Mom_gain - Mom_loss - Mom_drag
h = 0.2 - 0.00025*10
D = 0.00025
kappa = 0.4
# CD_air = 8e-3
rho_a = 1.225
rho_sand = 2650
nu_a = 1.46e-5
Shields = np.linspace(0.02, 0.06, 5)
u_star = np.sqrt(Shields * (2650-1.225)*9.81*D/1.225)
t = np.linspace(0, 5, 501)
dt = np.mean(np.diff(t))

# Containers for storing results
Ua_dpm_all = []
S_obs_all = []

def r2_score(y, ypred):
    ss_res = np.sum((y - ypred)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    return 1.0 - ss_res/ss_tot

def tau_bottom_phi_model(Ua_arr, l_eff, phi_b):
    """Height-averaged mixing-length bottom stress"""
    tau_bed = rho_a * (nu_a + (l_eff**2) * np.abs(Ua_arr)/h) * Ua_arr / h
    return tau_bed * (1-phi_b)

# Loop over conditions S002 to S006
for i in range(2, 7):
    shields_val = i * 0.01  # Convert index to Shields value
    
    # ---- Load data ----
    file_ua = f'TotalDragForce/Uair_ave-tS00{i}Dryh02.txt'
    Ua_dpm = np.loadtxt(file_ua, delimiter='\t')[:, 1]
    Ua0 = Ua_dpm[0]
    dUa_dt = np.gradient(Ua_dpm, dt) 

    file_fd = f'TotalDragForce/FD_S00{i}dry.txt'
    data_FD = np.loadtxt(file_fd)
    FD_dpm = data_FD / (100 * D * 2 * D)
    
    file_c = f'CGdata/Shields00{i}dry.txt'
    data_dpm = np.loadtxt(file_c)
    c_dpm = data_dpm[:, 1]
    phi = c_dpm/(rho_sand*h)

    # ---- Store results ----
    Ua_dpm_all.append(Ua_dpm)
    tau_top = np.ones(len(FD_dpm))*rho_a*u_star[i-2]**2
    S_obs_all.append(tau_top-FD_dpm-rho_a * h * (1-phi) * dUa_dt)

Ua_all = np.concatenate(Ua_dpm_all) 
S_obs = np.concatenate(S_obs_all)

p0 = [0.005, 0.2]  # initial guess for l_eff
bounds=([1e-6, 0.0], [0.05,  0.64])
params, cov = curve_fit(tau_bottom_phi_model, Ua_all, S_obs, p0=p0, bounds=bounds, maxfev=20000)
l_eff, phi_b = params
print(f"l_eff={l_eff:.4f} m, phi_b={phi_b:.3f}")
S_b_fit_all = tau_bottom_phi_model(Ua_all, *params)
print("R2 B =", r2_score(S_obs, S_b_fit_all))

S_b = []
for i in range(5):
    tau_b = tau_bottom_phi_model(Ua_dpm_all[i], *params)
    S_b.append(tau_b)
    
# ---- Plotting ----
plt.close('all')
plt.figure(figsize=(12, 10))
for i in range(5):
    plt.subplot(3, 2, i + 1)
    plt.plot(t, S_b[i], label='fit', lw=1.8)
    plt.plot(t, S_obs[501*i:501*(i+1)], label='DPM', lw=1.5)
    plt.title(f"S00{i+2} Dry")
    plt.xlabel("Time [s]")
    plt.ylabel(r"$\tau_b(1-\phi_b)$ [N/m$^2$]")
    plt.grid(True)
    plt.legend()

plt.tight_layout()
# plt.suptitle("Comparison of Computed and DPM $U_a$ Across Conditions", fontsize=16, y=1.02)
plt.show()

