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
dUadt_dpm_all = []
FD_dpm_all = []
c_dpm_all = []
tau_top_all = []
S_obs_all = []
Hs_dpm_all =[]
#   # ---- Define ODE ----
# def dUa_dt_ODE(t, Ua, u_star, M_D_interp, c_interp):
#     # M_loss = (1 - 2e-2)*rho_air * (1.46e-5 + D* Ua/h*2) * Ua/h*2
#     Re = Ua * D / (1.46e-5)
#     CD_bed = 1.05e-6 * Re**2
#     M_loss = 0.5 * rho_a * CD_bed * Ua * abs(Ua)
#     # z = 12*D
#     # z0 = D/30
#     # zr = 0.00115
#     # dudz = Ua / zr *1/(np.log(h/z0) - 1)
#     # phi_b = 0.4
#     # M_loss = rho_air * kappa**2 * z**2 * dudz**2 * (1-phi_b)
#     tau_top = rho_a * u_star**2
#     RHS = tau_top - M_loss - M_D_interp(t)
#     return RHS / (h * rho_a * (1 - c_interp(t) / (rho_sand * h)))

def r2_score(y, ypred):
    ss_res = np.sum((y - ypred)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    return 1.0 - ss_res/ss_tot

def phi_b_saturating(c, hs, phi_min, phi_max, gamma):
    # smooth, bounded  [phi_min, phi_max]
    x = c / (rho_sand * hs)                      # near-bed volume-fraction proxy
    return phi_min + (phi_max - phi_min) * (1.0 - np.exp(-gamma * x))

def tau_bottom_phi_model(X, l_eff, phi_min, phi_max, gamma):
    Ua_arr, c, hs = X
    phib = phi_b_saturating(c, hs, phi_min, phi_max, gamma)
    """Height-averaged mixing-length bottom stress"""
    tau_bed = rho_a * (nu_a + (l_eff**2) * np.abs(Ua_arr)/h) * Ua_arr / h
    return tau_bed * (1-phib)

def smooth_hs(hs_vals, window_length=7, polyorder=2):
    hs_vals = np.asarray(hs_vals)
    
    # Ensure valid window length
    if len(hs_vals) < window_length:
        window_length = len(hs_vals) if len(hs_vals) % 2 == 1 else len(hs_vals) - 1
        if window_length < 3:
            return hs_vals  # not enough points to smooth
    
    return savgol_filter(hs_vals, window_length=window_length, polyorder=polyorder)

    
# Loop over conditions S002 to S006
# Load Hs data from CSV
hs_file = "Hsals_Qs_timeseries.csv"  # CSV with columns: Shields, Liquid, Hs, Q
df_hs = pd.read_csv(hs_file)
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
    
    file_c = f'Shields00{i}dry.txt'
    data_dpm = np.loadtxt(file_c)
    c_dpm = data_dpm[:, 1]
    phi = c_dpm/(rho_sand*h)
    
    # ---- Load Hs for this Shields from CSV ----
    subset_hs = df_hs[(np.isclose(df_hs['Shields'], shields_val, atol=1e-6)) &
                      (df_hs['Liquid'] == 0)]
    hs_vals = subset_hs['Hsal'].values
    hs_smooth = smooth_hs(hs_vals, window_length=7, polyorder=2)

    # ---- Store results ----
    Ua_dpm_all.append(Ua_dpm)
    dUadt_dpm_all.append(dUa_dt)
    FD_dpm_all.append(FD_dpm)
    c_dpm_all.append(c_dpm)
    tau_top = np.ones(len(FD_dpm))*rho_a*u_star[i-2]**2
    tau_top_all.append(np.ones(len(FD_dpm))*rho_a*u_star[i-2]**2)
    S_obs_all.append(tau_top-FD_dpm-rho_a * h * (1-phi) * dUa_dt)
    Hs_dpm_all.append(hs_smooth)

Ua_all = np.concatenate(Ua_dpm_all)
duadt_all = np.concatenate(dUadt_dpm_all)  
FD_all = np.concatenate(FD_dpm_all)  
c_all = np.concatenate(c_dpm_all)
tau_top = np.concatenate(tau_top_all)
S_obs = np.concatenate(S_obs_all)
Hs_all = np.concatenate(Hs_dpm_all)

Xdata = np.vstack((Ua_all, c_all, Hs_all))
p0 = [0.005, 0, 0.4, 5.0]  # initial guess for l_eff
bounds=([1e-6, 0.0, 0.05, 1e-6],     # lower
        [0.05,  0.10, 0.64, 1e3])    # upper
params, cov = curve_fit(tau_bottom_phi_model, Xdata, S_obs, p0=p0, bounds=bounds, maxfev=20000)
l_eff, phi_min, phi_max, gamma = params
print(f"l_eff={l_eff:.4f} m, phi_min={phi_min:.3f}, phi_max={phi_max:.3f}, gamma={gamma:.3f}")
S_b_fit_all = tau_bottom_phi_model(Xdata, *params)
print("R2 B =", r2_score(S_obs, S_b_fit_all))

S_b = []
for i in range(5):
    X = Ua_dpm_all[i], c_dpm_all[i], Hs_dpm_all[i]
    tau_b = tau_bottom_phi_model(X, *params)
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

