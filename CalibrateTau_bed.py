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
mp = 2650 * np.pi/6 * D**3 #particle mass
t = np.linspace(0, 5, 501)
dt = np.mean(np.diff(t))

def r2_score(y, ypred):
    ss_res = np.sum((y - ypred)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    return 1.0 - ss_res/ss_tot

# def tau_bottom_phi_model(Ua_arr, l_eff, phi_b):
#     """Height-averaged mixing-length bottom stress"""
#     tau_bed = rho_a * (nu_a + (l_eff**2) * np.abs(Ua_arr)/h) * Ua_arr / h
#     return tau_bed * (1-phi_b)

def tau_bed_dragform(x, beta, K):
    Ua, U, c = x
    Ubed = beta*U
    tau_b_oneminusphib = rho_a * K * c/(rho_sand*D) * abs(Ua-Ubed) * (Ua-Ubed) 
    return tau_b_oneminusphib

def BintaubUa(Ua, RHS, Uabin):
    Ua = np.asarray(Ua, dtype=float)
    RHS = np.asarray(RHS, dtype=float)
    edges = np.asarray(Uabin, dtype=float)

    if Ua.shape != RHS.shape:
        raise ValueError("Ua and RHS must have the same shape.")
    if edges.ndim != 1 or edges.size < 2:
        raise ValueError("Uabin must be a 1D array of length >= 2 (bin edges).")
    if not np.all(np.diff(edges) > 0):
        raise ValueError("Uabin must be strictly increasing.")

    # keep only finite pairs
    m = np.isfinite(Ua) & np.isfinite(RHS)
    Ua = Ua[m]
    RHS = RHS[m]

    nbins = edges.size - 1
    RHS_mean   = np.full(nbins, np.nan, dtype=float)
    RHS_se     = np.full(nbins, np.nan, dtype=float)
    Ua_median = np.full(nbins, np.nan, dtype=float)

    for i in range(nbins):
        lo, hi = edges[i], edges[i+1]
        # right-inclusive on the last bin
        if i < nbins - 1:
            sel = (Ua >= lo) & (Ua < hi)
        else:
            sel = (Ua >= lo) & (Ua <= hi)

        if not np.any(sel):
            continue

        RHS_i = RHS[sel]
        Ua_i = Ua[sel]
        n = RHS_i.size

        RHS_mean[i] = np.mean(RHS_i)
        if n > 1:
            sd = np.std(RHS_i, ddof=1)
            se = sd / np.sqrt(n)
            RHS_se[i] = np.nan if se == 0 else se
        else:
            RHS_se[i] = np.nan

        Ua_median[i] = np.median(Ua_i)
    return RHS_mean, RHS_se, Ua_median

def weighted_r2(y_true, y_pred, weights):
    y_avg = np.average(y_true, weights=weights)
    ss_res = np.sum(weights * (y_true - y_pred)**2)
    ss_tot = np.sum(weights * (y_true - y_avg)**2)
    return 1 - ss_res / ss_tot

l_eff = 0.025
phi_b = 0.4
Ua_bin = np.linspace(0, 13, 21)
CDbed, Ua_c, n = 0.11, 5, 1.75 # for dragform
# Containers for storing results
Ua_all_S, RHS_se_all_S, RHS_all_S, LHS_all_S = [], [], [], []
U_all_S, c_all_S = [],[]
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
    U_dpm = data_dpm[:, 2]
    phi = c_dpm/(rho_sand*h)
    
    #---- compute RHS ----
    tau_top = np.ones(len(FD_dpm))*rho_a*u_star[i-2]**2
    RHS = tau_top-FD_dpm-rho_a * h * (1-phi) * dUa_dt
    # RHS_binned, RHS_se, Ua_binned = BintaubUa(Ua_dpm, RHS, Ua_bin)
    
    #----- compute LHS -----
    LHS = tau_bed_dragform((Ua_dpm, U_dpm, c_dpm), 0.2, 0.05)

    # ---- Store results ----
    Ua_all_S.append(Ua_dpm)
    U_all_S.append(U_dpm)
    c_all_S.append(c_dpm)
    RHS_all_S.append(RHS)
    # RHS_se_all_S.append(RHS_se)
    LHS_all_S.append(LHS)

plt.close('all')
plt.figure(figsize=(12, 10))
for i in range(5):
    plt.subplot(3, 2, i + 1)
    plt.plot(Ua_all_S[i], RHS_all_S[i], '.', label='DPM')
    plt.plot(Ua_all_S[i], LHS_all_S[i], '.', label='proposed')
    plt.title(f"S00{i+2} Dry")
    # plt.xlabel("Ua [m/s]")
    plt.ylabel(r"$\tau_b(1-\phi_b)$ [N/m$^2$]")
    plt.ylim(0,3)
    plt.xlim(0,13.5)
    plt.grid(True)
    plt.legend()
plt.tight_layout()
plt.show()


Ua_all= np.concatenate(Ua_all_S)
U_all= np.concatenate(U_all_S)
c_all= np.concatenate(c_all_S)
RHS_all = np.concatenate(RHS_all_S)
# RHS_se_all = np.concatenate(RHS_se_all_S)
mask = np.isfinite(Ua_all) & np.isfinite(RHS_all) #& np.isfinite(RHS_se_all)
Ua_all, RHS_all = Ua_all[mask], RHS_all[mask] #RHS_se_all = RHS_se_all[mask] 
U_all, c_all = U_all[mask], c_all[mask]

popt, _ = curve_fit(tau_bed_dragform, (Ua_all, U_all, c_all), RHS_all)
beta, K = popt
# RHS_pred = tau_bed_dragform(Ua_all, CDbed, Ua_c, n)
# r2 = weighted_r2(RHS_all, RHS_pred, 1/RHS_se_all**2)
# print('r2', r2)
    
# ---- Plotting ----
plt.close('all')
plt.figure(figsize=(12, 10))
for i in range(5):
    plt.subplot(3, 2, i + 1)
    LHS = tau_bed_dragform((Ua_all_S[i], U_all_S[i], c_all_S[i]), beta, K)
    plt.plot(Ua_all_S[i], RHS_all_S[i], '.', label='DPM')
    plt.plot(Ua_all_S[i], LHS, '.', label='fit')
    # plt.errorbar(Ua_all_S[i], RHS_all_S[i], yerr=RHS_se_all[i], fmt='o', capsize=5, label='DPM')
    # plt.plot(Ua_all_S[i], LHS, 'o', label='fit')
    plt.title(f"S00{i+2} Dry")
    plt.xlabel("Ua [m/s]")
    plt.ylabel(r"$\tau_b(1-\phi_b)$ [N/m$^2$]")
    # plt.ylabel(r'$\tau_b=CD_b|cU_a|cU_a$ [N/m$^2$]')
    plt.ylim(0,3)
    plt.xlim(0,13)
    plt.grid(True)
    plt.legend()

plt.tight_layout()
# plt.suptitle("Comparison of Computed and DPM $U_a$ Across Conditions", fontsize=16, y=1.02)
plt.show()

# plt.figure(figsize=(6, 5))
# for i in range(5):
#     plt.plot(Ua_dpm_all[i], RHS_all[i], '.', label=f'Shields=0.0{i+2}')
# plt.xlabel("Ua [m/s]")
# plt.ylabel(r"$\tau_b(1-\phi_b)$ [N/m$^2$]")
# # plt.ylabel(r'$\tau_b=CD_b|cU_a|cU_a$ [N/m$^2$]')
# plt.ylim(0,3)
# plt.xlim(0, 13)
# plt.grid(True)
# plt.legend()
# plt.tight_layout()