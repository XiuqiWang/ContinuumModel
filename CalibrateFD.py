# -*- coding: utf-8 -*-
"""
Created on Tue Aug  5 16:37:40 2025

@author: WangX3
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

### FROM Ua to FD; Calibrate expression of mom_drag
h = 0.2 - 0.00025*13.5
D = 0.00025
kappa = 0.4
rho_air = 1.225
rho_sand = 2650
nu_a = 1.45e-6
Shields = np.linspace(0.02, 0.06, 5)
u_star = np.sqrt(Shields * (2650-1.225)*9.81*D/1.225)
t = np.linspace(0, 5, 501)
mp = 2650 * np.pi/6 * D**3 #particle mass

def drag_model(x, b, CD):
    Ua, U = x
    ueff = b*Ua
    U_rel = ueff - U
    return np.pi*D**2/8 * rho_air * CD * np.abs(U_rel) * U_rel  

def drag_model_ori(x, b_Ua, b_U):
    Ua, U, c = x
    # z0 = D/30
    # b = np.log(z1/z0) / ((0.2*np.log(0.2/z0) - 0.2 - (13.5*D*np.log(13.5*30) - 13.5*D)) / (0.2-13.5*D))
    # print(f'b={b:.4f}')
    Ueff = b_Ua * Ua #* np.sqrt(1-c/(c+c_s)) 
    Urel = Ueff - U * b_U 
    Re = abs(Urel)*D/nu_a
    Ruc = 24
    Cd_inf = 0.5
    Cd = (np.sqrt(Cd_inf)+np.sqrt(Ruc/Re))**2   
    # print('Cd=',Cd)
    fdrag = np.pi/8 * D**2 * rho_air * Urel * abs(Urel) * Cd
    return fdrag

def BinfdUa(Ua, fd, U, Uabin):
    Ua = np.asarray(Ua, dtype=float)
    fd = np.asarray(fd, dtype=float)
    U  = np.asarray(U, dtype=float)
    edges = np.asarray(Uabin, dtype=float)

    if Ua.shape != fd.shape:
        raise ValueError("Ua and fd must have the same shape.")
    if edges.ndim != 1 or edges.size < 2:
        raise ValueError("Uabin must be a 1D array of length >= 2 (bin edges).")
    if not np.all(np.diff(edges) > 0):
        raise ValueError("Uabin must be strictly increasing.")

    # keep only finite pairs
    m = np.isfinite(Ua) & np.isfinite(fd)
    Ua = Ua[m]
    fd = fd[m]
    U = U[m]

    nbins = edges.size - 1
    fd_mean   = np.full(nbins, np.nan, dtype=float)
    fd_se     = np.full(nbins, np.nan, dtype=float)
    U_mean    = np.full(nbins, np.nan, dtype=float)
    Ua_mean = np.full(nbins, np.nan, dtype=float)

    for i in range(nbins):
        lo, hi = edges[i], edges[i+1]
        # right-inclusive on the last bin
        if i < nbins - 1:
            sel = (Ua >= lo) & (Ua < hi)
        else:
            sel = (Ua >= lo) & (Ua <= hi)

        if not np.any(sel):
            continue

        fd_i = fd[sel]
        Ua_i = Ua[sel]
        U_i = U[sel]
        n = fd_i.size

        fd_mean[i] = np.mean(fd_i)
        U_mean[i] = np.mean(U_i)
        if n > 1:
            sd = np.std(fd_i, ddof=1)
            se = sd / np.sqrt(n)
            fd_se[i] = np.nan if se == 0 else se
        else:
            fd_se[i] = np.nan

        Ua_mean[i] = np.mean(Ua_i)
    return fd_mean, fd_se, U_mean, Ua_mean

def weighted_r2(y_true, y_pred, weights):
    y_avg = np.average(y_true, weights=weights)
    ss_res = np.sum(weights * (y_true - y_pred)**2)
    ss_tot = np.sum(weights * (y_true - y_avg)**2)
    return 1 - ss_res / ss_tot

def r2_score(y, ypred):
    ss_res = np.sum((y - ypred)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    return 1.0 - ss_res/ss_tot

Ua_bin = np.linspace(0, 13, 21)
C_all_S, U_all_S, Ua_all_S, MD_all_S, fd_ori, fd_ori_se, fd_com = [], [], [], [], [], [], []
for i in range(2, 7):
    # ---- Load data ----
    file_fd = f'TotalDragForce/Mdrag/FD_S00{i}Dry.txt'
    data_FD = np.loadtxt(file_fd)
    MD = data_FD
    file_c = f'CGdata/hb=13.5d/Shields00{i}Dry-135d.txt'
    data_dpm = np.loadtxt(file_c)
    C_dpm = data_dpm[:, 1]
    U_dpm = data_dpm[:, 2]
    file_ua = f'TotalDragForce/Ua-t/Uair_ave-tS00{i}Dry.txt'
    Ua_dpm = np.loadtxt(file_ua, delimiter='\t')[:, 1]
    
    # binning Ua and getting the mean of RHS
    fd_dpm = MD*mp/C_dpm
    # fd_dpm_binned, fd_dpm_se, U_dpm_binned, Ua_dpm_binned = BinfdUa(Ua_dpm, fd_dpm, U_dpm, Ua_bin)
    
    # Combine
    U_all_S.append(U_dpm)
    Ua_all_S.append(Ua_dpm)
    C_all_S.append(C_dpm)
    fd_ori.append(fd_dpm)
    # fd_ori_se.append(fd_dpm_se)

U_all = np.concatenate(U_all_S)
Ua_all = np.concatenate(Ua_all_S)
C_all = np.concatenate(C_all_S)
fd_all = np.concatenate(fd_ori)
# fd_se_all = np.concatenate(fd_ori_se)
mask = np.isfinite(U_all) & np.isfinite(Ua_all) & np.isfinite(C_all) & np.isfinite(fd_all) 
U_all, Ua_all, C_all, fd_all = U_all[mask], Ua_all[mask], C_all[mask], fd_all[mask]

# popt, _ = curve_fit(drag_model_ori, (Ua_all, U_all, C_all), fd_all, absolute_sigma=True, maxfev=20000)
# b_ua, b_u = popt
# print(f'b_ua={b_ua:.4f}, b_u={b_u:.4f}')

# fd_pred = drag_model_ori((Ua_all, U_all, C_all), b_ua, b_u)
# r2 = r2_score(fd_all, fd_pred)
# print('r2', r2)

# b_ua, b_u = 0.30, 0.40
# b_ua, b_u = 0.40, 0.30

plt.close('all')
plt.figure(figsize=(10, 8))
for i in range(5):
    plt.subplot(3, 2, i + 1)
    fd_com = drag_model_ori((Ua_all_S[i], U_all_S[i], C_all_S[i]), b_ua, b_u)
    plt.plot(Ua_all_S[i], fd_ori[i], 'o', label='DPM')
    plt.plot(Ua_all_S[i], fd_com, 'o', label='fit')
    plt.title(f"S00{i+2} Dry")
    plt.xlabel('Ua [m/s]')
    plt.ylabel(r'$f_d$ [N]')
    plt.ylim(0, 4e-7)
    plt.xlim(4,14)
    plt.grid(True)
    plt.legend()
plt.tight_layout()
plt.show()

# plt.figure(figsize=(10, 8))
# for i in range(5):
#     plt.subplot(3, 2, i + 1)
#     plt.plot(t, Ua_all_S[i]-U_all_S[i], 'o', label='DPM')
#     plt.title(f"S00{i+2} M20")
#     plt.xlabel('t [s]')
#     plt.ylabel(r'$U_a-U$ [m/s]')
#     plt.ylim(0, 12)
#     plt.xlim(left=0)
#     plt.grid(True)
#     plt.legend()
# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(10, 8))
# for i in range(5):
#     plt.subplot(3, 2, i + 1)
#     plt.plot(t, fd_ori[i], 'o', label='DPM')
#     fd_com = drag_model_ori((Ua_all_S[i], U_all_S[i], C_all_S[i]), b)
#     # plt.plot(t, fd_com, 'o', label='fit')
#     plt.title(f"S00{i+2} M20")
#     plt.xlabel('t [s]')
#     plt.ylabel(r'$f_d$ [N]')
#     plt.ylim(0, 4e-7)
#     plt.xlim(left=0)
#     plt.grid(True)
#     plt.legend()
# plt.tight_layout()
# plt.show()

# Omega = [0, 0.01, 0.05, 0.1, 0.2]
# plt.figure()
# plt.plot(Omega, [0.30, 0.40, 0.39, 0.40, 0.43], 'o')
# plt.xlabel(r'$\Omega$');plt.ylabel(r'$b_{Ua}$')

# plt.figure()
# plt.plot(Omega, [0.40, 0.33, 0.26, 0.26, 0.33], 'o')
# plt.xlabel(r'$\Omega$');plt.ylabel(r'$b_{U}$')