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
h = 0.2 - 0.00025*10
D = 0.00025
kappa = 0.4
# CD_air = 8e-3
rho_air = 1.225
rho_sand = 2650
Shields = np.linspace(0.02, 0.06, 5)
u_star = np.sqrt(Shields * (2650-1.225)*9.81*D/1.225)
t = np.linspace(0, 5, 501)
mp = 2650 * np.pi/6 * D**3 #particle mass

def drag_model(Ua, U, b, k):
    # ueff = Ua/(np.log(h/z0)-1) * np.log(zr/z0)
    ueff = b*Ua
    # ueff = Ua * (np.log(hs/z0) - 1) / (np.log(h/z0) - 1)
    # print('Ua', Ua)
    # print('zr', zr)
    # print('ueff', ueff)
    U_rel = ueff - U
    return k * np.abs(U_rel) * U_rel  

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

        Ua_median[i] = np.median(Ua_i)
    return fd_mean, fd_se, U_mean, Ua_median

Ua_bin = np.linspace(0, 13, 21)
b = 0.4
k = 5e-9
C_all_S, U_all_S, Ua_all_S, MD_all_S, fd_ori, fd_ori_se, fd_com = [], [], [], [], [], [], []
for i in range(2, 7):
    # ---- Load data ----
    file_fd = f'TotalDragForce/FD_S00{i}dry.txt'
    data_FD = np.loadtxt(file_fd)
    MD = data_FD/(100 * D * 2 * D)
    file_c = f'CGdata/Shields00{i}dry.txt'
    data_dpm = np.loadtxt(file_c)
    Q_dpm = data_dpm[:, 0]
    C_dpm = data_dpm[:, 1]
    U_dpm = data_dpm[:, 2]
    file_ua = f'TotalDragForce/Uair_ave-tS00{i}Dryh02.txt'
    Ua_dpm = np.loadtxt(file_ua, delimiter='\t')[:, 1]
    
    
    # binning Ua and getting the mean of RHS
    fd_dpm = MD*mp/C_dpm
    fd_dpm_binned, fd_dpm_se, U_dpm_binned, Ua_dpm_binned = BinfdUa(Ua_dpm, fd_dpm, U_dpm, Ua_bin)
    #------ compute LHD ------
    fd = drag_model(Ua_dpm_binned, U_dpm_binned, b, k)
    
    # Combine
    C_all_S.append(C_dpm)
    U_all_S.append(U_dpm)
    Ua_all_S.append(Ua_dpm_binned)
    fd_ori.append(fd_dpm_binned)
    fd_ori_se.append(fd_dpm_se)
    fd_com.append(fd)

plt.close('all')
plt.figure(figsize=(12, 10))
for i in range(5):
    plt.subplot(3, 2, i + 1)
    plt.errorbar(Ua_all_S[i], fd_ori[i], yerr=fd_ori_se[i], fmt='o', capsize=5, label='DPM')
    plt.plot(Ua_all_S[i], fd_com[i], 'o', label='fit')
    # plt.plot(t_con, FD_continuum, label='Continuum')
    plt.title(f"S00{i+2} Dry")
    plt.xlabel('Ua [m/s]')
    plt.ylabel(r'$f_d$ [N]')
    plt.ylim(0,1.4e-7)
    plt.xlim(0,13)
    plt.grid(True)
    plt.legend()
plt.tight_layout()
plt.show()
