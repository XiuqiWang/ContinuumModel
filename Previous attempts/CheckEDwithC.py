# -*- coding: utf-8 -*-
"""
Created on Wed Oct 15 13:48:51 2025

@author: WangX3
"""

import numpy as np
import matplotlib.pyplot as plt

cases = range(2, 7)  # S002..S006
C_list, dcdt_list, E_list, D_list = [], [], [], []
U_list = []
t_dpm = np.linspace(0.01, 5, 501)
dt = 0.01
for i in cases:
    # CG data: columns -> Q, C, U (501 rows)
    cg = np.loadtxt(f"CGdata/hb=13d,M20/new/Shields{i:03d}M20-13d.txt")
    C_list.append(cg[:, 1])
    U_list.append(cg[:, 2])
    dcdt = np.gradient(cg[:, 1], dt)
    dcdt_list.append(dcdt)
    # E and D
    ED = np.loadtxt(f"dcdt/1D-M20/new/S{i:03d}M20RimEandD.txt")
    # ED = np.loadtxt(f"dcdt/2D-allparticles/S{i:03d}RimEandD.txt")
    # t_Dis = np.linspace(5/len(ED[:,0]), 5, len(ED[:,0]))
    # # make E and D same length as U and C
    # E_on_U = np.interp(t_dpm, t_Dis, ED[:,0])
    # D_on_U = np.interp(t_dpm, t_Dis, ED[:,1])
    E, D = ED[:,1], ED[:,2]
    E_list.append(E)
    D_list.append(D)

c_com_list = []
N = len(C_list[0])    

for i in range(5):    
    c0 = C_list[i][0]
    c = np.zeros(N)
    c[0] = c0
    for j in range(N-1):
        dcdt = E_list[i][j] - D_list[i][j]
        c[j+1] = c[j] + dcdt * dt
    c_com_list.append(c)


# plt.close('all')
plt.figure(figsize=(10,9))
for i in range(5):
    plt.subplot(3,2,i+1)
    plt.plot(t_dpm, C_list[i], label='DPM CG data')
    plt.plot(t_dpm, c_com_list[i], label='computed from E and D')
    plt.xlabel('t [s]')
    plt.ylabel(r'$C$ [kg/m$^2$]')
    plt.ylim(0,0.3)
plt.legend()
plt.tight_layout()

# plt.figure(figsize=(10,9))
# for i in range(5):
#     plt.subplot(3,2,i+1)
#     plt.plot(t_dpm[:-1], E_list[i], label='E')
#     plt.plot(t_dpm[:-1], D_list[i], label='D')
#     plt.xlabel('t [s]')
#     plt.ylabel(r'$C$ [kg/m$^2$]')
# plt.legend()
# plt.tight_layout()

# C_center_list = [0.5 * (C_component[:-1] + C_component[1:]) for C_component in C_list]
# U_center_list = [0.5 * (U_component[:-1] + U_component[1:]) for U_component in U_list]
# E_over_c = [E/c for E,c in zip(E_list,C_center_list)]
# D_over_c = [D/c for D,c in zip(D_list,C_center_list)]

# plt.figure(figsize=(10,9))
# for i in range(5):
#     plt.subplot(3,2,i+1)
#     plt.plot(U_center_list[i], E_over_c[i], '.', label='E')
#     plt.plot(U_center_list[i], D_over_c[i], '.', label='D')
#     plt.xlabel('U [m/s]')
#     plt.ylabel(r'$E/c$ $&$ $D/c$ [/s]')
# plt.legend()
# plt.tight_layout()

# bin the U and E/C and D/c
# def BintaubUa(U, E_over_c_i, D_over_c_i, Ubin):
#     U = np.asarray(U, dtype=float)
#     E_over_c_i = np.asarray(E_over_c_i, dtype=float)
#     D_over_c_i = np.asarray(D_over_c_i, dtype=float)
#     edges = np.asarray(Ubin, dtype=float)

#     if U.shape != E_over_c_i.shape:
#         raise ValueError("U and E_over_c_i must have the same shape.")
#     if edges.ndim != 1 or edges.size < 2:
#         raise ValueError("Ubin must be a 1D array of length >= 2 (bin edges).")
#     if not np.all(np.diff(edges) > 0):
#         raise ValueError("Ubin must be strictly increasing.")

#     # keep only finite pairs
#     m = np.isfinite(U) & np.isfinite(E_over_c_i) & np.isfinite(D_over_c_i)
#     U = U[m]
#     E_over_c_i = E_over_c_i[m]
#     D_over_c_i = D_over_c_i[m]

#     nbins = edges.size - 1
#     Eoverc_mean   = np.full(nbins, np.nan, dtype=float)
#     Eoverc_se     = np.full(nbins, np.nan, dtype=float)
#     Doverc_mean   = np.full(nbins, np.nan, dtype=float)
#     Doverc_se     = np.full(nbins, np.nan, dtype=float)
#     U_mean  = np.full(nbins, np.nan, dtype=float)

#     for i in range(nbins):
#         lo, hi = edges[i], edges[i+1]
#         # right-inclusive on the last bin
#         if i < nbins - 1:
#             sel = (U >= lo) & (U < hi)
#         else:
#             sel = (U >= lo) & (U <= hi)

#         if not np.any(sel):
#             continue

#         E_over_c_i_inbin = E_over_c_i[sel]
#         D_over_c_i_inbin = D_over_c_i[sel]
#         U_i = U[sel]
#         n_E = E_over_c_i_inbin.size
#         n_D = D_over_c_i_inbin.size

#         Eoverc_mean[i] = np.mean(E_over_c_i_inbin)
#         Doverc_mean[i] = np.mean(D_over_c_i_inbin)
#         if n_E > 1:
#             sd_E = np.std(E_over_c_i_inbin, ddof=1)
#             se_E = sd_E / np.sqrt(n_E)
#             Eoverc_se[i] = np.nan if se_E == 0 else se_E
#         else:
#             Eoverc_se[i] = np.nan
            
#         if n_D > 1:
#             sd_D = np.std(D_over_c_i_inbin, ddof=1)
#             se_D = sd_D / np.sqrt(n_D)
#             Doverc_se[i] = np.nan if se_D == 0 else se_D
#         else:
#             Doverc_se[i] = np.nan    

#         U_mean[i] = np.mean(U_i)
#     return Eoverc_mean, Eoverc_se, Doverc_mean, Doverc_se, U_mean

# U_bin = np.linspace(np.nanmin(np.concatenate(U_center_list)), np.nanmax(np.concatenate(U_center_list)), 10)
# E_c = [np.full(len(U_bin), np.nan) for _ in range(5)]
# D_c, E_c_se, D_c_se, U_mean = [np.full(len(U_bin), np.nan) for _ in range(5)], [np.full(len(U_bin), np.nan) for _ in range(5)], [np.full(len(U_bin), np.nan) for _ in range(5)], [np.full(len(U_bin), np.nan) for _ in range(5)]
# for i in range(5):
#     E_c[i], E_c_se[i], D_c[i], D_c_se[i], U_mean[i] = BintaubUa(U_center_list[i], E_over_c[i], D_over_c[i], U_bin)
 
# plt.figure(figsize=(10,9))
# for i in range(5):
#     plt.subplot(3,2,i+1)
#     plt.errorbar(U_mean[i], E_c[i], yerr=E_c_se[i], fmt='o', capsize=5, label='E')
#     plt.errorbar(U_mean[i], D_c[i], yerr=D_c_se[i], fmt='o', capsize=5, label='D')
#     plt.xlabel('U [m/s]')
#     plt.ylabel(r'$E/c$ $&$ $D/c$ [/s]')
# plt.legend()
# plt.tight_layout()    

# smooth out E and D while keeping the first value the same
from scipy.signal import savgol_filter
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

# E_s, D_s = [], []
# for i in range(5):
#     E_smooth = savgol_keep_first(E_list[i], window_length=41, polyorder=2)
#     D_smooth = savgol_keep_first(D_list[i], window_length=41, polyorder=2)
#     E_s.append(E_smooth)
#     D_s.append(D_smooth)
    
# plt.figure(figsize=(10,9))
# for i in range(5):
#     plt.subplot(3,2,i+1)
#     plt.plot(t_dpm[:-1], E_s[i], label='E')
#     plt.plot(t_dpm[:-1], D_s[i], label='D')
#     plt.xlabel('t [s]')
#     plt.ylabel(r'$C$ [kg/m$^2$]')
# plt.legend()
# plt.tight_layout()

# c_com_smooth = [] 

# for i in range(5):    
#     c0 = C_list[i][0]
#     c = np.zeros(N)
#     c[0] = c0
#     for j in range(N-1):
#         dcdt = E_s[i][j] - D_s[i][j]
#         c[j+1] = c[j] + dcdt * dt
#     c_com_smooth.append(c)
    
# plt.figure(figsize=(10,9))
# for i in range(5):
#     plt.subplot(3,2,i+1)
#     plt.plot(t_dpm, C_list[i], label='DPM CG data')
#     plt.plot(t_dpm, c_com_smooth[i], label='computed from E and D')
#     plt.xlabel('t [s]')
#     plt.ylabel(r'$C$ [kg/m$^2$]')
# plt.legend()
# plt.tight_layout()

# E_over_c_smooth = [E/c for E,c in zip(E_s,C_center_list)]
# D_over_c_smooth = [D/c for D,c in zip(D_s,C_center_list)]

# plt.figure(figsize=(10,9))
# for i in range(5):
#     plt.subplot(3,2,i+1)
#     plt.plot(U_center_list[i], E_over_c_smooth[i], '.', label='E')
#     plt.plot(U_center_list[i], D_over_c_smooth[i], '.', label='D')
#     plt.xlabel('U [m/s]')
#     plt.ylabel(r'$E/c$ $&$ $D/c$ [/s]')
# plt.legend()
# plt.tight_layout()

#save U, smoothed E/c and D/c for S006
# data = np.column_stack((U_center_list[4], E_over_c_smooth[4], D_over_c_smooth[4]))
# # Save to file
# np.savetxt('U_ED_c_smoothedS006.txt', data, fmt='%.6f')
    
# plt.figure(figsize=(10,9))
# for i in range(5):
#     plt.subplot(3,2,i+1)
#     plt.plot(t_dpm, dcdt_list[i], label='DPM CG data')
#     plt.plot(t_dpm[5:], E_list[i][5:]-D_list[i][5:], label='computed from E and D')
#     plt.xlabel('t [s]')
#     plt.ylabel(r'$dC/dt$ [kg/m$^2$/s]')
# plt.legend()
# plt.tight_layout()