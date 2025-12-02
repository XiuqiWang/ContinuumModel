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
import math

### FROM Ua to FD; Calibrate expression of mom_drag
h = 0.2 - 0.00025*13.5
D = 0.00025
kappa = 0.4
rho_air = 1.225
rho_sand = 2650
nu_a = 1.46e-5
Ruc = 24
Cd_inf = 0.5
Shields = np.linspace(0.02, 0.06, 5)
u_star = np.sqrt(Shields * (2650-1.225)*9.81*D/1.225)
t = np.linspace(0, 5, 501)
mp = 2650 * np.pi/6 * D**3 #particle mass

# def drag_model(x, b_Ua, b_urel):
#     Ua, U, c = x
#     Urel = b_urel*(b_Ua * Ua - U)
#     Re = abs(Urel)*D/nu_a
#     Ruc = 24
#     Cd_inf = 0.5
#     Cd = (np.sqrt(Cd_inf)+np.sqrt(Ruc/Re))**2   
#     fdrag = np.pi/8 * D**2 * rho_air * Urel * abs(Urel) * Cd
#     return fdrag

# def drag_model_ori(x, Cref, Cref_urel):
#     Ua, U, c = x
#     b_urel = 1/np.sqrt(1 + c/Cref_urel)
#     b_Ua = np.sqrt(1-c/(c+Cref))
#     Urel = b_urel*(b_Ua * Ua - U)
#     Re = abs(Urel)*D/nu_a
#     Cd = (np.sqrt(Cd_inf)+np.sqrt(Ruc/Re))**2   
#     fdrag = np.pi/8 * D**2 * rho_air * Urel * abs(Urel) * Cd
#     return fdrag

def CalMdrag(x, b):
    Ua, U, c = x
    Urel = b * Ua - U
    Re = abs(Urel)*D/nu_a
    Cd = (np.sqrt(Cd_inf)+np.sqrt(Ruc/Re))**2   
    Mdrag = np.pi/8 * D**2 * rho_air * Urel * abs(Urel) * Cd * c/mp
    return Mdrag

def CalUrel(x, Cref, Cref_urel):
    Ua, U, c = x
    b_urel = 1/np.sqrt(1 + c/Cref_urel)
    b_Ua = np.sqrt(1-c/(c+Cref))
    Urel = b_urel*(b_Ua * Ua - U)
    return Urel

def compute_b(Mdrag, c, U, Ua, max_iter=50, tol=1e-6):    
    Mdrag = np.asarray(Mdrag, dtype=float)
    c     = np.asarray(c,     dtype=float)
    U     = np.asarray(U,     dtype=float)
    Ua    = np.asarray(Ua,    dtype=float)

    # Avoid division by zero
    eps = 1e-16
    Ua_safe = Ua + eps
    c_safe  = c  + eps

    # Initial guess
    b = np.ones_like(Mdrag)

    # Pre-factor for X (everything except Cd)
    K = Mdrag * mp * 8.0 / (np.pi * D**2 * rho_air * c_safe)

    for _ in range(max_iter):
        # Using current b, compute U_rel, Re, Cd
        Urel = b * Ua_safe - U
        Re   = np.abs(Urel) * D / (nu_a + eps) + eps
        Cd   = (np.sqrt(Cd_inf) + np.sqrt(Ruc / Re))**2

        # Solve for Urel_new from Mdrag expression with this Cd
        X = K / Cd                      # X = U_rel * |U_rel|
        Urel_new = np.sign(X) * np.sqrt(np.abs(X))

        # Update b
        b_new = (U + Urel_new) / Ua_safe

        # Check convergence
        if np.all(np.abs(b_new - b) < tol):
            b = b_new
            break

        b = b_new

    return b

def Fitb(x, b0, b_inf, k0, lamda):
    U, c = x
    k = k0/(1+lamda*c)
    b = b0 + (b_inf - b0)*(1 - np.exp(-k*U))
    return b

def weighted_r2(y_true, y_pred, weights):
    y_avg = np.average(y_true, weights=weights)
    ss_res = np.sum(weights * (y_true - y_pred)**2)
    ss_tot = np.sum(weights * (y_true - y_avg)**2)
    return 1 - ss_res / ss_tot

def r2_score(y, ypred):
    ss_res = np.sum((y - ypred)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    return 1.0 - ss_res/ss_tot

def BinbandU(U, b, Ubin):
    b = np.asarray(b, dtype=float)
    U  = np.asarray(U, dtype=float)
    edges = np.asarray(Ubin, dtype=float)

    if U.shape != b.shape:
        raise ValueError("U and b must have the same shape.")
    if edges.ndim != 1 or edges.size < 2:
        raise ValueError("Ubin must be a 1D array of length >= 2 (bin edges).")
    if not np.all(np.diff(edges) > 0):
        raise ValueError("Uabin must be strictly increasing.")

    # keep only finite pairs
    m = np.isfinite(U) & np.isfinite(b)
    U = U[m]
    b = b[m]

    nbins = edges.size - 1
    b_mean   = np.full(nbins, np.nan, dtype=float)
    b_se     = np.full(nbins, np.nan, dtype=float)
    U_mean    = np.full(nbins, np.nan, dtype=float)

    for i in range(nbins):
        lo, hi = edges[i], edges[i+1]
        # right-inclusive on the last bin
        if i < nbins - 1:
            sel = (U >= lo) & (U < hi)
        else:
            sel = (U >= lo) & (U <= hi)

        if not np.any(sel):
            continue

        b_i = b[sel]
        U_i = U[sel]
        n = b_i.size

        b_mean[i] = np.mean(b_i)
        U_mean[i] = np.mean(U_i)
        if n > 1:
            sd = np.std(b_i, ddof=1)
            se = sd / np.sqrt(n)
            b_se[i] = np.nan if se == 0 else se
        else:
            b_se[i] = np.nan

        U_mean[i] = np.mean(U_i)
    return b_mean, b_se, U_mean

C_all_S, U_all_S, Ua_all_S, MD_all_S, fd_ori, fd_ori_se, fd_com, Md_all_S, b_all_S = [], [], [], [], [], [], [], [], []
Ubin_all_S, bbin_all_S, bbinse_all_S = [], [], []
omega_labels = ['Dry', 'M1', 'M5', 'M10', 'M20']
Omega = [0, 1, 5, 10, 20]

    # ---- Load data ----
for label in omega_labels:
    for i in range(2, 7):
        file_fd = f'TotalDragForce/Mdrag/FD_S00{i}{label}.txt'
        data_FD = np.loadtxt(file_fd)
        MD = data_FD
        file_path = f'CGdata/hb=13.5d/Shields00{i}{label}-135d.txt'
        data_dpm = np.loadtxt(file_path)
        C_dpm = data_dpm[:, 1]
        U_dpm = data_dpm[:, 2]
        file_ua = f'TotalDragForce/Ua-t/Uair_ave-tS00{i}{label}.txt'
        Ua_dpm = np.loadtxt(file_ua, delimiter='\t')[:, 1]
        
        # binning Ua and getting the mean of RHS
        # fd_dpm = MD*mp/C_dpm
        
        b_dpm = compute_b(MD, C_dpm, U_dpm, Ua_dpm)
        # U_bin = np.linspace(min(U_dpm), max(U_dpm), math.ceil((max(U_dpm) - min(U_dpm)) / 0.5))
        # b_mean, b_se, U_mean = BinbandU(U_dpm, b_dpm, U_bin)
        
        # Combine
        U_all_S.append(U_dpm)
        Ua_all_S.append(Ua_dpm)
        C_all_S.append(C_dpm)
        # fd_ori.append(fd_dpm)
        Md_all_S.append(MD)
        b_all_S.append(b_dpm)
        
        # binned
        # Ubin_all_S.append(U_mean)
        # bbin_all_S.append(b_mean)
        # bbinse_all_S.append(b_se)

U_all = np.concatenate(U_all_S)
# Ureal_all = np.concatenate(U_all_S)
# Ua_all = np.concatenate(Ua_all_S)
C_all = np.concatenate(C_all_S)
# fd_all = np.concatenate(fd_ori)
# breal_all = np.concatenate(b_all_S)
b_all = np.concatenate(b_all_S)
# bse_all = np.concatenate(bbinse_all_S)

mask = np.isfinite(U_all) & np.isfinite(b_all) & np.isfinite(C_all)
U_all, b_all, C_all = U_all[mask], b_all[mask], C_all[mask]

popt, _ = curve_fit(Fitb, [U_all, C_all], b_all, absolute_sigma=True, maxfev=20000)
b0, b_inf, k0, lamda = popt
print(f'b0={b0:.4f}, b_inf={b_inf:.4f}, k0 = {k0:.4f}, lamda = {lamda:.4f}')

# Cref, Cref_urel = 1.0, 0.024#1.0, 0.0088
b_pred = Fitb([U_all, C_all], b0, b_inf, k0, lamda)
r2 = r2_score(b_all, b_pred)
print('r2', r2)

plt.close('all')
# Cref, Cref_urel = 1.0, 0.024#1.0, 0.0088
for i in range(5): #Omega
    plt.figure(figsize=(12, 8))
    for j in range(5): #Shields
        plt.subplot(3, 2, j+1)
        index_byS = i*5+j 
        b_com = Fitb([U_all_S[index_byS], C_all_S[index_byS]], b0, b_inf, k0, lamda)
        plt.plot(U_all_S[index_byS], b_all_S[index_byS], 'o', label=r'DPM $\hat{b}$ = f($\hat{M}_{drag}$, $\hat{U}_a$, $\hat{U}$, $\hat{c}$)')
        plt.plot(U_all_S[index_byS], b_com, 'o', label=r'Computed $b = b_{0} + (b_{inf} - b_{0})\cdot(1 - \exp(-k\hat{U}))$''\n' r'$k = k_0/(1+\lambda \hat{c})$')
        plt.title(fr"$\tilde{{\Theta}}$=0.0{j+2}, $\Omega$={Omega[i]}%")
        plt.xlabel(r'$\hat{U}$ [m/s]')
        plt.ylabel(r'$b$ [-]')
        plt.ylim(0, 1.3)
        plt.xlim(0, 10)
        plt.grid(True)
        if j == 0:
            plt.legend(fontsize=9, loc='lower right')
    plt.tight_layout()
    plt.show()
    
# for i in range(5): #Omega
#     plt.figure(figsize=(12, 8))
#     for j in range(5): #Shields
#         plt.subplot(3, 2, j+1)
#         index_byS = i*5+j 
#         b_com = Fitb([U_all_S[index_byS], C_all_S[index_byS]], b0, b_inf, k0, lamda)
#         plt.plot(t, b_all_S[index_byS], 'o', label=r'DPM $\hat{b}$ = f($\hat{M}_{drag}$, $\hat{U}_a$, $\hat{U}$, $\hat{c}$)')
#         plt.plot(t, b_com, 'o', label=r'Computed $b = b_{0} + (b_{inf} - b_{0})\cdot(1 - \exp(-k\hat{U}))$''\n' r'$k = k_0/(1+\lambda \hat{c})$')
#         plt.title(fr"$\tilde{{\Theta}}$=0.0{j+2}, $\Omega$={Omega[i]}%")
#         plt.xlabel(r'$t$ [s]')
#         plt.ylabel(r'$b$ [-]')
#         plt.ylim(0, 1.5)
#         plt.xlim(0, 5)
#         plt.grid(True)
#         if j == 0:
#             plt.legend(fontsize=9, loc='upper right')
#     plt.tight_layout()
#     plt.show()
    
# # plot to find out the dependence of c/U on b    
# for i in range(5): #Omega
#     plt.figure(figsize=(8, 8))
#     for j in range(5): #Shields
#         plt.subplot(3, 2, j+1)
#         index_byS = i*5+j 
#         Md_dpm = Md_all_S[index_byS]
#         b_dpm = compute_b(Md_dpm, C_all_S[index_byS], U_all_S[index_byS], Ua_all_S[index_byS])
#         plt.plot(C_all_S[index_byS], b_dpm, 'o', label='DPM')
#         plt.title(fr"$\tilde{{\Theta}}$=0.0{j+2}, $\Omega$={Omega[i]}%")
#         plt.xlabel(r'$\hat{c}$ [kg/m$^2$]')
#         plt.ylabel(r'$\hat{b}$ [-]')
#         plt.ylim(0, 1.1)
#         plt.xlim(0, 0.3)
#         plt.grid(True)
#         plt.legend()
#     plt.tight_layout()
#     plt.show()
    
# for i in range(5): #Omega
#     plt.figure(figsize=(8, 8))
#     for j in range(5): #Shields
#         plt.subplot(3, 2, j+1)
#         index_byS = i*5+j 
#         Md_dpm = Md_all_S[index_byS]
#         b_dpm = compute_b(Md_dpm, C_all_S[index_byS], U_all_S[index_byS], Ua_all_S[index_byS])
#         plt.plot(U_all_S[index_byS], b_dpm, 'o', label='DPM')
#         plt.title(fr"$\tilde{{\Theta}}$=0.0{j+2}, $\Omega$={Omega[i]}%")
#         plt.xlabel(r'$\hat{U}$ [m/s]')
#         plt.ylabel(r'$\hat{b}$ [-]')
#         plt.ylim(0, 1.1)
#         plt.xlim(0, 9)
#         plt.grid(True)
#         plt.legend()
#     plt.tight_layout()
#     plt.show()    

# plot b,U,c - t
# for i in range(5): #Omega
#     plt.figure(figsize=(8, 8))
#     for j in range(5): #Shields
#         plt.subplot(3, 2, j+1)
#         index_byS = i*5+j 
#         Md_dpm = Md_all_S[index_byS]
#         b_dpm = compute_b(Md_dpm, C_all_S[index_byS], U_all_S[index_byS], Ua_all_S[index_byS])
#         plt.plot(t, b_dpm, 'o', label=r'$\hat{b}$')
#         plt.plot(t, U_all_S[index_byS], 'o', label=r'$\hat{U}$')
#         plt.plot(t, C_all_S[index_byS], 'o', label=r'$\hat{c}$')
#         plt.title(f"S00{j+2} {omega_labels[i]}")
#         plt.xlabel(r'$t$ [s]')
#         plt.ylabel(r'$\hat{b}$, $\hat{U}$, $\hat{c}$')
#         plt.ylim(0, 10.0)
#         plt.xlim(0, 5)
#         plt.grid(True)
#         plt.legend()
#     plt.tight_layout()
#     plt.show()    

for i in range(5): #Omega
    plt.figure(figsize=(10, 8))
    for j in range(5): #Shields
        plt.subplot(3, 2, j+1)
        index_byS = i*5+j 
        b_com = Fitb([U_all_S[index_byS], C_all_S[index_byS]], b0, b_inf, k0, lamda)
        Mdrag = CalMdrag([Ua_all_S[index_byS], U_all_S[index_byS], C_all_S[index_byS]], b_com)
        plt.plot(t, Md_all_S[index_byS], '.', label=r'DPM $\hat{M}_{drag}$')
        plt.plot(t, Mdrag, '.', label='Computed $M_{drag}=f(\hat{U_{a}}, \hat{U}, \hat{c}, $b$)$')
        plt.title(f"S00{j+2} {omega_labels[i]}")
        plt.xlabel(r'$t$ [s]')
        plt.ylabel(r'$M_{drag}$ [N/m$^2$]')
        plt.ylim(0, 2.2)
        plt.xlim(0, 5)
        plt.grid(True)
        plt.legend()
    plt.tight_layout()
    plt.show()
    
