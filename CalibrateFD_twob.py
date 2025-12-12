# -*- coding: utf-8 -*-
"""
Created on Fri Dec  5 16:07:17 2025

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
const = np.sqrt(9.81*D)

def CalMdrag(x, b_inf, k, cref):
    Ua, U, c = x
    b = b_inf * (1.0 - np.exp(-k * U/const))
    burel = 1/(1+c/cref)
    Urel = burel*(b * Ua - U)
    Re = abs(Urel)*D/nu_a
    Re = np.maximum(Re, 1e-6)
    Cd = (np.sqrt(Cd_inf)+np.sqrt(Ruc/Re))**2   
    Mdrag = np.pi/8 * D**2 * rho_air * Urel * abs(Urel) * Cd * c/mp
    return Mdrag

def r2_score(y, ypred):
    ss_res = np.sum((y - ypred)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    return 1.0 - ss_res/ss_tot

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
        
        # Combine
        U_all_S.append(U_dpm)
        Ua_all_S.append(Ua_dpm)
        C_all_S.append(C_dpm)
        Md_all_S.append(MD)

U_all = np.concatenate(U_all_S)
Ua_all = np.concatenate(Ua_all_S)
C_all = np.concatenate(C_all_S)
Md_all = np.concatenate(Md_all_S)

mask = np.isfinite(U_all) & np.isfinite(Ua_all) & np.isfinite(C_all) & np.isfinite(Md_all)
U_all, Ua_all, C_all, Md_all = U_all[mask], Ua_all[mask], C_all[mask], Md_all[mask]

popt, _ = curve_fit(CalMdrag, [Ua_all, U_all, C_all], Md_all, absolute_sigma=True, maxfev=100000)
b_inf, k, cref = popt
print(f'b_inf={b_inf:.4f}, k = {k:.4f}, cref={cref:.2f}')
Md_pred = CalMdrag([Ua_all, U_all, C_all], b_inf, k, cref)
r2 = r2_score(Md_all, Md_pred)
print('r2', r2)

plt.close('all')
    
for i in range(5): #Omega
    plt.figure(figsize=(10, 8))
    for j in range(5): #Shields
        plt.subplot(3, 2, j+1)
        index_byS = i*5+j 
        Mdrag = CalMdrag([Ua_all_S[index_byS], U_all_S[index_byS], C_all_S[index_byS]], b_inf, k, cref)
        plt.plot(t, Md_all_S[index_byS], '.', label=r'$\hat{M}_\mathrm{sal}$')
        plt.plot(t, Mdrag, '.', label=r'$\breve{M}_{\mathrm{sal}}=f(\hat{U_\mathrm{air}}, \hat{U}, \hat{c}, b_\mathrm{Urel}, b)$')
        plt.title(fr"$\tilde{{\Theta}}$=0.0{j+2}")
        plt.xlabel(r'$t$ [s]')
        plt.ylabel(r'$M_\mathrm{sal}$ [N/m$^2$]')
        plt.ylim(0, 2.5)
        plt.xlim(0, 5)
        plt.grid(True)
        if j == 0:
            plt.legend(fontsize=9, loc='upper right')
    plt.suptitle(fr'$\Omega$={Omega[i]}%')
    plt.tight_layout()
    plt.show()
    