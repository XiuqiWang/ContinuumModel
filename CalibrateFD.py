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

def drag_model(Ua, U, hs, z0, k, n):
    # ueff = Ua/(np.log(h/z0)-1) * np.log(zr/z0)
    # ueff = b*Ua
    ueff = Ua * (np.log(hs/z0) - 1) / (np.log(h/z0) - 1)
    # print('Ua', Ua)
    # print('zr', zr)
    # print('ueff', ueff)
    U_rel = ueff - U
    return k * np.abs(U_rel) * U_rel**(n-1)  # |U|*U^(n-1) = U*|U|^(n-1)

# Predicted drag function for fitting
def total_drag_fit(X, hs, z0, k, n):
    Ua, U, C = X
    return (C / mp) * drag_model(Ua, U, hs, z0, k, n)

C_all_S, U_all_S, Ua_all_S, MD_all_S = [], [], [], []
for i in range(2, 7):
    # ---- Load data ----
    file_fd = f'TotalDragForce/FD_S00{i}dry.txt'
    data_FD = np.loadtxt(file_fd)
    MD = data_FD/(100 * D * 2 * D)
    file_c = f'Shields00{i}dry.txt'
    data_dpm = np.loadtxt(file_c)
    Q_dpm = data_dpm[:, 0]
    C_dpm = data_dpm[:, 1]
    U_dpm = data_dpm[:, 2]
    file_ua = f'TotalDragForce/Uair_ave-tS00{i}Dryh02.txt'
    Ua_dpm = np.loadtxt(file_ua, delimiter='\t')[:, 1]
    
    # Combine
    C_all_S.append(C_dpm)
    U_all_S.append(U_dpm)
    Ua_all_S.append(Ua_dpm)
    MD_all_S.append(MD)

C_all = np.concatenate(C_all_S)
U_all = np.concatenate(U_all_S)
Ua_all = np.concatenate(Ua_all_S)
MD_all = np.concatenate(MD_all_S)
    
Xdata = np.vstack((Ua_all, U_all, C_all))
p0 = [0.01, 2.5e-6, 1e-6, 2.0]  # initial guesses for b, k, n
params, cov = curve_fit(total_drag_fit, Xdata, MD_all, p0=p0, maxfev=5000)
print("Fitted parameters: zr = {:.3e}, z0 = {:.3e}, k = {:.3e}, n = {:.3f}".format(*params))
# ------------------ Compute fitted MD ------------------
MD_fit_all = total_drag_fit(Xdata, *params)
# Compute residuals
residuals = MD_all - MD_fit_all
# Total sum of squares
ss_tot = np.sum((MD_all - np.mean(MD_all))**2)
# Residual sum of squares
ss_res = np.sum(residuals**2)
# R^2
R2 = 1 - ss_res/ss_tot
print(f"R^2 = {R2:.4f}")

MD_fit = []
for i in range(5):
    Xdata = np.vstack((Ua_all_S[i], U_all_S[i], C_all_S[i]))
    MD_fit.append(total_drag_fit(Xdata, *params))

plt.close('all')
plt.figure(figsize=(12, 10))
for i in range(5):
    plt.subplot(3, 2, i + 1)
    plt.plot(t, MD_all_S[i], label='fit')
    plt.plot(t, MD_fit[i], label='DPM')
    # plt.plot(t_con, FD_continuum, label='Continuum')
    plt.title(f"S00{i+2} Dry")
    plt.xlabel('time [s]')
    plt.ylabel(r'$M_{drag}$ [N/m$^2$]')
    plt.ylim(0,1)
    plt.grid(True)
    plt.legend()
plt.tight_layout()
plt.show()