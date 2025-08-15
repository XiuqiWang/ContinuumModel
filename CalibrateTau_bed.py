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
minusphi_dpm_all = []
tau_top_all = []
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

def tau_bottom_phi_model(Ua_arr, l_eff, phi_b):
    """Height-averaged mixing-length bottom stress"""
    tau_bed = rho_a * (nu_a + (l_eff**2) * np.abs(Ua_arr)/h) * Ua_arr / h
    return tau_bed * (1-phi_b)
    
# Loop over conditions S002 to S006
for i in range(2, 7):
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

    # ---- Store results ----
    Ua_dpm_all.append(Ua_dpm)
    dUadt_dpm_all.append(dUa_dt)
    FD_dpm_all.append(FD_dpm)
    minusphi_dpm_all.append(1-phi)
    tau_top_all.append(np.ones(len(FD_dpm))*rho_a*u_star[i-2]**2)

Ua_all = np.concatenate(Ua_dpm_all)
duadt_all = np.concatenate(dUadt_dpm_all)  
FD_all = np.concatenate(FD_dpm_all)  
minusphi_all = np.concatenate(minusphi_dpm_all)
tau_top = np.concatenate(tau_top_all)

# observed S = tau_bed*(1-phi_bed). We approximate phi_bed ~ phi(t)
S_obs = tau_top - FD_all - rho_a * h * minusphi_all * duadt_all

p0 = [0.005, 0.4]  # initial guess for l_eff (m)
params_B, cov_B = curve_fit(lambda Ua_arr, l_eff, phi_b: tau_bottom_phi_model(Ua_arr, l_eff, phi_b), Ua_all, S_obs, p0=p0, bounds=(1e-6, np.inf), maxfev=20000)
l_eff_hat, phi_b_hat = params_B
S_B_fit = tau_bottom_phi_model(Ua_all, l_eff_hat, phi_b_hat)

print("(mixing-length) fit: l_eff =", l_eff_hat, "m")
print("R2 B =", r2_score(S_obs, S_B_fit))

# # ---- Plotting ----
# plt.close('all')
# plt.figure(figsize=(12, 10))
# for i in range(5):
#     plt.subplot(3, 2, i + 1)
#     plt.plot(t, Ua_computed_all[i], label='Computed', lw=1.8)
#     plt.plot(t, Ua_dpm_all[i], '--', label='DPM', lw=1.5)
#     plt.title(f"S00{i+2} Dry")
#     plt.xlabel("Time [s]")
#     plt.ylabel("$U_a$ [m/s]")
#     plt.grid(True)
#     plt.legend()

# plt.tight_layout()
# # plt.suptitle("Comparison of Computed and DPM $U_a$ Across Conditions", fontsize=16, y=1.02)
# plt.show()

# plt.figure()
# for i in range(5):
#     plt.subplot(3, 2, i + 1)
#     plt.plot(t, dUa_dt_ODE_all[i], label='Computed $U_a$', lw=1.8)
#     plt.plot(t, dUa_dt_numerical_all[i], label='DPM $U_a$', lw=1.5)
#     plt.title(f"S00{i+2} Dry")
#     plt.xlabel("Time [s]")
#     plt.ylabel("$dU_a/dt$ [m/s]")
#     plt.grid(True)
#     plt.legend()
# plt.tight_layout()