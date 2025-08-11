# -*- coding: utf-8 -*-
"""
Created on Tue Aug  5 16:37:40 2025

@author: WangX3
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# from FD to Ua; Calibrate h dUa/dt = Mom_gain - Mom_loss - Mom_drag
h = 0.2 - 0.00025*10
D = 0.00025
# CD_air = 8e-3
rho_air = 1.225
rho_sand = 2650
Shields = np.linspace(0.02, 0.06, 5)
u_star = np.sqrt(Shields * (2650-1.225)*9.81*D/1.225)
t = np.linspace(0, 5, 501)

# Containers for storing results
Ua_dpm_all = []
FD_dpm_all = []
Ua_computed_all = []
dUa_dt_numerical_all = []
dUa_dt_ODE_all = []

 # ---- Define ODE ----
def dUa_dt_ODE(t, Ua, u_star, M_D_interp, c_interp):
    # M_loss = (1 - 2e-2)*rho_air * (1.46e-5 + D* Ua/h*2) * Ua/h*2
    Re = Ua * D / (1.46e-5)
    CD_bed = 1.05e-6 * Re**2
    M_loss = 0.5 * rho_air * CD_bed * Ua * abs(Ua)
    tau_top = rho_air * u_star**2
    RHS = tau_top - M_loss - M_D_interp(t)
    return RHS / (h * rho_air * (1 - c_interp(t) / (rho_sand * h)))

# def dUa_dt(t, Ua, u_star):
#     M_loss = 0.5 * rho_air * CD_bed * Ua * abs(Ua)
#     tau_top = rho_air * u_star**2
#     RHS = tau_top - M_loss - M_D_interp(t)
#     return RHS / (h * rho_air * (1 - c_interp(t) / (rho_sand * h)))
    
# Loop over conditions S002 to S006
for i in range(2, 7):
    # ---- Load data ----
    file_ua = f'TotalDragForce/Uair_ave-tS00{i}Dryh02.txt'
    Ua_dpm = np.loadtxt(file_ua, delimiter='\t')[:, 1]
    Ua0 = Ua_dpm[0]

    file_fd = f'TotalDragForce/FD_S00{i}dry.txt'
    data_FD = np.loadtxt(file_fd)
    FD_dpm = data_FD / (100 * D * 2 * D)
    
    file_c = f'Shields00{i}dry.txt'
    data_dpm = np.loadtxt(file_c)
    c_dpm = data_dpm[:, 1]

    # Interpolate FD_dpm for ODE solver
    M_D_interp = interp1d(t, FD_dpm, bounds_error=False, fill_value="extrapolate")
    c_interp = interp1d(t, c_dpm, bounds_error=False, fill_value="extrapolate")

    # ---- Solve ODE ----
    # sol = solve_ivp(dUa_dt, [t[0], t[-1]], [Ua0], args=(u_star[i-2],), t_eval=t, method='RK45')
    # Ua_computed = sol.y[0]
    # Compute ODE-based dUa/dt
    
    dt = t[1] - t[0]  # Time step
    dUa_dt_ODE_values = np.zeros_like(t-1)
    for j in range(len(t-1)):
        dUa_dt_ODE_values[j] = dUa_dt_ODE(t[j], Ua_dpm[j], u_star[i-2], M_D_interp, c_interp)
        
    Ua_computed = np.empty_like(t, dtype=float)
    Ua_computed[0] = Ua0
    for n in range(len(t)-1):
        Ua_computed[n+1] = Ua_computed[n] + dUa_dt_ODE_values[n]*dt
    
    # Compute numerical dUa/dt (finite difference)
    
    dUa_dt_numerical = np.gradient(Ua_dpm, dt)  # Central difference

    # ---- Store results ----
    Ua_dpm_all.append(Ua_dpm)
    FD_dpm_all.append(FD_dpm)
    Ua_computed_all.append(Ua_computed)
    dUa_dt_ODE_all.append(dUa_dt_ODE_values)
    dUa_dt_numerical_all.append(dUa_dt_numerical)

# ---- Plotting ----
plt.close('all')
plt.figure(figsize=(12, 10))
for i in range(5):
    plt.subplot(3, 2, i + 1)
    plt.plot(t, Ua_computed_all[i], label='Computed', lw=1.8)
    plt.plot(t, Ua_dpm_all[i], '--', label='DPM', lw=1.5)
    plt.title(f"S00{i+2} Dry")
    plt.xlabel("Time [s]")
    plt.ylabel("$U_a$ [m/s]")
    plt.grid(True)
    plt.legend()

plt.tight_layout()
# plt.suptitle("Comparison of Computed and DPM $U_a$ Across Conditions", fontsize=16, y=1.02)
plt.show()

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

# ### FROM Ua to FD; Calibrate expression of mom_drag
# C_drag_adjusted = 0.08
# mp = 2650 * np.pi/6 * D**3 #particle mass
# def Calfd(u_air, u_sal):
#     Re = abs(u_air - u_sal) * D/(1.45e-6) + 1e-12
#     C_D = (np.sqrt(0.5) + np.sqrt(24 / Re))**2
#     fd = 0.5* np.pi/8 * 1.225 * D**2 * C_D * (u_air - u_sal)* abs(u_air - u_sal)
#     return fd

# MD_com_all = []
# MD_all = []
# for i in range(2, 7):
#     # ---- Load data ----
#     file_fd = f'TotalDragForce/FD_S00{i}dry.txt'
#     data_FD = np.loadtxt(file_fd)
#     MD = data_FD/(100 * D * 2 * D)
#     data = np.loadtxt('Shields006dry.txt')
#     Q_dpm = data[:, 0]
#     C_dpm = data[:, 1]
#     U_dpm = data[:, 2]
#     file_ua = f'TotalDragForce/Uair_ave-tS00{i}Dryh02.txt'
#     Ua_dpm = np.loadtxt(file_ua, delimiter='\t')[:, 1]

#     fd = Calfd(Ua_dpm, U_dpm)
#     MD_com = fd * C_dpm/mp * C_drag_adjusted
#     MD_com_all.append(MD_com)
#     MD_all.append(MD)

# plt.figure(figsize=(12, 10))
# for i in range(5):
#     plt.subplot(3, 2, i + 1)
#     plt.plot(t, MD_com_all[i], label='computed')
#     plt.plot(t, MD_all[i], label='DPM')
#     # plt.plot(t_con, FD_continuum, label='Continuum')
#     plt.title(f"S00{i+2} Dry")
#     plt.xlabel('time [s]')
#     plt.ylabel(r'$Mom_{drag}$ [N/m$^2$]')
#     plt.ylim(0,2)
#     plt.grid(True)
#     plt.legend()
# plt.tight_layout()
# plt.show()