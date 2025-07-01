# -*- coding: utf-8 -*-
"""
Created on Wed Jun 11 16:12:21 2025

@author: WangX3

extended for moist cases
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import pandas as pd

# constants
D = 0.00025
constant = np.sqrt(9.81 * D)
CD_air = 9e-3
CD_drag_reduce = 0.5
CD_bed = 3e-4
# mass of air per unit area
hsal = 0.2 - 0.00025*10
mass_air = 1.225 * hsal
# Environmential conditions
Shields = np.linspace(0.01, 0.06, 6)
u_star = np.sqrt(Shields * (2650-1.225)*9.81*0.00025/1.225)
Omega = np.array([0, 0.01, 0.05, 0.1, 0.2])

# numerically solves Uim from Usal
def solveUim(Uim, u_sal, Omega):
    alpha = 39.21 - 2.31*Omega**0.53 
    beta = 15.74 * (1-np.exp(-102.58*Omega)) + 105.73
    denom = Uim / constant + beta
    arg = alpha / denom
    if np.abs(arg) > 1:
        return np.inf
    theta = np.arcsin(arg)
    return Uim * np.cos(theta) - u_sal

def Calfd(u_air, u_sal):
    C_D = (np.sqrt(0.5) + np.sqrt(24 / (abs(u_air - u_sal) * D/(1.45e-6))))**2
    fd = 0.5* np.pi/8 * 1.225 * D**2 * C_D * (u_air - u_sal)* abs(u_air - u_sal)
    return fd

def make_odefun(u_star, Omega, splash_history):
    def odefun(t, y):
        # air velocity
        u_air = y[2] / mass_air
        # saltating layer velocity
        u_sal = y[1] / (y[0] + 1e-9)
        
        # solve Uim and theta_im from u_sal
        U_guess = u_sal 
        # Solve Uim from Usal
        Uim_solution = fsolve(lambda Uim: solveUim(Uim, u_sal, Omega), U_guess)[0]
        # print("Usal, Uim =", u_sal, Uim_solution)
        # impact angle
        arg_im = np.arcsin((39.21 - 2.31*Omega**0.53) / (abs(Uim_solution) / constant + (15.74 * (1-np.exp(-102.58*Omega)) + 105.73)))
        arg_im_clipped = np.clip(arg_im, -1.0, 1.0)
        if np.any((arg_im < -1) | (arg_im > 1)):
            print("Warning: arg_im out of domain, clipping applied")
        theta_im = np.arcsin(arg_im_clipped)
        
        # ejection angle
        cos_thetaej = np.cos(47/180*np.pi)
        #deposition angle
        theta_dep = 25/180*np.pi
        
        # time scale for collision and deposition
        u_dep = Uim_solution*0.18
        Tim = 1e-9 + 2*abs(Uim_solution)*np.sin(theta_im)/9.81
        Tdep = 1e-9 + 2*abs(u_dep)*np.sin(theta_dep)/9.81
        
        # splash functions
        COR = 5.17 * (abs(Uim_solution) / constant)**(-0.6) + 0.11*np.log(1 + 591.22*Omega) * np.exp(-(Uim_solution/constant - 158.44)**2/(2*54.87**2))
        Pr = 0.94*np.exp(-7.11*np.exp(-0.11 * abs(Uim_solution) / constant))
        arg_re = (-0.0006 - 0.004*Omega**0.38) * abs(Uim_solution) / constant + 0.65
        arg_re_clipped = np.clip(arg_re, -1.0, 1.0)
        if np.any((arg_re < -1) | (arg_re > 1)):
            print("Warning: arg_re out of domain, clipping applied")
        theta_re = np.arcsin(arg_re_clipped)
        NE = (0.04 - 0.04*Omega**0.23) * abs(Uim_solution) / constant 
        if Omega == 0:
            if Uim_solution >= 0:
                UE = 5.02 * constant 
            else:
                UE = -5.02 * constant 
        elif Uim_solution >= 0:
            UE = (10.09*Omega + 5.02)*(Uim_solution/constant)**(-0.28*Omega + 0.07) * constant
        else:
            UE = -(10.09*Omega + 5.02)*(abs(Uim_solution)/constant)**(-0.28*Omega + 0.07) * constant
        Ure = Uim_solution * COR
        cos_thetare = np.cos(theta_re)
        
        # momentum is transferred quickly from air to saltating layer due to drag
        # (the quicker, the bigger the overshoot in momentum)
        fd_sal = Calfd(u_air, u_sal) 
        mp = 2650 * np.pi/6 * D**3 #particle mass
        # print('fd_sal:', fd_sal)
        mom_drag = CD_drag_reduce * y[0]*fd_sal/mp
        # mass is gained through sand erosion, mom is gained through erosion and rebound
        mass_ero =  NE * y[0] 
        mom_ero = mass_ero * UE * cos_thetaej 
        mom_re = y[0] * Pr * Ure * cos_thetare 
        # mass is lost through deposition, mom is lost through incident motion
        mass_dep = (1-Pr) * y[0]
        mom_inc =  y[0] * Pr * u_sal/Tim + y[0] * (1 - Pr) * u_dep*np.cos(theta_dep)/Tdep 
        
        # momentum of air gets replenished slowly through shear at the top boundary
        u_am = u_star/0.4 * np.log((hsal-0.00025*10)/(0.00025/30)) #law of the wall (COMSALT)
        mom_air_gain =  0.5* CD_air * 1.225 * (u_am - u_air) *abs(u_am - u_air) # 1.225 * u_star **2
        # momentum of air gets lost slowly from bed shear
        mom_air_loss = 0.5* 1.225 * CD_bed * u_air * abs(u_air) 
        
        # record splash-related terms
        splash_history.append({'t': t,'Uim': Uim_solution,'Pr': Pr,'COR': COR,'cos_thetare': cos_thetare,'NE': NE, 'UE': UE})
        
        # assemble source terms
        dydt = [mass_ero/Tim - mass_dep/Tdep,
                mom_drag + mom_ero/Tim + mom_re/Tim - mom_inc,
                mom_air_gain - mom_air_loss - mom_drag]
        
        return dydt
    return odefun
    
# Initial conditions
c0 = 0.0139
Usal0 = 2.9279
Uair0 = [5.1, 7.4, 9.2, 10.7, 12.0, 13.1] #h=0.2
# Time span
t_span = [0, 20]
t_eval = np.linspace(t_span[0], t_span[1], 500)

y_eval_all = []
splash_history_all = []

for omega in Omega:
    y_eval = []
    splash_history_omega = []
    for i in range(len(Uair0)):
        splash_history = []
        y0 = [c0, c0*Usal0, Uair0[i]*mass_air] #mass sal, momentum sal, momentum air
        odefun = make_odefun(u_star[i], omega, splash_history)
        sol = solve_ivp(odefun, t_span, y0, method='Radau', dense_output=True)
        y_eval.append(sol.sol(t_eval))
        splash_history_omega.append(splash_history)
    y_eval_all.append(y_eval)
    splash_history_all.append(splash_history_omega)

plt.close('all')
plt.figure(figsize=(15, 8))
for i in range(len(u_star)):
    plt.subplot(2,3,i+1)
    for j in range(len(Omega)):
        plt.plot(t_eval, y_eval_all[j][i][1], label=f"$\\Omega$ = {Omega[j]*100:.1f}%")
    plt.legend()
    plt.xlabel('Time [s]')
    plt.xlim(left=0)
    plt.ylabel(r'$Q$ [kg/m/s]')
    # plt.ylim(0,0.8)
    plt.xlim(left=0)
    plt.title(f"$\Theta$=0.0{i+1}")
plt.tight_layout()
plt.show()

plt.figure(figsize=(15, 8))
for i in range(len(u_star)):
    plt.subplot(2,3,i+1)
    for j in range(len(Omega)):
        plt.plot(t_eval, y_eval_all[j][i][0], label=f"$\\Omega$ = {Omega[j]*100:.1f}%")
    plt.legend()
    plt.xlabel('Time [s]')
    plt.xlim(left=0)
    plt.ylabel(r'$C$ [kg/m$^2$]')
    # plt.ylim(0,0.8)
    plt.xlim(left=0)
    plt.title(f"$\Theta$=0.0{i+1}")
plt.tight_layout()
plt.show()

plt.figure(figsize=(15, 8))
for i in range(len(u_star)):
    plt.subplot(2,3,i+1)
    for j in range(len(Omega)):
        plt.plot(t_eval, y_eval_all[j][i][1]/(y_eval_all[j][i][0] + 1e-9), label=f"$\\Omega$ = {Omega[j]*100:.1f}%")
    plt.legend()
    plt.xlabel('Time [s]')
    plt.xlim(left=0)
    plt.ylabel(r'$U$ [m/s]')
    # plt.ylim(0,0.8)
    plt.xlim(left=0)
    plt.title(f"$\Theta$=0.0{i+1}")
plt.tight_layout()
plt.show()


#evaluate splash terms
fig, axs = plt.subplots(2, 3, figsize=(15, 8))
# Define the labels and keys once for reuse
ylabels = ['Uim [m/s]', 'Pr', 'COR', 'cos_thetare', 'NE', 'UE [m/s]']
keys = ['Uim', 'Pr', 'COR', 'cos_thetare', 'NE', 'UE']
# Plot in a loop
for ax, key, ylabel in zip(axs.flat, keys, ylabels):
    for i in range(len(Omega)):
        splash_df = pd.DataFrame(splash_history_all[i][-1])
        ax.plot(splash_df['t'], splash_df[key], label=f"$\\Omega$ = {Omega[i]*100:.1f}%")
        ax.set_xlabel('Time [s]')
        ax.set_ylabel(ylabel)
        ax.set_xlim(left=0)
        # ax.set_ylim(bottom=0)
        ax.grid(True)
        if key == 'Uim':
            ax.legend()
fig.suptitle(r"$\Theta$=0.06")
plt.tight_layout()
plt.show()

# calculate steady Q
Q_steady_dpm = [0.0047, 0.0137, 0.0188, 0.0257, 0.0398, 0.0392] #dpm 
Q_steady_all,C_steady_all,U_steady_all = [],[],[]

for i in range(len(u_star)):
    Q_steady,C_steady,U_steady = [],[],[]
    for j in range(len(Omega)):
        Cs = np.mean(y_eval_all[j][i][0][400:])
        Qs = np.mean(y_eval_all[j][i][1][400:])
        Us = np.mean(y_eval_all[j][i][1][400:]/(y_eval_all[j][i][0][400:]+1e-9))
        Q_steady.append(Qs)
        C_steady.append(Cs)
        U_steady.append(Us)
    Q_steady_all.append(Q_steady)
    C_steady_all.append(C_steady)
    U_steady_all.append(U_steady)

plt.figure(figsize=(15,5))
plt.plot(Shields, Q_steady_dpm, 'xk', label='DPM simulation')
plt.subplot(1,3,1)
for i in range(len(u_star)):
    plt.plot(Omega*100, Q_steady_all[i], 'o', label=f"$\Theta$=0.0{i+1}")
plt.legend()
plt.xlabel(r'$\Omega$')
plt.ylabel(r'$Q_\mathrm{steady}$ [kg/m/s]')
# plt.xlim(left=0)
plt.ylim(-0.01, 0.1)
plt.xticks([0, 1, 5, 10, 20])
plt.subplot(1,3,2)
for i in range(len(u_star)):
    plt.plot(Omega*100, C_steady_all[i], 'o')
plt.xlabel(r'$\Omega$')
plt.ylabel(r'$C_\mathrm{steady}$ [kg/m$^2$]')
# plt.xlim(left=0)
plt.ylim(-0.01, 0.08)
plt.xticks([0, 1, 5, 10, 20])
plt.subplot(1,3,3)
for i in range(len(u_star)):
    plt.plot(Omega*100, U_steady_all[i], 'o')
plt.xlabel(r'$\Omega$')
plt.ylabel(r'$U_\mathrm{sal,steady}$ [m/s]')
# plt.xlim(left=0)
plt.ylim(0, 2)
plt.xticks([0, 1, 5, 10, 20])
plt.ylim(bottom=0)
plt.legend()
plt.tight_layout()
plt.show()
        

#Testing the splash functions with Omega
# Uim = np.linspace(0.01, 4, 100)
# data_all = {key: [] for key in ['Pr', 'COR', 'cos_thetare', 'NE', 'UE']}
# for j, omega in enumerate(Omega):
#     COR = 5.17 * (Uim / constant)**(-0.6) + 0.11*np.log(1 + 591.22*omega) * np.exp(-(Uim/constant - 158.44)**2/(2*54.87**2))
#     Ure = Uim * COR
#     Pr = 0.94*np.exp(-7.11*np.exp(-0.11 * Uim / constant))
#     theta_re = np.arcsin((-0.0006 - 0.004*omega**0.38) * Uim / constant + 0.65)
#     cos_thetare = np.cos(theta_re)
#     NE = (0.04 - 0.04*omega**0.23) * Uim / constant 
#     if omega == 0:
#         UE = 5.02 * constant * np.ones(len(Uim))
#     else:
#         UE = (10.09*omega + 5.02)*(Uim/constant)**(-0.28*omega + 0.07) *constant
#     # Store results
#     data_all['Pr'].append(Pr)
#     data_all['COR'].append(COR)
#     data_all['cos_thetare'].append(cos_thetare)
#     data_all['NE'].append(NE)
#     data_all['UE'].append(UE)
    

# fig, axs = plt.subplots(2, 3, figsize=(15, 8))
# # Define the labels and keys once for reuse
# ylabels = ['Pr', 'COR', 'cos_thetare', 'NE', 'UE [m/s]']
# keys = ['Pr', 'COR', 'cos_thetare', 'NE', 'UE']
# # Plot in a loop
# for ax, key, ylabel in zip(axs.flat, keys, ylabels):
#     for i in range(len(Omega)):
#         ax.plot(Uim, data_all[key][i], label=f"$\Omega$={Omega[i]*100:.1f}")
#         ax.set_xlabel('Uim m/s')
#         ax.set_ylabel(ylabel)
#         if key == 'Pr':
#             ax.legend()
#         ax.set_xlim(left=0)
#         # ax.set_ylim(bottom=0)
#         ax.grid(True)
# plt.tight_layout()