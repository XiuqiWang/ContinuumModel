# -*- coding: utf-8 -*-
"""
Created on Tue Jul  1 14:44:20 2025

@author: WangX3
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
from scipy.stats import lognorm
from scipy.integrate import quad

# constants
D = 0.00025
constant = np.sqrt(9.81 * D)
Shields = np.linspace(0.01, 0.06, 6)
u_star = np.sqrt(Shields * (2650-1.225)*9.81*D/1.225)

# mass of air per unit area
hsal = 0.1 - 0.00025*10
mass_air = 1.225 * hsal
CD_air = 8e-3
CD_bed = 3e-4
CD_drag_reduce = 0.431
alpha_ero = 1#0.786
alpha_dep = 1#1.235
alpha_im = 1#1.684
cos_thetaej = np.cos(47/180*np.pi)
theta_dep = 25/180*np.pi

# numerically solves Uim from Usal
# def solveUim(Uim, u_sal):
#     denom = Uim / constant + 105.73
#     arg = 39.21 / denom
#     if np.abs(arg) > 1:
#         return np.inf
#     theta = np.arcsin(arg)
#     return Uim * np.cos(theta) - u_sal
def solveUim(Uim, u_sal):
    Pr = 0.94 * np.exp(-7.11 * np.exp(-0.11 * abs(Uim) / constant)) 
    Udep = 0.18*Uim
    denom = Uim / constant + 105.73
    arg = 39.21 / denom
    if np.abs(arg) > 1:
        return np.inf
    theta = np.arcsin(arg)
    return Uim*Pr*np.cos(theta) + (1-Pr)*Udep*np.cos(theta_dep) - u_sal

def Calfd(u_air, u_sal):
    C_D = (np.sqrt(0.5) + np.sqrt(24 / (abs(u_air - u_sal) * D/(1.45e-6))))**2
    fd = 0.5* np.pi/8 * 1.225 * D**2 * C_D * (u_air - u_sal)* abs(u_air - u_sal)
    return fd

def make_odefun(u_star):
    def odefun(t, y):
        # air velocity
        u_air = y[2] / mass_air
        # saltating layer velocity
        u_sal = y[1] / (y[0] + 1e-9)
        # solve Uim and theta_im from u_sal
        U_guess = u_sal 
        # Solve Uim from Usal
        Uim_solution = fsolve(lambda Uim: solveUim(Uim, u_sal), U_guess)[0]
        # impact angle
        arg_im = 39.21 / (abs(Uim_solution) / constant + 105.73)
        arg_im_clipped = np.clip(arg_im, -1.0, 1.0)
        if np.any((arg_im < -1) | (arg_im > 1)):
            print("Warning: arg_im out of domain, clipping applied")
        theta_im = np.arcsin(arg_im_clipped)
        
        # time scale for collision and deposition
        u_dep = Uim_solution*0.18
        Tim = 1e-9 + 2*abs(Uim_solution)*np.sin(theta_im)/9.81
        Tdep = 1e-9 + 2*abs(u_dep)*np.sin(theta_dep)/9.81
        
        # splash functions
        COR = 5.17 * (abs(Uim_solution) / constant + 1e-9)**(-0.6) 
        Pr = 0.94 * np.exp(-7.11 * np.exp(-0.11 * abs(Uim_solution) / constant)) 
        arg_re = -0.0006 * abs(Uim_solution) / constant + 0.65 
        arg_re_clipped = np.clip(arg_re, -1.0, 1.0)
        if np.any((arg_re < -1) | (arg_re > 1)):
            print("Warning: arg_re out of domain, clipping applied")
        theta_re = np.arcsin(arg_re_clipped)
        NE = 0.04 * abs(Uim_solution) / constant 
        if Uim_solution >= 0:
            UE = 5.02 * constant
        else:
            UE = -5.02 * constant 
        Ure = Uim_solution * COR
        cos_thetare = np.cos(theta_re)
        
        # momentum is transferred quickly from air to saltating layer due to drag
        # (the quicker, the bigger the overshoot in momentum)
        fd_sal = Calfd(u_air, u_sal) # the drag force on saltating particles
        mp = 2650 * np.pi/6 * D**3 #particle mass
        mom_drag = CD_drag_reduce * y[0]*fd_sal/mp
        # mass is gained through sand erosion, mom is gained through erosion and rebound
        mass_ero =  alpha_ero * NE * y[0] 
        mom_ero = mass_ero * UE * cos_thetaej 
        mass_im = alpha_im * y[0] * Pr
        mom_re = mass_im * Ure * cos_thetare 
        # mass is lost through deposition, mom is lost through incident motion
        mass_dep = alpha_dep * (1-Pr) * y[0]
        mom_inc =  mass_im * Uim_solution*np.cos(theta_im)/Tim + mass_dep * u_dep*np.cos(theta_dep)/Tdep 
        
        # momentum of air gets replenished slowly through shear at the top boundary
        u_am = u_star/0.4 * np.log(hsal/(D/30)) #law of the wall (COMSALT)
        mom_air_gain = 0.5* CD_air * 1.225 * (u_am - u_air) *abs(u_am - u_air) # 1.225 * u_star **2
        # momentum of air gets lost slowly from bed shear
        mom_air_loss = 0.5* 1.225 * CD_bed * u_air * abs(u_air) 
        
        # assemble source terms
        dydt = [mass_ero/Tim - mass_dep/Tdep,
                mom_drag + mom_ero/Tim + mom_re/Tim - mom_inc,
                mom_air_gain - mom_air_loss - mom_drag]
        return dydt
    return odefun

# Read the data at Shields=0.06
data = np.loadtxt('Shields006dry.txt', delimiter=',')
Q_dpm = data[20:, 0]
C_dpm = data[20:, 1]
U_dpm = data[20:, 2]
t_dpm = np.linspace(0,5,482)
data_ua = np.loadtxt('TotalDragForce/Uair_ave-tS006Dryh01.txt', delimiter='\t')
# Ua_dpm = np.insert(data_ua[20:,1], 0, Uair0[-1])
Ua_dpm = data_ua[19:,1]    
# Initial conditions
c0 =  C_dpm[0] # 0.147 # 0.0139
Usal0 = U_dpm[0] # 0.55 # 2.9279
# Uair0 = [4.6827, 6.8129, 8.4490, 9.8194, 11.0206, 12.1046]  #h=0.1, u_air_end = 5.4162 m/s for Shields=0.06
Uair0 = [5.1, 7.4, 9.2, 10.7, 12.0, Ua_dpm[0]] #h=0.2 # Shields=0.01 uair0=5.1 #13.1 #0.01s 3.93 #12.9983
# Uair0 = [1.45, 4.47, 5.70, 6.62, 7.40, 8.11] # Uair derived from total drag force
# Time span
t_span = [0, 5]
t_eval = np.linspace(t_span[0], t_span[1], 482)

y_eval = []

for i in range(len(Uair0)):
    y0 = [c0, c0*Usal0, Uair0[i]*mass_air] #mass sal, momentum sal, momentum air
    odefun = make_odefun(u_star[i])
    sol = solve_ivp(odefun, t_span, y0, method='Radau', dense_output=True)
    y_eval.append(sol.sol(t_eval))

#calibrate splash functions with DPM data
plt.close('all')
# Define the labels and keys once for reuse
ylabels = ['Q [kg/m/s]', 'C [kg/m^2]', 'U_sal [m/s]', 'U_air [m/s]']
keys_dpm = [Q_dpm, C_dpm, U_dpm, Ua_dpm]
keys_continuum = []
keys_continuum.append([y_eval[-1][1], y_eval[-1][0], y_eval[-1][1] / (y_eval[-1][0] + 1e-9), y_eval[-1][2]/mass_air]) #Q, C, U, Ua
fig, axs = plt.subplots(2, 2, figsize=(10, 10))
axs = axs.flatten()
# Plot in a loop
for i, ax in enumerate(axs):
    ax.plot(t_dpm, keys_dpm[i], color='k', label='DPM simulation')
    ax.plot(t_eval, keys_continuum[0][i], label='Continuum')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel(ylabels[i])
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.grid(True)
    if i == 0:
        ax.legend()
fig.suptitle(r'$\Theta$=0.06, CD_air=9e-3, CD_drag_reduced=0.3, Usal=Pr*Uim,x+(1-Pr)*Udep,x')
plt.tight_layout()
plt.show()

plt.figure()
for i in range(len(u_star)):
    plt.plot(t_eval, y_eval[i][1], label=f"$\Theta$=0.0{i+1}")
plt.legend()
plt.xlabel('Time [s]')
plt.xlim(left=0)
plt.ylabel(r'$Q$ [kg/m/s]')
plt.ylim(bottom=0)
plt.xlim(left=0)
plt.title('CD_air = 9e-3, CD_drag_reduced=0.3, Usal=Pr*Uim,x+(1-Pr)*Udep,x')
plt.tight_layout()
plt.show()

# calculate steady Q
Q_steady_dpm = [0.0047, 0.0137, 0.0188, 0.0257, 0.0398, 0.0392] #dpm Shields=0.01 Qs=0.0047
Q_steady = np.zeros(len(u_star))
for i in range(len(u_star)):
    Qs = np.mean(y_eval[i][1][300:])
    Q_steady[i] = Qs

plt.figure()
plt.plot(Shields, Q_steady_dpm, 'ok', label='DPM simulation')
plt.plot(Shields, Q_steady, 'x', label='Continuum')
plt.xlabel(r'$\Theta$')
plt.ylabel(r'$Q_\mathrm{steady}$ [kg/m/s]')
plt.xlim(left=0)
plt.ylim(bottom=0)
plt.legend()
plt.title(r'$\Theta$=0.06, CD_air=9e-3, CD_drag_reduced=0.3, Usal=Pr*Uim,x+(1-Pr)*Udep,x')
plt.tight_layout()
plt.show()