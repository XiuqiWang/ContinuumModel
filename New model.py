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


###### To do: angles should be between 0-90!!!
######        mind the power laws to go infinity for low Uim/Uinc 


# constants
D = 0.00025
constant = np.sqrt(9.81 * D)
Shields = np.linspace(0.01, 0.06, 6)
u_star = np.sqrt(Shields * (2650-1.225)*9.81*D/1.225)

# mass of air per unit area
hsal = 0.2 - 0.00025*10
mass_air = 1.225 * hsal
CD_air = 8e-3
CD_bed = 3e-4
CD_drag_reduce = 1 #0.431
alpha_ero = 1#0.786
alpha_dep = 1#1.235
alpha_im = 1#1.684
cos_thetaej = np.cos(47/180*np.pi)
# ratio_Udep = 0.18

# numerically solves Uim from Usal
# def solveUim(Uim, u_sal):
#     denom = Uim / constant + 105.73
#     arg = 39.21 / denom
#     if np.abs(arg) > 1:
#         return np.inf
#     theta = np.arcsin(arg)
#     return Uim * np.cos(theta) - u_sal
# def solveUim(Uim, u_sal):
#     Pr = 0.94 * np.exp(-7.11 * np.exp(-0.11 * abs(Uim) / constant)) 
#     Udep = 0.18*Uim
#     denom = Uim / constant + 105.73
#     arg = 39.21 / denom
#     if np.abs(arg) > 1:
#         return np.inf
#     theta = np.arcsin(arg)
#     theta_dep = 25/180*np.pi
#     return Uim*Pr*np.cos(theta) + (1-Pr)*Udep*np.cos(theta_dep) - u_sal
# def solveUinc(u_sal):
#     Uinc = (abs(u_sal)/constant/0.38)**(1/1.18)*constant
#     return Uinc

def solveUim(u_sal): #R2=0.74
    u_sal = np.array(u_sal)
    Uim = 0.96*(abs(u_sal)/constant) + 15.07
    return Uim*constant

def solveUD(u_sal): #R2=0.83
    UD = 0.96*(abs(u_sal)/constant) + 3.41
    return UD*constant

def solveUinc(Uim, Udep, Tim, Tdep):
    def equation(Uinc):
        # Calculate Pr based on Uinc
        Pr = 0.94 * np.exp(-7.11 * np.exp(-0.11 * Uinc / constant))
        # Compute weights
        denom = Pr * Tim + (1 - Pr) * Tdep
        weight_im = (Pr * Tim) / denom
        weight_D = 1 - weight_im
        # Expected Uinc based on Pr
        Uinc_model = weight_im * Uim + weight_D * Udep
        return Uinc - Uinc_model

    # Initial guess
    Uinc_guess = Uim
    # Solve using fsolve
    Uinc_solution, = fsolve(equation, Uinc_guess)
    return Uinc_solution

# def CalUim(Uinc, Pr):
#     Uim = Uinc/(Pr+(1-Pr)*ratio_Udep)
#     return Uim

def Calfd(u_air, u_sal):
    u_eff = 0.3*u_air
    Re = abs(u_eff - u_sal) * D/(1.45e-6) + 1e-12
    C_D = (np.sqrt(0.5) + np.sqrt(24 / Re))**2
    fd = 0.5* np.pi/8 * 1.225 * D**2 * C_D * (u_eff - u_sal)* abs(u_eff - u_sal)
    return fd

def CalCDbed(u_air):
    Re = u_air * D / (1.46e-5)
    CD_bed = 1.05e-6 * Re**2
    return CD_bed

def make_odefun(u_star, mom_drag_list):
    def odefun(t, y):
        # air velocity
        u_air = y[2] / mass_air
        # saltating layer velocity
        u_sal = y[1] / (y[0] + 1e-9)
        # solve Uim and theta_im from u_sal
        # U_guess = u_sal 
        # Solve Uim from Usal
        # Uim = fsolve(lambda Uim: solveUim(Uim, u_sal), U_guess)[0]
        # Udep = 0.18*Uim
        # Uinc = solveUinc(u_sal)
        Uim = solveUim(u_sal)
        Udep = solveUD(u_sal)
        print('Usal=',u_sal)
        print('Uim=', Uim)
        print('Udep=',Udep)
        
        # impact angle
        arg_im = 39.21 / (abs(Uim) / constant + 105.73)
        arg_im_clipped = np.clip(arg_im, -1.0, 1.0)
        if np.any((arg_im < -1) | (arg_im > 1)):
            print("Warning: arg_im out of domain, clipping applied")
        theta_im = np.arcsin(arg_im_clipped)
        print('theta_im=',np.rad2deg(theta_im))
        # deposition angle
        arg_dep = 96.36/ (abs(Udep) / constant + 83.42)
        arg_dep_clipped = np.clip(arg_dep, -1.0, 1.0)
        if np.any((arg_dep < -1) | (arg_dep > 1)):
            print("Warning: arg_im out of domain, clipping applied")
        theta_dep = 0.33 * np.arcsin(arg_dep_clipped)
        print('theta_dep=',np.rad2deg(theta_dep))
        
        # time scale for collision and deposition
        # Udep = Uim*0.5#*ratio_Udep
        Tim = 1e-9 + 2*abs(Uim)*np.sin(theta_im)/9.81
        Tdep = 1e-9 + 2*abs(Udep)*np.sin(theta_dep)/9.81
        
        Uinc = solveUinc(Uim,Udep,Tim,Tdep)
        # print('Uinc=', Uinc)
        
        # splash functions
        Pr = 0.94 * np.exp(-7.11 * np.exp(-0.11 * abs(Uinc) / constant)) 
        NE = 0.04 * abs(Uinc) / constant 
        if Uim >= 0:
            UE = 5.02 * constant
        else:
            UE = -5.02 * constant 
        
        # Uim = Uinc#CalUim(Uinc, Pr)
        COR = 3.41 * (abs(Uim) / constant + 1e-9)**(-0.5) 
        ###### angles should be between 0-90!!!
        arg_re = -0.0006 * abs(Uim) / constant + 0.65 
        arg_re_clipped = np.clip(arg_re, -1.0, 1.0)
        if np.any((arg_re < -1) | (arg_re > 1)):
            print("Warning: arg_re out of domain, clipping applied")
        theta_re = np.arcsin(arg_re_clipped)
        Ure = Uim * COR
        cos_thetare = np.cos(theta_re)
        
        #concentrations
        cim = y[0]  * Pr*Tim/(Pr*Tim + (1-Pr)*Tdep)
        cdep = y[0] - cim
        
        # momentum is transferred quickly from air to saltating layer due to drag
        # (the quicker, the bigger the overshoot in momentum)
        # Usal = u_sal*y[0] / (cim + cdep*Udep/Uim)
        # Urep = Usal * Udep/Uim
        # print('Usal',Usal,'Urep',Urep)
        # fd_sal = Calfd(u_air, Usal) # the drag force on one saltating particle
        mp = 2650 * np.pi/6 * D**3 #particle mass
        # fd_rep = Calfd(u_air, Urep)
        fd_sal = Calfd(u_air, u_sal)
        mom_drag = CD_drag_reduce * y[0]*fd_sal/mp #cdep*fd_rep/mp  #
        mom_drag_list.append(mom_drag)
        # mom_drag_wind = cdep*fd_sal/mp
        # mass is gained through sand erosion, mom is gained through erosion and rebound
        mass_ero =  alpha_ero * NE * cim #y[0] 
        mom_ero = mass_ero * UE * cos_thetaej 
        mass_im = cim # alpha_im * y[0] * Pr
        mom_re = mass_im * Ure * cos_thetare 
        # mass is lost through deposition, mom is lost through incident motion
        mass_dep = cdep #alpha_dep * (1-Pr) * y[0]
        # mom_im =  mass_im * Uim*np.cos(theta_im)
        # mom_dep = mass_dep * Udep*np.cos(theta_dep) 
        mom_inc = y[0] * Uinc * np.cos(theta_im)
        
        # momentum of air gets replenished slowly through shear at the top boundary
        # u_am = u_star/0.4 * np.log(hsal/(D/30)) #law of the wall (COMSALT)
        mom_air_gain = 1.225 * u_star **2 # 0.5* CD_air * 1.225 * (u_am - u_air) *abs(u_am - u_air)
        # momentum of air gets lost slowly from bed shear
        CD_bed = CalCDbed(u_air)
        mom_air_loss = 0.5* 1.225 * CD_bed * u_air * abs(u_air) 
        
        # assemble source terms
        dydt = [mass_ero/Tim - mass_dep/Tdep,
                mom_drag + mom_ero/Tim + mom_re/Tim - mom_inc/Tim, #mom_im/Tim - mom_dep/Tdep,
                mom_air_gain - mom_air_loss - mom_drag]
        return dydt
    return odefun

# Read the data at Shields=0.06
data = np.loadtxt('Shields006dry.txt')
Q_dpm = data[:, 0]
C_dpm = data[:, 1]
U_dpm = data[:, 2]
t_dpm = np.linspace(0,5,501)
data_ua = np.loadtxt('TotalDragForce/Uair_ave-tS006Dryh02.txt', delimiter='\t')
# Ua_dpm = np.insert(data_ua[20:,1], 0, Uair0[-1])
Ua_dpm = data_ua[0:,1]    
# Initial conditions
c0 =  C_dpm[0] # 0.147 # 0.0139
Usal0 = U_dpm[0] # 0.55 # 2.9279
# Uair0 = [4.6827, 6.8129, 8.4490, 9.8194, 11.0206, 12.1046]  #h=0.1, u_air_end = 5.4162 m/s for Shields=0.06 
Uair0 = [5.1, 7.4, 9.2, 10.7, 12.0, 13.0] #h=0.2 # Shields=0.01 uair0=5.1 #13.1 #0.01s 3.93 #12.9983
# Time span
t_span = [0, 5]
t_eval = np.linspace(t_span[0], t_span[1], 501)

y_eval = []
mom_drag_all = []
for i in range(len(Uair0)):
    mom_drag_list = []
    y0 = [c0, c0*Usal0, Uair0[i]*mass_air] #mass sal, momentum sal, momentum air
    odefun = make_odefun(u_star[i], mom_drag_list)
    sol = solve_ivp(odefun, t_span, y0, method='Radau', dense_output=True)
    y_eval.append(sol.sol(t_eval))
    mom_drag_all.append(mom_drag_list)

plt.close('all')
#calibrate splash functions with DPM data
# plt.close('all')
# Define the labels and keys once for reuse
ylabels = ['Q [kg/m/s]', 'C [kg/m^2]', 'U_sal [m/s]', 'U_air [m/s]']
keys_dpm = [Q_dpm, C_dpm, U_dpm, Ua_dpm]
keys_continuum = []
keys_continuum.append([y_eval[-1][1], y_eval[-1][0], y_eval[-1][1] / (y_eval[-1][0] + 1e-9), y_eval[-1][2]/mass_air]) #Q, C, U, Ua
fig, axs = plt.subplots(2, 2, figsize=(8, 8))
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

# plt.figure()
# for i in range(len(u_star)-1):
#     plt.plot(t_eval, y_eval[i][1], label=f"$\Theta$=0.0{i+1}")
# plt.legend()
# plt.xlabel('Time [s]')
# plt.xlim(left=0)
# plt.ylabel(r'$Q$ [kg/m/s]')
# plt.ylim(bottom=0)
# plt.xlim(left=0)
# plt.title('CD_air = 9e-3, CD_drag_reduced=0.3, Usal=Pr*Uim,x+(1-Pr)*Udep,x')
# plt.tight_layout()
# plt.show()

# # calculate steady Q
# Q_steady_dpm = [0.0047, 0.0137, 0.0188, 0.0257, 0.0398, 0.0392] #dpm Shields=0.01 Qs=0.0047
# Q_steady = np.zeros(len(u_star))
# for i in range(len(u_star)):
#     Qs = np.mean(y_eval[i][1][300:])
#     Q_steady[i] = Qs

# plt.figure()
# plt.plot(Shields, Q_steady_dpm, 'ok', label='DPM simulation')
# plt.plot(Shields, Q_steady, 'x', label='Continuum')
# plt.xlabel(r'$\Theta$')
# plt.ylabel(r'$Q_\mathrm{steady}$ [kg/m/s]')
# plt.xlim(left=0)
# plt.ylim(bottom=0)
# plt.legend()
# plt.title(r'$\Theta$=0.06, CD_air=9e-3, CD_drag_reduced=0.3, Usal=Pr*Uim,x+(1-Pr)*Udep,x')
# plt.tight_layout()
# plt.show()