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
######        mind the power laws might go infinity for low Uim/Uinc 

# constants
D = 0.00025
constant = np.sqrt(9.81 * D)
Shields = np.linspace(0.01, 0.06, 6)
u_star = np.sqrt(Shields * (2650-1.225)*9.81*D/1.225)

# mass of air per unit area
h = 0.2 - 0.00025*10
mass_air = 1.225 * h
rho_a = 1.225
rho_p = 2650
nu_a = 1.46e-5
CD_air = 8e-3
CD_bed = 3e-4
CD_drag_reduce = 1 #0.431
alpha_ero = 1#0.786
alpha_dep = 1#1.235
alpha_im = 1#1.684
cos_thetaej = np.cos(42/180*np.pi)
eps = 1e-9
U_on = 1e-3

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
def solveUinc(Uim, UD, Tim, Tlast):
    def equation(Uinc):
        # Calculate Pr based on Uinc
        Pr = 0.94 * np.exp(-7.11 * np.exp(-0.11 * abs(Uinc) / constant))
        # Compute weights
        denom = Pr * Tim + (1 - Pr) * Tlast
        weight_im = (Pr * Tim) / denom
        weight_D = 1 - weight_im
        # Expected Uinc based on Pr
        Uinc_model = weight_im * abs(Uim) + weight_D * abs(UD)
        return abs(Uinc) - Uinc_model
    # Initial guess
    Uinc_guess = abs(Uim)
    # Solve using fsolve
    Uinc_solution, = fsolve(equation, Uinc_guess)
    return Uinc_solution

def solveUim(u_sal):
    # Uim = 0.94*(abs(u_sal)/constant) + 10.50
    Umag = abs(u_sal/constant)**0.55 + 26.15
    return np.sign(u_sal) * Umag * constant

def solveUD(u_sal):
    # UD = 0.66*(abs(u_sal)/constant) + 1.34
    Umag = abs(u_sal/constant)**0.41 + 2.76
    return np.sign(u_sal) * Umag * constant

def Calfd(u_air, u_sal):
    hs = 6.514e-4
    z0 = 2.386e-6
    u_eff = u_air * (np.log(hs/z0)-1)/(np.log(h/z0)-1)
    k = 3.216e-9
    n = 2.194
    Δu = u_eff - u_sal
    fd = k * np.abs(Δu)**(n-1) * Δu
    # u_eff = 0.3*u_air
    # Re = abs(u_eff - u_sal) * D/(1.45e-6) + 1e-12
    # C_D = (np.sqrt(0.5) + np.sqrt(24 / Re))**2
    # fd = 0.5* np.pi/8 * 1.225 * D**2 * C_D * (u_eff - u_sal)* abs(u_eff - u_sal)
    return fd

def taub_minusphib(u_air):
    l_eff = 0.0319
    phi_b = 0.64
    tau_b = rho_a * (nu_a + l_eff**2*abs(u_air/h))*u_air/h
    return tau_b*(1-phi_b) 

# def CalCDbed(u_air):
#     Re = u_air * D / (1.46e-5)
#     CD_bed = 1.05e-6 * Re**2
#     return CD_bed

def make_odefun(u_star, mom_drag_list):
    dir_state = 1.0
    U_state = 0.0  # persistent
    def odefun(t, y):
        # air velocity
        u_air = y[2]
        # saltation concentration
        c = y[0]
        # saltating layer velocity
        c_eff = max(c, 1e-3)      # for computing u_sal only
        u_sal = y[1] / c_eff
        # u_sal = y[1] / (y[0] + eps)
        # solve Uim and theta_im from u_sal
        # U_guess = u_sal 
        # Solve Uim from Usal
        # Uim = fsolve(lambda Uim: solveUim(Uim, u_sal), U_guess)[0]
        # Udep = 0.18*Uim
        # Uinc = solveUinc(u_sal)
        Uim = solveUim(u_sal)
        UD = solveUD(u_sal)
        print('Usal=',u_sal)
        print('Uim=', Uim)
        print('UD=',UD)
        
        speed_im, speed_dep = abs(Uim), abs(UD)
        
        # impact angle
        arg_im = 50.40 / (speed_im / constant + 159.33)
        arg_im_clipped = np.clip(arg_im, -1.0, 1.0)
        if np.any((arg_im < -1) | (arg_im > 1)):
            print("Warning: arg_im out of domain, clipping applied")
        theta_im = np.arcsin(arg_im_clipped)
        # print('theta_im=',np.rad2deg(theta_im))
        # deposition angle
        arg_dep = 163.68/ (speed_dep / constant + 154.65)
        arg_dep_clipped = np.clip(arg_dep, -1.0, 1.0)
        if np.any((arg_dep < -1) | (arg_dep > 1)):
            print("Warning: arg_im out of domain, clipping applied")
        theta_dep = 0.28 * np.arcsin(arg_dep_clipped)
        # print('theta_dep=',np.rad2deg(theta_dep))
        
        # time scale for collision and deposition
        # Tim = 1e-9 + 2*abs(Uim)*np.sin(theta_im)/9.81
        # Tlast = 1e-9 + 2*abs(UD)*np.sin(theta_dep)/9.81
        Tim = 0.04*speed_im**0.84 + eps
        Tlast = 0.07*speed_dep**0.66 + eps
        Tlast = max(Tlast, 1.2*Tim)
        
        # splash functions
        # Pr = 0.94 * np.exp(-7.12 * np.exp(-0.10 * speed_im / constant)) 
        # Pr = 0.16 * abs(u_sal)**0.52 # temporary new formula
        # Pr = 0.94 * np.exp(-7.12 * np.exp(-0.10 * abs(u_sal) / constant)) # testing
        Pr = 0.2 + (0.9-0.2) * (1-np.exp(-(u_sal/1))**1)        
        NE_im = 0.04 * speed_im / constant 
        NE_D = 0.04 * speed_dep / constant
        UE_im = 4.53 * constant * np.sign(Uim)
        UE_D = 4.53 * constant * np.sign(UD)
        
        # Uim = Uinc#CalUim(Uinc, Pr)
        COR = 3.05 * (speed_im / constant + 1e-9)**(-0.47) 
        ###### angles should be between 0-90!!!
        arg_re = -0.0003 * speed_im / constant + 0.52 
        arg_re_clipped = np.clip(arg_re, -1.0, 1.0)
        if np.any((arg_re < -1) | (arg_re > 1)):
            print("Warning: arg_re out of domain, clipping applied")
        theta_re = np.arcsin(arg_re_clipped)
        Ure = Uim * COR
        cos_thetare = np.cos(theta_re)
        
        #concentrations
        cim = c * Pr*Tim/(Pr*Tim + (1-Pr)*Tlast + 1e-12)
        cdep = c - cim
        
        # momentum is transferred quickly from air to saltating layer due to drag
        # (the quicker, the bigger the overshoot in momentum)
        # Usal = u_sal*y[0] / (cim + cdep*Udep/Uim)
        # Urep = Usal * Udep/Uim
        # print('Usal',Usal,'Urep',Urep)
        # fd_sal = Calfd(u_air, Usal) # the drag force on one saltating particle
        mp = 2650 * np.pi/6 * D**3 #particle mass
        # fd_rep = Calfd(u_air, Urep)
        fd_sal = Calfd(u_air, u_sal)
        mom_drag = CD_drag_reduce * c*fd_sal/mp #cdep*fd_rep/mp  #
        # mom_drag_wind = cdep*fd_sal/mp

        nonlocal dir_state
        if   u_sal > +U_on: dir_state = +1.0
        elif u_sal < -U_on: dir_state = -1.0
        dir = dir_state
        
        # mass is gained through sand erosion, mom is gained through erosion and rebound
        mass_ero =  NE_im * cim/Tim + NE_D * cdep/Tlast
        mass_dep = cdep/Tlast #alpha_dep * (1-Pr) * y[0]
        mom_ero = (NE_im * cim/Tim * UE_im * cos_thetaej + NE_D * cdep/Tlast * UE_D * cos_thetaej) * dir
        mom_re = cim * Ure * cos_thetare / Tim * dir
        mom_im =  cim * Uim*np.cos(theta_im) / Tim* dir
        mom_dep = mass_dep * UD*np.cos(theta_dep) * dir
        
        dc_dt   = mass_ero - mass_dep
        d_cU_dt = mom_drag + mom_ero + mom_re - mom_im - mom_dep
        
        # momentum of air gets replenished slowly through shear at the top boundary
        # u_am = u_star/0.4 * np.log(h/(D/30)) #law of the wall (COMSALT)
        # mom_air_gain = 1.225 * u_star **2 # 0.5* CD_air * 1.225 * (u_am - u_air) *abs(u_am - u_air)
        # momentum of air gets lost slowly from bed shear
        # mom_air_loss = taub_minusphib(u_air) 
        # CD_bed = CalCDbed(u_air)
        # mom_air_loss = 0.5* 1.225 * CD_bed * u_air * abs(u_air) 
        # Stresses on air
        tau_top = rho_a * u_star**2
        tau_bed_eff = taub_minusphib(u_air) 
        # Effective air mass per area
        chi = max(1e-6, 1.0 - c/(rho_p*h))           # keep positive
        m_air_eff = rho_a * h * chi
        # Air acceleration from the PDE
        dU_air_dt = (tau_top - tau_bed_eff - mom_drag) / m_air_eff
        
        mom_drag_list.append(mom_drag)
        
        # assemble source terms
        dydt = [dc_dt, d_cU_dt, dU_air_dt]
        return dydt
    return odefun

# Read the data at Shields=0.06
data = np.loadtxt('CGdata/Shields006dry.txt')
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
# mom_drag_all = []
# for i in range(len(Uair0)):
mom_drag_list = []
y0 = [c0, c0*Usal0, Uair0[-1]] #mass sal, momentum sal, momentum air
odefun = make_odefun(u_star[-1], mom_drag_list)
sol = solve_ivp(odefun, t_span, y0, method='Radau', dense_output=True)
y_eval.append(sol.sol(t_eval))
    # mom_drag_all.append(mom_drag_list)

plt.close('all')
#calibrate splash functions with DPM data
# plt.close('all')
# Define the labels and keys once for reuse
ylabels = ['Q [kg/m/s]', 'C [kg/m^2]', 'U_sal [m/s]', 'U_air [m/s]']
keys_dpm = [Q_dpm, C_dpm, U_dpm, Ua_dpm]
keys_continuum = []
keys_continuum.append([y_eval[-1][1], y_eval[-1][0], y_eval[-1][1] / (y_eval[-1][0] + 1e-9), y_eval[-1][2]]) #Q, C, U, Ua
fig, axs = plt.subplots(2, 2, figsize=(8, 8))
axs = axs.flatten()
# Plot in a loop
for i, ax in enumerate(axs):
    ax.plot(t_dpm, keys_dpm[i], color='k', label='DPM simulation')
    ax.plot(t_eval, keys_continuum[0][i], label='Continuum')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel(ylabels[i])
    ax.set_xlim(left=0)
    # ax.set_ylim(bottom=0)
    ax.grid(True)
    if i == 0:
        ax.legend()
fig.suptitle(r'$\Theta$=0.06')
plt.tight_layout()
plt.show()

file_fd = 'TotalDragForce/FD_S006dry.txt'
data_FD = np.loadtxt(file_fd)
FD_dpm = data_FD / (100 * D * 2 * D)
t_drag = np.linspace(0, 5, len(mom_drag_list))
plt.figure()
plt.plot(t_drag, mom_drag_list)
plt.plot(t_dpm, FD_dpm)
plt.xlabel('t [s]')
plt.ylabel(r'$M_{drag}$ [N/m$^2$]')


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