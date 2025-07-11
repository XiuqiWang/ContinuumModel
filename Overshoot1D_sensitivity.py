# -*- coding: utf-8 -*-
"""
Created on Wed Jun 11 12:01:26 2025

@author: WangX3
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
from scipy.stats import lognorm
from scipy.stats import truncnorm
from scipy.stats import expon
from scipy.integrate import quad
from scipy.interpolate import interp1d

# constants
D = 0.00025
constant = np.sqrt(9.81 * D)
Shields = 0.06 #np.linspace(0.01, 0.06, 6)
u_star = np.sqrt(Shields * (2650-1.225)*9.81*D/1.225)

# mass of air per unit area
# hsal = 0.2 - 0.00025*10
# mass_air = 1.225 * hsal

# parameters for calibration
# CD_air = 9e-3
CD_bed = 3e-4

# numerically solves Uim from Usal
def solveUim(Uim, u_sal):
    denom = Uim / constant + 105.73
    arg = 39.21 / denom
    if np.abs(arg) > 1:
        return np.inf
    theta = np.arcsin(arg)
    return Uim * np.cos(theta) - u_sal

# #test
# u_sal = 3
# U_guess = 3
# # Solve Uim from Usal
# Uim_solution = fsolve(lambda Uim: solveUim(Uim, u_sal), U_guess)[0]
# Uim_computed = u_sal/np.cos(np.arcsin(39.21/(Uim_solution/constant+105.73)))
# print("Uim = , Uim_computed = ", Uim_solution, Uim_computed)

def Calfd(u_air, u_sal):
    C_D = (np.sqrt(0.5) + np.sqrt(24 / (abs(u_air - u_sal) * D/(1.45e-6))))**2
    fd = 0.5* np.pi/8 * 1.225 * D**2 * C_D * (u_air - u_sal)* abs(u_air - u_sal)
    return fd


def make_odefun(CD_air, alpha_drag, alpha_ero, alpha_dep, alpha_im):
    def odefun(t, y):
        #saltation velocity
        u_sal = y[1] / (y[0] + 1e-9)
        #saltation layer height
        hsal = u_sal**2/9.81 - 0.0025
        #air mass
        mass_air = 1.225 * hsal
        #air velocity
        u_air = y[2] / mass_air
        # print('Ua', y[2]/mass_air)
        # print('C:',y[0])
        # print('CU', y[1])
        # solve Uim and theta_im from u_sal
        U_guess = u_sal 
        # Solve Uim from Usal
        Uim_solution = fsolve(lambda Uim: solveUim(Uim, u_sal), U_guess)[0]
        # print("Usal, Uim =", u_sal, Uim_solution)
        # print('Uair = ', u_air)
        # impact angle
        arg_im = 39.21 / (abs(Uim_solution) / constant + 105.73)
        arg_im_clipped = np.clip(arg_im, -1.0, 1.0)
        if np.any((arg_im < -1) | (arg_im > 1)):
            print("Warning: arg_im out of domain, clipping applied")
        theta_im = np.arcsin(arg_im_clipped)
        # theta_im = 15/180*np.pi
        # Uim_solution = u_sal / np.cos(theta_im)
        # print("Usal, Uim =", u_sal, Uim_solution)
        
        # ejection angle
        cos_thetaej = np.cos(47/180*np.pi)
        #deposition angle
        theta_dep = 25/180*np.pi
        
        # time scale for collision and deposition
        u_dep = Uim_solution*0.18
        Tim = 1e-9 + 2*abs(Uim_solution)*np.sin(theta_im)/9.81
        Tdep = 1e-9 + 2*abs(u_dep)*np.sin(theta_dep)/9.81
        
        # splash functions
        COR = 5.17 * (abs(Uim_solution) / constant + 1e-9)**(-0.6) #5.17 # 0.7469*np.exp(0.1374*1.5)*(Uim_solution/constant)**(-0.0741*np.exp(0.2140*1.5)) # np.sqrt(0.45)
        Pr = 0.94 * np.exp(-7.11 * np.exp(-0.11 * abs(Uim_solution) / constant)) # MeanPr(Uim_solution, 0.64) # 0.96*(1-np.exp(- Uim_solution)) 
        arg_re = -0.0006 * abs(Uim_solution) / constant + 0.65 # 45/180*np.pi
        arg_re_clipped = np.clip(arg_re, -1.0, 1.0)
        if np.any((arg_re < -1) | (arg_re > 1)):
            print("Warning: arg_re out of domain, clipping applied")
        theta_re = np.arcsin(arg_re_clipped)
        # theta_re_rad = theta_re/180*np.pi
        NE = 0.04 * abs(Uim_solution) / constant # 0.04
        if Uim_solution >= 0:
            UE = 5.02 * constant#0.15/0.02*(1-np.exp(-Uim_solution/constant/40))*constant# # 5.02  #   # 1.18*(Uim_solution/constant)**0.25*constant
        else:
            UE = -5.02 * constant #0.15/0.02*(1-np.exp(-Uim_solution/constant/40))*constant#
        Ure = Uim_solution * COR
        cos_thetare = np.cos(theta_re)
        
        # momentum is transferred quickly from air to saltating layer due to drag
        # (the quicker, the bigger the overshoot in momentum)
        # u_sal_mean = MeanUlognorm(u_sal, 0.64, plot_pdf=False)
        fd_sal = Calfd(u_air, u_sal) # the drag force on high-energy saltating particles
        # fd_dep = Calfd(u_air, u_dep*np.cos(theta_dep)) # the drag force on low-energy depositing particles
        mp = 2650 * np.pi/6 * D**3 #particle mass
        mom_drag = alpha_drag * y[0]*fd_sal/mp
        # mass is gained through sand erosion, mom is gained through erosion and rebound
        mass_ero =  alpha_ero * NE * y[0] 
        mom_ero = mass_ero * UE * cos_thetaej # MeanUlognorm(UE, 0.7)
        mass_im = alpha_im * y[0] * Pr
        mom_re = mass_im * Ure * cos_thetare # MeanUlognorm(Ure, 0.66)
        # mass is lost through deposition, mom is lost through incident motion
        mass_dep = alpha_dep * (1-Pr) * y[0]
        mom_inc = mass_im * Uim_solution*np.cos(theta_im)/Tim + mass_dep * u_dep*np.cos(theta_dep)/Tdep #  # MeanUlognorm(u_sal, 0.66)
        
        # momentum of air gets replenished slowly through shear at the top boundary
        u_am = u_star/0.4 * np.log((hsal)/(0.00025/30)) #law of the wall (COMSALT)
        mom_air_gain =  0.5* CD_air * 1.225 * (u_am - u_air) *abs(u_am - u_air) # 1.225 * u_star **2
        # momentum of air gets lost slowly from bed shear
        mom_air_loss = 0.5* 1.225 * CD_bed * u_air * abs(u_air) 
        
        # assemble source terms
        dydt = [mass_ero/Tim - mass_dep/Tdep,
                mom_drag + mom_ero/Tim + mom_re/Tim - mom_inc,
                mom_air_gain - mom_air_loss - mom_drag]
        return dydt
    return odefun
    
# Initial conditions
c0 = 0.147 #0.0139
Usal0 = 0.55 #2.9279
# Uair0 = [4.6827, 6.8129, 8.4490, 9.8194, 11.0206, 12.1046]  #h=0.1, u_air_end = 5.4162 m/s for Shields=0.06
# Uair0 = [5.1, 7.4, 9.2, 10.7, 12.0, 13.1] #h=0.2 # Shields=0.01 uair0=5.1
Uair0 = 12.9983 #11.55 #9.9
mass_air_ini = (0.012-0.0025)*1.225

# Time span
t_span = [0, 5]
t_eval = np.linspace(t_span[0], t_span[1], 501)

data = np.loadtxt('Shields006dry.txt', delimiter=',')
Q_dpm = data[1:, 0]
C_dpm = data[1:, 1]
U_dpm = data[1:, 2]
t_dpm = np.linspace(0,5,501)
data_ua_dpm = np.loadtxt('TotalDragForce/Uair_ave-tS006Dryh02.txt', delimiter='\t')
Ua_dpm = data_ua_dpm[:,1]
hsal_dpm = np.loadtxt("Hsal-t-S006Dry.txt") - 0.0025

# --- Parameter-specific ranges ---
param_config = {
    'CD_air': np.linspace(3e-3, 1e-2, 10),
    'alpha_drag': np.linspace(0.1, 1, 50),
    'alpha_ero':  np.linspace(0.5, 1.5, 50),
    'alpha_dep':  np.linspace(0.5, 2, 50),
    'alpha_im':   np.linspace(0.5, 2.5, 50)
}

# --- To store all results ---
results = {}

# --- Loop through each parameter ---
for param_name, param_range in param_config.items():
    param_results = {
        'values': [],
        'errors': [],
        'C_all': [],
        'Usal_all': [],
        'Q_all': [],
        'Ua_all': []
    }

    for val in param_range:
        # Default values
        CD_air = 8e-3
        alpha_drag = 1.0
        alpha_ero = 1.0
        alpha_dep = 1.0
        alpha_im = 1.0

        if param_name == 'CD_air':
            CD_air = val
        if param_name == 'alpha_drag':
            alpha_drag = val
        elif param_name == 'alpha_ero':
            alpha_ero = val
        elif param_name == 'alpha_dep':
            alpha_dep = val
        elif param_name == 'alpha_im':
            alpha_im = val

        # --- Solve the ODE system ---
        y0 = [c0, c0 * Usal0, Uair0 * mass_air_ini]
        sol = solve_ivp(
            make_odefun(CD_air, alpha_drag, alpha_ero, alpha_dep, alpha_im),
            t_span, y0, method='Radau', dense_output=True
        )
        y_eval = sol.sol(t_eval)

        C = y_eval[0, :]
        Usal = y_eval[1, :] / (y_eval[0, :] + 1e-9)
        hsal = Usal * 0.1 - 0.0025
        mass_air_t = 1.225*(hsal)
        Ua = y_eval[2, :] / mass_air_t
        Q = y_eval[1, :]

        # --- Compute average relative error ---
        error = abs(C - C_dpm) / C_dpm + abs(Usal - U_dpm) / U_dpm + abs(Ua - Ua_dpm) / Ua_dpm
        error_mean = np.mean(error)

        # --- Store all outputs ---
        param_results['values'].append(val)
        param_results['errors'].append(error_mean)
        param_results['C_all'].append(C)
        param_results['Usal_all'].append(Usal)
        param_results['Q_all'].append(Q)
        param_results['Ua_all'].append(Ua)

    results[param_name] = param_results

param_names = ['CD_{air}','alpha_{drag}','alpha_{ero}', 'alpha_{dep}', 'alpha_{im}']
plt.close('all')
plt.figure(figsize=(12, 8))
for i, (param_name, data) in enumerate(results.items()):
    plt.subplot(2, 3, i + 1)
    plt.plot(data['values'], data['errors'], 'o-')
    plt.xlabel(f'{param_name}', fontsize=14)
    plt.ylabel(r"<Error>")
    plt.tight_layout()
plt.show()


best_params = {}

for param_name, data in results.items():
    min_idx = np.argmin(data["errors"])  # Index of the minimum error
    best_value = data["values"][min_idx]
    min_error = data["errors"][min_idx]
    best_params[param_name] = {"value": best_value, "error": min_error}

# Plotting C, Usal, Q, Ua over time for best parameters
for param_name, result in best_params.items():
    min_idx = np.argmin(results[param_name]["errors"])
    t = t_eval  # already defined earlier

    C_best = results[param_name]["C_all"][min_idx]
    Usal_best = results[param_name]["Usal_all"][min_idx]
    Q_best = results[param_name]["Q_all"][min_idx]
    Ua_best = results[param_name]["Ua_all"][min_idx]

    fig, axs = plt.subplots(2, 2, figsize=(8, 6), sharex=True)
    fig.suptitle(f"Time evolution for best {param_name} = {result['value']:.3f}")

    axs[0,0].plot(t, Q_dpm, color='k', label='DPM')
    axs[0,0].plot(t, Q_best, label="Continuum", color="tab:blue")
    axs[0,0].set_ylabel("Q [kg/m/s]")
    axs[0,0].set_ylim(bottom=0)
    axs[0,0].set_xlim(left=0)
    axs[0,0].legend()
    axs[0, 0].grid(True)
    
    axs[0,1].plot(t, C_dpm, color='k', label='DPM simulation')
    axs[0,1].plot(t, C_best, label="C", color="tab:blue")
    axs[0,1].set_ylabel("C [kg/m^2]")
    axs[0,1].set_ylim(bottom=0)
    axs[0,1].set_xlim(left=0)
    axs[0, 1].grid(True)

    axs[1,0].plot(t, U_dpm, color='k')
    axs[1,0].plot(t, Usal_best, label="Usal", color="tab:blue")
    axs[1,0].set_ylabel("Usal [m/s]")
    axs[1,0].set_ylim(bottom=0)
    axs[1,0].set_xlim(left=0)
    axs[1, 0].grid(True)
    
    axs[1,1].plot(t, Ua_dpm, color='k')
    axs[1,1].plot(t, Ua_best, label="Ua", color="tab:blue")
    axs[1,1].set_xlabel("Time [s]")
    axs[1,1].set_ylabel("Ua [m/s]")
    axs[1,1].set_ylim(bottom=0)
    axs[1,1].set_xlim(left=0)
    axs[1, 1].grid(True)

    plt.tight_layout()  # leave space for suptitle
    plt.show()




# keys_dpm = [Q_dpm, C_dpm, U_dpm, Ua_dpm]
# keys_continuum = [Q_all[18], C_all[18], Usal_all[18], Ua_all[18]]
# ylabels = ['Q [kg/m/s]', 'C [kg/m^2]', 'U_sal [m/s]', 'U_air [m/s]']
# fig, axs = plt.subplots(2, 2, figsize=(10, 10))
# axs = axs.flatten()
# # Plot in a loop
# for i, ax in enumerate(axs):
#     ax.plot(t_dpm, keys_dpm[i], color='k', label='DPM simulation')
#     ax.plot(t_eval, keys_continuum[i], label='Continuum')
#     ax.set_xlabel('Time [s]')
#     ax.set_ylabel(ylabels[i])
#     ax.set_xlim(left=0)
#     ax.set_ylim(bottom=0)
#     ax.grid(True)
#     if i == 0:
#         ax.legend()
# fig.suptitle(fr'$\Theta$=0.06, CD_air={CD_air}, $\alpha_{{drag}}$={alpha_drag_vals[18]:.2f}')
# plt.tight_layout()
# plt.show()

#plot 
# Q_steady_dpm = [0.0047, 0.0137, 0.0188, 0.0257, 0.0398, 0.0392]

# plt.close('all')
# fig, axes = plt.subplots(2, 3, figsize=(15, 8))
# for i, ax in enumerate(axes.flatten()):
#     im = ax.contourf(P1, P2, q_ss_all[i] - Q_steady_dpm[i], levels=20, cmap='viridis')
#     ax.set_title(f"$\Theta$ = {Shields[i]:.2f}")
#     ax.set_xlabel("C_drag_reduce")
#     ax.set_ylabel("C_mass_im")
#     cb = fig.colorbar(im, ax=ax)
#     cb.set_label(r"$\Delta Q_s$ [kg/m/s]")
# plt.tight_layout()
# plt.show()
          
 

# #calibrate splash functions with DPM data
# # Read the data at Shields=0.06
# data = np.loadtxt('Shields006.txt', delimiter=',')
# Q_dpm = data[:, 0]
# C_dpm = data[:, 1]
# U_dpm = data[:, 2]
# t_dpm = np.linspace(0,5,502)
# data_ua = np.loadtxt('Uair_ave-t.txt', delimiter='\t')
# Ua_dpm = np.insert(data_ua[:, 1], 0, Uair0[-1])

# plt.close('all')
# # Define the labels and keys once for reuse
# ylabels = ['Q [kg/m/s]', 'C [kg/m^2]', 'U_sal [m/s]', 'U_air [m/s]']
# keys_dpm = [Q_dpm, C_dpm, U_dpm, Ua_dpm]
# keys_continuum = []
# for i in range(len(UE0_list)):
#     keys_continuum.append([y_eval_all[i][-1][1],y_eval_all[i][-1][0],y_eval_all[i][-1][1] / (y_eval_all[i][-1][0] + 1e-9), y_eval_all[i][-1][2]/mass_air]) #Q, C, U, Ua
# fig, axs = plt.subplots(2, 2, figsize=(10, 10))
# axs = axs.flatten()
# # Plot in a loop
# for i, ax in enumerate(axs):
#     ax.plot(t_dpm, keys_dpm[i], color='k', label='DPM simulation')
#     for j in range(len(UE0_list)):
#         ax.plot(t_eval, keys_continuum[j][i], label=fr'Continuum $UE_{{0}}$=${UE0_list[j]}$')
#     ax.set_xlabel('Time [s]')
#     ax.set_ylabel(ylabels[i])
#     ax.set_xlim(left=0)
#     ax.set_ylim(bottom=0)
#     ax.grid(True)
#     if i == 0:
#         ax.legend()
# fig.suptitle(r'$\Theta$=0.06, CD_air={CD_air}, CD_drag_reduced={CD_drag_reduce}')
# plt.tight_layout()
# plt.show()

# # plt.figure()
# # for i in range(len(u_star)):
# #     plt.plot(t_eval, y_eval_all[-1][i][1], label=f"$\Theta$=0.0{i+2}")
# # plt.legend()
# # plt.xlabel('Time [s]')
# # plt.xlim(left=0)
# # plt.ylabel(r'$Q$ [kg/m/s]')
# # plt.ylim(bottom=0)
# # plt.xlim(left=0)
# # plt.title('CD_air = 5e-3')
# # plt.tight_layout()
# # plt.show()

# #calculate steady Q
# Q_steady_dpm = [0.0047, 0.0137, 0.0188, 0.0257, 0.0398, 0.0392] #dpm Shields=0.01 Qs=0.0047
# Q_steady_all = []

# for idx, a in enumerate(UE0_list):
#     Q_steady = np.zeros(len(u_star))
#     for i in range(len(u_star)):
#         Qs = np.mean(y_eval_all[idx][i][1][300:])
#         Q_steady[i] = Qs
#     Q_steady_all.append(Q_steady)

# plt.figure()
# plt.plot(Shields, Q_steady_dpm, 'ok', label='DPM simulation')
# for idx, Q_steady in enumerate(Q_steady_all):
#     label = f'$UE_{{0}}$ = {UE0_list[idx]}'
#     plt.plot(Shields, Q_steady, 'x', color=colors[idx], label=label)
# plt.xlabel(r'$\Theta$')
# plt.ylabel(r'$Q_\mathrm{steady}$ [kg/m/s]')
# plt.xlim(left=0)
# plt.ylim(bottom=0)
# plt.legend()
# plt.title(fr'$\Theta$=0.06, CD_air={CD_air}, CD_drag_reduced={CD_drag_reduce}')
# plt.tight_layout()
# plt.show()
    

# def MeanPr(Uim_mu, Uim_sigma):
#     # Create lognormal distribution (Uim > 0)
#     # The 'scale' parameter is exp(mu) for lognorm in scipy
#     log_mu = np.log(Uim_mu) - Uim_sigma**2*0.5
#     dist = lognorm(s=Uim_sigma, scale=np.exp(log_mu))
#     # lower, upper = 0, np.inf
#     # a, b = (lower - Uim_mu) / Uim_sigma, (upper - Uim_mu) / Uim_sigma
#     # # Create truncated normal distribution
#     # dist = truncnorm(a, b, loc=Uim_mu, scale=Uim_sigma) 
#     # --- Rebound probability function ---
#     def Pr(Uim):
#         return 0.94 * np.exp(-7.11 * np.exp(-0.11 * Uim / constant))
#     # --- Integrand: p(Uim) * Pr(Uim) ---
#     def integrand_Pr(Uim):
#         return dist.pdf(Uim) * Pr(Uim)   
#     Pr_bar, error_Pr = quad(integrand_Pr, 0, np.inf)
#     # thetare_bar, error_thetare = quad(integrand_thetare, 0, np.inf)
#     return Pr_bar

# Uim = np.linspace(0, 8, 100)
# Pr = 0.94 * np.exp(-7.11 * np.exp(-0.11 * Uim / constant))
# Pr_bar = [MeanPr(uim, 0.8) for uim in Uim]

# plt.figure()
# plt.plot(Uim, Pr, label='individual')
# plt.plot(Uim, Pr_bar, label='mean')
# plt.xlabel('Uim')
# plt.ylabel('Pr')
# plt.legend()

# def MeanUlognorm(U, U_sigma, plot_pdf=True):
#     U_mu = np.log(U) - U_sigma**2*0.5
#     # print('U_mu:', U_mu)
#     dist = lognorm(s=U_sigma, scale=np.exp(U_mu))
#     def integrand(U):
#         return U * dist.pdf(U)
#     U_mean, _ = quad(integrand, 0, np.inf)
    
#     # Plot PDF if requested
#     if plot_pdf:
#         x_vals = np.linspace(0.01, U + 4 * U, 1000)
#         pdf_vals = dist.pdf(x_vals)
#         plt.figure(figsize=(7, 4))
#         plt.plot(x_vals, pdf_vals, label='Lognormal PDF')
#         plt.axvline(U_mean, color='red', linestyle='--', label=f'Mean = {U_mean:.2f}')
#         plt.xlabel('U')
#         plt.ylabel('PDF')
#         plt.title('PDF of U (Lognormal Distribution)')
#         plt.legend()
#         plt.grid(True)
#         plt.tight_layout()
#         plt.show()
    
#     return U_mean    

# def MeanUElognorm(mu, sigma, plot_pdf=True):
#     dist = lognorm(s=sigma, scale=np.exp(mu))
#     def integrand(U):
#         return U * dist.pdf(U)
#     U_mean, _ = quad(integrand, 0, np.inf)
#     return U_mean



# MeanUlognorm(1.3,0.63)

# def MeanUsal(Usal, Uim_sigma, plot_pdf=True):
#     # Define bounds and parameters for truncated normal
#     lower, upper = 0, np.inf
#     a, b = (lower - Usal) / Uim_sigma, (upper - Usal) / Uim_sigma
#     dist = truncnorm(a, b, loc=Usal, scale=Uim_sigma)

#     # Compute mean by integrating U * PDF(U)
#     def integrand(U):
#         return U * dist.pdf(U)
#     U_mean, _ = quad(integrand, 0, np.inf)

#     # Plot PDF if requested
#     if plot_pdf:
#         x_vals = np.linspace(0, Usal + 4 * Uim_sigma, 1000)
#         pdf_vals = dist.pdf(x_vals)
#         plt.figure(figsize=(7, 4))
#         plt.plot(x_vals, pdf_vals, label='Truncated Normal PDF')
#         plt.axvline(U_mean, color='red', linestyle='--', label=f'Mean = {U_mean:.2f}')
#         plt.xlabel('Usal')
#         plt.ylabel('PDF')
#         plt.title('PDF of Usal (Truncated Normal)')
#         plt.legend()
#         plt.grid(True)
#         plt.tight_layout()
#         plt.show()

#     return U_mean

# def MeanUexpo(U_mu):
#     # Define the exponential distribution with mean = Usal_mu
#     scale = U_mu  # scale = 1/lambda
#     dist = expon(scale=scale)
#     def integrand(U):
#         return U * dist.pdf(U)
#     # Compute the integral numerically
#     U_mean, _ = quad(integrand, 0, np.inf)
#     return U_mean

# other splash functions
#COMSALT
# COR = np.sqrt(0.45)
# Pr = 0.96*(1-np.exp(- Uim_solution))
# Ure = Uim_solution * np.sqrt(0.45) #kinetic energy of rebound is 45%+-22% of that of impact 
# cos_thetare = np.cos(40/180*np.pi) #exponential distribution with mean of 40
# NE = 0.02 * Uim_solution / constant
# UE = 0.15/0.02*(1-np.exp(-Uim_solution/constant/40))*constant 
# cos_thetaej = np.cos(50/180*np.pi) 
       
# Beladijne 
# COR = 0.87 - 0.72*np.sin(theta_im)
# if abs(Uim_solution) / constant  >= 40:
#     NE = 13*(1-COR**2)*(abs(Uim_solution) / constant - 40) #0.04 * (Uim_solution / constant - 12)
# else:
#     NE = 0
# Pr = 0.96*(1-np.exp(- abs(Uim_solution)))
# theta_re = 45/180*np.pi
# cos_thetare = np.cos(theta_re)
# UE = 1.18*(Uim_solution/constant)**0.25*constant
# Ure = Uim_solution * COR