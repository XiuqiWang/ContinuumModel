# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 16:25:52 2025

@author: WangX3
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
import pandas as pd
from scipy.signal import savgol_filter

h = 0.2 - 0.00025*12
D = 0.00025
g = 9.81
rho_a = 1.225
rho_sand = 2650
nu_a = 1.46e-5
Shields = np.linspace(0.02, 0.06, 5)
u_star = np.sqrt(Shields * (2650-1.225)*9.81*D/1.225)
mp = 2650 * np.pi/6 * D**3 #particle mass
const = np.sqrt(g*D)

def Uim_from_U(U):
    """U_im from instantaneous saltation-layer velocity U (includes Ω effect)."""
    Uim_mag = 0.04*(abs(U)/const)**1.54 + 32.23
    return np.sign(U) * (Uim_mag*const)

def UD_from_U(U):
    """U_D from U. PDF only provides Dry (Ω=0). We use Dry law for all Ω unless you add a wet fit."""
    UD_mag = 6.66         
    return np.sign(U) * (UD_mag*const)

def Tim_from_Uim(Uim):
    return 0.04 * (abs(Uim))**0.84         

def TD_from_UD(UD):
    # A_td, p_td, Pmax, Uc, p, A_NE = params
    return 0.06 * (abs(UD))**0.66  
    # return 0.44 * (abs(UD))** 0.18
    
def calc_T_jump_ballistic_assumption(Uinc, theta_inc_degree):
    Uy0 = Uinc*np.sin(theta_inc_degree/180*np.pi)
    Tjump = 2*Uy0/g
    return Tjump     

def Preff_of_U(U, Pmax, Uc, p):
    """State-conditioned rebound fraction. PDF marks 'needs calibration'—placeholder Gompertz-like."""
    # P_min + (P_max-P_min) * (1 - exp(-(U/Uc)^p))
    # Pmin, Pmax, Uc, p = 0, 0.999, 3.84, 0.76 
    Pmin = 0
    U = abs(U)
    return Pmin + (Pmax - Pmin)*(1.0 - np.exp(-(U/Uc)**p)) 

def NE_from_Uinc(Uinc, A_NE): 
    # return (0.04-0.04*Omega**0.23) * (abs(Uinc)/const) 
    return A_NE * (abs(Uinc)/const) # try

def calc_N_E_test3(Uinc, A_NE, p_NE, Uinc_half_min):
        N_E_Xiuqi = NE_from_Uinc(Uinc, A_NE)
        # p = 8
        ##
        # p2 = 2
        # A = 100
        Uinc_half_min = 1.0
        # Uinc_half_max = 2.0
        Uinc_half = Uinc_half_min #+ (Uinc_half_max - Uinc_half_min)*(A*Omega)**p2/((A*Omega)**p2+1)
        ##
        #Uinc_half = 0.5+40*Omega**0.5
        B = 1/Uinc_half**p_NE
        N_E = (1-1./(1.+B*Uinc**p_NE))*N_E_Xiuqi
        return N_E         

def CalRimfromU(U, c_vec, Pmax, Uc, p):
    Uim, UD = Uim_from_U(U), UD_from_U(U)
    Tim = calc_T_jump_ballistic_assumption(Uim, 15)
    TD = calc_T_jump_ballistic_assumption(UD, 25)
    Pr = Preff_of_U(U, Pmax, Uc, p)
    cim = Pr*Tim/(Pr*Tim + (1-Pr)*TD) * c_vec
    Rim = cim/Tim
    return Rim

def CalEfromU(x, A_NE, p_NE, Uinc_half_min):
    U, c_vec = x
    Uim, UD = Uim_from_U(U), UD_from_U(U)
    Tim = calc_T_jump_ballistic_assumption(Uim, 15)
    TD = calc_T_jump_ballistic_assumption(UD, 25)
    Pr = Preff_of_U(U, 0.45, 0.66, 1.55)
    cim = Pr*Tim/(Pr*Tim + (1-Pr)*TD) * c_vec
    Rim = cim/Tim
    cD = (1-Pr)*TD/(Pr*Tim + (1-Pr)*TD) * c_vec
    D = cD/TD
    Nim, ND = calc_N_E_test3(Uim, A_NE, p_NE, Uinc_half_min), calc_N_E_test3(UD, A_NE, p_NE, Uinc_half_min)
    E = Rim*Nim + D*ND
    return E

# def OutputEfromU(U, c_vec, A_NE, p_NE, Uinc_half_min):
#     Uim, UD = Uim_from_U(U), UD_from_U(U)
#     Tim = calc_T_jump_ballistic_assumption(Uim, 15)
#     TD = calc_T_jump_ballistic_assumption(UD, 25)
#     Pr = Preff_of_U(U, 0.60, 0.46, 3.20)
#     cim = Pr*Tim/(Pr*Tim + (1-Pr)*TD) * c_vec
#     Rim = cim/Tim
#     cD = (1-Pr)*TD/(Pr*Tim + (1-Pr)*TD) * c_vec
#     D = cD/TD
#     Nim, ND = calc_N_E_test3(Uim, A_NE, p_NE, Uinc_half_min), calc_N_E_test3(UD, A_NE, p_NE, Uinc_half_min)
#     E = Rim*Nim + D*ND
#     return E, cim, cD, Rim, D, Nim, ND, Tim, TD
    
def CalDfromU(x, Pmax, Uc, p):
    U, c_vec = x
    Uim, UD = Uim_from_U(U), UD_from_U(U)
    Tim = calc_T_jump_ballistic_assumption(Uim, 15)
    TD = calc_T_jump_ballistic_assumption(UD, 25)
    Pr = Preff_of_U(U, Pmax, Uc, p)
    cD = (1-Pr)*TD/(Pr*Tim + (1-Pr)*TD) * c_vec
    D = cD/TD
    return D

def savgol_keep_first(x, window_length=21, polyorder=2):
    x = np.asarray(x, dtype=float)

    # Handle NaNs if any
    if np.isnan(x).any():
        idx = np.arange(len(x))
        x = np.interp(idx, idx[~np.isnan(x)], x[~np.isnan(x)])

    n = len(x)
    w = int(window_length)
    if w % 2 == 0:
        w += 1
    w = min(w, n if n % 2 == 1 else n - 1)
    w = max(w, polyorder + 2 if (polyorder + 2) % 2 == 1 else polyorder + 3)

    # Smooth only from 2nd element onward
    y = np.empty_like(x)
    y[0] = x[0]
    if n > 1:
        y[1:] = savgol_filter(x[1:], window_length=w if w < n else n - 1, polyorder=polyorder, mode='interp')
    return y

def r2_score(y, ypred):
    ss_res = np.sum((y - ypred)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    return 1.0 - ss_res/ss_tot

cases = range(2, 7)  # S002..S006
C_list, E_list, D_list = [], [], []
U_list = []
C_ori_list = []
E_smooth_list, D_smooth_list = [], []
t_dpm = np.linspace(0.01, 5, 501)
t = np.linspace(0.01, 5, 500)
dt = 0.01
for i in cases:
    # discrete data: columns -> C, U (501 rows)
    cg = np.loadtxt(f"dcdt/Discrete_CU/S{i:03d}Drydiscrete.txt")
    c, u = cg[:,0], cg[:,1]
    # c_center = 0.5*(c[:-1]+c[1:])
    # U_center = 0.5*(U[:-1]+U[1:])
    C_ori_list.append(c)
    C_list.append(c[:-1])
    U_list.append(u[:-1])
   
    # Rim, E and D
    ED = np.loadtxt(f"dcdt/ED/S{i:03d}DryEandD.txt")
    E, D = ED[:,0], ED[:,1]
    E_list.append(E)
    D_list.append(D)
    E_i = savgol_keep_first(E, window_length=21, polyorder=2) #smooth
    D_i = savgol_keep_first(D, window_length=21, polyorder=2) #smooth
    E_smooth_list.append(E_i)
    D_smooth_list.append(D_i)
    
for i in cases:
    cg = np.loadtxt(f"dcdt/Discrete_CU/S{i:03d}M20discrete.txt")
    c, u = cg[:,0], cg[:,1]
    # c_center = 0.5*(c[:-1]+c[1:])
    # U_center = 0.5*(U[:-1]+U[1:])
    C_ori_list.append(c)
    C_list.append(c[:-1])
    U_list.append(u[:-1])
   
    # Rim, E and D
    ED = np.loadtxt(f"dcdt/ED/S{i:03d}M20EandD.txt")
    E, D = ED[:,0], ED[:,1]
    E_list.append(E)
    D_list.append(D)
    E_i = savgol_keep_first(E, window_length=21, polyorder=2) #smooth
    D_i = savgol_keep_first(D, window_length=21, polyorder=2) #smooth
    E_smooth_list.append(E_i)
    D_smooth_list.append(D_i)
    
# plot E/c and D/c (10 in length now)
E_over_c = [E/c for E,c in zip(E_smooth_list,C_list)]
D_over_c = [D/c for D,c in zip(D_smooth_list,C_list)]

plt.figure(figsize=(8, 6))
for i in range(5):
    plt.subplot(3,2,i+1)
    plt.plot(U_list[i], E_over_c[i], '.', label='E/c')
    plt.plot(U_list[i], D_over_c[i], '.', label='D/c')
    plt.xlabel('U');plt.ylabel('E/c and D/c')
    plt.xlim(0, 3);plt.ylim(-5,80)
    plt.legend()
    plt.title(f'Shields=0.0{i+2}')
plt.suptitle('Dry')
plt.tight_layout()

plt.figure(figsize=(8, 6))
for i in range(5,10):
    plt.subplot(3,2,i-4)
    plt.plot(U_list[i], E_over_c[i], '.', label='E/c')
    plt.plot(U_list[i], D_over_c[i], '.', label='D/c')
    plt.xlabel('U');plt.ylabel('E/c and D/c')
    plt.xlim(0, 10);plt.ylim(-30,120)
    plt.legend()
    plt.title(f'Shields=0.0{i-3}')
plt.suptitle(r'$\Omega$=20 $\%$')
plt.tight_layout()

plt.figure(figsize=(8, 6))
for i in range(5):
    plt.subplot(3,2,i+1)
    plt.plot(U_list[i], E_over_c[i]-D_over_c[i], '.', label='E/c')
    # plt.plot(U_list[i], D_over_c[i], '.', label='D/c')
    plt.xlabel('U');plt.ylabel('E/c - D/c')
    plt.grid()
    plt.xlim(0, 3);plt.ylim(-32,30)
    plt.title(f'Shields=0.0{i+2}')
plt.suptitle('Dry')
plt.tight_layout()

plt.figure(figsize=(8, 6))
for i in range(5,10):
    plt.subplot(3,2,i-4)
    plt.plot(U_list[i], E_over_c[i]-D_over_c[i], '.', label='E/c')
    # plt.plot(U_list[i], D_over_c[i], '.', label='D/c')
    plt.xlabel('U');plt.ylabel('E/c - D/c')
    plt.grid()
    plt.xlim(0, 10);plt.ylim(-32,40)
    plt.title(f'Shields=0.0{i-3}')
plt.suptitle(r'$\Omega$=20 $\%$')
plt.tight_layout()
    
plt.figure(figsize=(8, 6))
for i in range(5):
    plt.subplot(3,2,i+1)
    plt.plot(U_list[i], E_list[i], '.', label='E')
    plt.plot(U_list[i], D_list[i], '.', label='D')
    plt.xlabel('U');plt.ylabel('E and D')
    plt.xlim(0, 3);plt.ylim(-2,10)
    plt.legend()
    plt.title(f'Shields=0.0{i+2}')
plt.suptitle('Dry')
plt.tight_layout()

plt.figure(figsize=(8, 6))
for i in range(5,10):
    plt.subplot(3,2,i-4)
    plt.plot(U_list[i], E_list[i], '.', label='E')
    plt.plot(U_list[i], D_list[i], '.', label='D')
    plt.xlabel('U');plt.ylabel('E and D')
    plt.xlim(0, 10);plt.ylim(-2,5)
    plt.legend()
    plt.title(f'Shields=0.0{i+2}')
plt.suptitle(r'$\Omega$=20 $\%$')
plt.tight_layout()

plt.figure(figsize=(8, 6))
for i in range(5):
    plt.subplot(3,2,i+1)
    plt.plot(C_list[i], E_list[i], '.', label='E')
    plt.plot(C_list[i], D_list[i], '.', label='D')
    plt.xlabel('C');plt.ylabel('E and D')
    plt.xlim(0, 0.35);plt.ylim(-2,10)
    plt.legend()
    plt.title(f'Shields=0.0{i+2}')
plt.suptitle('Dry')
plt.tight_layout()

plt.figure(figsize=(8, 6))
for i in range(5,10):
    plt.subplot(3,2,i-4)
    plt.plot(C_list[i], E_list[i], '.', label='E')
    plt.plot(C_list[i], D_list[i], '.', label='D')
    plt.xlabel('C');plt.ylabel('E and D')
    plt.xlim(0, 0.2);plt.ylim(-2,5)
    plt.legend()
    plt.title(f'Shields=0.0{i+2}')
plt.suptitle(r'$\Omega$=20 $\%$')
plt.tight_layout()

def CalphiD(U, C):
    Uim, UD = Uim_from_U(U), UD_from_U(U)
    print('UD',UD)
    Tim = calc_T_jump_ballistic_assumption(Uim, 15)
    TD = calc_T_jump_ballistic_assumption(UD, 25)
    Pr = Preff_of_U(U, 0.45, 0.66, 1.55)
    phiD = (1-Pr)*TD/(Pr*Tim + (1-Pr)*TD)
    return phiD

TD = calc_T_jump_ballistic_assumption(0.33, 25)
plt.figure(figsize=(10, 9))
for i in range(5):
    plt.subplot(3,2,i+1)
    plt.plot(C_list[i]/TD, D_smooth_list[i],'.', label='E')
    plt.xlabel('C/TD');plt.ylabel('D')
plt.legend()
plt.tight_layout()

plt.figure(figsize=(10, 9))
for i in range(5,10):
    plt.subplot(3,2,i-4)
    plt.plot(t, D_smooth_list[i]/C_list[i]*TD, '.', label='D/(C/TD)')
    # phiD = CalphiD(U_list[i], C_list[i])
    # plt.plot(t, phiD, '.', label='computed phiD')
    plt.xlabel('U');plt.ylabel(r'$\phi_D$')
    plt.title(f'Shields=0.0{i+2}')
plt.tight_layout()

plt.figure(figsize=(10, 9))
for i in range(5):
    plt.subplot(3,2,i+1)
    plt.plot(t, D_smooth_list[i], '.', label='D')
    plt.plot(t, C_list[i]/TD, '.', label='C/TD')
    plt.xlabel('t')
    plt.title(f'Shields=0.0{i+2}')
    plt.legend()
plt.suptitle('Dry')
plt.tight_layout()

TD_M20 = calc_T_jump_ballistic_assumption(0.4, 22)
plt.figure(figsize=(10, 9))
for i in range(5,10):
    plt.subplot(3,2,i-4)
    plt.plot(t, D_smooth_list[i], '.', label='D')
    plt.plot(t, C_list[i]/TD_M20, '.', label='C/TD')
    plt.xlabel('t')
    plt.title(f'Shields=0.0{i-3}')
    plt.legend()
plt.suptitle(r'$\Omega$=20 $\%$')
plt.tight_layout()

# # # find optimized parameters    
# E_all = np.concatenate(E_smooth_list)
# D_all = np.concatenate(D_smooth_list)
# U_all = np.concatenate(U_list)
# c_all = np.concatenate(C_list)

# popt, _ = curve_fit(CalDfromU, (U_all, c_all), D_all, maxfev=10000)
# Pmax, Uc, p = popt
# print('Pmax', Pmax, 'Uc', Uc,'p', p)
# D_pred = CalDfromU((U_all, c_all), Pmax, Uc, p)
# r2 = r2_score(D_all, D_pred)
# print('r2', r2)    

# popt, _ = curve_fit(CalEfromU, (U_all, c_all), E_all, maxfev=10000)
# A_NE, p_NE, Uinc_half_min = popt
# print('A_NE', A_NE, 'p_NE', p_NE, 'Uinc_half_min', Uinc_half_min)
# E_pred = CalEfromU((U_all, c_all), A_NE, p_NE, Uinc_half_min)
# r2 = r2_score(E_all, E_pred)
# print('r2', r2)  

# E_check, cim_check, cD_check, Rim_check, D_check, Nim_check, ND_check, Tim_check, TD_check = OutputEfromU(U_list[4], C_list[4], A_NE, p_NE, Uinc_half_min)   
# plt.figure(figsize=(12,8))
# plt.subplot(2,1,1)
# plt.plot(t, E_check, label='E')
# plt.plot(t, C_list[4], label='c')
# plt.plot(t, cim_check, label='cim')
# plt.plot(t, Tim_check, label='Tim')
# plt.plot(t, Rim_check, label='Rim')
# plt.plot(t, Nim_check, label='Nim')
# plt.plot(t, Rim_check*Nim_check, label='Rim*Nim')
# plt.ylim(0,2.75)
# plt.xlim(0,5)
# plt.xlabel('t [s]')
# plt.legend() 
# plt.subplot(2,1,2)
# plt.plot(t, E_check, label='E')
# plt.plot(t, C_list[4], label='c')
# plt.plot(t, cD_check, label='cD')
# plt.plot(t, TD_check, label='TD')
# plt.plot(t, D_check, label='D')
# plt.plot(t, ND_check, label='ND')
# plt.plot(t, D_check*ND_check, label='D*ND')
# plt.xlabel('t [s]')
# plt.ylim(0,2.75)
# plt.xlim(0,5)
# plt.legend() 
# plt.tight_layout()

# E_com_list, D_com_list = [], []
# c_com_list = []
# N = len(C_list[0]+1)
# for i in range(5):
#     E_com = CalEfromU((U_list[i], C_list[i]), A_NE, p_NE, Uinc_half_min)
#     D_com = CalDfromU((U_list[i], C_list[i]), Pmax, Uc, p)
#     E_com_list.append(E_com)
#     D_com_list.append(D_com)    
#     c0 = C_list[i][0]
#     c = np.zeros(N)
#     c[0] = c0
#     for j in range(N-1):
#         E = CalEfromU((U_list[i][j], c[j]), A_NE, p_NE, Uinc_half_min)
#         D = CalDfromU((U_list[i][j], c[j]), Pmax, Uc, p)
#         dcdt = E - D
#         c[j+1] = c[j] + dcdt * dt
#     c_com_list.append(c)

# plt.figure(figsize=(12, 6))
# for i in range(5):  # 5 groups
#     plt.subplot(2, 3, i+1)
#     E_i = E_smooth_list[i]
#     plt.plot(t, E_i, '.', label='DPM')
#     plt.plot(t, E_com_list[i], '.', label='computed')
#     plt.title(f'$\Theta$=0.0{i+2}')
#     plt.ylabel(r'$E$ [kg/m$^2$/s]')
#     plt.xlabel(r't [s]')
#     plt.xlim(left=0)
#     plt.ylim(bottom=0)
#     plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(12, 6))
# for i in range(5):  # 5 groups
#     plt.subplot(2, 3, i+1)
#     D_i = D_smooth_list[i]
#     plt.plot(t, D_i, '.', label='DPM')
#     plt.plot(t, D_com_list[i], '.', label='computed')
#     plt.title(f'$\Theta$=0.0{i+2}')
#     plt.ylabel(r'$D$ [kg/m$^2$/s]')
#     plt.xlabel(r't [s]')
#     plt.xlim(left=0)
#     plt.ylim(bottom=0)
#     plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(12, 6))
# for i in range(5):  # 5 groups
#     plt.subplot(2, 3, i+1)
#     plt.plot(t_dpm[:-1], C_list[i], '.', label='DPM')
#     plt.plot(t_dpm[:-1], c_com_list[i], '.', label='computed')
#     plt.title(f'$\Theta$=0.0{i+2}')
#     plt.ylabel(r'$C$ [kg/m$^2$]')
#     plt.xlabel(r't [s]')
#     plt.xlim(left=0)
#     plt.ylim(bottom=0)
#     plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.show()