# -*- coding: utf-8 -*-
"""
Created on Fri Nov 14 11:40:43 2025

@author: WangX3
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# -------------------- constants & inputs --------------------
g = 9.81
d = 0.00025                # grain diameter [m]  
const = np.sqrt(g*d)       # √(g d)
h = 0.2 - 13.5*d           # domain height [m] 
Shields = np.linspace(0.03, 0.06, 4)
ustar_list = np.sqrt(Shields * (2650-1.225)*9.81*d/1.225)
Omega_list = [0.0, 0.01, 0.05, 0.10, 0.20]

rho_a  = 1.225             # air density [kg/m^3] 
nu_a   = 1.46e-5           # air kinematic viscosity [m^2/s]
rho_p  = 2650.0            # particle density [kg/m^3]   
Ruc    = 24
Cd_inf = 0.5

# Particle mass
mp = rho_p * (np.pi/6.0) * d**3         
thetaE = np.deg2rad(24)

colors = plt.cm.viridis(np.linspace(1, 0, 5))  # 5 colors

# -------------------- closures from the PDF --------------------
def CalUincfromU(U, c):
    A = 1/(1 + c/0.19)
    Uinc = A*U
    return Uinc

def calc_T_jump_ballistic_assumption(Uinc, theta_inc):
    Uy0 = Uinc*np.sin(theta_inc)
    Tjump = 2*Uy0/g
    return Tjump 

def calc_Pr(Uinc, params, Omega):
    # kA, kB = 0, 0 #params
    # # A0 = 0.74
    # # B0 = 3.66
    # A0, B0, _ = params
    # A = A0 * (1.0 + kA * Omega)
    # B = B0 * (1.0 + kB * Omega)
    # # ensure A <= 1 for all Omega <= 0.2
    # A = min(A, 0.999)
    # Pr = A*np.exp(-B*np.exp(-0.10*Uinc/const))
    # Pr = 0.74*np.exp(-4.46*np.exp(-0.1*abs(Uinc)/const))
    # ap, bp, cp = params#0.74, 4.46, 0.10
    # Pr = ap*np.exp(-bp*np.exp(-cp*abs(Uinc)/const))
    ap, bp, _,_,_ = params
    Pr = ap * (1-np.exp(-bp*Uinc))
    # Pr = 0.95*(1-np.exp(-2*Uinc))
    # Pr = 0.97*(1-np.exp(-3.5*Uinc))
    return Pr

def e_COR_from_Uim(Uim, Omega):
    mu = 312.48
    sigma = 156.27
    e_com = 3.18*(abs(Uim)/const)**(-0.50) + 0.12*np.log(1 + 1061.81*Omega)*np.exp(- (abs(Uim)/const - mu)**2/(2*sigma**2) )     
    e = min(e_com, 1.0)
    return e          

def theta_inc_from_Uinc(Uinc, Omega):
    alpha = 0.27 - 0.05*Omega**0.15
    beta = -0.0029
    x = alpha * np.exp(beta * abs(Uinc)/const)  
    theta_inc = np.arcsin(x)
    if theta_inc < 0 or theta_inc > np.pi / 2: 
        print('The value of theta_inc is not logical')                                     
    return theta_inc

def theta_reb_from_Uim(Uim, Omega):
    x = (-0.0032*Omega**0.39)*(abs(Uim)/const) + 0.43 
    theta_re = np.arcsin(x)
    if theta_re < 0 or theta_re > np.pi / 2: 
        print('The value of theta_re is not logical')                              
    return theta_re

def NE_from_Uinc(Uinc, Omega, params): 
    # NE = (0.03 - 0.025*Omega**0.21) * (abs(Uinc)/const) 
    _,_,ane, bne, cne = params #0.03, 0.025, 0.21
    NE = (ane - bne*Omega**cne) * (abs(Uinc)/const) 
    return NE

def UE_from_Uinc(Uinc, Omega):
    A = -1.51*Omega + 4.62
    B = 0.37*Omega**0.26 + 0.019
    U_E = A*(abs(Uinc)/const)**B * const
    return U_E * np.sign(Uinc)  

def tau_top(u_star):                                                 
    return rho_a * u_star**2     

# def Fitb(U, c, b0=0.015, b_inf=0.79, k0=0.53, lamda=4.86):
#     k = k0/(1+lamda*c)
#     b = b0 + (b_inf - b0)*(1 - np.exp(-k*U))
#     return b

# def calc_Mdrag(Ua, U, c):
#     b = Fitb(U, c)
#     Urel = b * Ua - U
#     Re = abs(Urel)*d/nu_a
#     Cd = (np.sqrt(Cd_inf)+np.sqrt(Ruc/Re))**2   
#     Mdrag = np.pi/8 * d**2 * rho_a * Urel * abs(Urel) * Cd * c/mp
#     return Mdrag

def calc_Mdrag(Ua, U, c):
    b = 0.84*(1-np.exp(-0.49*U))
    burel = 1/(1+c/0.09)
    Urel = burel*(b * Ua - U)
    Re = abs(Urel)*d/nu_a
    # Re = np.maximum(Re, 1e-6)
    Cd = (np.sqrt(Cd_inf)+np.sqrt(Ruc/Re))**2   
    Mdrag = np.pi/8 * d**2 * rho_a * Cd * Urel * abs(Urel) * c/mp
    return Mdrag

def calc_Mcreep(Ua, U, c):
    Mdrag = calc_Mdrag(Ua, U, c)
    M_bed = 1.21*Mdrag 
    return M_bed

def calc_Maero(Ua):
    Maero = 0.5*0.0037*rho_a*abs(Ua)*Ua
    return Maero

# MEASUREMENT–DRIVEN START TIME DETECTION
# ============================================================
def detect_start_idx(C_meas, Cref):
    above = np.where(C_meas > Cref)[0]
    if len(above) == 0:
        return None
    return above[0]

def detect_peak_idx_smoothed(C, window=5):
    C = np.asarray(C)
    Csmooth = np.convolve(C, np.ones(window)/window, mode='same')
    return np.argmax(Csmooth)

def rhs_cmUa(t, y, u_star, Omega, params):
    eps=1e-16
    c, m, Ua = y
    U = m/(c + eps)

    Uinc = CalUincfromU(U, c)
    thetainc = theta_inc_from_Uinc(Uinc, Omega)
    Tim  = calc_T_jump_ballistic_assumption(Uinc, thetainc) + eps
    Pr = calc_Pr(Uinc, params, Omega)

    # mixing fractions and rates
    r_im = c/Tim 

    # ejection numbers and rebound kinematics
    NEim = NE_from_Uinc(Uinc, Omega, params) 
    # NEim = max(1e-6, NEim)
    eCOR = e_COR_from_Uim(Uinc, Omega); Ure = Uinc*eCOR
    th_re = theta_reb_from_Uim(Uinc, Omega)
    UE = UE_from_Uinc(Uinc, Omega)

    # scalar sources
    E = r_im*NEim 
    D = r_im*(1-Pr) 
    
    if E<0:
            print('E = ',E)

    # momentum sources (streamwise)
    M_drag = calc_Mdrag(Ua, U, c) 
    M_eje  = E * UE * np.cos(thetaE)
    M_re   = r_im * Pr * Ure*np.cos(th_re) 
    M_im   = r_im * Pr * Uinc*np.cos(thetainc) 
    M_dep  = D * Uinc*np.cos(thetainc) 
    
    if M_dep > D*U:
            print('More momentum is leaving per particle by deposition, than that there is on average in the saltation layer')
            print('M_dep =',M_dep)
            print('D*U', D*U)
            print('D =',D)
            print('U =',U)
            print('Uinc =',Uinc)
            print('-----------')
            
    if D<0:
        print('D=',D)
        
    if M_eje>U*E:
        print('M_eje =',M_eje)
        print('U*E = ', U*E)
        print('U =',U)
        print('E = ',E)
        print('----')
        
    if M_re>M_im:
        print('M_re =',M_re)
        print('M_im =',M_im)
        print('----')

    # ODEs
    dc_dt = E - D
    dm_dt = M_drag + M_eje + M_re - M_im - M_dep

    phi_term  = 1.0 - c/(rho_p*h)
    m_air_eff = rho_a*h*phi_term
    M_creep = calc_Mcreep(Ua, U, c)
    M_aero = calc_Maero(Ua)
    dUa_dt    = (tau_top(u_star) - M_aero - M_creep - M_drag) / m_air_eff

    return [dc_dt, dm_dt, dUa_dt]

def euler_forward(rhs, y0, t_span, dt, u_star, Omega, params):
    t0, t1 = t_span
    nsteps = int(np.ceil((t1 - t0) / dt))
    t = np.empty(nsteps + 1, dtype=float)
    y = np.empty((3, nsteps + 1), dtype=float)

    t[0]   = t0
    y[:,0] = np.asarray(y0, dtype=float)
    
    # Mdrag_eff_list = []
    for k in range(nsteps):
        tk   = t[k]
        yk   = y[:,k].copy()

        # Euler step
        rhs_out = rhs_cmUa(tk, yk, u_star, Omega, params)
        f = rhs_out
        y_next = yk + dt * np.asarray(f, dtype=float)
        
        # Mdrageff = rhs_out[-1]
        # Mdrag_eff_list.append(Mdrageff)

        t[k+1]   = min(tk + dt, t1)
        y[:,k+1] = y_next

    return t, y

omega_labels = ['Dry', 'M1', 'M5', 'M10', 'M20']
# Initialize lists to hold results for each Omega
Q_dpm = []
C_dpm = []
U_dpm = []
Ua_dpm = []

# Loop over all moisture levels
for label in omega_labels:
    for i in range(3, 7):
        # --- Load sediment transport data ---
        file_path = f'CGdata/hb=13.5d/Shields00{i}{label}-135d.txt'
        data = np.loadtxt(file_path)
        C = data[:, 1]
        U = data[:, 2]
        Q = C * U
        # Append to lists
        C_dpm.append(C)
        U_dpm.append(U)
        Q_dpm.append(Q)
        # --- Load corresponding air velocity data ---
        file_path_ua = f'TotalDragForce/Ua-t/Uair_ave-tS00{i}{label}.txt'
        data_ua = np.loadtxt(file_path_ua, delimiter='\t')
        Ua = data_ua[:, 1]
        Ua_dpm.append(Ua)

def simulate_model(params):
    model_output = {Omega: {} for Omega in Omega_list}
    dt = 1e-2
    
    for i, Omega in enumerate(Omega_list):
        for j, ustar in enumerate(ustar_list):
        # j, ustar = 3, ustar_list[-1]
            index = i*len(ustar_list)+j
    
            C_meas = C_dpm[index]
            U_meas = U_dpm[index]
            Ua_meas = Ua_dpm[index]
    
            start_idx = detect_start_idx(C_meas, 0.075)
            if start_idx is None:
                # no saltation → ignore
                continue
    
            # initial conditions set from data at start point
            C0 = C_meas[start_idx]
            U0 = U_meas[start_idx]
            Ua0 = Ua_meas[start_idx]
    
            y0 = (C0, C0*U0, Ua0)
    
            # simulate model from that initial point over remaining time
            total_time = 5.0 - start_idx*dt
            t, Y = euler_forward(rhs_cmUa, y0, (0.0, total_time), dt, ustar, Omega, params)
    
            c  = Y[0]
            m  = Y[1]
            Ua = Y[2]
            U  = m / np.maximum(c, 1e-16)
    
            model_output[Omega][ustar] = {
                't': t,
                'idx0': start_idx,
                'c': c,
                'U': U,
                'Ua': Ua,
            }
    return model_output

def cost_function(params):
    model = simulate_model(params)
    eps = 1e-8
    cost = 0.0

    for i, Omega in enumerate(Omega_list):
        for j, ustar in enumerate(ustar_list):
            index = i*len(ustar_list)+j

            C_meas = C_dpm[index]
            U_meas = U_dpm[index]
            Ua_meas = Ua_dpm[index]

            start_idx = detect_start_idx(C_meas, 0.075)
            if start_idx is None:
                continue

            # data after overshoot
            C_meas_seg  = C_meas[start_idx:]
            U_meas_seg  = U_meas[start_idx:]
            Ua_meas_seg = Ua_meas[start_idx:]

            # model
            c_mod  = model[Omega][ustar]['c'][:len(C_meas_seg)]
            U_mod  = model[Omega][ustar]['U'][:len(U_meas_seg)]
            Ua_mod = model[Omega][ustar]['Ua'][:len(Ua_meas_seg)]

            # ---- TIME WEIGHTING ----
            n = len(C_meas_seg)
            t_weights = np.linspace(0.3, 1.0, n)

            cost += np.sum(t_weights * (c_mod - C_meas_seg)**2) / (np.sum(t_weights * C_meas_seg**2) + eps)
            cost += np.sum(t_weights * (U_mod - U_meas_seg)**2) / (np.sum(t_weights * U_meas_seg**2) + eps)
            cost += np.sum(t_weights * (Ua_mod - Ua_meas_seg)**2) / (np.sum(t_weights * Ua_meas_seg**2) + eps) 

    return cost

# initial guess
# x0 = [0.90, 1.00, 0.03, 0.025, 0.21]#[0.99, 1.00, 0.13, 0.02, 0.02, 0.1]
# bounds = [(0.2, 0.99), (1.0, 10.0), (0.02, 0.035), (0.02, 0.03), (0.1, 0.25)]#[(0.10, 0.99),(1e-6, 15.0), (0.001, 0.5),#(1e-6, 1.0), (1e-6, 1.0), (1e-6, None), (1e-6, None)]#, (0.01, 1.0), (0.01, 2.0)]

# res = minimize(cost_function, x0, bounds=bounds, method='L-BFGS-B')

# param_names = ['a_Pr', 'b_Pr', 'c_Pr','a_NE', 'b_NE', 'c_NE']

# for name, value in zip(param_names, res.x):
#     print(f"{name:>6} = {value:.6f}")

model_run = simulate_model([0.99, 3.5, 0.02, 0.02, 0.1]) #
t_mod = np.linspace(0, 5, 501)
dt = 0.01

plt.close('all')
for ui, ustar in enumerate(ustar_list):
# ui, ustar = 3, ustar_list[-1]

    fig, axes = plt.subplots(3, 1, figsize=(7, 10), sharex=True)
    axC, axU, axUa = axes
    
    for oi, Omega in enumerate(Omega_list):
    
        # measured series
        idx     = oi*len(ustar_list) + ui
        C_meas  = C_dpm[idx]
        U_meas  = U_dpm[idx]
        Ua_meas = Ua_dpm[idx]
        t_meas  = np.linspace(0, 5, len(C_meas))
    
        # detect start idx
        start_idx = detect_start_idx(C_meas, 0.075)
        if start_idx is None:
            continue
    
        # model series (already trimmed in simulate_model)
        C_mod  = model_run[Omega][ustar]['c']
        U_mod  = model_run[Omega][ustar]['U']
        Ua_mod = model_run[Omega][ustar]['Ua']
        t_mod  = model_run[Omega][ustar]['t']
    
        # Now SHIFT model time to actual real time
        t_mod_shifted = t_mod + (start_idx * dt)
    
        label = f'Ω={Omega}'
    
        # --- Plot C ---
        axC.plot(t_meas, C_meas, '--', color=colors[oi])
        axC.plot(t_mod_shifted, C_mod, color=colors[oi], label=f'{label} – model')
    
        # --- Plot U ---
        axU.plot(t_meas, U_meas, '--', color=colors[oi])
        axU.plot(t_mod_shifted, U_mod, color=colors[oi])
    
        # --- Plot Ua ---
        axUa.plot(t_meas, Ua_meas, '--', color=colors[oi])
        axUa.plot(t_mod_shifted, Ua_mod, color=colors[oi])
    
    axC.plot([], [], color='black', label=r"Continuum")
    axC.plot([], [], '--', color='black', label=r"DPM")
    axC.set_ylabel('C [kg/m²]')
    axC.set_title(fr'$\Theta$ = {Shields[ui]:.2f}')
    axC.grid(True)
    axC.set_ylim(0, 0.30)
    axC.legend(fontsize=8)
    
    axU.set_ylabel('U [m/s]')
    axU.set_ylim(0, 9.5)
    axU.grid(True)
    
    axUa.set_ylabel('Ua [m/s]')
    axUa.set_xlabel('t [s]')
    axUa.set_ylim(0, 13.5)
    axUa.grid(True)
    
    plt.tight_layout()
    plt.show()

    
# plt.figure()
# for i in range(4):
#     plt.plot(t_mod, ustar_list[i]*np.ones(len(t_mod)))
#     plt.plot(t_mod[:-1], model_run[0][ustar_list[i]]['Mdrag_eff'])
# plt.xlabel('t')
# plt.ylabel('balance between tau_top and Mdrag,eff')

# # steady C
# mean_cs, mean_us, mean_uas = [], [], []

# for Omega in model_run:
#     for ustar in model_run[Omega]:
#         c_array = model_run[Omega][ustar]['c']
#         u_array = model_run[Omega][ustar]['U']
#         ua_array = model_run[Omega][ustar]['Ua']
#         mean_cs.append(np.mean(c_array[100:]))
#         mean_us.append(np.mean(u_array[100:]))
#         mean_uas.append(np.mean(ua_array[100:]))
        
# Uinc = CalUincfromU(mean_us[-1], Omega_list)
# thetainc = theta_inc_from_Uinc(Uinc, Omega_list)
# Tim  = calc_T_jump_ballistic_assumption(Uinc, thetainc) 
# r_im = mean_cs[-1]/Tim

# NEim = NE_from_Uinc(Uinc, Omega_list)*(1-0.5*Omega_list)      
# Pr   = calc_Pr(Uinc)*(1-2.0*Omega_list)

# E = r_im * NEim
# D = r_im * (1 - Pr)
# print(Omega_list, E/D)
