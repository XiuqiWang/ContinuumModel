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

# Particle mass
mp = rho_p * (np.pi/6.0) * d**3         
thetaE = np.deg2rad(24)

colors = plt.cm.viridis(np.linspace(1, 0, 5))  # 5 colors

# -------------------- closures from the PDF --------------------
def CalUincfromU(U, c, Omega):
    Cref, Cref_urel, B, p = 1.0, 0.0088, 9.52, 18.05 
    # a1, b1 = 0.01, 0.99
    a0, b0 = 0.92, 0.58
    a1, b1 = 0,0
    # A = a0 + a1*Omega
    # n = b0 + b1*Omega
    # if Omega == 0:
    #     # Uinc = 0.43*U
    #     Uinc = 0.61*U**0.44
    # else:
    #     # Uinc = 0.85*U
    #     Uinc = 0.44*U**1.36
    Cref_Uinc, Cref_Uinc_n = 0.20, 0.05
    A = 1/np.sqrt(1 + c/Cref_Uinc)
    n = 1/np.sqrt(1 + c/Cref_Uinc_n)
    Uinc = A*U**n
    return Uinc

def calc_T_jump_ballistic_assumption(Uinc, theta_inc):
    Uy0 = Uinc*np.sin(theta_inc)
    Tjump = 2*Uy0/g
    return Tjump 

def calc_Pr(Uinc, Omega, params):
    kA, kB = 0, 0 #params
    # A0 = 0.74
    # B0 = 3.66
    A0, B0, _ = params
    A = A0 * (1.0 + kA * Omega)
    B = B0 * (1.0 + kB * Omega)
    # ensure A <= 1 for all Omega <= 0.2
    A = min(A, 0.999)
    Pr = A*np.exp(-B*np.exp(-0.10*Uinc/const))
    return Pr

# Uinc = np.linspace(0, 5, 100)
# Pr = calc_Pr(Uinc, 0, 0.5, 0.5)
# Pr_wet = calc_Pr(Uinc, 0.2, 0.5, 0.5)
# plt.figure()
# plt.plot(Uinc, Pr)
# plt.plot(Uinc, Pr_wet)

def e_COR_from_Uim(Uim, Omega):
    mu = 312.48
    sigma = 156.27
    e_com = 3.18*(abs(Uim)/const)**(-0.50) + 0.12*np.log(1 + 1061.81*Omega)*np.exp(- (abs(Uim)/const - mu)**2/(2*sigma**2) )   
    e = min(e_com, 1.0)
    return e          

def theta_inc_from_Uinc(Uinc, Omega):
    alpha = -113.89*Omega + 67.78 
    beta = -260.77 * Omega + 248.51
    x = alpha / (abs(Uinc)/const + beta)  
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
    _, _, a_ne = params
    # NE = (0.03-0.028*Omega**0.19) * (abs(Uinc)/const) 
    NE = a_ne * (abs(Uinc)/const) 
    return NE

def UE_from_Uinc(Uinc, Omega):
    A = -2.13*Omega + 4.60
    B = 0.40*Omega**0.24 + 0.008
    U_E = A*(abs(Uinc)/const)**B * const
    return U_E * np.sign(Uinc)  

def tau_top(u_star):                                                 
    return rho_a * u_star**2     

def calc_Mdrag(c, Uair, U, Omega, params):
    Cref, Cref_urel, B, p = 1.0, 0.0088, 9.52, 18.05
    b = np.sqrt(1 - c/(c+Cref))
    b_urel = 1/np.sqrt(1 + c/Cref_urel)
    Urel = b_urel * (b*Uair - U)
    Re = abs(Urel)*d/nu_a
    Ruc = 24
    Cd_inf = 0.5
    Cd = (np.sqrt(Cd_inf)+np.sqrt(Ruc/Re))**2
    Agrain = np.pi*(d/2)**2
    Ngrains = c/mp
    Mdrag = 0.5*rho_a*Urel*abs(Urel)*Cd*Agrain*Ngrains
    return Mdrag

def MD_eff(Ua, U, c, Omega, params):
    Cref, Cref_urel, B, p = 1.0, 0.0088, 9.52, 18.05
    Mdrag = calc_Mdrag(c, Ua, U, Omega, params)
    CD_bed = 0.0037
    # B, p = 3.5, 7.0
    tau_basic = 0.5 * rho_a * CD_bed * Ua * abs(Ua)
    M_bed = tau_basic * (1/(1+(B*Mdrag)**p)) # to guarantee that there is a balancing term with tau_top when c=0
    M_final = Mdrag + M_bed
    return M_final

def rhs_cmUa(t, y, u_star, Omega, params):
    eps=1e-16
    c, m, Ua = y
    U = m/(c + eps)

    Uinc = CalUincfromU(U, c, Omega)
    thetainc = theta_inc_from_Uinc(Uinc, Omega)
    Tim  = calc_T_jump_ballistic_assumption(Uinc, thetainc) + eps
    Pr = calc_Pr(Uinc, Omega, params)

    # mixing fractions and rates
    r_im = c/Tim 

    # ejection numbers and rebound kinematics
    NEim = NE_from_Uinc(Uinc, Omega, params) 
    NEim = max(1e-6, NEim)
    eCOR = e_COR_from_Uim(Uinc, Omega); Ure = Uinc*eCOR
    th_re = theta_reb_from_Uim(Uinc, Omega)
    UE = UE_from_Uinc(Uinc, Omega)

    # scalar sources
    E = r_im*NEim 
    D = r_im*(1-Pr) 
    
    if E<0:
            print('E = ',E)

    # momentum sources (streamwise)
    M_drag = calc_Mdrag(c, Ua, U, Omega, params) 
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
    Mdrageff = MD_eff(Ua, U, c, Omega, params)
    dUa_dt    = (tau_top(u_star) - Mdrageff) / m_air_eff

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

    for i, Omega in enumerate(Omega_list):
        for j, ustar in enumerate(ustar_list):
            index = i*4+j
            y0 = (C_dpm[index][0], C_dpm[index][0]*U_dpm[index][0], Ua_dpm[index][0])
            t, Y = euler_forward(rhs_cmUa, y0, (0.0, 5.0), 1e-2, ustar, Omega, params)
    
            c  = Y[0]
            m  = Y[1]
            Ua = Y[2]
            U  = m / np.maximum(c, 1e-16)
    
            model_output[Omega][ustar] = {
                'c': c,
                'U': U,
                'Ua': Ua,
            }

    return model_output

def cost_function(params):
    model = simulate_model(params)
    
    eps = 1e-8
    cost = 0.0
    
    # for i, Omega in enumerate(Omega_list):
    i, Omega = 0, 0.0
    for j, ustar in enumerate(ustar_list):
        index = i*4+j
        # measured
        C_meas = C_dpm[index]
        U_meas = U_dpm[index]
        Ua_meas = Ua_dpm[index]

        # model
        c_mod  = model[Omega][ustar]['c'][:len(C_meas)]
        U_mod  = model[Omega][ustar]['U'][:len(U_meas)]
        Ua_mod = model[Omega][ustar]['Ua'][:len(Ua_meas)]

        # squared error
        cost += np.sum((c_mod - C_meas)**2 / (C_meas + eps))
        cost += np.sum((U_mod - U_meas)**2 / (U_meas + eps))
        cost += np.sum((Ua_mod - Ua_meas)**2 / (Ua_meas + eps))

    return cost

# initial guess
x0 = [0.75, 3.66, 0.03]#[0.2, 0.05]#[0.05, 0.05, 3.5, 7.0]#, 0.50, 1.0]
bounds = [(0.1, 0.99), (1e-6, 10.0), (0.01, 0.1)]#[(1e-6, 1.0), (1e-6, 1.0)]#[(1e-6, 1.0), (1e-6, 1.0), (1e-6, None), (1e-6, None)]#, (0.01, 1.0), (0.01, 2.0)]

res = minimize(cost_function, x0, bounds=bounds, method='L-BFGS-B')

param_names = ['A0_pr', 'B0_pr']

for name, value in zip(param_names, res.x):
    print(f"{name:>6} = {value:.6f}")

model_run = simulate_model(res.x)
t_mod = np.linspace(0, 5, 501)

plt.close('all')
for ui, ustar in enumerate(ustar_list):

    fig, axes = plt.subplots(3, 1, figsize=(7, 10), sharex=True)
    axC, axU, axUa = axes

    for oi, Omega in enumerate(Omega_list):

        # measured data index: i*5 + j
        idx = oi*len(ustar_list) + ui
        
        # measured
        C_meas  = C_dpm[idx]
        U_meas  = U_dpm[idx]
        Ua_meas = Ua_dpm[idx]
        t_meas  = np.linspace(0, 5, len(C_meas))

        # model
        C_mod   = model_run[Omega][ustar]['c']
        U_mod   = model_run[Omega][ustar]['U']
        Ua_mod  = model_run[Omega][ustar]['Ua']

        label = f'Ω={Omega}'

        # --- Plot C ---
        axC.plot(t_mod, C_mod, color=colors[oi], label=f'{label} – model')
        axC.plot(t_meas, C_meas, '--', color=colors[oi], label=f'{label} – data')

        # --- Plot U ---
        axU.plot(t_mod, U_mod, color=colors[oi])
        axU.plot(t_meas, U_meas, '--', color=colors[oi])

        # --- Plot Ua ---
        axUa.plot(t_mod, Ua_mod, color=colors[oi])
        axUa.plot(t_meas, Ua_meas, '--', color=colors[oi])

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
