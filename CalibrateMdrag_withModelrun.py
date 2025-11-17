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
thetaE = np.deg2rad(25)

colors = plt.cm.viridis(np.linspace(1, 0, 5))  # 5 colors

# -------------------- closures from the PDF --------------------
def CalUincfromU(U, Omega):
    if Omega == 0:
        # Uinc = 0.43*U
        Uinc = 0.61*U**0.44
    else:
        # Uinc = 0.85*U
        Uinc = 0.44*U**1.36
    return Uinc

def calc_T_jump_ballistic_assumption(Uinc, theta_inc):
    Uy0 = Uinc*np.sin(theta_inc)
    Tjump = 2*Uy0/g
    return Tjump 

def calc_Pr(Uinc):
    Pr = 0.74*np.exp(-3.66*np.exp(-0.10*Uinc/const))
    if Pr <0 or Pr>1:
        print('Warning: Pr is not between 0 and 1')
    return Pr

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

def NE_from_Uinc(Uinc, Omega): 
    return (0.03-0.028*Omega**0.19) * (abs(Uinc)/const) 

def UE_from_Uinc(Uinc, Omega):
    A = -2.13*Omega + 4.60
    B = 0.40*Omega**0.24 + 0.008
    U_E = A*(abs(Uinc)/const)**B * const
    return U_E * np.sign(Uinc)  

def tau_top(u_star):                                                 
    return rho_a * u_star**2     

def calc_Mdrag(c, Uair, U, Omega, params):
    Cref, Cref_urel, B, p = params
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
    Cref, Cref_urel, B, p = params
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

    Uinc = 0.5*U#CalUincfromU(U, Omega)
    thetainc = theta_inc_from_Uinc(Uinc, Omega)
    Tim  = calc_T_jump_ballistic_assumption(Uinc, thetainc) + eps
    Pr = calc_Pr(Uinc) 

    # mixing fractions and rates
    r_im = c/Tim

    # ejection numbers and rebound kinematics
    NEim = NE_from_Uinc(Uinc, Omega)
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
    dUa_dt    = (tau_top(u_star) - MD_eff(Ua, U, c, Omega, params)) / m_air_eff

    return [dc_dt, dm_dt, dUa_dt]

def euler_forward(rhs, y0, t_span, dt, u_star, Omega, params):
    t0, t1 = t_span
    nsteps = int(np.ceil((t1 - t0) / dt))
    t = np.empty(nsteps + 1, dtype=float)
    y = np.empty((3, nsteps + 1), dtype=float)

    t[0]   = t0
    y[:,0] = np.asarray(y0, dtype=float)

    for k in range(nsteps):
        tk   = t[k]
        yk   = y[:,k].copy()

        # Euler step
        f = rhs_cmUa(tk, yk, u_star, Omega, params)
        y_next = yk + dt * np.asarray(f, dtype=float)

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
                'Ua': Ua
            }

    return model_output

def cost_function(params):
    model = simulate_model(params)

    cost = 0.0
    
    for i, Omega in enumerate(Omega_list):
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
            cost += np.sum((c_mod - C_meas)**2)
            cost += np.sum((U_mod - U_meas)**2)
            cost += np.sum((Ua_mod - Ua_meas)**2)

    return cost

# initial guess
x0 = [0.05, 0.05, 3.5, 7.0]
bounds = [(1e-6, 1.0), (1e-6, 1.0), (1e-6, None), (1e-6, None)]

res = minimize(cost_function, x0, bounds=bounds, method='L-BFGS-B')

param_names = ["Cref", "Cref_urel", "B", "p"]

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
