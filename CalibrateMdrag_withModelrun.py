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
h = 0.197                 # domain height [m]
u_star = 0.56              # shear velocity [m/s]

rho_a  = 1.225             # air density [kg/m^3] 
nu_a   = 1.46e-5           # air kinematic viscosity [m^2/s]
rho_p  = 2650.0            # particle density [kg/m^3]   

# Particle mass
mp = rho_p * (np.pi/6.0) * d**3         
thetaE = np.deg2rad(25)

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
    Pr = 0.74*np.exp(-4.46*np.exp(-0.1*Uinc/const))
    # if Pr <0 or Pr>1:
    #     print('Warning: Pr is not between 0 and 1')
    return Pr

def e_COR_from_Uim(Uim, Omega):
    # return 3.05 * (abs(Uim)/const + 1e-12)**(-0.47)   
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
    # if theta_inc < 0 or theta_inc > np.pi / 2: 
    #     print('The value of theta_inc is not logical')                                     
    return theta_inc

def theta_reb_from_Uim(Uim, Omega):
    x = (-0.0032*Omega**0.39)*(abs(Uim)/const) + 0.43 
    theta_re = np.arcsin(x)
    if theta_re < 0 or theta_re > np.pi / 2: 
        print('The value of theta_re is not logical')                              
    return theta_re

def NE_from_Uinc(Uinc, Omega): 
    return (0.03-0.025*Omega**0.21) * (abs(Uinc)/const) 

def UE_from_Uinc(Uinc, Omega):
    A = -1.51*Omega + 4.62
    if Omega>0:
        B = 0.56*Omega+0.15
    else:
        B = 0
    U_E = A*(abs(Uinc)/const)**B * const
    return U_E * np.sign(Uinc)  

def tau_top(u_star):                                                 
    return rho_a * u_star**2     

def calc_Mdrag(c, Uair, U, Omega, Cref, b):
    b_Urel = 1/(1 + c/Cref)
    Urel = b_Urel * (b*Uair - U)
    Re = abs(Urel)*d/nu_a
    Ruc = 24
    Cd_inf = 0.5
    Cd = (np.sqrt(Cd_inf)+np.sqrt(Ruc/Re))**2
    Agrain = np.pi*(d/2)**2
    Ngrains = c/mp
    Mdrag = 0.5*rho_a*Urel*abs(Urel)*Cd*Agrain*Ngrains
    return Mdrag

def MD_eff(Ua, U, c, Omega):
    Mdrag = calc_Mdrag(c, Ua, U, Omega)
    CD_bed = 0.0037
    B, p = 3.5, 7.0
    tau_basic = 0.5 * rho_a * CD_bed * Ua * abs(Ua)
    M_bed = tau_basic * (1/(1+(B*Mdrag)**p)) # to guarantee that there is a balancing term with tau_top when c=0
    M_final = Mdrag + M_bed
    return M_final

def rhs_cmUa(t, y, u_star, Omega, Cref, b):
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
    M_drag = calc_Mdrag(c, Ua, U, Omega, Cref, b) 
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
    dUa_dt    = (tau_top(u_star) - MD_eff(Ua, U, c, Omega)) / m_air_eff

    return [dc_dt, dm_dt, dUa_dt]

def euler_forward(rhs, y0, t_span, dt, u_star, Omega, Cref, b):
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
        f = rhs_cmUa(tk, yk, u_star, Omega, Cref, b)
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
    for i in range(2, 7):
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

def simulate_model(Cref, b):
    Omega_list = [0.0, 0.01, 0.05, 0.10, 0.20]

    model_output = {}

    for i, Omega in enumerate(Omega_list):
        y0 = (C_dpm[i][0], C_dpm[i][0]*U_dpm[i][0], Ua_dpm[i][0])
        t, Y = euler_forward(rhs_cmUa, y0, (0.0, 5.0), 1e-2, u_star, Omega, Cref, b)

        c  = Y[0]
        m  = Y[1]
        Ua = Y[2]
        U  = m / np.maximum(c, 1e-16)

        model_output[Omega] = {
            't': t,
            'c': c,
            'U': U,
            'Ua': Ua
        }

    return model_output

def cost_function(params):
    Cref, b = params
    model = simulate_model(Cref, b)

    cost = 0.0
    
    for i, Omega in enumerate([0.0, 0.01, 0.05, 0.10, 0.20]):
        # measured
        C_meas = C_dpm[i]
        U_meas = U_dpm[i]
        Ua_meas = Ua_dpm[i]

        # model
        c_mod  = model[Omega]['c'][:len(C_meas)]
        U_mod  = model[Omega]['U'][:len(U_meas)]
        Ua_mod = model[Omega]['Ua'][:len(Ua_meas)]

        # squared error
        cost += np.sum((c_mod - C_meas)**2)
        cost += np.sum((U_mod - U_meas)**2)
        cost += np.sum((Ua_mod - Ua_meas)**2)

    return cost

# 参数的初始猜测
x0 = [0.02, 0.6]   # example: Cref=0.02, b=0.6

# 给参数加约束 (Cref>0, 0<b<1)
bounds = [(1e-6, None), (0.1, 1.0)]

res = minimize(cost_function, x0, bounds=bounds, method='L-BFGS-B')

print("Optimized parameters:")
print("Cref =", res.x[0])
print("b    =", res.x[1])
