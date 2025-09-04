# -*- coding: utf-8 -*-
"""
Created on Wed Sep  3 13:47:44 2025

@author: WangX3
"""
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# -------------------- constants & inputs --------------------
g = 9.81
d = 0.00025                # grain diameter [m]  
const = np.sqrt(g*d)       # √(g d)
h = 0.1975                 # domain height [m]
D = 0.00025                # particle size [m]
Shields = np.linspace(0.02, 0.06, 5) #Shields number
u_star = np.sqrt(Shields * (2650-1.225)*9.81*D/1.225) #shear velocity [m/s]

rho_a  = 1.225             # air density [kg/m^3] 
nu_a   = 1.46e-5           # air kinematic viscosity [m^2/s]
rho_p  = 2650.0            # particle density [kg/m^3]

# Drag calibration from DPM (single-particle drag law)
k_drag = 3.22e-9          
n_drag = 1.19             

# Particle mass
mp = rho_p * (np.pi/6.0) * d**3

def tau_top(u_star):                                                 
    return rho_a * u_star**2                                        

def tau_bed(Uair):                    
    # (1-phi_bed) * rho_a * (nu_a + l_eff_bed**2 * abs(Uair/h)) * (Uair/h)   
    CDbed, Ua_c, n = 0.13, 5, 1.75 
    tau_bed = CDbed*abs(Uair - Ua_c)**n * np.sign(Uair - Ua_c)                            
    return tau_bed

def Mdrag(c, Uair, U):
    """Momentum exchange (air→saltation): c * f_drag / m_p."""
    b = 0.4
    k_drag = 5e-9
    Ueff = b * Uair
    dU = Ueff - U
    fdrag = k_drag * abs(dU) * dU                     
    return (c * fdrag) / mp



dt = 0.01
# Explicit Euler
# for i in range(len(t_dpm)-1):
#     M_drag = Mdrag(C_dpm[i], Ua[i], U_dpm[i])
#     chi        = max(1e-6, 1.0 - C_dpm[i]/(rho_p*h))
#     m_air_eff  = rho_a * h * chi
#     dUair_dt   = (tau_top(u_star) - tau_bed(Ua[i]) - M_drag) / m_air_eff
#     Ua[i+1]    = Ua[i] + dUair_dt*dt

# linearly implicit Euler
def numdiff(f, x, h=1e-6):
    # central difference with step relative to magnitude of x
    h = max(h, 1e-6*(1.0 + abs(x)))
    return (f(x + h) - f(x - h)) / (2*h)

Ua_S, Ua_dpm_S = [], []
for i in range(2, 7):
    tauTop = tau_top(u_star[i-2])
    #----------- DPM data ----------
    file_c = f'CGdata/Shields00{i}dry.txt'
    data = np.loadtxt(file_c)
    Q_dpm = data[:, 0]
    C_dpm = data[:, 1]
    U_dpm = data[:, 2]
    t_dpm = np.linspace(0,5,501)
    file_ua = f'TotalDragForce/Uair_ave-tS00{i}Dryh02.txt'
    Ua_dpm = np.loadtxt(file_ua, delimiter='\t')[:, 1]
    Ua_dpm_S.append(Ua_dpm)
    
    Ua = np.zeros_like(Q_dpm)
    Ua[0] = Ua_dpm[0]
    for i in range(len(t_dpm)-1):
        C_i = C_dpm[i]
        U_i = U_dpm[i]
        Ua_i = Ua[i]
    
        # effective air mass in the layer
        chi       = max(1e-6, 1.0 - C_i/(rho_p*h))   # keep positive
        m_air_eff = rho_a * h * chi
    
        # values at Ua_i
        tb_i = tau_bed(Ua_i)
        Md_i = Mdrag(C_i, Ua_i, U_i)
    
        # Jacobians wrt Ua (finite-diff if analytic not available)
        Kb = numdiff(lambda x: tau_bed(x), Ua_i)
        Kd = numdiff(lambda x: Mdrag(C_i, x, U_i), Ua_i)
    
        # semi-implicit closed form update
        num = (m_air_eff/dt)*Ua_i + tauTop - tb_i - Md_i + (Kb + Kd)*Ua_i
        den = (m_air_eff/dt) + Kb + Kd
    
        # safety: avoid division by tiny den
        if den <= 1e-12:
            # fallback to explicit small step if pathological
            dUdt = (tauTop - tb_i - Md_i) / max(m_air_eff, 1e-12)
            Ua[i+1] = Ua_i + dt * dUdt
        else:
            Ua[i+1] = num / den
            
    Ua_S.append(Ua)  

    
plt.close('all')
plt.figure(figsize=(12, 10))
for i in range(5):
    plt.subplot(3, 2, i + 1)
    plt.plot(t_dpm, Ua_dpm_S[i], label='DPM')
    plt.plot(t_dpm, Ua_S[i], label='computed')
    plt.xlabel('t [s]')
    plt.ylabel(r'$U_\mathrm{a}$ [m/s]')
    plt.title(f'Shields=0.0{i+2}')
    plt.legend()
    plt.ylim(4,13)
    plt.xlim(0,5)
plt.tight_layout()