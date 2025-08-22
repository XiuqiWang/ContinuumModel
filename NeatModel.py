# -*- coding: utf-8 -*-
"""
Created on Fri Aug 22 15:45:16 2025

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
u_star = 0.56              # shear velocity [m/s]

rho_a  = 1.225             # air density [kg/m^3] 
nu_a   = 1.46e-5           # air kinematic viscosity [m^2/s]
rho_p  = 2650.0            # particle density [kg/m^3]
phi_bed = 0.64             # bed volume fraction 

# Drag calibration from DPM (single-particle drag law)
k_drag = 3.22e-9          
n_drag = 1.19             

# Effective velocity profile constants 
hs = 6.514e-4               
z0 = 2.386e-6             

# Bed shear closure 
l_eff_bed = 0.0319        

# Particle mass
mp = rho_p * (np.pi/6.0) * d**3

# Ejection speed magnitude from PDF
UE_mag = 5.02 * const     
thetaE_deg = 40.0         
cos_thetaE = np.cos(np.deg2rad(thetaE_deg))

# Moisture (Ω) for U_im mapping; affects only U_im per PDF
Omega = 0.0  

# --------------tuning parameters--------------
alpha_Eim=1.8
alpha_ED=1.0
gamma_drag = 2    # try 1.5–3.0 after fixing mass

# -------------------- closures from the PDF --------------------
def Uim_from_U(U):
    """U_im from instantaneous saltation-layer velocity U (includes Ω effect)."""
    Uabs_nd = abs(U)/const
    expo = Omega*0.54 + 0.55
    bump = 1567.94*Omega*np.exp(-Omega/0.02)
    Uim_nd = (Uabs_nd**expo) + 26.15 + bump  
    return np.sign(U) * (Uim_nd*const)

def UD_from_U(U):
    """U_D from U. PDF only provides Dry (Ω=0). We use Dry law for all Ω unless you add a wet fit."""
    Uabs_nd = abs(U)/const
    UD_nd = (Uabs_nd**0.41) + 2.76           
    return np.sign(U) * (UD_nd*const)

def Tim_from_Uim(Uim):
    return 0.04 * (abs(Uim))**0.84           

def TD_from_UD(UD):
    return 0.07 * (abs(UD))**0.66             

def Preff_of_U(U):
    """State-conditioned rebound fraction. PDF marks 'needs calibration'—placeholder Gompertz-like."""
    # P_min + (P_max-P_min) * (1 - exp(-(U/Uc)^p))
    Pmin, Pmax, Uc, p = 0.30, 0.9, 0.5, 1   # tune to your binned counts
    U = abs(U)
    return Pmin + (Pmax - Pmin)*(1.0 - np.exp(-(U/max(Uc,1e-6))**p))  

# def Pr_neutral(U):
#     Uim, UD = Uim_from_U(U), UD_from_U(U)
#     return (1.0 - NE_from_Uinc(UD)) / (NE_from_Uinc(Uim) + 1.0 - NE_from_Uinc(UD))

# # pre-run diagnostic
# Ugrid = np.linspace(0.0, 2.0, 200)        # pick a range covering your U
# Pr_star = np.clip([Pr_neutral(U) for U in Ugrid], 0, 1)
# Pr_fit  = np.clip([Preff_of_U(U) for U in Ugrid], 0, 1)

# plt.figure()
# plt.plot(Ugrid, Pr_star, label=r"$P_r^\star(U)$ (neutral)")
# plt.plot(Ugrid, Pr_fit,  label=r"$P_r^{\rm eff}(U)$ (your fit)")
# plt.axvline(0.55, ls="--", c="k", alpha=0.5, label="U₀")
# plt.ylim(0,1); plt.xlabel("U [m/s]"); plt.ylabel("Pr")
# plt.legend(); plt.tight_layout()

def e_COR_from_Uim(Uim):
    return 3.05 * (abs(Uim)/const + 1e-12)**(-0.47)                  

def theta_im_from_Uim(Uim):
    x = 50.40 / (abs(Uim)/const + 159.33)                            
    return np.arcsin(np.clip(x, -1.0, 1.0))

def theta_D_from_UD(UD):
    x = 96.36 / (abs(UD)/const + 83.42)                             
    return 0.33 * np.arcsin(np.clip(x, -1.0, 1.0))                   

def theta_reb_from_Uim(Uim):
    x = -0.0003*(abs(Uim)/const) + 0.52                              
    return np.arcsin(np.clip(x, -1.0, 1.0))

def NE_from_Uinc(Uinc):
    return 0.04 * (abs(Uinc)/const)                                  

def Ueff_from_Uair(Uair):
    """PDF effective air speed in the saltation layer (algebraic log profile)."""
    return Uair * (np.log(hs/z0) - 1.0) / (np.log(h/z0) - 1.0)       

def tau_top(u_star):                                                 
    return rho_a * u_star**2                                        

def tau_bed(Uair):                                                   
    return rho_a * (nu_a + l_eff_bed**2 * abs(Uair/h)) * (Uair/h)   

def Mdrag(c, Uair, U):
    """Momentum exchange (air→saltation): c * f_drag / m_p."""
    Ueff = Ueff_from_Uair(Uair)
    dU = Ueff - U
    fdrag = k_drag * (abs(dU)**(n_drag-1)) * dU                     
    return (c * fdrag) / mp

# -------------------- ODE (y = [c, U, U_air]) --------------------
def rhs(t, y, u_star, c_floor=1e-3, U_switch=1e-3):
    c, U, Uair = y

    # direction (smooth sign):
    dir = np.tanh(U / max(U_switch,1e-9))

    # incident speeds, angles, COR
    Uim = Uim_from_U(U); UD = UD_from_U(U)
    th_im = theta_im_from_Uim(Uim)
    th_D  = theta_D_from_UD(UD)
    th_re = theta_reb_from_Uim(Uim)
    eCOR  = e_COR_from_Uim(Uim)
    Ure   = Uim * eCOR

    # timescales
    Tim = Tim_from_Uim(Uim)
    TD  = TD_from_UD(UD)

    # split c into cim, cD using Preff(U)
    Pr = np.clip(Preff_of_U(U), 0.0, 1.0)
    phi_im = (Pr*Tim) / (Pr*Tim + (1.0-Pr)*TD + 1e-12)
    cim = c * phi_im
    cD  = c - cim

    # rates
    r_im  = cim / Tim
    r_dep = cD  / TD

    # ejection mass & deposition mass (E and D)
    E = alpha_Eim * r_im*NE_from_Uinc(Uim) + alpha_ED * r_dep*NE_from_Uinc(UD)              
    D = r_dep
    
    # inside rhs after computing NE_im, NE_D, Pr:
    ED_ratio = E/D
    print('ED_ratio', ED_ratio)

    # momentum sources (streamwise; drag already has its sign)
    M_drag = gamma_drag * Mdrag(c, Uair, U)
    M_eje  = (alpha_Eim * r_im*NE_from_Uinc(Uim)*UE_mag*cos_thetaE + alpha_ED * r_dep*NE_from_Uinc(UD)*UE_mag*cos_thetaE ) * dir
    M_re   = r_im  * ( Ure * np.cos(th_re) ) * dir
    M_im   = r_im  * ( Uim * np.cos(th_im) ) * dir
    M_dep  = r_dep * ( UD  * np.cos(th_D)  ) * dir

    # mass eqn
    dc_dt = E - D

    # momentum balance for U (no division by tiny c)
    S_U   = M_drag + M_eje + M_re - M_im - M_dep
    c_den = max(c, c_floor)
    dU_dt = (S_U / c_den) - (U * dc_dt / c_den)

    # air momentum
    chi        = max(1e-6, 1.0 - c/(rho_p*h))                         # (1 - c/(h ρp))
    m_air_eff  = rho_a * h * chi
    dUair_dt   = (tau_top(u_star) - (1.0 - phi_bed)*tau_bed(Uair) - M_drag) / m_air_eff

    return [dc_dt, dU_dt, dUair_dt]

# -------------------- runner --------------------
def run_continuum(T=5.0, y0=(0.147, 0.55, 12.99), method="Radau"):
    sol = solve_ivp(lambda t, y: rhs(t, y, u_star),
                    (0.0, T), y0, method=method,
                    rtol=1e-6, atol=[1e-9, 1e-8, 1e-8], max_step=1e-2)
    return sol

# Example usage:
sol = run_continuum(T=5.0, y0=(0.147, 0.55, 12.99))
t, c, U, Uair = sol.t, sol.y[0], sol.y[1], sol.y[2]

data = np.loadtxt('CGdata/Shields006dry.txt')
Q_dpm = data[:, 0]
C_dpm = data[:, 1]
U_dpm = data[:, 2]
t_dpm = np.linspace(0,5,501)
data_ua = np.loadtxt('TotalDragForce/Uair_ave-tS006Dryh02.txt', delimiter='\t')
Ua_dpm = data_ua[0:,1]    

plt.close('all')
ylabels = ['Q [kg/m/s]', 'C [kg/m^2]', 'U_sal [m/s]', 'U_air [m/s]']
keys_dpm = [Q_dpm, C_dpm, U_dpm, Ua_dpm]
keys_continuum = []
keys_continuum.append([c*U, c, U, Uair]) #Q, C, U, Ua
fig, axs = plt.subplots(2, 2, figsize=(8, 8))
axs = axs.flatten()
# Plot in a loop
for i, ax in enumerate(axs):
    ax.plot(t_dpm, keys_dpm[i], color='k', label='DPM simulation')
    ax.plot(t, keys_continuum[0][i], label='Continuum')
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
