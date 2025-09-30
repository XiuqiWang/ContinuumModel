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
h = 0.197                 # domain height [m]
u_star = 0.56              # shear velocity [m/s]

rho_a  = 1.225             # air density [kg/m^3] 
nu_a   = 1.46e-5           # air kinematic viscosity [m^2/s]
rho_p  = 2650.0            # particle density [kg/m^3]
# phi_bed = 0.64             # bed volume fraction           

# Effective velocity profile constants 
# hs = 6.514e-4               
# z0 = 2.386e-6             

# Bed shear closure 
# l_eff_bed = 0.0319        

# Particle mass
mp = rho_p * (np.pi/6.0) * d**3

# Ejection speed magnitude from PDF
UE_mag = 4.53 * const     
thetaE_deg = 40.0         
cos_thetaE = np.cos(np.deg2rad(thetaE_deg))

# Moisture (Ω) for U_im mapping; affects only U_im per PDF
Omega = 0.0  

# --------------tuning parameters--------------
alpha_Eim=1
alpha_ED=1
gamma_drag = 1    # try 1.5–3.0 after fixing mass

# -------------------- closures from the PDF --------------------
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
    return 0.07 * (abs(UD))**0.66             

def Preff_of_U(U):
    """State-conditioned rebound fraction. PDF marks 'needs calibration'—placeholder Gompertz-like."""
    # P_min + (P_max-P_min) * (1 - exp(-(U/Uc)^p))
    # Pmin, Pmax, Uc, p = 0, 0.999, 10.038, 0.429   
    Pmin, Pmax, Uc, p = 0, 0.59, 2.09, 0.74   # tuned
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
    # return 3.05 * (abs(Uim)/const + 1e-12)**(-0.47)       
    return 3.09 * (abs(Uim)/const + 1e-12)**(-0.47)                  

def theta_im_from_Uim(Uim):
    x = 50.40 / (abs(Uim)/const + 159.33)                            
    return np.arcsin(np.clip(x, -1.0, 1.0))

def theta_D_from_UD(UD):
    x = 163.68 / (np.abs(UD) / const + 156.65)
    x = np.clip(x, 0.0, 1.0)
    theta = 0.28 * np.arcsin(x)
    return np.clip(theta, 0.0, 0.5 * np.pi)                   

def theta_reb_from_Uim(Uim):
    x = -0.0003*(abs(Uim)/const) + 0.52                              
    return np.arcsin(np.clip(x, -1.0, 1.0))

def NE_from_Uinc(Uinc):
    # return 0.04 * (abs(Uinc)/const)     
    return 0.063 * (abs(Uinc)/const)                                  

# def Ueff_from_Uair(Uair):
#     """PDF effective air speed in the saltation layer (algebraic log profile)."""
#     return Uair * (np.log(hs/z0) - 1.0) / (np.log(h/z0) - 1.0)       

def tau_top(u_star):                                                 
    return rho_a * u_star**2                                        

# def tau_bed(Uair, U, c):                    
#     # (1-phi_bed) * rho_a * (nu_a + l_eff_bed**2 * abs(Uair/h)) * (Uair/h)   
#     K = 0.08
#     beta = 0.9
#     U_eff = beta*Uair
#     tau_bed_minusphib = rho_a*K*c/(rho_p*d) * abs(U_eff - U)*(U_eff - U)                            
#     return tau_bed_minusphib

def Mdrag(c, Uair, U):
    """Momentum exchange (air→saltation): c * f_drag / m_p."""
    b = 0.58
    C_D = 0.18
    Ueff = b * Uair
    dU = Ueff - U
    fdrag = np.pi/8 * d**2 * rho_a * C_D * abs(dU) * dU                     
    return (c * fdrag) / mp

def MD_eff(Ua, U, c):
    Uaeff = 0.87*Ua
    MD_eff = rho_a * 0.19 * c/(rho_p*d) *abs(Uaeff - U) * (Uaeff - U)
    return MD_eff

# --- RHS in (c, m=cU, Ua) ---
def rhs_cmUa(t, y, u_star, eps=1e-4):
    c, m, Ua = y
    c = max(c, 0.0)                         # clip for safety
    U = np.clip(m / max(c, 1e-4), 0.0, 20.0)     # recover U

    Uim, UD = Uim_from_U(U), UD_from_U(U)
    Tim, TD  = max(1e-9, Tim_from_Uim(Uim)), max(1e-9, TD_from_UD(UD))
    Pr = np.clip(Preff_of_U(U), 0.0, 1.0)

    # mixing fractions and rates
    phi_im = (Pr*Tim) / (Pr*Tim + (1.0-Pr)*TD + 1e-12)
    cim, cD = c*phi_im, c*(1.0-phi_im)
    r_im, r_dep = cim/Tim, cD/TD

    # ejection numbers and rebound kinematics
    NEim, NEd = NE_from_Uinc(Uim), NE_from_Uinc(UD)
    eCOR = e_COR_from_Uim(Uim); Ure = Uim*eCOR
    th_im, th_D, th_re = theta_im_from_Uim(Uim), theta_D_from_UD(UD), theta_reb_from_Uim(Uim)

    # scalar sources
    E = alpha_Eim*r_im*NEim + alpha_ED*r_dep*NEd
    D = r_dep

    # momentum sources (streamwise)
    M_drag = gamma_drag * Mdrag(c, Ua, U)
    M_eje  = (alpha_Eim*r_im*NEim + alpha_ED*r_dep*NEd) * UE_mag * cos_thetaE
    M_re   = r_im * ( Ure*np.cos(th_re) )
    M_im   = r_im * ( Uim*np.cos(th_im) )
    M_dep  = r_dep* ( UD *np.cos(th_D ) )

    # ODEs
    dc_dt = E - D
    dm_dt = M_drag + M_eje + M_re - M_im - M_dep

    chi       = max(1e-6, 1.0 - c/(rho_p*h))
    m_air_eff = rho_a*h*chi
    dUa_dt    = (tau_top(u_star) - MD_eff(Ua, U, c)) / m_air_eff

    return [dc_dt, dm_dt, dUa_dt]


data = np.loadtxt('CGdata/hb=12d/Shields006dry.txt')
Q_dpm = data[:, 0]
C_dpm = data[:, 1]
U_dpm = data[:, 2]
t_dpm = np.linspace(0,5,501)
data_ua = np.loadtxt('TotalDragForce/Uair_ave-tS006Dryh02.txt', delimiter='\t')
Ua_dpm = data_ua[0:,1]  
# --- integrate with an implicit method (stiff & steady-seeking) ---
y0 = (C_dpm[0], C_dpm[0]*U_dpm[0], Ua_dpm[0])
sol = solve_ivp(lambda t,y: rhs_cmUa(t,y,u_star),
                (0, 5.0), y0, method='BDF', rtol=1e-6, atol=1e-8, max_step=5e-3)

t  = sol.t
c  = sol.y[0]
m  = sol.y[1]
Ua = sol.y[2]
U  = m/np.maximum(c,1e-12)

# quick sanity check against your steady balances:
print("final ~steady:",
      "c =", c[-1], "U =", U[-1], "Ua =", Ua[-1])  

# 
# ylabels = ['Q [kg/m/s]', 'C [kg/m^2]', 'U [m/s]', 'U_air [m/s]']
# keys_dpm = [Q_dpm, C_dpm, U_dpm, Ua_dpm]
# keys_continuum = []
# keys_continuum.append([c*U, c, U, Uair]) #Q, C, U, Ua
# fig, axs = plt.subplots(2, 2, figsize=(8, 8))
# axs = axs.flatten()
# # Plot in a loop
# for i, ax in enumerate(axs):
#     ax.plot(t_dpm, keys_dpm[i], color='k', label='DPM simulation')
#     ax.plot(t, keys_continuum[0][i], label='Continuum')
#     ax.set_xlabel('Time [s]')
#     ax.set_ylabel(ylabels[i])
#     ax.set_xlim(left=0)
#     # ax.set_ylim(bottom=0)
#     ax.grid(True)
#     if i == 0:
#         ax.legend()
# fig.suptitle(r'$\Theta$=0.06')
# plt.tight_layout()
# plt.show()
plt.close('all')
plt.figure(figsize=(8,6))
plt.subplot(3,1,1); plt.plot(t,c);  plt.plot(t_dpm,C_dpm); plt.ylabel('c')
plt.subplot(3,1,2); plt.plot(t,U);  plt.plot(t_dpm,U_dpm); plt.ylabel('U')
plt.subplot(3,1,3); plt.plot(t,Ua); plt.plot(t_dpm,Ua_dpm); plt.ylabel('Ua'); plt.xlabel('t [s]')
plt.tight_layout(); plt.show()

