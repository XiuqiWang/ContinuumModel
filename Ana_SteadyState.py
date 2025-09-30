# -*- coding: utf-8 -*-
"""
Created on Mon Sep 15 10:22:12 2025

@author: WangX3
"""

from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.optimize import lsq_linear

# -------------------- constants & inputs --------------------
g = 9.81
d = 0.00025                # grain diameter [m]  
const = np.sqrt(g*d)       # √(g d)
h = 0.197                 # domain height [m]
Shields = np.linspace(0.02, 0.06, 5)
u_star = np.sqrt(Shields * (2650-1.225)*9.81*d/1.225)

rho_a  = 1.225             # air density [kg/m^3] 
nu_a   = 1.46e-5           # air kinematic viscosity [m^2/s]
rho_p  = 2650.0            # particle density [kg/m^3]           

# Particle mass
mp = rho_p * (np.pi/6.0) * d**3

# Ejection speed magnitude from PDF
UE_mag = 4.53 * const     
thetaE_deg = 40.0         
cos_thetaE = np.cos(np.deg2rad(thetaE_deg))

# Moisture (Ω) for U_im mapping; affects only U_im per PDF
Omega = 0.0  

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
    return 0.06 * (abs(UD))**0.66         

def theta_im_from_Uim(Uim):
    x = 50.40 / (abs(Uim)/const + 159.33)                            
    return np.arcsin(np.clip(x, -1.0, 1.0))

def theta_D_from_UD(UD):
    x = 163.68 / (abs(UD)/const + 156.65)                             
    return 0.28 * np.arcsin(np.clip(x, -1.0, 1.0))                   

def theta_reb_from_Uim(Uim):
    x = -0.0003*(abs(Uim)/const) + 0.52                              
    return np.arcsin(np.clip(x, -1.0, 1.0))

def Mdrag(c, Uair, U):
    """Momentum exchange (air→saltation): c * f_drag / m_p."""
    b = 0.55
    C_D = 0.1037
    Ueff = b * Uair
    dU = Ueff - U
    fdrag = np.pi/8 * rho_a * d**2 * C_D * abs(dU) * dU                     
    return (c * fdrag) / mp       

def tau_bed(Uair, U, c):                    
    K = 0.08
    beta = 0.9
    U_eff = beta*Uair
    tau_bed_minusphib = rho_a*K*c/(rho_p*d) * abs(U_eff - U)*(U_eff - U)                            
    return tau_bed_minusphib

def MD_eff(Ua, U, c):
    Uaeff = 0.85*Ua
    MD_eff = rho_a * 0.12 * c/(rho_p*d) *abs(Uaeff - U) * (Uaeff - U)
    return MD_eff

# tuning functions
def Preff_of_U(U, Pmax, Uc, p):
    """State-conditioned rebound fraction. PDF marks 'needs calibration'—placeholder Gompertz-like."""
    # P_min + (P_max-P_min) * (1 - exp(-(U/Uc)^p))
    # Pmin, Pmax, Uc, p = 0, 0.6, 2, 0.5 #1 #3.84, 0.76      # tune to your binned counts
    Pmin = 0
    U = abs(U)
    return Pmin + (Pmax - Pmin)*(1.0 - np.exp(-(U/max(Uc,1e-6))**p)) 

def e_COR_from_Uim(Uim):
    return 3.05 * (abs(Uim)/const + 1e-12)**(-0.47)  

def NE_from_Uinc(Uinc):
    NE = 0.04 * (abs(Uinc)/const)
    return NE             

# helper function
def Cal_EDM(U):
    # streamwise momentum terms
    Uim = Uim_from_U(U);    UD  = UD_from_U(U)
    if np.ndim(U) != 0:
        Tim = np.maximum(np.ones_like(U)*1e-6, Tim_from_Uim(Uim))
        TD  = np.maximum(np.ones_like(U)*1e-6, TD_from_UD(UD))
        NEim   = np.maximum(np.zeros_like(U), NE_from_Uinc(Uim))
        NEd    = np.maximum(np.zeros_like(U), NE_from_Uinc(UD))
    else:
        Tim = max(1e-6, Tim_from_Uim(Uim))
        TD  = max(1e-6, TD_from_UD(UD))
        NEim   = max(0.0, NE_from_Uinc(Uim))
        NEd    = max(0.0, NE_from_Uinc(UD))
    P   = np.clip(Preff_of_U(U, 1, 3.84, 0.76), 0.0, 1.0)
    # angles/COR (your closures)
    th_im = theta_im_from_Uim(Uim)
    th_D  = theta_D_from_UD(UD)
    th_re = theta_reb_from_Uim(Uim)
    eCOR  = e_COR_from_Uim(Uim)
    Ure   = Uim * eCOR
    
    phi_im = P * Tim / (P*Tim + (1.0-P)*TD + 1e-12)
    phi_D  = 1-phi_im
    E_overc = phi_im/Tim*NEim + phi_D/TD*NEd   
    D_overc = phi_D/TD
    M_eje_overc = E_overc * UE_mag * cos_thetaE
    M_re_overc  = phi_im/Tim  * ( Ure * np.cos(th_re) )
    M_im_overc  = phi_im/Tim  * ( Uim * np.cos(th_im) )
    M_dep_overc = D_overc * ( UD  * np.cos(th_D)  )
    M_bed = M_eje_overc + M_im_overc - M_re_overc - M_dep_overc
    
    return E_overc, D_overc , M_bed

#----------- DPM data ----------
cases = range(2, 7)  # S002..S006
C_list, U_list, Ua_list, E_list, D_list = [], [], [], [], []
t_dpm = np.linspace(0.01, 5, 501)
for i in cases:
    # CG data: columns -> Q, C, U (501 rows)
    cg = np.loadtxt(f"CGdata/hb=12d/Shields{i:03d}dry.txt")
    C_list.append(cg[:, 1])
    U_list.append(cg[:, 2])
    # depth-averaged air speed Ua (tab-separated, second column)
    ua = np.loadtxt(f"TotalDragForce/Uair_ave-tS{i:03d}Dryh02.txt", delimiter="\t")
    Ua_list.append(ua[:, 1])
    # E and D
    ED = np.loadtxt(f"dcdt/S{i:03d}EandD.txt")
    E_list.append(ED[:,0])
    D_list.append(ED[:,1])
t_Dis = np.linspace(5/len(E_list[0]), 5, len(E_list[0]))
# make E and D same length as U and C
E_on_U = np.interp(t_dpm, t_Dis, E_list[4])
D_on_U = np.interp(t_dpm, t_Dis, D_list[4])

Us_dpm = np.array([a[int(0.6*len(a)):].mean() for a in U_list])
cs_dpm = np.array([a[int(0.6*len(a)):].mean() for a in C_list])
Uas_dpm = np.array([a[int(0.6*len(a)):].mean() for a in Ua_list])

# ------- proposed --------
U_com = np.linspace(0.01, 2, 100)
Ua_com = np.linspace(5, 20, 100)
c_com = np.linspace(0, 0.1, 100)
Usteady, Uasteady, csteady = np.zeros(5), np.zeros(5), np.zeros(5)
for i in range(5):
    # Steady U
    E_overc, D_overc, _ = Cal_EDM(U_com)
    Usteady[i] = np.interp(0, E_overc - D_overc, U_com)
    
    # Steady Ua
    dU_com = 0.55*Ua_com - Usteady[i]
    _,_,M_bed_steady = Cal_EDM(Usteady[i])
    M_drag_overc = np.pi/8 * d**2 * 0.1037 * abs(dU_com) * dU_com / mp      
    Uasteady[i] = np.interp(M_bed_steady, M_drag_overc, Ua_com)
    
    # Steady c
    M_drag_com = Mdrag(c_com, Uasteady[i], Usteady[i])
    # tau_bed_com = tau_bed(Uasteady[i], Usteady[i], c_com)
    Mdrag_eff_com = MD_eff(Uasteady[i], Usteady[i], c_com)
    tau_top = rho_a * u_star[i] **2
    csteady[i] = np.interp(tau_top, Mdrag_eff_com, c_com)

plt.close('all')
plt.figure(figsize=(15,4))
plt.subplot(1,3,1)
plt.plot(Shields, Usteady, label='computed')
plt.scatter(Shields, Us_dpm, label='DPM')
plt.ylim(bottom=0)
plt.xlabel('Shields number')
plt.ylabel('Usteady')
plt.legend()
plt.subplot(1,3,2)
plt.plot(Shields, Uasteady, label='computed')
plt.scatter(Shields, Uas_dpm, label='DPM')
plt.xlabel('Shields number')
plt.ylabel('Uasteady')
plt.legend()
plt.subplot(1,3,3)
plt.plot(Shields, csteady, label='computed')
plt.scatter(Shields, cs_dpm, label='DPM')
plt.xlabel('Shields number')
plt.ylabel('c_steady')
plt.legend()


U_emp = U_list[4]
C_emp = C_list[4]
E_emp_overc = E_on_U/C_emp
D_emp_overc = D_on_U/C_emp

plt.figure(figsize=(15,4))
plt.subplot(1,3,1)
plt.plot(U_com, E_overc, color='C0', label='E/c')
plt.plot(U_com, D_overc, color='C1', label='D/c')
plt.plot(U_emp, E_emp_overc, 'C0.', label='E/c DPM')
plt.plot(U_emp, D_emp_overc, 'C1.', label='D/c DPM')
plt.xlabel('U [m/s]')
plt.ylabel('E/c & D/c')
plt.title(f'Computed Usteady = {Usteady[4]:.2f} m/s')
plt.legend()
plt.subplot(1,3,2)
plt.plot(dU_com, M_drag_overc, color='C0', label='M_drag/c')
plt.plot(dU_com, M_bed_steady*np.ones_like(dU_com), color='C1', label='M_bed/c')
# plt.plot(U_emp, E_emp_overc, 'C0.', label='E/c DPM')
# plt.plot(U_emp, D_emp_overc, 'C1.', label='D/c DPM')
plt.xlabel('0.55Ua - U [m/s]')
plt.ylabel('M/c')
plt.title(f'Computed Uasteady = {Uasteady[4]:.2f} m/s')
plt.legend()
plt.subplot(1,3,3)
plt.plot(c_com, Mdrag_eff_com, color='C0', label=r'$M_{D,eff}$')
plt.plot(c_com, tau_top*np.ones_like(c_com), color='C1', label='tau_top')
plt.xlabel(r'$c$ [kg/m$^2$]')
plt.ylabel('Source/sink in air equation')
plt.title(rf'Computed c_steady = {csteady[4]:.4f} kg/m$^2$')
plt.legend()
plt.suptitle('Shields=0.06')