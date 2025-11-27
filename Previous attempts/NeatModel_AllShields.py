# -*- coding: utf-8 -*-
"""
Created on Wed Oct  8 14:35:28 2025

@author: WangX3
"""

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

rho_a  = 1.225             # air density [kg/m^3] 
nu_a   = 1.46e-5           # air kinematic viscosity [m^2/s]
rho_p  = 2650.0            # particle density [kg/m^3]   

# Particle mass
mp = rho_p * (np.pi/6.0) * d**3

# Shields number and shear velocity
Shields_all = np.linspace(0.02, 0.06, 5)
u_star_all  = np.sqrt(Shields_all * (rho_p - rho_a) * g * d / rho_a)

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

def calc_T_jump_ballistic_assumption(Uinc, theta_inc_degree):
    Uy0 = Uinc*np.sin(theta_inc_degree/180*np.pi)
    Tjump = 2*Uy0/g
    return Tjump 
    
def calc_T_jump_Test(Uinc, theta_inc_degree, mode):
    Tjump_ballistic = calc_T_jump_ballistic_assumption(Uinc, theta_inc_degree)
    # branch based on mode string (case-insensitive)
    mode = mode.lower()
    if mode == "im":
        Tjump_Xiuqi = Tim_from_Uim(Uinc)
    elif mode == "de":
        Tjump_Xiuqi = TD_from_UD(Uinc)
    else:
        raise ValueError(f"Invalid mode '{mode}'. Expected 'im' or 'de'.")
    
    Tjump  = 0.5*Tjump_Xiuqi + 0.5*Tjump_ballistic
    # print('Tjump_Xiuqi', Tjump_Xiuqi, 'Tjump_ballistic', Tjump_ballistic, 'Tjump', Tjump)
    return Tjump

def calc_N_E_test(Uinc):
       sqrtgd = np.sqrt(g*d)
       # N_E = np.sqrt(Uinc/sqrtgd)*(0.04-0.04*Omega**0.23)*5
       # N_E = (1-(10*Omega+0.2)/(Uinc**2+(10*Omega+0.2)))*np.sqrt(Uinc/sqrtgd)*(0.04-0.04*Omega**0.23)*7
       p = 8
       ##
       p2 = 2
       A = 100
       Uinc_half_min = 1.0
       Uinc_half_max = 6
       Uinc_half = Uinc_half_min + (Uinc_half_max - Uinc_half_min)*(A*Omega)**p2/((A*Omega)**p2+1)
       ##
       #Uinc_half = 0.5+40*Omega**0.5
       B = 1/Uinc_half**p
       N_E = (1-1./(1.+B*Uinc**p))*2.5 * Uinc**(1/10)
       return N_E                              

def theta_im_from_Uim(Uim):
    x = 50.40 / (abs(Uim)/const + 159.33)                            
    return np.arcsin(np.clip(x, -1.0, 1.0))

def theta_D_from_UD(UD):
    x = 163.68 / (abs(UD) / const + 156.65)
    x = np.clip(x, 0.0, 1.0)
    theta = 0.28 * np.arcsin(x)
    return np.clip(theta, 0.0, 0.5 * np.pi)                   

def theta_reb_from_Uim(Uim):
    x = -0.0003*(abs(Uim)/const) + 0.52                              
    return np.arcsin(np.clip(x, -1.0, 1.0))

def tau_top(u_star):
    return rho_a * u_star**2

def calc_Mdrag(c, Uair, U):
    alpha_drag = 0.25 #0.32
    Ngrains = c/mp
    Ueff = alpha_drag*Uair
    Urel = Ueff-U
    Re = abs(Urel)*d/nu_a
    Ruc = 24
    Cd_inf = 0.5
    Cd = (np.sqrt(Cd_inf)+np.sqrt(Ruc/Re))**2
    
    Agrain = np.pi*(d/2)**2
    Mdrag = 0.5*rho_a*Urel*abs(Urel)*Cd*Agrain*Ngrains # drag term based on uniform velocity
    return Mdrag

def MD_eff(Ua, U, c):
    # _Pmax, _Uc, _pshape, _A_NE, B, p = params
    # alpha, K = 1.20, 0.040
    # Uaeff = alpha*Ua
    # MD_eff = rho_a * K * c/(rho_p*d) *abs(Uaeff - U) * (Uaeff - U)
    Mdrag = calc_Mdrag(c, Ua, U)
    CD_bed = 0.0037
    B, p = 4.46, 7.70
    tau_basic = 0.5 * rho_a * CD_bed * Ua * abs(Ua)
    M_bed = tau_basic * (1/(1+(B*Mdrag)**p)) # to guarantee that there is a balancing term with tau_top when c=0
    M_final = Mdrag + M_bed
    return M_final

# parameters optimized              
def Preff_of_U(U):
    """State-conditioned rebound fraction. PDF marks 'needs calibration'—placeholder Gompertz-like."""
    # P_min + (P_max-P_min) * (1 - exp(-(U/Uc)^p))
    # Pmin, Pmax, Uc, p = 0, 0.999, 10.038, 0.429 
    Pmin, Pmax, Uc, p = 0.0, 0.999, 3.84, 0.76 
    Pmin, Pmax, Uc, p = 0.0, 0.75, 2, 0.4  # try
    # Pmin, Pmax, Uc, p = 0, 0.5934, 2.0939, 0.7368   # tuned for steady solutions
    U = abs(U)
    return Pmin + (Pmax - Pmin)*(1.0 - np.exp(-(U/Uc)**p))  

def calc_Pr_Xiuqi_paper2(Usal):  # from Xiuqi paper 2
    Uinc = Usal / np.cos(15/180*np.pi)
    Pr = 0.94*np.exp(-7.12*np.exp(-0.1*Uinc/np.sqrt(g*d)))
    return Pr

def NE_from_Uinc(Uinc):
    # return 0.04 * (abs(Uinc)/const)     
    return 0.035 * (abs(Uinc)/const) # try
    # return 0.0635 * (abs(Uinc)/const)   

def calc_N_E_test3(Uinc):
        N_E_Xiuqi = NE_from_Uinc(Uinc)
        p = 8
        ##
        p2 = 2
        A = 100
        Uinc_half_min = 1.0
        Uinc_half_max = 2.0
        Uinc_half = Uinc_half_min + (Uinc_half_max - Uinc_half_min)*(A*Omega)**p2/((A*Omega)**p2+1)
        ##
        #Uinc_half = 0.5+40*Omega**0.5
        B = 1/Uinc_half**p
        N_E = (1-1./(1.+B*Uinc**p))*N_E_Xiuqi
        return N_E        

def e_COR_from_Uim(Uim):
    if abs(Uim) != 0 :
        e = 3.05 * (abs(Uim)/const + 1e-12)**(-0.47) 
    else:
        e = 0  
    # return 10.0 * (abs(Uim)/const + 1e-12)**(-0.74)   
    # return 3.0932 * (abs(Uim)/const + 1e-12)**(-0.4689) # tuned for steady solutions   
    return e                          

# def Ueff_from_Uair(Uair):
#     """PDF effective air speed in the saltation layer (algebraic log profile)."""
#     return Uair * (np.log(hs/z0) - 1.0) / (np.log(h/z0) - 1.0)                                            

# def tau_bed(Uair, U, c):                    
#     # (1-phi_bed) * rho_a * (nu_a + l_eff_bed**2 * abs(Uair/h)) * (Uair/h)   
#     K = 0.08
#     beta = 0.9
#     U_eff = beta*Uair
#     tau_bed_minusphib = rho_a*K*c/(rho_p*d) * abs(U_eff - U)*(U_eff - U)                            
#     return tau_bed_minusphib

# def Mdrag(c, Uair, U):
#     """Momentum exchange (air→saltation): c * f_drag / m_p."""
#     b = 0.55 #0.5842 # tuned for steady solutions         
#     C_D = 0.1037 #0.1776 # tuned for steady solutions         
#     Ueff = b * Uair
#     dU = Ueff - U
#     fdrag = np.pi/8 * d**2 * rho_a * C_D * abs(dU) * dU                     
#     return (c * fdrag) / mp

# Ua = 8; U = 1
# c = np.linspace(0, 1, 100)
# M, MD, Mtune = MD_eff(Ua, U, c)
# plt.figure()
# plt.plot(c, Mtune)
# plt.xlabel('c [kg/m^2]')
# plt.ylabel('M_tune [N/m^2]')

# --- Calculate rhs of the 3 equations ---
E_all, D_all = [], []
def rhs(t, y, eps, u_star):
    c, m, Ua = y
    U = m/(c + eps)

    Uim, UD = Uim_from_U(U), UD_from_U(U)
    Tim, TD  = calc_T_jump_ballistic_assumption(Uim, 15), calc_T_jump_ballistic_assumption(Uim, 15) # changed 
    Pr = Preff_of_U(U) # changed

    # mixing fractions and rates
    phi_im = (Pr*Tim) / (Pr*Tim + (1.0-Pr)*TD)
    cim, cD = c*phi_im, c*(1.0-phi_im)

    if cD<0:
            print('cD=',cD)
    r_im, r_dep = cim/Tim, cD/TD

    # ejection numbers and rebound kinematics
    NEim, NEd = calc_N_E_test3(Uim), calc_N_E_test3(UD) # changed
    eCOR = e_COR_from_Uim(Uim); Ure = Uim*eCOR
    th_im, th_D, th_re = theta_im_from_Uim(Uim), theta_D_from_UD(UD), theta_reb_from_Uim(Uim)

    # scalar sources
    E = r_im*NEim + r_dep*NEd
    D = r_dep
    
    if E<0:
            print('E = ',E)

    # momentum sources (streamwise)
    M_drag = calc_Mdrag(c, Ua, U) # changed
    M_eje  = (r_im*NEim + r_dep*NEd) * UE_mag * cos_thetaE
    M_re   = r_im * ( Ure*np.cos(th_re) )
    M_im   = r_im * ( Uim*np.cos(th_im) )
    M_dep  = r_dep* ( UD *np.cos(th_D ) )
    
    if M_dep > D*U:
            print('More momentum is leaving per particle by deposition, than that there is on average in the saltation layer')
            print('M_dep =',M_dep)
            print('D*U', D*U)
            print('D =',D)
            print('U =',U)
            print('UD =',UD)
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
    dUa_dt    = (tau_top(u_star) - MD_eff(Ua, U, c)) / m_air_eff

    return [dc_dt, dm_dt, dUa_dt]

# ---------- simple Euler–forward integrator ----------
def euler_forward(rhs, y0, t_span, dt, u_star):
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
        eps=1e-16
        f = rhs(tk, yk, eps, u_star)
        y_next = yk + dt * np.asarray(f, dtype=float)

        t[k+1]   = min(tk + dt, t1)
        y[:,k+1] = y_next

    return t, y

# -------------------- load DPM for a given Shields --------------------
def load_dpm_case(shields_value):
    # Map Shields to your file naming, e.g., 0.02 → S002, ..., 0.06 → S006
    label = int(round(shields_value*100))  # 2,3,4,5,6
    cg = np.loadtxt(f"CGdata/hb=12d/Shields{label:03d}dry.txt")
    C_dpm = cg[:,1]; U_dpm = cg[:,2]
    data_ua = np.loadtxt(f"TotalDragForce/Uair_ave-tS{label:03d}Dryh02.txt", delimiter="\t")
    Ua_dpm = data_ua[:,1]
    return C_dpm, U_dpm, Ua_dpm

# ---------- run it ----------
T0, T1 = 0.0, 5.0
dt     = 0.01  
t_dpm = np.linspace(0, 5, 501)
C_pred, U_pred, Ua_pred = [], [], []
DPM_C, DPM_U, DPM_Ua = [], [], []
for k, Theta in enumerate(Shields_all):
    ustar = u_star_all[k]
    C_dpm, U_dpm, Ua_dpm = load_dpm_case(Theta)
    y0 = (C_dpm[0], C_dpm[0]*U_dpm[0], Ua_dpm[0])  # per-case initial condition
    t, y_samp = euler_forward(rhs, y0, (0.0, T1), dt, ustar)
    c, U, Ua = y_samp[0], y_samp[1]/np.maximum(y_samp[0], 1e-16), y_samp[2]
    C_pred.append(c); U_pred.append(U); Ua_pred.append(Ua)
    DPM_C.append(C_dpm); DPM_U.append(U_dpm); DPM_Ua.append(Ua_dpm)

# ---------- plot ----------
plt.close('all')
plt.figure(figsize=(12, 9))
for k, Theta in enumerate(Shields_all):
    plt.subplot(3, 2, k + 1)
    plt.plot(t, C_pred[k], label='Continuum')
    plt.plot(t_dpm, DPM_C[k], '--', label='DPM')
    plt.title(f'Θ={Theta:.2f} Dry')
    plt.xlabel('t [s]')
    plt.ylabel(r'$C$ [kg/m$^2$]')
    plt.ylim(0, 0.35)
    plt.xlim(left=0)
    plt.grid(True)
    plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 9))
for k, Theta in enumerate(Shields_all):
    plt.subplot(3, 2, k + 1)
    plt.plot(t, U_pred[k], label='Continuum')
    plt.plot(t_dpm, DPM_U[k], '--', label='DPM')
    plt.title(f'Θ={Theta:.2f} Dry')
    plt.xlabel('t [s]')
    plt.ylabel(r'$U$ [m/s]')
    plt.ylim(0, 2.1)
    plt.xlim(left=0)
    plt.grid(True)
    plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 9))
for k, Theta in enumerate(Shields_all):
    plt.subplot(3, 2, k + 1)
    plt.plot(t, Ua_pred[k], label='Continuum')
    plt.plot(t_dpm, DPM_Ua[k], '--', label='DPM')
    plt.title(f'Θ={Theta:.2f} Dry')
    plt.xlabel('t [s]')
    plt.ylabel(r'$U_a$ [m/s]')
    plt.ylim(0,14)
    plt.xlim(left=0)
    plt.grid(True)
    plt.legend()
plt.tight_layout()
plt.show()

Shields = [0.06]
colors = plt.cm.RdBu(np.linspace(0, 1, 5))
plt.figure(figsize=(6,10))

# C
plt.subplot(3,1,1)
plt.plot(t_dpm, DPM_C[-1], '.', ms=2, alpha=0.7, label='DPM', color=colors[0])
plt.plot(t, C_pred[-1], '-', lw=1.5, alpha=0.9, label='Continuum', color=colors[0])
plt.xlabel('t [s]'); plt.ylabel('C [-]'); plt.grid(True)
plt.legend()
# U
plt.subplot(3,1,2)
plt.plot(t_dpm, DPM_U[-1], '.', ms=2, alpha=0.7, label=None, color=colors[0])
plt.plot(t, U_pred[-1], '-', lw=1.5, alpha=0.9, label=None, color=colors[0])
plt.xlabel('t [s]'); plt.ylabel('U [m/s]'); plt.grid(True)

# Ua
plt.subplot(3,1,3)
plt.plot(t_dpm, DPM_Ua[-1], '.', ms=2, alpha=0.7, label=None, color=colors[0])
plt.plot(t, Ua_pred[-1], '-', lw=1.5, alpha=0.9, label=None, color=colors[0])
plt.xlabel('t [s]'); plt.ylabel('Ua [m/s]'); plt.grid(True)

# simple legend (two entries) outside
lines = [plt.Line2D([0],[0], color='k', marker='.', linestyle='None', label='DPM'),
         plt.Line2D([0],[0], color='C0', linestyle='-', label='Model')]
plt.legend(handles=lines, loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=2)
plt.tight_layout()
plt.show()