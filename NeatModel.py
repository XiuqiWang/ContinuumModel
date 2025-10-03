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

def calc_T_jump_ballistic_assumption(Usal):
    Uy0 = Usal*np.tan(15/180*np.pi)
    Tjump = 0.1+np.sqrt(4*Uy0/g) 
    return Tjump
    # Uy0 = Usal*np.tan(15/180*np.pi)
    # Tjump = 2*Uy0/g
    # return Tjump
    
def calc_T_jump_Xiuqi(Usal):
    U_im = Uim_from_U(Usal)
    Tjump = 0.04*U_im**0.84
    return Tjump
    
def calc_T_jump_Test(Usal):
    Tjump_ballistic = calc_T_jump_ballistic_assumption(Usal)
    Tjump_Xiuqi = calc_T_jump_Xiuqi(Usal)
    print('Tjump_ballistic=',Tjump_ballistic,'Tjump_Xiuqi=', Tjump_Xiuqi)
    Tjump  = 0.5*Tjump_Xiuqi + 0.5*Tjump_ballistic
    return Tjump
              
def Preff_of_U(U):
    """State-conditioned rebound fraction. PDF marks 'needs calibration'—placeholder Gompertz-like."""
    # P_min + (P_max-P_min) * (1 - exp(-(U/Uc)^p))
    # Pmin, Pmax, Uc, p = 0, 0.999, 10.038, 0.429   
    Pmin, Pmax, Uc, p = 0, 0.5934, 2.0939, 0.7368   # tuned
    U = abs(U)
    return Pmin + (Pmax - Pmin)*(1.0 - np.exp(-(U/max(Uc,1e-6))**p))  

def calc_Pr_Xiuqi_paper2(Usal):  # from Xiuqi paper 2
        Pr = 0.94*np.exp(-7.12*np.exp(-0.1*Usal/np.sqrt(g*d)))
        return Pr

def e_COR_from_Uim(Uim):
    # return 3.05 * (abs(Uim)/const + 1e-12)**(-0.47)       
    return 3.0932 * (abs(Uim)/const + 1e-12)**(-0.4689)                  

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

def NE_from_Uinc(Uinc):
    return 0.04 * (abs(Uinc)/const)     
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
    b = 0.5842
    C_D = 0.1776
    Ueff = b * Uair
    dU = Ueff - U
    fdrag = np.pi/8 * d**2 * rho_a * C_D * abs(dU) * dU                     
    return (c * fdrag) / mp

def MD_eff(Ua, U, c):
    Uaeff = 0.8741*Ua
    MD_eff = rho_a * 0.1931 * c/(rho_p*d) *abs(Uaeff - U) * (Uaeff - U)
    return MD_eff

# --- keep all your definitions above (constants, closures, rhs_cmUa) unchanged ---
def rhs_cmUa(t, y, u_star, eps=1e-16):
    c, m, Ua = y
    # c = max(c, 0.0)                         # clip for safety
    # U = np.clip(m / max(c, 1e-4), 0.0, 20.0)     # recover U
    U = m/(c + eps)

    Uim, UD = Uim_from_U(U), UD_from_U(U)
    Tim, TD  = max(1e-9, calc_T_jump_Test(Uim)), max(1e-9, calc_T_jump_Test(UD)) # changed
    # Pr = np.clip(Preff_of_U(U), 0.0, 1.0)
    Pr = calc_Pr_Xiuqi_paper2(U) # changed

    # mixing fractions and rates
    phi_im = (Pr*Tim) / (Pr*Tim + (1.0-Pr)*TD + 1e-12)
    cim, cD = c*phi_im, c*(1.0-phi_im)
    if cD<0:
            print('cD=',cD)
    r_im, r_dep = cim/Tim, cD/TD

    # ejection numbers and rebound kinematics
    NEim, NEd = NE_from_Uinc(Uim), NE_from_Uinc(UD) 
    eCOR = e_COR_from_Uim(Uim); Ure = Uim*eCOR
    th_im, th_D, th_re = theta_im_from_Uim(Uim), theta_D_from_UD(UD), theta_reb_from_Uim(Uim)

    # scalar sources
    E = r_im*NEim + r_dep*NEd
    D = r_dep
    
    if E<0:
            print('E = ',E)

    # momentum sources (streamwise)
    M_drag = Mdrag(c, Ua, U)
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

    phi_term  = 1.0 - c/(rho_p*h)#max(1e-6, 1.0 - c/(rho_p*h))
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
        f = rhs_cmUa(tk, yk, u_star)
        y_next = yk + dt * np.asarray(f, dtype=float)

        # # minimal safety/projection:
        # # keep c >= 0, keep U recovery stable by preventing c ~ 0
        # y_next[0] = max(y_next[0], 0.0)                       # c >= 0
        # # limit absurd growth of |m| to avoid overflows
        # y_next[1] = np.clip(y_next[1], -1e9, 1e9)
        # # bound Ua to a reasonable range
        # y_next[2] = np.clip(y_next[2], 0.0, 50.0)

        t[k+1]   = min(tk + dt, t1)
        y[:,k+1] = y_next

        # if t[k+1] >= t1:
        #     # truncate (in case last step hits t1 early)
        #     t = t[:k+2]
        #     y = y[:, :k+2]
        #     break

    return t, y

data = np.loadtxt('CGdata/hb=12d/Shields006dry.txt')
Q_dpm = data[:, 0]
C_dpm = data[:, 1]
U_dpm = data[:, 2]
t_dpm = np.linspace(0,5,501)
data_ua = np.loadtxt('TotalDragForce/Uair_ave-tS006Dryh02.txt', delimiter='\t')
Ua_dpm = data_ua[0:,1]  

# ---------- run it ----------
# initial condition 
y0 = (C_dpm[0], C_dpm[0]*U_dpm[0], Ua_dpm[0])

T0, T1 = 0.0, 20.0
dt     = 1e-2   # try 1e-4 or 1e-5 for stability; increase if looks stable

t, Y = euler_forward(rhs_cmUa, y0, (T0, T1), dt, u_star)
c  = Y[0]
m  = Y[1]
Ua = Y[2]
U  = m / np.maximum(c, 1e-12)

print("final ~steady:", "c =", c[-1], "U =", U[-1], "Ua =", Ua[-1])

# ---------- plot ----------
plt.close('all')
plt.figure(figsize=(8,6))
plt.subplot(3,1,1); plt.plot(t, c,  label='model'); plt.plot(t_dpm, C_dpm,  label='DPM', alpha=0.5)
plt.ylabel('c'); plt.grid(True); plt.legend()

plt.subplot(3,1,2); plt.plot(t, U,  label='model'); plt.plot(t_dpm, U_dpm,  label='DPM', alpha=0.5)
plt.ylabel('U [m/s]'); plt.grid(True); plt.legend()

plt.subplot(3,1,3); plt.plot(t, Ua, label='model'); plt.plot(t_dpm, Ua_dpm, label='DPM', alpha=0.5)
plt.ylabel('Ua [m/s]'); plt.xlabel('t [s]'); plt.grid(True); plt.legend()

plt.tight_layout(); plt.show()


# # --- RHS in (c, m=cU, Ua) ---
# def rhs_cmUa(t, y, u_star, eps=1e-4):
#     c, m, Ua = y
#     c = max(c, 0.0)                         # clip for safety
#     U = np.clip(m / max(c, 1e-4), 0.0, 20.0)     # recover U

#     Uim, UD = Uim_from_U(U), UD_from_U(U)
#     Tim, TD  = max(1e-9, Tim_from_Uim(Uim)), max(1e-9, TD_from_UD(UD))
#     Pr = np.clip(Preff_of_U(U), 0.0, 1.0)

#     # mixing fractions and rates
#     phi_im = (Pr*Tim) / (Pr*Tim + (1.0-Pr)*TD + 1e-12)
#     cim, cD = c*phi_im, c*(1.0-phi_im)
#     r_im, r_dep = cim/Tim, cD/TD

#     # ejection numbers and rebound kinematics
#     NEim, NEd = NE_from_Uinc(Uim), NE_from_Uinc(UD)
#     eCOR = e_COR_from_Uim(Uim); Ure = Uim*eCOR
#     th_im, th_D, th_re = theta_im_from_Uim(Uim), theta_D_from_UD(UD), theta_reb_from_Uim(Uim)

#     # scalar sources
#     E = r_im*NEim + r_dep*NEd
#     D = r_dep

#     # momentum sources (streamwise)
#     M_drag = Mdrag(c, Ua, U)
#     M_eje  = (r_im*NEim + r_dep*NEd) * UE_mag * cos_thetaE
#     M_re   = r_im * ( Ure*np.cos(th_re) )
#     M_im   = r_im * ( Uim*np.cos(th_im) )
#     M_dep  = r_dep* ( UD *np.cos(th_D ) )

#     # ODEs
#     dc_dt = E - D
#     dm_dt = M_drag + M_eje + M_re - M_im - M_dep

#     chi       = max(1e-6, 1.0 - c/(rho_p*h))
#     m_air_eff = rho_a*h*chi
#     dUa_dt    = (tau_top(u_star) - MD_eff(Ua, U, c)) / m_air_eff

#     return [dc_dt, dm_dt, dUa_dt]


# data = np.loadtxt('CGdata/hb=12d/Shields006dry.txt')
# Q_dpm = data[:, 0]
# C_dpm = data[:, 1]
# U_dpm = data[:, 2]
# t_dpm = np.linspace(0,5,501)
# data_ua = np.loadtxt('TotalDragForce/Uair_ave-tS006Dryh02.txt', delimiter='\t')
# Ua_dpm = data_ua[0:,1]  
# # --- integrate with an implicit method (stiff & steady-seeking) ---
# y0 = (C_dpm[0], C_dpm[0]*U_dpm[0], Ua_dpm[0])
# sol = solve_ivp(lambda t,y: rhs_cmUa(t,y,u_star),
#                 (0, 5.0), y0, method='BDF', rtol=1e-6, atol=1e-7)

# print(sol.success)        # False means it bailed early
# print(sol.message)        # Reason it stopped
# print(sol.t[-1])          # Last successful time
# print(np.any(~np.isfinite(sol.y)))  # NaN/Inf?

# t  = sol.t
# c  = sol.y[0]
# m  = sol.y[1]
# Ua = sol.y[2]
# U  = m/np.maximum(c,1e-12)

# # quick sanity check against your steady balances:
# print("final ~steady:",
#       "c =", c[-1], "U =", U[-1], "Ua =", Ua[-1])  

# plt.close('all')
# plt.figure(figsize=(8,6))
# plt.subplot(3,1,1); plt.plot(t,c);  plt.plot(t_dpm,C_dpm); plt.ylabel('c')
# plt.subplot(3,1,2); plt.plot(t,U);  plt.plot(t_dpm,U_dpm); plt.ylabel('U')
# plt.subplot(3,1,3); plt.plot(t,Ua); plt.plot(t_dpm,Ua_dpm); plt.ylabel('Ua'); plt.xlabel('t [s]')
# plt.tight_layout(); plt.show()
