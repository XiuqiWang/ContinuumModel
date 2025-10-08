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

# Ejection properties
thetaE_deg = 40.0         
cos_thetaE = np.cos(np.deg2rad(thetaE_deg))

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
    Tjump = 2*Uy0/g
    return Tjump 
    
def calc_T_jump_Xiuqi(Usal):
    U_im = Uim_from_U(Usal)
    Tjump = 0.04*U_im**0.84
    return Tjump
    
def calc_T_jump_Test(Usal):
    Tjump_ballistic = calc_T_jump_ballistic_assumption(Usal)
    Tjump_Xiuqi = calc_T_jump_Xiuqi(Usal)
    Tjump  = 0.5*Tjump_Xiuqi + 0.5*Tjump_ballistic
    return Tjump
              
def Preff_of_U(U):
    """State-conditioned rebound fraction. PDF marks 'needs calibration'—placeholder Gompertz-like."""
    # P_min + (P_max-P_min) * (1 - exp(-(U/Uc)^p))
    # Pmin, Pmax, Uc, p = 0, 0.999, 10.038, 0.429 
    Pmin, Pmax, Uc, p = 0, 0.999, 3.84, 0.76 
    Pmin, Pmax, Uc, p = 0, 0.85, 0.1, 0.2  # try
    # Pmin, Pmax, Uc, p = 0, 0.5934, 2.0939, 0.7368   # tuned for steady solutions
    U = abs(U)
    return Pmin + (Pmax - Pmin)*(1.0 - np.exp(-(U/Uc)**p))  

def calc_Pr_Xiuqi_paper2(Usal):  # from Xiuqi paper 2
    Uinc = Usal / np.cos(15/180*np.pi)
    Pr = 0.94*np.exp(-7.12*np.exp(-0.1*Uinc/np.sqrt(g*d)))
    return Pr

def e_COR_from_Uim(Uim, Omega):
    # return 3.05 * (abs(Uim)/const + 1e-12)**(-0.47)   
    mu = 312.48
    sigma = 156.27
    e = 3.05*(abs(Uim)/const)**(-0.47) + 0.12*np.log(1 + 1061.81*Omega)*np.exp(- (abs(Uim)/const - mu)**2/(2*sigma**2) )    
    return e
    # return 3.0932 * (abs(Uim)/const + 1e-12)**(-0.4689) # tuned for steady solutions                 

def theta_im_from_Uim(Uim, Omega):
    alpha = 50.40 - 25.53*Omega**0.5 
    beta = -100.12*(1-np.exp(-2.34*Omega)) + 159.33
    x = alpha / (abs(Uim)/const + beta)                            
    return np.arcsin(np.clip(x, -1.0, 1.0))

def theta_D_from_UD(UD):
    x = 163.68 / (abs(UD) / const + 156.65)
    x = np.clip(x, 0.0, 1.0)
    theta = 0.28 * np.arcsin(x)
    return np.clip(theta, 0.0, 0.5 * np.pi)                   

def theta_reb_from_Uim(Uim, Omega):
    x = (-0.0003 - 0.00027*Omega**0.28)*(abs(Uim)/const) + 0.52                              
    return np.arcsin(np.clip(x, -1.0, 1.0))

def NE_from_Uinc(Uinc, Omega):
    # return 0.04 * (abs(Uinc)/const)     
    return (0.012-0.04*Omega**0.23) * (abs(Uinc)/const) # try
    # return (0.012-0.012*Omega**0.23) * (abs(Uinc)/const) # try
    # return 0.0635 * (abs(Uinc)/const)   

def calc_N_E_test(Uinc, Omega):
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

def calc_N_E_test3(Uinc, Omega):
        N_E_Xiuqi = NE_from_Uinc(Uinc, Omega)
        
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

def UE_from_Uinc(Uinc, Omega):
    A = 9.02*Omega + 4.53
    if Omega>0:
        B = -0.24*Omega+0.07
    else:
        B = 0
    U_E_im = const*A*(abs(Uinc)/const)**B
    return U_E_im * np.sign(Uinc)          

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

# def Mdrag(c, Uair, U):
#     """Momentum exchange (air→saltation): c * f_drag / m_p."""
#     b = 0.55 #0.5842 # tuned for steady solutions         
#     C_D = 0.1037 #0.1776 # tuned for steady solutions         
#     Ueff = b * Uair
#     dU = Ueff - U
#     fdrag = np.pi/8 * d**2 * rho_a * C_D * abs(dU) * dU                     
#     return (c * fdrag) / mp

def MD_eff(Ua, U, c):
    alpha, K = 1.20, 0.040
    alpha, K = 1.18, 0.04 # try
    Uaeff = alpha*Ua
    MD_eff = rho_a * K * c/(rho_p*d) *abs(Uaeff - U) * (Uaeff - U)
    CD_bed = 0.053
    CD_bed = 0.05 # try
    tau_basic = 0.5 * rho_a * CD_bed * Ua * abs(Ua)
    B, p = 1.07e+25, 0.033
    M_tune = tau_basic * (1/(1+(B*MD_eff)**p)) # to guarantee that there is a balancing term with tau_top when c=0
    M_final = MD_eff + M_tune
    return M_final

def calc_Mdrag_geert(c, Uair, U):
    Ngrains = c/mp
    alpha = 0.32
    alpha = 0.4 # try
    Ueff = alpha*Uair
    Urel = Ueff-U
    Re = abs(Urel)*d/nu_a
    Ruc = 24
    Cd_inf = 0.5
    Cd = (np.sqrt(Cd_inf)+np.sqrt(Ruc/Re))**2
    
    Agrain = np.pi*(d/2)**2
    Mdrag = 0.5*rho_a*Urel*abs(Urel)*Cd*Agrain*Ngrains # drag term based on uniform velocity
    return Mdrag

# --- Calculate rhs of the 3 equations ---
E_all, D_all = [], []
def rhs_cmUa(t, y, u_star, Omega, eps=1e-16):
    c, m, Ua = y
    U = m/(c + eps)

    Uim, UD = Uim_from_U(U), UD_from_U(U)
    Tim, TD  = calc_T_jump_Test(Uim), calc_T_jump_Test(UD) # changed 
    Pr = Preff_of_U(U) # changed

    # mixing fractions and rates
    phi_im = (Pr*Tim) / (Pr*Tim + (1.0-Pr)*TD)
    cim, cD = c*phi_im, c*(1.0-phi_im)
    if cD<0:
            print('cD=',cD)
    r_im, r_dep = cim/Tim, cD/TD

    # ejection numbers and rebound kinematics
    NEim, NEd = calc_N_E_test3(Uim, Omega), calc_N_E_test3(UD, Omega)# changed
    eCOR = e_COR_from_Uim(Uim, Omega); Ure = Uim*eCOR
    th_im, th_D, th_re = theta_im_from_Uim(Uim, Omega), theta_D_from_UD(UD), theta_reb_from_Uim(Uim, Omega)

    # scalar sources
    E = r_im*NEim + r_dep*NEd
    D = r_dep
    
    if E<0:
            print('E = ',E)

    # momentum sources (streamwise)
    M_drag = calc_Mdrag_geert(c, Ua, U) # changed
    M_eje  = r_im*NEim * UE_from_Uinc(Uim, Omega) * cos_thetaE + r_dep*NEd * UE_from_Uinc(UD, Omega) * cos_thetaE
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
    
    E_all.append(E)
    D_all.append(D)

    return [dc_dt, dm_dt, dUa_dt]
# ---------- simple Euler–forward integrator ----------
def euler_forward(rhs, y0, t_span, dt, u_star, Omega):
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
        f = rhs_cmUa(tk, yk, u_star, Omega)
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

T0, T1 = 0.0, 5.0
dt     = 1e-2   

Omega_dry = 0.0
Omega_wet = 0.01

# dry
t, Y = euler_forward(rhs_cmUa, y0, (T0, T1), dt, u_star, Omega_dry)
c  = Y[0]
m  = Y[1]
Ua = Y[2]
U  = m / np.maximum(c, 1e-16)
# wet
t, Y_wet = euler_forward(rhs_cmUa, y0, (T0, T1), dt, u_star, Omega_wet)
c_wet  = Y_wet[0]
m_wet  = Y_wet[1]
Ua_wet = Y_wet[2]
U_wet  = m_wet / np.maximum(c_wet, 1e-16)

# ---------- plot ----------
plt.close('all')
plt.figure(figsize=(8,8))
plt.subplot(4,1,1); plt.plot(t, m,  label='dry'); plt.plot(t, m_wet, label='wet')
# plt.plot(t_dpm, C_dpm,  label='DPM', alpha=0.5)
plt.ylabel('Q [kg/m/s]'); plt.grid(True); plt.legend()

plt.subplot(4,1,2); plt.plot(t, c,  label='dry'); plt.plot(t, c_wet, label='wet')
# plt.plot(t_dpm, C_dpm,  label='DPM', alpha=0.5)
plt.ylabel(r'c [kg/m$^2$]'); plt.grid(True); plt.legend()

plt.subplot(4,1,3); plt.plot(t, U,  label='dry'); plt.plot(t, U_wet, label='wet')
# plt.plot(t_dpm, U_dpm,  label='DPM', alpha=0.5)
plt.ylabel('U [m/s]'); plt.grid(True); plt.legend()

plt.subplot(4,1,4); plt.plot(t, Ua, label='dry'); plt.plot(t, Ua_wet, label='wet')
# plt.plot(t_dpm, Ua_dpm, label='DPM', alpha=0.5)
plt.ylabel('Ua [m/s]'); plt.xlabel('t [s]'); plt.grid(True); plt.legend()
plt.tight_layout(); plt.show()



# # E and D
# ED = np.loadtxt("dcdt/S006EandD.txt")
# E_dpm = ED[:,0]
# D_dpm = ED[:,1]
# t_dis = np.linspace(0, 5, len(E_dpm))
    
# plt.figure()
# plt.plot(U[:-1], E_all, '.', label='E')
# plt.plot(U[:-1], D_all, '.', label='D')
# plt.xlabel('U [m/s]')
# plt.ylabel('E & D')
# plt.title('Shields = 0.06 Dry')
# plt.legend()

# plt.figure()
# plt.plot(t[:-1], E_all, '.', color='C0', label='E')
# plt.plot(t[:-1], D_all, '.', color='C1', label='D')
# plt.plot(t_dis, E_dpm, color='C0', label='E (DPM)')
# plt.plot(t_dis, D_dpm, color='C1', label='D (DPM)')
# plt.title('Shields = 0.06 Dry')
# plt.xlabel('t [s]')
# plt.ylabel('E & D')
# plt.legend()