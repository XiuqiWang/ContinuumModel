# -*- coding: utf-8 -*-
"""
Created on Wed Nov  5 14:46:00 2025

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

# Particle mass
mp = rho_p * (np.pi/6.0) * d**3

# Ejection properties
thetaE = np.deg2rad(25.0)         
thetaim = np.deg2rad(13.0)
thetaD = np.deg2rad(13.0)

# -------------------- closures from the PDF --------------------
def calc_T_jump_ballistic_assumption(Uinc, theta_inc):
    Uy0 = Uinc*np.sin(theta_inc)
    Tjump = 2*Uy0/g
    return Tjump 

def calc_Pr(Uinc):
    Pr = 0.74*np.exp(-4.46*np.exp(-0.1*Uinc/np.sqrt(g*d)))
    if Pr <0 or Pr>1:
        print('Warning: Pr is not between 0 and 1')
    return Pr

def e_COR_from_Uim(Uim, Omega):
    # return 3.05 * (abs(Uim)/const + 1e-12)**(-0.47)   
    mu = 312.48
    sigma = 156.27
    e_com = 3.18*(abs(Uim)/const)**(-0.50) + 0.12*np.log(1 + 1061.81*Omega)*np.exp(- (abs(Uim)/const - mu)**2/(2*sigma**2) )   
    e = min(e_com, 1.0)
    return e          

# def theta_im_from_Uim(Uim, Omega):
#     alpha = 50.40 - 25.53*Omega**0.5 
#     beta = -100.12*(1-np.exp(-2.34*Omega)) + 159.33
#     x = alpha / (abs(Uim)/const + beta)                            
#     return np.arcsin(np.clip(x, -1.0, 1.0))

# def theta_D_from_UD(UD):
#     x = 163.68 / (abs(UD) / const + 156.65)
#     x = np.clip(x, 0.0, 1.0)
#     theta = 0.28 * np.arcsin(x)
#     return np.clip(theta, 0.0, 0.5 * np.pi)                   

def theta_reb_from_Uim(Uim, Omega):
    x = (-0.0032*Omega**0.39)*(abs(Uim)/const) + 0.43 
    theta_re = np.arcsin(x)
    if theta_re < 0 or theta_re > np.pi / 2: 
        print('The value of theta_re is not logical')                              
    return theta_re

def NE_from_Uinc(Uinc, Omega): 
    return (0.03-0.025*Omega**0.21) * (abs(Uinc)/const) # try

def UE_from_Uinc(Uinc, Omega):
    A = -1.51*Omega + 4.62
    if Omega>0:
        B = 0.56*Omega+0.15
    else:
        B = 0
    U_E = A*(abs(Uinc)/const)**B * const
    return U_E * np.sign(Uinc)  

# def calc_N_E_test(Uinc, Omega):
#        sqrtgd = np.sqrt(g*d)
#        # N_E = np.sqrt(Uinc/sqrtgd)*(0.04-0.04*Omega**0.23)*5
#        # N_E = (1-(10*Omega+0.2)/(Uinc**2+(10*Omega+0.2)))*np.sqrt(Uinc/sqrtgd)*(0.04-0.04*Omega**0.23)*7
#        p = 8
#        ##
#        p2 = 2
#        A = 100
#        Uinc_half_min = 1.0
#        Uinc_half_max = 6
#        Uinc_half = Uinc_half_min + (Uinc_half_max - Uinc_half_min)*(A*Omega)**p2/((A*Omega)**p2+1)
#        ##
#        #Uinc_half = 0.5+40*Omega**0.5
#        B = 1/Uinc_half**p
#        N_E = (1-1./(1.+B*Uinc**p))*2.5 * Uinc**(1/10)
#        return N_E 

# def calc_N_E_test3(Uinc, Omega):
#         N_E_Xiuqi = NE_from_Uinc(Uinc, Omega)
        
#         p = 8
#         ##
#         p2 = 2
#         A = 100
#         Uinc_half_min = 1.0
#         Uinc_half_max = 2.0
#         Uinc_half = Uinc_half_min + (Uinc_half_max - Uinc_half_min)*(A*Omega)**p2/((A*Omega)**p2+1)
#         ##
#         #Uinc_half = 0.5+40*Omega**0.5
#         B = 1/Uinc_half**p
#         N_E = (1-1./(1.+B*Uinc**p))*N_E_Xiuqi
#         return N_E         

def tau_top(u_star):                                                 
    return rho_a * u_star**2                                        

# def tau_bed(Uair, U, c):                    
#     # (1-phi_bed) * rho_a * (nu_a + l_eff_bed**2 * abs(Uair/h)) * (Uair/h)   
#     K = 0.08
#     beta = 0.9
#     U_eff = beta*Uair
#     tau_bed_minusphib = rho_a*K*c/(rho_p*d) * abs(U_eff - U)*(U_eff - U)                            
#     return tau_bed_minusphib

# def Mdrag_simp(c, Uair, U, Omega):
#     """Momentum exchange (air→saltation): c * f_drag / m_p."""
#     if Omega == 0:
#         b = 0.91    
#     else:
#         b = 1.30
#     C_D = 0.10      
#     Ueff = b * Uair
#     dU = Ueff - U
#     fdrag = np.pi/8 * d**2 * rho_a * C_D * abs(dU) * dU                     
#     return (c * fdrag) / mp

def calc_Mdrag(c, Uair, U, Omega):
    if Omega == 0:
        alpha_drag = 0.4 # ?
    else:
        alpha_drag = 0.4 # ?
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

def MD_eff(Ua, U, c, Omega):
    # _Pmax, _Uc, _pshape, _A_NE, B, p = params
    # alpha, K = 1.20, 0.040
    # Uaeff = alpha*Ua
    # MD_eff = rho_a * K * c/(rho_p*d) *abs(Uaeff - U) * (Uaeff - U)
    Mdrag = calc_Mdrag(c, Ua, U, Omega)
    CD_bed = 0.0037
    B, p = 4.46, 7.70
    tau_basic = 0.5 * rho_a * CD_bed * Ua * abs(Ua)
    M_bed = tau_basic * (1/(1+(B*Mdrag)**p)) # to guarantee that there is a balancing term with tau_top when c=0
    M_final = Mdrag + M_bed
    return M_final

# --- Calculate rhs of the 3 equations ---
E_all, D_all = [], []
Rim_all = []
def rhs_cmUa(t, y, u_star, Omega, eps=1e-16):
    c, m, Ua = y
    U = m/(c + eps)

    Uim = U*0.7/np.cos(thetaim)
    UD = Uim#Uim/3
    Tim  = calc_T_jump_ballistic_assumption(Uim, thetaim) 
    Pr = calc_Pr(Uim) 

    # mixing fractions and rates
    r_im = c/Tim

    # ejection numbers and rebound kinematics
    NEim = NE_from_Uinc(Uim, Omega)
    eCOR = e_COR_from_Uim(Uim, Omega); Ure = Uim*eCOR
    th_re = theta_reb_from_Uim(Uim, Omega)

    # scalar sources
    E = r_im*NEim 
    D = r_im*(1-Pr)
    
    if E<0:
            print('E = ',E)

    # momentum sources (streamwise)
    M_drag = calc_Mdrag(c, Ua, U, Omega) # changed
    M_eje  = E * UE_from_Uinc(Uim, Omega) * np.cos(thetaE)
    M_re   = r_im * Pr * ( Ure*np.cos(th_re) )
    M_im   = r_im * Pr * ( Uim*np.cos(thetaim) )
    M_dep  = D * ( UD*np.cos(thetaD) )
    
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
    dUa_dt    = (tau_top(u_star) - MD_eff(Ua, U, c, Omega)) / m_air_eff
    
    E_all.append(E)
    D_all.append(D)
    Rim_all.append(r_im)

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

        t[k+1]   = min(tk + dt, t1)
        y[:,k+1] = y_next

    return t, y

data = np.loadtxt('CGdata/hb=13.5d/Shields006Dry135d.txt')
Q_dpm = data[:, 1]*data[:, 2]
C_dpm = data[:, 1]
U_dpm = data[:, 2]
data_M20 = np.loadtxt('CGdata/hb=13.5d/Shields006M20-135d.txt')
Q_dpmM20 = data_M20[:, 1]*data_M20[:, 2]
C_dpmM20 = data_M20[:, 1]
U_dpmM20 = data_M20[:, 2]
t_dpm = np.linspace(0.01,5,501)
data_ua = np.loadtxt('TotalDragForce/Ua-t/Uair_ave-tS006Dry.txt', delimiter='\t')
Ua_dpm = data_ua[0:,1]  
data_uaM20 = np.loadtxt('TotalDragForce/Ua-t/Uair_ave-tS006M20.txt', delimiter='\t')
Ua_dpmM20 = data_uaM20[0:,1]  

# ---------- run it ----------
# initial condition 
y0 = (C_dpm[0], C_dpm[0]*U_dpm[0], Ua_dpm[0])
y0_wet = (C_dpmM20[0], C_dpmM20[0]*U_dpmM20[0], Ua_dpmM20[0])

T0, T1 = 0.0, 5.0
dt     = 1e-2   

Omega_dry = 0.0
Omega_wet = 0.2

# dry
t, Y = euler_forward(rhs_cmUa, y0, (T0, T1), dt, u_star, Omega_dry)
c  = Y[0]
m  = Y[1]
Ua = Y[2]
U  = m / np.maximum(c, 1e-16)
# wet
t, Y_wet = euler_forward(rhs_cmUa, y0_wet, (T0, T1), dt, u_star, Omega_wet)
c_wet  = Y_wet[0]
m_wet  = Y_wet[1]
Ua_wet = Y_wet[2]
U_wet  = m_wet / np.maximum(c_wet, 1e-16)

# ---------- plot ----------
# plt.close('all')
plt.figure(figsize=(8,8))
plt.subplot(4,1,1); plt.plot(t, m, color='tab:blue', label='dry'); plt.plot(t, m_wet, color='tab:orange', label=r'$\Omega$=20$\%$')
plt.plot(t_dpm, Q_dpm,  '--', color='tab:blue', label='dry DPM'); plt.plot(t_dpm, Q_dpmM20,  '--', color='tab:orange', label='M20 DPM')
plt.ylabel('Q [kg/m/s]'); plt.grid(True); plt.legend()

plt.subplot(4,1,2); plt.plot(t, c,  label='dry'); plt.plot(t, c_wet, label=r'$\Omega$=20$\%$')
plt.plot(t_dpm, C_dpm,  '--', color='tab:blue', label='dry DPM'); plt.plot(t_dpm, C_dpmM20,  '--', color='tab:orange', label='M20 DPM')
plt.ylabel(r'c [kg/m$^2$]'); plt.grid(True); plt.legend()

plt.subplot(4,1,3); plt.plot(t, U,  label='dry'); plt.plot(t, U_wet, label=r'$\Omega$=20$\%$')
plt.plot(t_dpm, U_dpm,  '--', color='tab:blue', label='dry DPM'); plt.plot(t_dpm, U_dpmM20,  '--', color='tab:orange', label='M20 DPM')
plt.ylabel('U [m/s]'); plt.grid(True); plt.legend()

plt.subplot(4,1,4); plt.plot(t, Ua, label='dry'); plt.plot(t, Ua_wet, label=r'$\Omega$=20$\%$')
plt.plot(t_dpm, Ua_dpm,  '--', color='tab:blue', label='dry DPM'); plt.plot(t_dpm, Ua_dpmM20,  '--', color='tab:orange', label='M20 DPM')
plt.ylabel('Ua [m/s]'); plt.xlabel('t [s]'); plt.grid(True); plt.legend()

for ax in plt.gcf().axes:
    ax.set_ylim(bottom=0)
    
plt.tight_layout(); plt.show()