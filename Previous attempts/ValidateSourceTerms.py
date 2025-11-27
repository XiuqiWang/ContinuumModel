# -*- coding: utf-8 -*-
"""
Created on Tue Oct 14 11:52:09 2025

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
thetaE_deg = 40.0         
cos_thetaE = np.cos(np.deg2rad(thetaE_deg))

alpha_drag = 0.25

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
              
def Preff_of_U(U):
    """State-conditioned rebound fraction. PDF marks 'needs calibration'—placeholder Gompertz-like."""
    # P_min + (P_max-P_min) * (1 - exp(-(U/Uc)^p))
    Pmin, Pmax, Uc, p = 0, 0.999, 3.84, 0.76 
    U = abs(U)
    return Pmin + (Pmax - Pmin)*(1.0 - np.exp(-(U/Uc)**p))  

def Preff_of_U_test(U):
    """State-conditioned rebound fraction. PDF marks 'needs calibration'—placeholder Gompertz-like."""
    # P_min + (P_max-P_min) * (1 - exp(-(U/Uc)^p))
    Pmin, Pmax, Uc, p = 0, 0.75, 2, 0.4
    U = abs(U)
    return Pmin + (Pmax - Pmin)*(1.0 - np.exp(-(U/Uc)**p))  

def calc_Pr_Xiuqi_paper2(U):  # from Xiuqi paper 2
    Uinc = U / np.cos(15/180*np.pi)
    Pr = 0.94*np.exp(-7.12*np.exp(-0.1*Uinc/np.sqrt(g*d)))
    return Pr

def e_COR_from_Uim(Uim, Omega):
    # return 3.05 * (abs(Uim)/const + 1e-12)**(-0.47)   
    mu = 312.48
    sigma = 156.27
    e_value = 3.05*(abs(Uim)/const)**(-0.47) + 0.12*np.log(1 + 1061.81*Omega)*np.exp(- (abs(Uim)/const - mu)**2/(2*sigma**2) )    
    # e = np.where(Uim > 0.0, e_value, 0.0)
    return e_value

def e_COR_from_Uim_test(Uim, Omega):
    # return 3.05 * (abs(Uim)/const + 1e-12)**(-0.47)   
    mu = 312.48
    sigma = 156.27
    e_value = 4*(abs(Uim)/const)**(-0.47) + 0.12*np.log(1 + 1061.81*Omega)*np.exp(- (abs(Uim)/const - mu)**2/(2*sigma**2) )    
    # e = np.where(Uim > 0.0, e_value, 0.0)
    return e_value          

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
    # return (0.04-0.04*Omega**0.23) * (abs(Uinc)/const) 
    return (0.035-0.04*Omega**0.23) * (abs(Uinc)/const) # try

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

def tau_top(u_star):                                                 
    return rho_a * u_star**2                                        

def calc_Mdrag(c, Uair, U):
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
    Mdrag = calc_Mdrag(c, Ua, U)
    CD_bed = 0.0037
    B, p = 4.46, 7.70
    tau_basic = 0.5 * rho_a * CD_bed * Ua * abs(Ua)
    M_bed = tau_basic * (1/(1+(B*Mdrag)**p)) # to guarantee that there is a balancing term with tau_top when c=0
    M_final = Mdrag + M_bed
    return M_final

# --- Calculate rhs of the 3 equations ---
E_all, D_all = [], []
Mdrag_all, Mbed_all, Mdrageff_all = [], [], []
def rhs_cmUa(t, y, u_star, Omega, eps=1e-16):
    c, m, Ua = y
    U = m/(c + eps)

    Uim, UD = Uim_from_U(U), UD_from_U(U)
    Tim, TD  = calc_T_jump_ballistic_assumption(Uim, 15), calc_T_jump_ballistic_assumption(Uim, 15) # changed 
    Pr = Preff_of_U_test(U) # changed

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
    M_drag = calc_Mdrag(c, Ua, U) # changed
    M_eje  = r_im*NEim * UE_from_Uinc(Uim, Omega) * cos_thetaE + r_dep*NEd * UE_from_Uinc(UD, Omega) * cos_thetaE
    M_re   = r_im * ( Ure*np.cos(th_re) )
    M_im   = r_im * ( Uim*np.cos(th_im) )
    M_dep  = r_dep* ( UD *np.cos(th_D ) )
    M_bed =  M_im + M_dep - M_eje - M_re
    
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
    dm_dt = M_drag - M_bed

    phi_term  = 1.0 - c/(rho_p*h)
    m_air_eff = rho_a*h*phi_term
    Mdrag_eff = MD_eff(Ua, U, c)
    dUa_dt    = (tau_top(u_star) - Mdrag_eff) / m_air_eff
    
    E_all.append(E)
    D_all.append(D)
    Mdrag_all.append(M_drag)
    Mbed_all.append(M_bed)
    Mdrageff_all.append(MD_eff(Ua, U, c))

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

# ---------- load DPM data --------- 
data = np.loadtxt('CGdata/hb=12d/Shields006dry.txt')
Q_dpm = data[:, 0]
C_dpm = data[:, 1]
U_dpm = data[:, 2]
t_dpm = np.linspace(0,5,501)
data_ua = np.loadtxt('TotalDragForce/Uair_ave-tS006Dryh02.txt', delimiter='\t')
Ua_dpm = data_ua[0:,1]  
MD_dpm = np.loadtxt("TotalDragForce/FD_S006dry.txt")
Mdrag_overc_dpm = MD_dpm/C_dpm
dt = 0.01
dcUdt_dpm = np.gradient(U_dpm*C_dpm, dt)
Mbed_overc_dpm = (MD_dpm - dcUdt_dpm)/C_dpm

phi = C_dpm/(rho_p*h)
dUa_dt = np.gradient(Ua_dpm, dt)
tau_top_emp = tau_top(u_star)
MDeff_dpm = tau_top_emp - rho_a * h * dUa_dt * (1-phi)
dU_dpm = alpha_drag*Ua_dpm - U_dpm

# ED = np.loadtxt("dcdt/check0.5/S006EandD.txt")
# E_dpm = ED[:,0]
# D_dpm = ED[:,1]
# C_inter = 0.5 * (C_dpm[:-1] + C_dpm[1:]) # central values 
# E_dpm_overc = E_dpm/C_dpm[:-1]
# D_dpm_overc = D_dpm/C_dpm[:-1]
# U_dpm_inter = 0.5 * (U_dpm[:-1] + U_dpm[1:])
ED = np.loadtxt("dcdt/U_ED_c_smoothedS006.txt")
U_dpm_center = ED[:,0]
E_dpm_overc = ED[:,1]
D_dpm_overc = ED[:,2]

# ---------- run it ----------
# initial condition 
y0 = (C_dpm[0], C_dpm[0]*U_dpm[0], Ua_dpm[0])

T0, T1 = 0, 5.0
dt = 0.01

Omega_dry = 0.0

# dry
t, Y = euler_forward(rhs_cmUa, y0, (T0, T1), dt, u_star, Omega_dry)
c  = Y[0]
m  = Y[1]
Ua = Y[2]
U  = m / np.maximum(c, 1e-16)

U_center = 0.5*(U[:-1]+U[1:])
dU = alpha_drag*Ua - U
dU_center = 0.5*(dU[:-1]+dU[1:])
c_center = 0.5*(c[:-1]+c[1:])
E_overc = E_all/c_center
D_overc = D_all/c_center
Mdrag_overc = Mdrag_all/c_center
Mbed_overc = Mbed_all/c_center

plt.close('all')
plt.figure(figsize=(15,6))
plt.subplot(1,3,1)
plt.plot(U_center, E_overc, color='C0', label='E/c')
plt.plot(U_center, D_overc, color='C1', label='D/c')
plt.plot(U_dpm_center, E_dpm_overc, 'C0.', label='E/c DPM')
plt.plot(U_dpm_center, D_dpm_overc, 'C1.', label='D/c DPM')
plt.xlabel('U [m/s]')
plt.ylabel('E/c & D/c')
plt.legend()
plt.subplot(1,3,2)
plt.plot(dU_center, Mdrag_overc, color='C0', label='M_drag/c')
plt.plot(dU_center, Mbed_overc, color='C1', label='M_bed/c')
plt.plot(dU_dpm, Mdrag_overc_dpm, '.', color='C0', label='M_drag/c DPM')
plt.plot(dU_dpm, Mbed_overc_dpm, '.', color='C1', label='M_bed/c DPM')
plt.xlabel(f'{alpha_drag:.2f}Ua - U [m/s]')
plt.ylabel('M/c')
plt.legend()
plt.subplot(1,3,3)
plt.plot(c_center, Mdrageff_all, color='C0', label=r'$M_{drag,eff}$')
plt.plot(c, tau_top_emp*np.ones_like(c), color='C1', label=r'$\tau_{top}$')
plt.plot(C_dpm, MDeff_dpm, '.', color='C0', label=r'$M_{drag,eff}$ DPM')
plt.xlabel(r'$c$ [kg/m$^2$]')
plt.ylabel('Source/sink in air equation')
plt.legend()
plt.suptitle('Shields=0.06')


plt.figure(figsize=(6,6))
plt.subplot(3,1,1); plt.plot(t, c,  label='Continuum'); plt.plot(t_dpm, C_dpm, label='DPM')
plt.ylabel(r'c [kg/m$^2$]'); plt.grid(True); plt.legend()

plt.subplot(3,1,2); plt.plot(t, U,  label='Continuum'); plt.plot(t_dpm, U_dpm, label='DPM')
plt.ylabel('U [m/s]'); plt.grid(True); plt.legend()

plt.subplot(3,1,3); plt.plot(t, Ua, label='Continuum'); plt.plot(t_dpm, Ua_dpm, label='DPM')
plt.ylabel('Ua [m/s]'); plt.xlabel('t [s]'); plt.grid(True); plt.legend()
plt.tight_layout(); plt.show()

# check dependency of Pr on U
U_test = np.linspace(0, 2, 100)
plt.figure()
plt.plot(U_test, Preff_of_U(U_test), label='paper3')
plt.plot(U_test, Preff_of_U_test(U_test), label='paper3 test')
plt.plot(U_test, calc_Pr_Xiuqi_paper2(U_test), label='paper2')
plt.xlabel('U [m/s]')
plt.ylabel('Pr')
plt.grid(True)
plt.xlim(left=0)
plt.ylim(bottom=0)
plt.legend()

Uinc_test = Uim_from_U(U_test)
# plt.figure()
# plt.plot(Uinc_test, NE_from_Uinc(Uinc_test, 0), label='paper2')
# plt.plot(Uinc_test, calc_N_E_test3(Uinc_test, 0), label='N_E_test3')
# plt.plot(Uinc_test, calc_N_E_test(Uinc_test, 0), label='N_E_test')
# plt.xlabel('U_inc [m/s]')
# plt.ylabel('NE')
# plt.grid(True)
# plt.xlim(left=0)
# plt.ylim(bottom=0)
# plt.legend()

plt.figure()
plt.plot(Uinc_test[1:], e_COR_from_Uim(Uinc_test[1:], 0), label='paper2')
plt.plot(Uinc_test[1:], e_COR_from_Uim_test(Uinc_test[1:], 0), label='paper2 test')
plt.xlabel('U_inc [m/s]')
plt.ylabel('e')
plt.grid(True)
plt.xlim(left=0)
plt.ylim(bottom=0)
plt.legend()
