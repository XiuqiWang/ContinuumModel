# -*- coding: utf-8 -*-
"""
Created on Tue Sep  2 12:10:57 2025

@author: WangX3
"""

from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.optimize import least_squares
from scipy.interpolate import interp1d

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
    # A_td, p_td, Pmax, Uc, p, A_NE = params
    # return 0.06 * (abs(UD))**0.66  
    return 0.44 * (abs(UD))** 0.18

# def calc_T_jump_ballistic_assumption(Uinc, theta_inc_degree):
#     Uy0 = Uinc*np.sin(theta_inc_degree/180*np.pi)
#     Tjump = 2*Uy0/g
#     return Tjump 
    
# def calc_T_jump_Test(Uinc, theta_inc_degree, mode):
#     Tjump_ballistic = calc_T_jump_ballistic_assumption(Uinc, theta_inc_degree)
#     # branch based on mode string (case-insensitive)
#     mode = mode.lower()
#     if mode == "im":
#         Tjump_Xiuqi = Tim_from_Uim(Uinc)
#     elif mode == "de":
#         Tjump_Xiuqi = TD_from_UD(Uinc)
#     else:
#         raise ValueError(f"Invalid mode '{mode}'. Expected 'im' or 'de'.")
    
#     Tjump  = 0.5*Tjump_Xiuqi + 0.5*Tjump_ballistic
#     # print('Tjump_Xiuqi', Tjump_Xiuqi, 'Tjump_ballistic', Tjump_ballistic, 'Tjump', Tjump)
#     return Tjump
              
def Preff_of_U(U, params):
    """State-conditioned rebound fraction. PDF marks 'needs calibration'—placeholder Gompertz-like."""
    # P_min + (P_max-P_min) * (1 - exp(-(U/Uc)^p))
    Pmax, Uc, p, A_NE = params
    Pmin = 0
    U = abs(U)
    return Pmin + (Pmax - Pmin)*(1.0 - np.exp(-(U/Uc)**p))  

def NE_from_Uinc(Uinc, Omega, params): 
    Pmax, Uc, p, A_NE = params
    return (A_NE-0.04*Omega**0.23) * (abs(Uinc)/const) # try

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

# --- Calculate rhs of the 3 equations ---
def rhs_cmUa(t, y, U, u_star, Omega, params):
    c = y

    Uim, UD = Uim_from_U(U), UD_from_U(U)
    Tim, TD  = Tim_from_Uim(Uim), TD_from_UD(UD) # changed 
    Pr = Preff_of_U(U, params) # changed

    # mixing fractions and rates
    # phi_im = (Pr*Tim) / (Pr*Tim + (1.0-Pr)*TD)
    cim, cD = c*Pr, c*(1.0-Pr)
    if cD<0:
            print('cD=', cD)
    r_im, r_dep = cim/Tim, cD/TD

    # ejection numbers and rebound kinematics
    NEim, NEd = NE_from_Uinc(Uim, Omega, params), NE_from_Uinc(UD, Omega, params)# changed

    # scalar sources
    E = r_im*NEim + r_dep*NEd
    D = r_dep
    
    if E<0:
            print('E = ', E)
            
    if D<0:
        print('D=', D)

    # ODE
    dc_dt = E - D
    
    return dc_dt, E/c, D/c

# ---------- simple Euler–forward integrator ----------
def euler_forward(rhs, U_vec, y0, t_span, dt, u_star, Omega, params):
    t0, t1 = t_span
    nsteps = int(np.ceil((t1 - t0) / dt))
    t = np.empty(nsteps + 1, dtype=float)
    y = np.empty((1, nsteps + 1), dtype=float) #only solved dcdt equation

    t[0]   = t0
    y[:,0] = np.asarray(y0, dtype=float)
    E_c = np.empty(nsteps, dtype=float)
    D_c = np.empty(nsteps, dtype=float)

    for k in range(nsteps):
        tk   = t[k]
        yk   = y[:,k].copy()

        # Euler step
        f = rhs_cmUa(tk, yk, U_vec[k], u_star, Omega, params)
        y_next = yk + dt * np.asarray(f[0], dtype=float)

        t[k+1]   = min(tk + dt, t1)
        y[:,k+1] = y_next
        
        E_c[k] = f[1]
        D_c[k] = f[2]

    return t, y, E_c, D_c

def simulate_and_sample_case(u_star_case, Omega, U, params, T0, T1, y0, dt):
    t, y_samp, E_c, D_c = euler_forward(rhs_cmUa, U, y0, (T0, T1), dt, u_star_case, Omega, params)
    c_mod = y_samp[0]
    return c_mod, E_c, D_c

# -------------------- per-case residual --------------------
def make_residuals_time_single(u_star_case, Omega, y0, T0, T1, dt, C_dpm, U_dpm, E_dpm_overc, D_dpm_overc,
                                wC=1.0, wE=1.0, wD=1.0, normalize=True):
    sC  = max(np.mean(np.abs(C_dpm)),  1e-6)
    sE  = max(np.mean(np.abs(E_dpm_overc)),  1e-6)
    sD  = max(np.mean(np.abs(D_dpm_overc)),  1e-6)
    def residuals(params):
        c_m, E_c, D_c = simulate_and_sample_case(u_star_case, Omega, U_dpm, params, T0, T1, y0, dt)
        rC  = (c_m  - C_dpm)  / (sC  if normalize else 1.0)
        rE = (E_c  - E_dpm_overc)  / (sE  if normalize else 1.0)
        rD = (D_c  - D_dpm_overc)  / (sD  if normalize else 1.0)
        return np.r_[wE*rE, wD*rD]
    return residuals


# ---------- load DPM data --------- 
data = np.loadtxt('CGdata/hb=12d/Shields006dry.txt')
Q_dpm = data[:, 0]
C_dpm = data[:, 1]
U_dpm = data[:, 2]
t_dpm = np.linspace(0,5,501)

ED = np.loadtxt("dcdt/U_ED_c_smoothedS006.txt")
U_dpm_center = ED[:,0]
E_dpm_overc = ED[:,1]
D_dpm_overc = ED[:,2]

# ---------- run it ----------
# initial guess and bounds
p0 = np.array([0.75, 2.0, 0.4, 0.03])    
lb = np.array([0.3, 0.1, 0.2, 0.01])
ub = np.array([1.0, 10.0, 3.0, 0.1])

# initial condition 
y0 = C_dpm[0]
T0, T1 = 0, 5.0
dt = 0.01
Omega_dry = 0.0

resfn = make_residuals_time_single(u_star, Omega_dry, y0, T0, T1, dt, C_dpm, U_dpm, E_dpm_overc, D_dpm_overc,
                                        wC=1.0, wE=1.0, wD=1.0, normalize=True)

opt = least_squares(resfn, p0, bounds=(lb, ub),
                        method='trf', loss='soft_l1', f_scale=1.0,
                        xtol=1e-7, ftol=1e-7, gtol=1e-7)

params_opt = np.round(opt.x, 3)
print('params_opt', params_opt)
c_m, E_c, D_c = simulate_and_sample_case(u_star, Omega_dry, U_dpm, params_opt, T0, T1, y0, dt)

plt.figure()
plt.plot(t_dpm, c_m)
plt.plot(t_dpm, C_dpm, '.')
plt.xlabel('t [s]')
plt.ylabel(r'$C$ [kg/m$^2$]')

plt.figure()
plt.plot(U_dpm_center, E_c, '.', color='C0', label='E/c')
plt.plot(U_dpm_center, E_dpm_overc, '.', color='C0', label='E/c DPM')
plt.plot(U_dpm_center, D_c, '.', color='C1', label='D/c')
plt.plot(U_dpm_center, D_dpm_overc, '.', color='C1', label='D/c DPM')
plt.xlabel('t [s]')
plt.ylabel(r'$C$ [kg/m$^2$]')
plt.legend()