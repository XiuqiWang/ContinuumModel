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
h = 0.2 - 13.5*d                 # domain height [m]
Shields = np.linspace(0.02, 0.06, 5)
ustar_list = np.sqrt(Shields * (2650-1.225)*9.81*d/1.225)

rho_a  = 1.225             # air density [kg/m^3] 
nu_a   = 1.46e-5           # air kinematic viscosity [m^2/s]
rho_p  = 2650.0            # particle density [kg/m^3]   
Ruc    = 24
Cd_inf = 0.5

# Particle mass
mp = rho_p * np.pi/6.0 * d**3         
thetaE = np.deg2rad(24)

Omega_list = [0.0, 0.01, 0.05, 0.10, 0.20]

# -------------------- closures from the PDF --------------------
def CalUincfromU(U, c):
    # if Omega == 0:
    #     # Uinc = 0.43*U
    #     Uinc = 0.61*U**0.44
    # else:
    #     # Uinc = 0.85*U
    #     Uinc = 0.44*U**1.36
    A = 1/(1 + c/0.19)
    Uinc = A*U
    return Uinc

def calc_T_jump_ballistic_assumption(Uinc, theta_inc):
    Uy0 = Uinc*np.sin(theta_inc)
    Tjump = 2*Uy0/g
    return Tjump 

def calc_Pr(Uinc, Omega):
    Pr = 0.74*np.exp(-4.46*np.exp(-0.10*Uinc/const))
    # a_pr = 0.70 + 0.23*Omega**0.12
    # b_pr = 0.01/(Omega + 0.0013)**1.59 + 3.18
    # c_pr = 0.41 - 0.42 * Omega **0.12
    # Pr = a_pr*np.exp(-b_pr*np.exp(-c_pr*Uinc/const))
    return Pr

def e_COR_from_Uim(Uim, Omega):
    mu = 312.48
    sigma = 156.27
    e_com = 3.18*(abs(Uim)/const)**(-0.50) + 0.12*np.log(1 + 1061.81*Omega)*np.exp(- (abs(Uim)/const - mu)**2/(2*sigma**2) )   
    e = min(e_com, 1.0)
    return e          

def theta_inc_from_Uinc(Uinc, Omega):
    # alpha = -113.89*Omega + 67.78 
    # beta = -260.77 * Omega + 248.51
    # x = alpha / (abs(Uinc)/const + beta)  
    # theta_inc = np.arcsin(x)
    alpha = 0.27 - 0.05*Omega**0.15
    beta = -0.0029
    x = alpha * np.exp(beta * abs(Uinc)/const)  
    theta_inc = np.arcsin(x)
    if theta_inc < 0 or theta_inc > np.pi / 2: 
        print('The value of theta_inc is not logical')                                     
    return theta_inc

# def theta_D_from_UD(UD):
#     x = 163.68 / (abs(UD) / const + 156.65)
#     x = np.clip(x, 0.0, 1.0)
#     theta = 0.28 * np.arcsin(x)
#     return np.clip(theta, 0.0, 0.5 * np.pi)                   

def theta_reb_from_Uim(Uim, Omega):
    x = -0.0032*Omega**0.39 *(abs(Uim)/const) + 0.43 
    theta_re = np.arcsin(x)
    if theta_re < 0 or theta_re > np.pi / 2: 
        print('The value of theta_re is not logical')                              
    return theta_re

def NE_from_Uinc(Uinc, Omega): 
    # return (0.03-0.028*Omega**0.19) * (abs(Uinc)/const) 
    NE = (0.03 - 0.025*Omega**0.21) * (abs(Uinc)/const) 
    return NE

def UE_from_Uinc(Uinc, Omega):
    # A = -2.13*Omega + 4.60
    # B = 0.40*Omega**0.24 + 0.15
    # U_E = A*(abs(Uinc)/const)**B * const
    A = -1.51*Omega + 4.62
    B = 0.37*Omega**0.26 + 0.019
    U_E = A*(abs(Uinc)/const)**B * const
    return U_E * np.sign(Uinc)  

# def calc_N_E_test(Uinc, Omega):
#         sqrtgd = np.sqrt(g*d)
#         # N_E = np.sqrt(Uinc/sqrtgd)*(0.04-0.04*Omega**0.23)*5
#         # N_E = (1-(10*Omega+0.2)/(Uinc**2+(10*Omega+0.2)))*np.sqrt(Uinc/sqrtgd)*(0.04-0.04*Omega**0.23)*7
#         p = 8
#         ##
#         p2 = 2
#         A = 100
#         Uinc_half_min = 1.0
#         Uinc_half_max = 6
#         Uinc_half = Uinc_half_min + (Uinc_half_max - Uinc_half_min)*(A*Omega)**p2/((A*Omega)**p2+1)
#         ##
#         #Uinc_half = 0.5+40*Omega**0.5
#         B = 1/Uinc_half**p
#         N_E = (1-1./(1.+B*Uinc**p))*2.5 * Uinc**(1/10)
#         return N_E 

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

# def Fitb(U, c, b0=0.015, b_inf=0.79, k0=0.53, lamda=4.86):
#     k = k0/(1+lamda*c)
#     b = b0 + (b_inf - b0)*(1 - np.exp(-k*U))
#     return b

def calc_Mdrag(Ua, U, c):
    b = 0.84*(1-np.exp(-0.49*U))
    burel = 1/(1+c/0.09)
    Urel = burel*(b * Ua - U)
    Re = abs(Urel)*d/nu_a
    Re = np.maximum(Re, 1e-6)
    Cd = (np.sqrt(Cd_inf)+np.sqrt(Ruc/Re))**2   
    Mdrag = np.pi/8 * d**2 * rho_a * Cd * Urel * abs(Urel) * c/mp
    return Mdrag

def calc_Mcreep(Ua, U, c):
    Mdrag = calc_Mdrag(Ua, U, c)
    M_bed = 1.21*Mdrag 
    return M_bed

def calc_Maero(Ua):
    Maero = 0.5*0.0037*rho_a*abs(Ua)*Ua
    return Maero

# --- Calculate rhs of the 3 equations ---
def rhs_cmUa(t, y, u_star, Omega, eps=1e-16):
    c, m, Ua = y
    U = m/(c + eps)

    Uinc = CalUincfromU(U, c)
    thetainc = theta_inc_from_Uinc(Uinc, Omega)
    Tim  = calc_T_jump_ballistic_assumption(Uinc, thetainc) + eps
    Pr = calc_Pr(Uinc, Omega) 

    # mixing fractions and rates
    r_im = c/Tim

    # ejection numbers and rebound kinematics
    NEim = NE_from_Uinc(Uinc, Omega)
    eCOR = e_COR_from_Uim(Uinc, Omega); Ure = Uinc*eCOR
    th_re = theta_reb_from_Uim(Uinc, Omega)
    UE = UE_from_Uinc(Uinc, Omega)

    # scalar sources
    E = r_im*NEim 
    D = r_im*(1-Pr)
    
    if E<0:
            print('E = ',E)

    # momentum sources (streamwise)
    M_drag = calc_Mdrag(Ua, U, c) 
    M_eje  = E * UE * np.cos(thetaE)
    M_re   = r_im * Pr * Ure*np.cos(th_re) 
    M_im   = r_im * Pr * Uinc*np.cos(thetainc) 
    M_dep  = D * Uinc*np.cos(thetainc) 
    
    if M_dep > D*U:
            print('More momentum is leaving per particle by deposition, than that there is on average in the saltation layer')
            print('M_dep =',M_dep)
            print('D*U', D*U)
            print('D =',D)
            print('U =',U)
            print('Uinc =',Uinc)
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
    M_creep = calc_Mcreep(Ua, U, c)
    M_aero = calc_Maero(Ua)
    dUa_dt    = (tau_top(u_star) - M_aero - M_creep - M_drag) / m_air_eff

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


omega_labels = ['Dry', 'M1', 'M5', 'M10', 'M20']
# Initialize lists to hold results for each Omega
Q_dpm = []
C_dpm = []
U_dpm = []
Ua_dpm = []

# Loop over all moisture levels
for label in omega_labels:
    for i in range(2, 7):
        # --- Load sediment transport data ---
        file_path = f'CGdata/hb=13.5d/Shields00{i}{label}-135d.txt'
        data = np.loadtxt(file_path)
        C = data[:, 1]
        U = data[:, 2]
        Q = C * U
        # Append to lists
        C_dpm.append(C)
        U_dpm.append(U)
        Q_dpm.append(Q)
        # --- Load corresponding air velocity data ---
        file_path_ua = f'TotalDragForce/Ua-t/Uair_ave-tS00{i}{label}.txt'
        data_ua = np.loadtxt(file_path_ua, delimiter='\t')
        Ua = data_ua[:, 1]
        Ua_dpm.append(Ua)

# ---------- run it ---------- 
def simulate_model():

    model_output = {Omega: {} for Omega in Omega_list}

    for i, Omega in enumerate(Omega_list):
        for j, ustar in enumerate(ustar_list):
            index = i*5+j
            y0 = (C_dpm[index][0], C_dpm[index][0]*U_dpm[index][0], Ua_dpm[index][0])
            t, Y = euler_forward(rhs_cmUa, y0, (0.0, 5.0), 1e-2, ustar, Omega)
    
            c  = Y[0]
            m  = Y[1]
            Ua = Y[2]
            U  = m / np.maximum(c, 1e-16)
    
            model_output[Omega][ustar] = {
                'c': c,
                'U': U,
                'Ua': Ua,
            }

    return model_output

model_run = simulate_model()
t_mod = np.linspace(0, 5, 501)

# ---------- plot ----------
colors = plt.cm.viridis(np.linspace(1, 0, 5))  # 5 colors

plt.close('all')
for ui, ustar in enumerate(ustar_list):

    fig, axes = plt.subplots(3, 1, figsize=(7, 8), sharex=True)
    axC, axU, axUa = axes

    for oi, Omega in enumerate(Omega_list):

        # measured data index: i*5 + j
        idx = oi*len(ustar_list) + ui
        
        # measured
        C_meas  = C_dpm[idx]
        U_meas  = U_dpm[idx]
        Ua_meas = Ua_dpm[idx]
        t_meas  = np.linspace(0, 5, len(C_meas))

        # model
        C_mod   = model_run[Omega][ustar]['c']
        U_mod   = model_run[Omega][ustar]['U']
        Ua_mod  = model_run[Omega][ustar]['Ua']

        label = fr'Ω={Omega*100} $\%$'

        # --- Plot C ---
        axC.plot(t_mod, C_mod, color=colors[oi], label=f'{label}')
        axC.plot(t_meas, C_meas, '--', color=colors[oi])

        # --- Plot U ---
        axU.plot(t_mod, U_mod, color=colors[oi])
        axU.plot(t_meas, U_meas, '--', color=colors[oi])

        # --- Plot Ua ---
        axUa.plot(t_mod, Ua_mod, color=colors[oi])
        axUa.plot(t_meas, Ua_meas, '--', color=colors[oi])
        
    axC.plot([], [], color='black', label=r"Continuum")
    axC.plot([], [], '--', color='black', label=r"DPM")
    axC.set_ylabel(r'$c$ [kg/m$^2$]')
    axC.set_title(fr'$\tilde{{\Theta}}$ = {Shields[ui]:.2f}')
    axC.grid(True)
    axC.set_ylim(0, 0.4)
    axC.set_xlim(0,5)
    axC.legend(fontsize=8)

    axU.set_ylabel(r'$U$ [m/s]')
    axU.set_ylim(0, 9.5)
    axU.set_xlim(0,5)
    axU.grid(True)

    axUa.set_ylabel(r'$U_\mathrm{air}$ [m/s]')
    axUa.set_xlabel(r'$t$ [s]')
    axUa.set_ylim(0, 13.5)
    axUa.set_xlim(0,5)
    axUa.grid(True)

    plt.tight_layout()
    plt.show()
    
# in one figure
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

fig = plt.figure(figsize=(12, 10), constrained_layout=True)

# Outer grid: 2 rows × 3 columns
outer = GridSpec(
    2, 3,
    figure=fig,
    width_ratios=[1, 1, 1]
)

for ui, ustar in enumerate(ustar_list):

    row = ui // 3
    col = ui % 3

    # ✅ Inner grid must use GridSpecFromSubplotSpec
    inner = GridSpecFromSubplotSpec(
        3, 1,
        subplot_spec=outer[row, col]
    )

    axC  = fig.add_subplot(inner[0])
    axU  = fig.add_subplot(inner[1], sharex=axC)
    axUa = fig.add_subplot(inner[2], sharex=axC)

    for oi, Omega in enumerate(Omega_list):

        idx = oi * len(ustar_list) + ui

        C_meas  = np.asarray(C_dpm[idx])
        U_meas  = np.asarray(U_dpm[idx])
        Ua_meas = np.asarray(Ua_dpm[idx])
        t_meas  = np.linspace(0, 5, len(C_meas))

        C_mod  = model_run[Omega][ustar]['c']
        U_mod  = model_run[Omega][ustar]['U']
        Ua_mod = model_run[Omega][ustar]['Ua']

        axC.plot(t_mod, C_mod, color=colors[oi], label=fr'{Omega*100} $\%$')
        axC.plot(t_meas, C_meas, '--', color=colors[oi])

        axU.plot(t_mod, U_mod, color=colors[oi])
        axU.plot(t_meas, U_meas, '--', color=colors[oi])

        axUa.plot(t_mod, Ua_mod, color=colors[oi])
        axUa.plot(t_meas, Ua_meas, '--', color=colors[oi])

    axC.set_title(fr'$\tilde{{\Theta}}$ = {Shields[ui]:.2f}')
    axC.set_ylim(0, 0.4)
    axC.set_xlim(0, 5)
    axC.grid(True)

    axU.set_ylim(0, 9.5)
    axU.set_xlim(0, 5)
    axU.grid(True)

    axUa.set_ylim(0, 13.5)
    axUa.set_xlim(0, 5)
    axUa.set_xlabel(r'$t$ [s]')
    axUa.grid(True)

    if col == 0:
        axC.set_ylabel(r'$c$ [kg/m$^2$]')
        axU.set_ylabel(r'$U$ [m/s]')
        axUa.set_ylabel(r'$U_\mathrm{air}$ [m/s]')
    else:
        for ax in (axC, axU, axUa):
            ax.set_yticklabels([])

# Hide unused bottom-right panel
ax_empty = fig.add_subplot(outer[1, 2])
ax_empty.axis('off')

# Legend once
axC.plot([], [], color='black', label='Continuum')
axC.plot([], [], '--', color='black', label='DPM')
axC.legend(fontsize=8, loc='upper right')

plt.show()


    

# chunk_size = 3000
# plt.figure(figsize=(8,8))
# plt.subplot(2,1,1)
# for i in range(len(Omega_list)):
#     start = i * chunk_size
#     end   = (i + 1) * chunk_size
#     plt.plot(t[1:], E_all[start:end], color=colors[i], label=rf"$\Omega$={Omega_list[i]*100} $\%$")
#     # plt.plot(t, C_dpm[i],  '--', color=colors[i], label=rf"$\Omega$={Omega_list[i]*100} $\%$ DPM")
# plt.ylim(bottom=0)
# plt.ylabel('E'); plt.grid(True); plt.legend()

# plt.subplot(2,1,2); 
# for i in range(len(Omega_list)):
#     start = i * chunk_size
#     end   = (i + 1) * chunk_size
#     plt.plot(t[1:], D_all[start:end], color=colors[i])
#     # plt.plot(t, C_dpm[i],  '--', color=colors[i])
# plt.ylim(bottom=0)
# plt.ylabel(r'D [kg/m$^2$/s]'); plt.grid(True)

# for ax in plt.gcf().axes:
#     ax.set_ylim(bottom=0)
    
# plt.tight_layout(); plt.show()

# # check E/c and D/c
# U_check = np.linspace(0.5, 6, 100)
# E_c_all, D_c_all = [], []
# for i in range(5):
#     Uinc_check = CalUincfromU(U_check, Omega_list[i])
#     theta_inc_check = theta_inc_from_Uinc(Uinc_check, Omega_list[i])
#     T_check = calc_T_jump_ballistic_assumption(Uinc_check, theta_inc_check)
#     NE_check = NE_from_Uinc(Uinc_check, Omega_list[i])
#     Pr_check = calc_Pr(Uinc_check)
    
#     E_c = NE_check/T_check
#     D_c = (1-Pr_check)/T_check
#     E_c_all.append(E_c)
#     D_c_all.append(D_c)
    
# plt.figure()
# for i in range(len(Omega_list)):
#     plt.plot(U_check, E_c_all[i], color=colors[i], label=f'Omega={Omega_list[i]} E')   
# for i in range(len(Omega_list)):
#     plt.plot(U_check, D_c_all[i], '--', color=colors[i], label=f'Omega={Omega_list[i]} D') 
# plt.legend()
# plt.xlabel('U [m/s]')
# plt.ylabel('E/c and D/c')
    