# -*- coding: utf-8 -*-
"""
Created on Tue Sep  9 16:20:32 2025

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
u_star = 0.56              # shear velocity [m/s]

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
  
def Preff_of_U(U, Pmax, Uc, p):
    """State-conditioned rebound fraction. PDF marks 'needs calibration'—placeholder Gompertz-like."""
    # P_min + (P_max-P_min) * (1 - exp(-(U/Uc)^p))
    # Pmin, Pmax, Uc, p = 0, 0.6, 2, 0.5 #1 #3.84, 0.76      # tune to your binned counts
    Pmin = 0
    U = abs(U)
    return Pmin + (Pmax - Pmin)*(1.0 - np.exp(-(U/max(Uc,1e-6))**p)) 

def e_COR_from_Uim(Uim):
    return 3.05 * (abs(Uim)/const + 1e-12)**(-0.47)                  

def theta_im_from_Uim(Uim):
    x = 50.40 / (abs(Uim)/const + 159.33)                            
    return np.arcsin(np.clip(x, -1.0, 1.0))

def theta_D_from_UD(UD):
    x = 163.68 / (abs(UD)/const + 156.65)                             
    return 0.28 * np.arcsin(np.clip(x, -1.0, 1.0))                   

def theta_reb_from_Uim(Uim):
    x = -0.0003*(abs(Uim)/const) + 0.52                              
    return np.arcsin(np.clip(x, -1.0, 1.0))

def NE_from_Uinc(Uinc):
    NE = 0.04 * (abs(Uinc)/const)     
    # if abs(Uinc)/const >= 10:
    #     NE = 0.8 * ((abs(Uinc)/const)-10)**0.25
    # else:
    #     NE = 0
    return NE 

def Mdrag(c, Uair, U):
    """Momentum exchange (air→saltation): c * f_drag / m_p."""
    b = 0.55
    C_D = 0.1037
    Ueff = b * Uair
    dU = Ueff - U
    fdrag = np.pi/8 * rho_a * d**2 * C_D * abs(dU) * dU                     
    return (c * fdrag) / mp       

#----------- DPM data ----------
cases = range(2, 7)  # S002..S006
C_list, U_list, Ua_list, E_list, D_list = [], [], [], [], []
t_dpm = np.linspace(0.01, 5, 501)
for i in cases:
    # CG data: columns -> Q, C, U (501 rows in your example)
    cg = np.loadtxt(f"CGdata/Shields{i:03d}dry.txt")
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

plt.figure()
plt.plot()

# def savgol_keep_first(x, window_length=21, polyorder=2):
#     x = np.asarray(x, dtype=float)

#     # handle NaNs (optional)
#     if np.isnan(x).any():
#         idx = np.arange(len(x))
#         x = np.interp(idx, idx[~np.isnan(x)], x[~np.isnan(x)])

#     # make window valid: odd, <= len(x), > polyorder
#     n = len(x)
#     w = int(window_length)
#     if w % 2 == 0: w += 1
#     w = min(w, n if n % 2 == 1 else n-1)
#     w = max(w, polyorder + 2 if (polyorder + 2) % 2 == 1 else polyorder + 3)

#     y = savgol_filter(x, window_length=w, polyorder=polyorder, mode='interp')
#     y[0] = x[0]             # keep the first value fixed
#     return y

# # example
# U_smooth = savgol_keep_first(U_dpm, window_length=21, polyorder=2)

dt = 0.01
# collect per-case series in lists first
C_com_list, U_com_list, m_com_list = [], [], []
E_com_list, D_com_list = [], []
r_im_list, r_dep_list = [], []
cim_list, cD_list = [], []
N = 501
eps, Tmin = 1e-12, 1e-6
for i in cases:
    # ---- allocate state arrays ----
    C_com = np.zeros(N); m_com = np.zeros(N)
    C_com[0] = C_list[i-2][0]
    m_com[0] = C_list[i-2][0] * U_list[i-2][0]

    E, D = np.zeros(N-1), np.zeros(N-1)
    r_im, r_dep = np.zeros(N-1), np.zeros(N-1)
    cim = np.zeros(N-1); cD = np.zeros(N-1)
    # ---- time stepping ----
    for k in range(N-1):
        U_com = m_com[k] / max(C_com[k], eps)
        Ua_k  = Ua_list[i-2][k]

        # mappings & timescales
        Uim = Uim_from_U(U_com);    UD  = UD_from_U(U_com)
        Tim = max(Tmin, Tim_from_Uim(Uim))
        TD  = max(Tmin, TD_from_UD(UD))
        P   = np.clip(Preff_of_U(U_com, 1, 3.84, 0.76), 0.0, 1.0)

        # angles/COR (your closures)
        th_im = theta_im_from_Uim(Uim)
        th_D  = theta_D_from_UD(UD)
        th_re = theta_reb_from_Uim(Uim)
        eCOR  = e_COR_from_Uim(Uim)
        Ure   = Uim * eCOR

        # mixing fraction and rates
        phi_im = P * Tim / (P*Tim + (1.0-P)*TD + 1e-12)
        x1, x2 = phi_im/Tim, (1.0-phi_im)/TD
        NEim   = max(0.0, NE_from_Uinc(Uim))
        NEd    = max(0.0, NE_from_Uinc(UD))

        a_im = x1 * NEim
        a_dep= x2 * NEd
        b    = x2
        lam  = (a_im + a_dep) - b

        # --- diagnostics ---
        cim_k = C_com[k] * phi_im
        cD_k  = C_com[k] - cim_k

        cim[k]     = cim_k
        cD[k]      = cD_k
        
        # mass rates
        r_im[k]  = cim_k / Tim
        r_dep[k] = cD_k  / TD
        # ejection mass & deposition mass (E and D)
        E[k] = r_im[k]*NEim + r_dep[k]*NEd           
        D[k] = r_dep[k]

        # streamwise momentum terms
        M_drag = Mdrag(C_com[k], Ua_k, U_com)
        M_eje = (r_im[k]*NEim + r_dep[k]*NEd) * UE_mag * cos_thetaE
        M_re  = r_im[k]  * ( Ure * np.cos(th_re) )
        M_im  = r_im[k]  * ( Uim * np.cos(th_im) )
        M_dep = r_dep[k] * ( UD  * np.cos(th_D)  )

        dm_dt = M_drag + M_eje + M_re - M_im - M_dep
        # exponential Euler for C (positivity), explicit for m
        C_com[k+1] = C_com[k] * np.exp(np.clip(lam*dt, -50.0, 50.0))
        m_com[k+1] = m_com[k] + dt*dm_dt

    # recover U_com
    U_com = m_com / np.maximum(C_com, eps)

    # store
    C_com_list.append(C_com)
    E_com_list.append(E)
    D_com_list.append(D)
    U_com_list.append(U_com)
    r_im_list.append(r_im)
    r_dep_list.append(r_dep)
    cim_list.append(cim)
    cD_list.append(cD)

# U_com = m_com / np.maximum(C_com, 1e-12)
# lam_emp = (C_dpm[1:] - C_dpm[:-1]) / (dt * np.maximum(C_dpm[:-1], 1e-12))
# lam_smooth = savgol_keep_first(lam_emp, window_length=21, polyorder=2)
# dcUdt_emp = (C_dpm[1:]*U_dpm[1:] - C_dpm[:-1]*U_dpm[:-1]) / dt
# dc_dt = (C_dpm[1:] - C_dpm[:-1]) / dt
# dUdt_emp = (dcUdt_emp - U_dpm[:-1]*dc_dt) / np.maximum(C_dpm[:-1], 1e-12)
# dUdt_smooth = savgol_keep_first(dUdt_emp, window_length=21, polyorder=2)

plt.close('all')
fig, axes = plt.subplots(2, 3, figsize=(12, 8), sharex=True)
axes = axes.ravel()  # flatten 2x3 -> 1D array of 6 Axes
for k in range(5):
    ax = axes[k]
    E = np.asarray(E_com_list[k])
    D = np.asarray(D_com_list[k])
    ax.plot(t_dpm[:-1], E, label='E computed')
    ax.plot(t_dpm[:-1], D, label='D computed')
    ax.plot(t_Dis, E_list[k], label='E DPM')
    ax.plot(t_Dis, D_list[k], label='D DPM')
    ax.set_ylabel(r'Rate [kg m$^{-2}$ s$^{-1}$]')
    ax.set_xlabel('t [s]')
    ax.set_title(rf'$\tilde{{\Theta}}$ = 0.0{k+2}')
    ax.grid(True, alpha=0.3)
# turn off any extra subplot (since you have 5 cases but 6 panels)
for k in range(5, len(axes)):
    axes[k].axis('off')
for ax in axes.ravel():          # flatten 2x3 into 1D
    ax.set_xlim(*(0,5))
    ax.set_ylim(*(0,6))
# shared legend (use the first real axes)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=2, frameon=False)
fig.tight_layout(rect=[0, 0, 1, 0.92])
plt.show()

fig, axes = plt.subplots(2, 3, figsize=(12, 8), sharex=True)
axes = axes.ravel()  # flatten 2x3 -> 1D array of 6 Axes
for k in range(5):
    ax = axes[k]
    r_im = np.asarray(r_im_list[k])
    r_dep = np.asarray(r_dep_list[k])
    ax.plot(t_dpm[1:], r_im, label='rim computed')
    ax.plot(t_dpm[1:], r_dep, label='rdep computed')
    ax.set_ylabel(r'Rate [kg m$^{-2}$ s$^{-1}$]')
    ax.set_xlabel('t [s]')
    ax.set_title(rf'$\tilde{{\Theta}}$ = 0.0{k+2}')
    ax.grid(True, alpha=0.3)
# turn off any extra subplot (since you have 5 cases but 6 panels)
for k in range(5, len(axes)):
    axes[k].axis('off')
for ax in axes.ravel():          # flatten 2x3 into 1D
    ax.set_xlim(*(0,5))
    ax.set_ylim(*(0,4))
# shared legend (use the first real axes)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=2, frameon=False)
fig.tight_layout(rect=[0, 0, 1, 0.92])
plt.show()

fig, axes = plt.subplots(2, 3, figsize=(12, 8), sharex=True)
axes = axes.ravel()  # flatten 2x3 -> 1D array of 6 Axes
for k in range(5):
    ax = axes[k]
    cim = np.asarray(cim_list[k])
    cD = np.asarray(cD_list[k])
    ax.plot(t_dpm[1:], cim, label='cim computed')
    ax.plot(t_dpm[1:], cD, label='cdep computed')
    ax.set_ylabel(r'c [kg/m$^{2}$]')
    ax.set_xlabel('t [s]')
    ax.set_title(rf'$\tilde{{\Theta}}$ = 0.0{k+2}')
    ax.grid(True, alpha=0.3)
# turn off any extra subplot (since you have 5 cases but 6 panels)
for k in range(5, len(axes)):
    axes[k].axis('off')
for ax in axes.ravel():          # flatten 2x3 into 1D
    ax.set_xlim(*(0,5))
    ax.set_ylim(*(0,0.2))
# shared legend (use the first real axes)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=2, frameon=False)
fig.tight_layout(rect=[0, 0, 1, 0.92])
plt.show()

fig, axes = plt.subplots(2, 3, figsize=(12, 8), sharex=True)
axes = axes.ravel()  # flatten 2x3 -> 1D array of 6 Axes
for k in range(5):
    ax = axes[k]
    ax.plot(t_dpm, C_com_list[k], label='computed')
    ax.plot(t_dpm, C_list[k],     label='DPM')
    ax.set_ylabel(r'C [kg m$^{-2}$]')
    ax.set_xlabel('t [s]')
    ax.set_title(rf'$\tilde{{\Theta}}$ = 0.0{k+2}')
    ax.grid(True, alpha=0.3)
# turn off any extra subplot (since you have 5 cases but 6 panels)
for k in range(5, len(axes)):
    axes[k].axis('off')
for ax in axes.ravel():          # flatten 2x3 into 1D
    ax.set_xlim(*(0,5))
    ax.set_ylim(*(0,0.45))
# shared legend (use the first real axes)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=2, frameon=False)
fig.tight_layout(rect=[0, 0, 1, 0.92])
plt.show()

plt.figure(figsize=(8,8))
plt.subplot(2,2,1)
plt.plot(t_dpm, C_com_list[4], label='computed')
plt.plot(t_dpm, C_list[4],     label='DPM')
plt.xlabel('t [s]')
plt.ylabel(r'C [kg m$^{-2}$]')
plt.legend()
plt.subplot(2,2,2)
plt.plot(t_dpm[:-1], E_com_list[4], color='C0', label='E computed')
plt.plot(t_dpm[:-1], D_com_list[4], 'C0--', label='D computed')
plt.plot(t_Dis, E_list[4], color='C1', label='E DPM')
plt.plot(t_Dis, D_list[4], color='C1', linestyle='--', label='D DPM')
plt.xlabel('t [s]')
plt.ylabel(r'Rate [kg m$^{-2}$ s$^{-1}$]')
plt.legend()
plt.subplot(2,2,3)
plt.plot(t_dpm[1:], cim, label='cim computed')
plt.plot(t_dpm[1:], cD, label='cD computed')
plt.xlabel('t [s]')
plt.ylabel(r'C [kg m$^{-2}$]')
plt.legend()
plt.subplot(2,2,4)
plt.plot(t_dpm[1:], r_im, label='cim/Tim computed')
plt.plot(t_dpm[1:], r_dep, label='cD/TD computed')
plt.xlabel('t [s]')
plt.ylabel(r'Rate [kg m$^{-2}$ s$^{-1}$]')
plt.legend()
plt.suptitle('Shields=0.06')
plt.tight_layout()

# plt.figure(figsize=(12,10))
# plt.subplot(2,2,1)
# plt.plot(t_dpm, C_dpm, label='DPM')
# plt.plot(t_dpm, C_com, label='computed')
# plt.xlabel('t [s]')
# plt.ylabel(r'C [kg/m$^2$]')
# plt.legend()
# plt.subplot(2,2,2)
# plt.plot(t_dpm[1:], lam_smooth, label='λ_DPM')
# plt.plot(t_dpm[1:], diag['lam'], label='λ_model')
# plt.xlabel('t [s]')
# plt.ylabel(r'$\lambda$ [s$^{-1}$]')
# plt.legend()
# plt.subplot(2,2,3)
# plt.plot(t_dpm, U_dpm, label='DPM')
# plt.plot(t_dpm, U_com, label='computed')
# plt.xlabel('t [s]')
# plt.ylabel(r'U [m/s]')
# plt.legend()
# plt.subplot(2,2,4)
# plt.plot(t_dpm[1:], dUdt_smooth, label='dUdt_DPM')
# plt.plot(t_dpm[1:], diag['dUdt'], label='dUdt_com')
# plt.xlabel('t [s]')
# plt.ylabel(r'dUdt [m/s$^{2}$]')
# plt.legend()
# plt.grid(True)


# plt.figure(figsize=(6,5))
# plt.plot(t_dpm[1:], diag['r_im'], label='r_im')
# plt.plot(t_dpm[1:], diag['r_dep'], label='r_dep')
# plt.plot(t_dpm[1:], diag['E'], label='E')
# plt.xlabel('t [s]')
# plt.legend(); plt.grid(True)


# plt.figure(figsize=(6,5))
# plt.plot(t_dpm[1:], lam_smooth, label='λ_DPM')
# plt.plot(t_dpm[1:], diag['lam'], label='λ_model')
# plt.plot(t_dpm[1:], [x + y for x, y in zip(diag['a_im'], diag['a_dep'])], label='a')
# # plt.plot(t_dpm[1:], diag['a_dep'], label='a_dep')
# plt.plot(t_dpm[1:], diag['b'], label='b')
# plt.xlabel('t [s]')
# plt.ylabel(r'$\lambda$ [s$^{-1}$]')
# plt.legend(); plt.grid(True)

# from scipy.optimize import least_squares

# # --- helpers: simulate C(t) for given Pr parameters ---
# def simulate_C(params, U_series, C0, dt):
#     Pmax, Uc, p = params
#     # guardrails
#     Pmax = float(np.clip(Pmax, 0.0, 0.999))
#     Uc   = float(max(Uc, 1e-6))
#     p    = float(max(p, 1e-3))

#     T = len(U_series)
#     C = np.zeros(T); C[0] = C0
#     lam = np.zeros(T-1)

#     for i in range(T-1):
#         U  = U_series[i]
#         Uim = Uim_from_U(U);  UD = UD_from_U(U)
#         Tim = max(Tmin, Tim_from_Uim(Uim))
#         TD  = max(Tmin, TD_from_UD(UD))

#         # Pr(U) with parameters to fit
#         Uabs = abs(U)
#         P = Pmax * (1.0 - np.exp(-(Uabs/Uc)**p))      # Gompertz/Weibull-like CDF
#         P = np.clip(P, 0.0, 0.999)

#         # mixture fraction and rates
#         phi = P * Tim / (P*Tim + (1.0-P)*TD + 1e-12)
#         x1  = phi/Tim
#         x2  = (1.0-phi)/TD

#         NEim = max(0.0, NE_from_Uinc(Uim))
#         NEd  = max(0.0, NE_from_Uinc(UD))

#         a_im  = x1 * NEim
#         a_dep = x2 * NEd
#         b     = x2
#         lam_i = (a_im + a_dep) - b
#         lam[i] = lam_i

#         # Exponential Euler step
#         C[i+1] = C[i] * np.exp(np.clip(lam_i*dt, -50.0, 50.0))

#     return C, lam

# # --- residuals for least-squares fit ---
# def residuals(params):
#     C_mod, lam_mod = simulate_C(params, U_smooth, C_dpm[0], dt)
#     # main misfit on C(t)
#     rC = (C_mod - C_dpm)

#     # weights: downweight tiny C, emphasize transient (0–1.5 s)
#     w_mag   = np.sqrt(np.clip(C_dpm, 1e-6, None))
#     w_trans = np.ones_like(C_dpm); w_trans[:int(1.5/dt)] = 3.0
#     w = w_mag * w_trans
#     rC *= w

#     # small steady-state constraint: mean lambda ≈ 0 (match DPM trend)
#     lam_emp = np.gradient(C_dpm, dt) / np.maximum(C_dpm, 1e-12)
#     # smooth a little to tame noise
#     win = 21 if len(C_dpm) >= 21 else (len(C_dpm)//2*2+1)
#     lam_emp = savgol_filter(lam_emp, win, 2, mode='interp')
#     Ns = min(100, len(C_dpm)//5)
#     ss_pen = (lam_mod[-Ns:].mean() - lam_emp[-Ns:].mean())
#     return np.hstack([rC, 0.1*ss_pen])   # 0.1 sets a light weight on the penalty

# # --- run the fit ---
# p0 = np.array([0.6, 2.0, 0.5])          # initial guess: [Pmax, Uc, p]
# lb = np.array([0.2, 0.1, 0.2])          # bounds (tune as needed)
# ub = np.array([0.999, 20.0, 5.0])

# res = least_squares(residuals, p0, bounds=(lb, ub), verbose=2, max_nfev=300)
# Pmax_hat, Uc_hat, p_hat = res.x
# print(f"Fitted: Pmax={Pmax_hat:.3f}, Uc={Uc_hat:.3f}, p={p_hat:.3f}")

# # --- recompute with fitted parameters and plot ---
# C_fit, lam_fit = simulate_C(res.x, U_smooth, C_dpm[0], dt)

# plt.figure(figsize=(12,5))
# t = t_dpm
# plt.subplot(1,2,1)
# plt.plot(t, C_dpm, label='DPM')
# plt.plot(t, C_fit, label='computed (fitted)')
# plt.xlabel('t [s]'); plt.ylabel(r'C [kg/m$^2$]'); plt.legend(); plt.grid(True)

# plt.subplot(1,2,2)
# lam_emp = np.gradient(C_dpm, dt) / np.maximum(C_dpm, 1e-12)
# plt.plot(t[1:], lam_emp[1:], label=r'$\lambda_{\rm DPM}$')
# plt.plot(t[1:], lam_fit,       label=r'$\lambda_{\rm model}$')
# plt.xlabel('t [s]'); plt.ylabel(r'$\lambda$ [s$^{-1}$]'); plt.legend(); plt.grid(True)


# # plt.figure()
# # plt.plot(t_dpm[1:], diag['Uim']/const)
# # plt.xlabel('t [s]')
# # plt.ylabel(r'$U_{im}/\sqrt{gd}$ [-]')

# # plt.figure()
# # plt.plot(t_dpm, U_smooth)
# # plt.xlabel('t [s]')
# # plt.ylabel(r'$U_{sal}$ [m/s]')


# U_s = np.linspace(0,max(U_smooth), 100)
# Pr = Preff_of_U(U_s, 1, 3.84, 0.76)
# Pr_adj = Preff_of_U(U_s, 0.999, 10.038, 0.429)
 
# plt.figure()
# plt.plot(U_s, Pr, label='ori')
# plt.plot(U_s, Pr_adj, label='adj')
# plt.xlabel(r'$U_{sal}$ [m/s]')
# plt.ylabel(r'$Pr_{eff}$')
# plt.legend()

# # # balance in steady state
# # Us = np.mean(U_smooth[200:])
# # Uim = Uim_from_U(Us);  UD = UD_from_U(Us)      # your mappings
# # Tim = max(Tmin, Tim_from_Uim(Uim))
# # TD  = max(Tmin, TD_from_UD(UD))
# # P   = np.clip(Preff_of_U(Us, 0.6, 2, 0.5), 0.0, 1.0)

# # phi = P * Tim / (P*Tim + (1-P)*TD)
# # x1  = phi/Tim
# # x2  = (1.0-phi)/TD
# # NEim = max(0.0, NE_from_Uinc(Uim))           # keep nonnegative
# # NEd  = max(0.0, NE_from_Uinc(UD))

# # a_im  = x1 * NEim 
# # a_dep = x2 * NEd 
# # a = a_im + a_dep
# # b = x2 
# # lams = a - b 
# # print('lam_S', lams)