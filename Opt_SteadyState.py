# -*- coding: utf-8 -*-
"""
Created on Mon Sep 15 16:12:01 2025

@author: WangX3
"""

import numpy as np
from scipy.optimize import brentq, least_squares
import matplotlib.pyplot as plt

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

def find_root_on_grid(fvals, xgrid, target=0.0):
    """
    Find x where f(x)=target given sampled fvals=f(xgrid).
    Returns np.nan if no sign change bracket is found.
    """
    g = fvals - target
    s = np.sign(g)
    # look for sign change between neighbors
    changes = np.where(s[:-1] * s[1:] < 0)[0]
    if len(changes) == 0:
        return np.nan
    i = changes[0]
    a, b = xgrid[i], xgrid[i+1]
    fa, fb = g[i], g[i+1]
    # refine with brentq on a local lambda built from linear interpolation of fvals
    def f_scalar(x):
        # local linear interpolation of sampled f
        return np.interp(x, xgrid, fvals) - target
    try:
        return brentq(f_scalar, a, b)
    except ValueError:
        # fallback to midpoint if brentq fails for numerical reasons
        return 0.5*(a+b)

def safe_interp(y_target, y, x):
    """
    Monotone-safe interpolation x(y)=? using numpy.interp on a
    monotonically increasing copy. If not monotone, fall back to argmin.
    """
    y = np.asarray(y); x = np.asarray(x)
    if np.all(np.diff(y) > 0) or np.all(np.diff(y) < 0):
        if y[0] > y[-1]:  # make increasing
            y = y[::-1]; x = x[::-1]
        y_target = np.clip(y_target, y.min(), y.max())
        return float(np.interp(y_target, y, x))
    # fallback: pick nearest
    idx = np.argmin(np.abs(y - y_target))
    return float(x[idx])

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
    return np.arcsin(np.clip(x, 0, 1.0))

def theta_D_from_UD(UD):
    x = 163.68 / (np.abs(UD) / const + 156.65)
    x = np.clip(x, 0.0, 1.0)
    theta = 0.28 * np.arcsin(x)
    return np.clip(theta, 0.0, 0.5 * np.pi)                  

def theta_reb_from_Uim(Uim):
    x = -0.0003*(abs(Uim)/const) + 0.52                              
    return np.arcsin(np.clip(x, 0, 1.0))

# def Mdrag(c, Uair, U):
#     """Momentum exchange (air→saltation): c * f_drag / m_p."""
#     b = 0.55
#     C_D = 0.1037
#     Ueff = b * Uair
#     dU = Ueff - U
#     fdrag = np.pi/8 * rho_a * d**2 * C_D * abs(dU) * dU                     
#     return (c * fdrag) / mp       

# def tau_bed(Uair, U, c):                    
#     K = 0.08
#     beta = 0.9
#     U_eff = beta*Uair
#     tau_bed_minusphib = rho_a*K*c/(rho_p*d) * abs(U_eff - U)*(U_eff - U)                            
#     return tau_bed_minusphib

# def MD_eff(Ua, U, c):
#     Uaeff = 0.85*Ua
#     MD_eff = rho_a * 0.12 * c/(rho_p*d) *abs(Uaeff - U) * (Uaeff - U)
#     return MD_eff

# tuning functions
def Preff_of_U_param(U, params):
    Pmax, Uc, pshape, _, _, _, _, _, _,_ = params
    Pmin = 0.0
    U = np.abs(U)
    return np.clip(Pmin + (Pmax - Pmin)*(1.0 - np.exp(-(U/max(Uc,1e-6))**pshape)), 0.0, 1.0)

def e_COR_from_Uim_param(Uim, params, const):
    _, _, _, A_e, B_e, _, _, _, _,_ = params
    return A_e * (np.abs(Uim)/const + 1e-12)**(-B_e)

def NE_from_Uinc_param(Uinc, params, const):
    _, _, _, _, _, A_NE, _, _, _,_ = params
    return A_NE * (np.abs(Uinc)/const) 

def Mdrag_overc_param(Uair, U, params):
    """Momentum exchange (air→saltation): c * f_drag / m_p."""
    _,_,_,_,_,_,b, C_D, _,_ = params
    Ueff = b * Uair
    dU = Ueff - U
    fdrag = np.pi/8 * rho_a * d**2 * C_D * abs(dU) * dU                     
    return fdrag / mp    

def MD_eff(Ua, U, c, params):
    _,_,_,_,_,_,_,_, b_eff, CD_eff = params
    Uaeff = b_eff*Ua
    MD_eff = rho_a * CD_eff * c/(rho_p*d) *abs(Uaeff - U) * (Uaeff - U)
    return MD_eff

# helper
def Cal_EDM_param(U, params, const, UE_mag, cos_thetaE,
                  Uim_from_U, UD_from_U, Tim_from_Uim, TD_from_UD,
                  theta_im_from_Uim, theta_D_from_UD, theta_reb_from_Uim):
    Uim = Uim_from_U(U);    UD  = UD_from_U(U)

    if np.ndim(U) != 0:
        Tim = np.maximum(np.ones_like(U)*1e-6, Tim_from_Uim(Uim))
        TD  = np.maximum(np.ones_like(U)*1e-6, TD_from_UD(UD))
        NEim   = np.maximum(np.zeros_like(U), NE_from_Uinc_param(Uim, params, const))
        NEd    = np.maximum(np.zeros_like(U), NE_from_Uinc_param(UD,  params, const))
    else:
        Tim = max(1e-6, Tim_from_Uim(Uim))
        TD  = max(1e-6, TD_from_UD(UD))
        NEim   = max(0.0, NE_from_Uinc_param(Uim, params, const))
        NEd    = max(0.0, NE_from_Uinc_param(UD,  params, const))

    P   = Preff_of_U_param(U, params)
    th_im = theta_im_from_Uim(Uim)
    th_D  = theta_D_from_UD(UD)
    th_re = theta_reb_from_Uim(Uim)
    eCOR  = e_COR_from_Uim_param(Uim, params, const)
    Ure   = Uim * eCOR

    phi_im = P * Tim / (P*Tim + (1.0-P)*TD + 1e-12)
    phi_D  = 1-phi_im

    E_overc = phi_im/Tim*NEim + phi_D/TD*NEd
    D_overc = phi_D/TD

    M_eje_overc = E_overc * UE_mag * cos_thetaE
    M_re_overc  = phi_im/Tim  * ( Ure * np.cos(th_re) )
    M_im_overc  = phi_im/Tim  * ( Uim * np.cos(th_im) )
    M_dep_overc = D_overc *   ( UD  * np.cos(th_D)  )
    M_bed = M_eje_overc + M_im_overc - M_re_overc - M_dep_overc

    return E_overc, D_overc, M_bed

def predict_steady(params, Shields, u_star, const, rho_a, mp,
                   UE_mag, cos_thetaE, d,
                   Uim_from_U, UD_from_U, Tim_from_Uim, TD_from_UD,
                   theta_im_from_Uim, theta_D_from_UD, theta_reb_from_Uim,
                   Mdrag_overc_param, MD_eff):
    Usteady = np.zeros_like(Shields)
    Uasteady = np.zeros_like(Shields)
    csteady = np.zeros_like(Shields)

    # grids for root/lookup
    U_grid  = np.linspace(0.01, 2.0, 400)
    Ua_grid = np.linspace(5.0,  25.0, 400)
    c_grid  = np.linspace(0.0,  0.2, 400)

    for i in range(len(Shields)):
        # 1) steady U: E_overc - D_overc = 0
        E_overc, D_overc, _ = Cal_EDM_param(U_grid, params, const, UE_mag, cos_thetaE,
                                            Uim_from_U, UD_from_U, Tim_from_Uim, TD_from_UD,
                                            theta_im_from_Uim, theta_D_from_UD, theta_reb_from_Uim)
        root_U = find_root_on_grid(E_overc - D_overc, U_grid)
        if np.isnan(root_U):
            # fallback to nearest zero of sampled curve
            root_U = safe_interp(0.0, E_overc - D_overc, U_grid)
        Usteady[i] = root_U

        # 2) steady Ua: M_bed(Usteady) = M_drag_overc(Ua, Usteady)
        _, _, M_bed_steady = Cal_EDM_param(Usteady[i], params, const, UE_mag, cos_thetaE,
                                           Uim_from_U, UD_from_U, Tim_from_Uim, TD_from_UD,
                                           theta_im_from_Uim, theta_D_from_UD, theta_reb_from_Uim)
        # dU = 0.55*Ua_grid - Usteady[i]
        # C_D = 0.1037
        M_drag_overc = Mdrag_overc_param(Ua_grid, Usteady[i], params)
        Uasteady[i] = safe_interp(M_bed_steady, M_drag_overc, Ua_grid)

        # 3) steady c: tau_top = Mdrag(c) + tau_bed(c)
        tau_top = rho_a * u_star[i]**2
        # M_drag_c = Mdrag(c_grid, Uasteady[i], Usteady[i])
        # tau_bed_c = tau_bed(Uasteady[i], Usteady[i], c_grid)
        M_drag_eff_c = MD_eff(Uasteady[i], Usteady[i], c_grid, params)
        csteady[i] = safe_interp(tau_top, M_drag_eff_c, c_grid)

    return Usteady, Uasteady, csteady

def make_residuals_fn(Shields, u_star, const, rho_a, mp, UE_mag, cos_thetaE, d,
                      Uim_from_U, UD_from_U, Tim_from_Uim, TD_from_UD,
                      theta_im_from_Uim, theta_D_from_UD, theta_reb_from_Uim,
                      Mdrag_overc_param, MD_eff,
                      Us_dpm, Uas_dpm, cs_dpm,
                      wU, wUa, wc):

    # targets for U and Ua are their DPM means
    Us_mean  = float(np.mean(Us_dpm))
    Uas_mean = float(np.mean(Uas_dpm))

    # scales to normalize residual magnitudes
    sU  = max(abs(Us_mean),  1e-6)
    sUa = max(abs(Uas_mean), 1e-6)
    sc  = max(np.mean(np.abs(cs_dpm)), 1e-6)

    def residuals(params):
        U, Ua, c = predict_steady(params, Shields, u_star, const, rho_a, mp,
                                  UE_mag, cos_thetaE, d,
                                  Uim_from_U, UD_from_U, Tim_from_Uim, TD_from_UD,
                                  theta_im_from_Uim, theta_D_from_UD, theta_reb_from_Uim,
                                  Mdrag_overc_param, MD_eff)
        # Pull U and Ua to their DPM means; keep c point-wise
        rU  = wU  * (U  - Us_mean)  / sU
        rUa = wUa * (Ua - Uas_mean) / sUa
        rc  = wc  * (c  - cs_dpm)   / sc
        return np.concatenate([rU, rUa, rc])

    return residuals

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

# --- initial guess and bounds for params: [Pmax, Uc, pshape, A_e, B_e, A_NE]
p0     = np.array([0.6,  2.0,  0.7,  3.0,  0.5, 0.05, 0.55, 0.15, 0.85, 0.12])
lb     = np.array([0.3,  0.1,  0.2,  0.5,  0.1, 0.005, 0.01, 0.001, 0.01, 0.001])
ub     = np.array([0.99, 8.0,  3.0, 10.0,  1.0, 0.2, 1.0, 0.5, 1.0, 0.5])

resfn = make_residuals_fn(Shields, u_star, const, rho_a, mp, UE_mag, cos_thetaE, d,
                          Uim_from_U, UD_from_U, Tim_from_Uim, TD_from_UD,
                          theta_im_from_Uim, theta_D_from_UD, theta_reb_from_Uim, Mdrag_overc_param, MD_eff,
                          Us_dpm, Uas_dpm, cs_dpm,
                          wU=1, wUa=1, wc=1)

opt = least_squares(resfn, p0, bounds=(lb, ub), method='trf', loss='soft_l1', f_scale=1.0, xtol=1e-8, ftol=1e-8)

print("Best-fit params [Pmax, Uc, p, A_e, B_e, A_NE, b, C_D, b_eff, C_D_eff] =", opt.x)
Ufit, Uafit, cfit = predict_steady(opt.x, Shields, u_star, const, rho_a, mp,
                                   UE_mag, cos_thetaE, d,
                                   Uim_from_U, UD_from_U, Tim_from_Uim, TD_from_UD,
                                   theta_im_from_Uim, theta_D_from_UD, theta_reb_from_Uim,
                                   Mdrag_overc_param, MD_eff)

plt.close('all')
plt.figure(figsize=(15,4))
plt.subplot(1,3,1)
plt.plot(Shields, Ufit, label='computed')
plt.scatter(Shields, Us_dpm, label='DPM')
plt.ylim(bottom=0)
plt.xlabel('Shields number')
plt.ylabel('Usteady')
plt.legend(loc='lower left')
plt.subplot(1,3,2)
plt.plot(Shields, Uafit, label='computed')
plt.scatter(Shields, Uas_dpm, label='DPM')
plt.xlabel('Shields number')
plt.ylabel('Uasteady')
plt.legend()
plt.subplot(1,3,3)
plt.plot(Shields, cfit, label='computed')
plt.scatter(Shields, cs_dpm, label='DPM')
plt.xlabel('Shields number')
plt.ylabel('c_steady')
plt.legend()