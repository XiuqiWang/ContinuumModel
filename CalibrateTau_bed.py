# -*- coding: utf-8 -*-
"""
Created on Thu Aug 14 17:37:15 2025

@author: WangX3
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
import pandas as pd
from scipy.signal import savgol_filter

# from FD to Ua; Calibrate h dUa/dt = Mom_gain - Mom_loss - Mom_drag
h = 0.2 - 0.00025*13.5
D = 0.00025
kappa = 0.4
rho_a = 1.225
rho_sand = 2650
nu_a = 1.46e-5
Ruc = 24
Cd_inf = 0.5
Shields = np.linspace(0.02, 0.06, 5)
u_star = np.sqrt(Shields * (2650-1.225)*9.81*D/1.225)
mp = 2650 * np.pi/6 * D**3 #particle mass
t = np.linspace(0, 5, 501)
dt = np.mean(np.diff(t))
#-------- check values!!!! -------
b0=0.015
b_inf=0.79
k0 = 0.53
lamda = 4.86

def r2_score(y, ypred):
    ss_res = np.sum((y - ypred)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    return 1.0 - ss_res/ss_tot

# def tau_bottom_phi_model(Ua_arr, l_eff, phi_b):
#     """Height-averaged mixing-length bottom stress"""
#     tau_bed = rho_a * (nu_a + (l_eff**2) * np.abs(Ua_arr)/h) * Ua_arr / h
#     return tau_bed * (1-phi_b)

# def tau_bed_dragform(x, beta, K):
#     Ua, U, c = x
#     Uabed = beta*Ua
#     tau_b_oneminusphib = rho_a * K * c/(rho_sand*D) * abs(Uabed-U) * (Uabed-U) 
#     return tau_b_oneminusphib

def CalMcreep(Mdrag, gamma):
    Mresidual = gamma*Mdrag
    return Mresidual 

def CalMdrag(x, b):
    Ua, U, c = x
    Urel = b * Ua - U
    Re = abs(Urel)*D/nu_a
    Cd = (np.sqrt(Cd_inf)+np.sqrt(Ruc/Re))**2   
    Mdrag = np.pi/8 * D**2 * rho_a * Urel * abs(Urel) * Cd * c/mp
    return Mdrag

# def Fitb(x, b0, b_inf, k0, lamda):
#     U, c = x
#     k = k0/(1+lamda*c)
#     b = b0 + (b_inf - b0)*(1 - np.exp(-k*U))
#     return b

def weighted_r2(y_true, y_pred, weights):
    y_avg = np.average(y_true, weights=weights)
    ss_res = np.sum(weights * (y_true - y_pred)**2)
    ss_tot = np.sum(weights * (y_true - y_avg)**2)
    return 1 - ss_res / ss_tot

# Containers for storing results
Ua_all_S, U_all_S, c_all_S = [], [], []
RHS_se_all_S, RHS_all_S, LHS_all_S = [], [], []
Uat_all_S , LHSt_all_S, RHSt_all_S = [], [], []
Mres_all_S, MD_all_S = [], []

omega_labels = ['Dry', 'M1', 'M5', 'M10', 'M20']
Omega = [0, 1, 5, 10, 20]

# ---- Load data ----
for label in omega_labels:
    for i in range(2, 7):
        file_ua = f'TotalDragForce/Ua-t/Uair_ave-tS00{i}{label}.txt'
        Ua_dpm = np.loadtxt(file_ua, delimiter='\t')[:, 1]
        dUa_dt = np.gradient(Ua_dpm, dt) 
        
        file_c = f'CGdata/hb=13.5d/Shields00{i}{label}-135d.txt'
        data_dpm = np.loadtxt(file_c)
        c_dpm = data_dpm[:, 1]
        U_dpm = data_dpm[:, 2]
        phi = c_dpm/(rho_sand*h)
        
        file_fd = f'TotalDragForce/Mdrag/FD_S00{i}{label}.txt'
        data_FD = np.loadtxt(file_fd)
        Mdrag_dpm = data_FD
        
        #---- compute RHS-t and binned RHS ----
        tau_top = np.ones(len(dUa_dt))*rho_a*u_star[i-2]**2
        M_aero = 0.5 * rho_a * 0.0037 * Ua_dpm * abs(Ua_dpm)
        RHS_t = tau_top - rho_a * h * dUa_dt * (1-phi) - Mdrag_dpm - M_aero
        M_residual = tau_top - rho_a * h * dUa_dt * (1-phi) - Mdrag_dpm
        # RHS_binned, RHS_se, U_binned, c_binned, Ua_binned = BintaubUa(Ua_dpm, U_dpm, c_dpm, RHS_t, Ua_bin)
        
        #----- compute LHS-t with the optimised parameters -----
        # LHS_t = tau_bed_dragform((Ua_dpm, U_dpm, c_dpm), 1.58, 0.08)
    
        # ---- Store results ----
        # first time-varying values
        Ua_all_S.append(Ua_dpm)
        U_all_S.append(U_dpm)
        c_all_S.append(c_dpm)
        MD_all_S.append(Mdrag_dpm)
        RHS_all_S.append(RHS_t)
        Mres_all_S.append(M_residual)


# Ua_all= np.concatenate(Ua_all_S)
MD_all = np.concatenate(MD_all_S)
RHS_all = np.concatenate(RHS_all_S)
# ustar_block = np.repeat(u_star, 501)   # one block of 5×501 = 2505 elements
# ustar_vec = np.tile(ustar_block, 5)   # repeat the block 5 times
mask = np.isfinite(MD_all) & np.isfinite(RHS_all) 
RHS_all, MD_all = RHS_all[mask], MD_all[mask]
# ustar_vec = ustar_vec[mask]

popt, _ = curve_fit(CalMcreep, MD_all, RHS_all, maxfev=20000)
gamma = popt[0]
print(f'gamma={gamma:.2f}')
RHS_pred = CalMcreep(MD_all, gamma)
r2 = r2_score(RHS_all, RHS_pred)
print('r2', r2)
    
# ---- Plotting ----
plt.close('all') 
    
for i in range(5): #Omega
    plt.figure(figsize=(10, 8))
    for j in range(5): #Shields
        plt.subplot(3, 2, j+1)
        index_byS = i*5+j 
        plt.plot(t, RHS_all_S[index_byS], '.', label=r'$\hat{M}_\mathrm{rep}=\hat{M}_\mathrm{top,air} -\rho_\mathrm{air}h(1-\frac{\hat{c}}{m_\mathrm{p}h})\frac{d\hat{U}_\mathrm{air}}{dt} - \hat{M}_{sal} - \hat{M}_\mathrm{bed,air}$')
        Mcreep = CalMcreep(MD_all_S[index_byS], gamma)
        plt.plot(t, Mcreep, '.', label=fr'$\breve{{M}}_{{\mathrm{{rep}}}} = {gamma:.2f}\,\hat{{M}}_{{\mathrm{{sal}}}}$')
        plt.title(fr"$\tilde{{\Theta}}$=0.0{j+2}")
        plt.xlabel(r"$t$ [s]")
        plt.ylabel(r"$M_\mathrm{rep}$ [N/m$^2$]")
        plt.ylim(0,2.5)
        plt.xlim(0,5)
        plt.grid(True)
        if j == 0:
            plt.legend(fontsize=9, loc='upper right')
    plt.suptitle(fr'$\Omega$={Omega[i]}%')
    plt.tight_layout()
    plt.show()

# # all in one figure
# colors = plt.cm.viridis(np.linspace(1, 0, 5))  # 5 colors
# fig, axes = plt.subplots(3, 2, figsize=(10, 8))
# axes = axes.flatten()  # for easy indexing
# for j in range(5):  # Shields index → one subplot per Shields
#     ax = axes[j]
#     for i in range(5):  # Omega index → multiple curves per subplot
#         index_byS = i * 5 + j
#         # ----- Measured (DPM residual) -----
#         ax.plot(
#             t,
#             RHS_all_S[index_byS],
#             '.', color=colors[i],
#             label=fr'$\Omega$={Omega[i]}%' if j == 0 else None
#         )
#         # ----- Computed -----
#         Mcreep = CalMcreep(MD_all_S[index_byS], gamma)
#         ax.plot(
#             t,
#             Mcreep,
#             '--', color=colors[i])
#     ax.plot([], [], '.', color='black', label=r"$\hat{M}_\mathrm{creep}$")
#     ax.plot([], [], '--', color='black', label=r"$M_\mathrm{creep}$")
#     ax.set_title(fr"$\tilde{{\Theta}}$=0.0{j+2}")
#     ax.set_xlabel("t [s]")
#     ax.set_ylabel(r"$M_{\mathrm{creep}}$ [N/m$^2$]")
#     ax.set_xlim(0, 5)
#     ax.set_ylim(0, 2.5)
#     ax.grid(True)
# # Hide unused 6th panel
# axes[5].axis("off")
# # ---- Single global legend (for Omega) ----
# handles, labels = axes[0].get_legend_handles_labels()
# fig.legend(handles, labels, fontsize=9, loc="upper right")
# plt.tight_layout()
# plt.show()

# for i in range(5): #Omega
#     plt.figure(figsize=(10, 10))
#     for j in range(5): #Shields
#         plt.subplot(3, 2, j+1)
#         index_byS = i*5+j 
#         # Mbedfriction = CalMbedfriction((Ua_all_S[index_byS], U_all_S[index_byS], c_all_S[index_byS], MD_all_S[index_byS], u_star[j]), B, p)
#         plt.plot(t, RHS_all_S[index_byS], '.', label=r'DPM $\hat{M}_{residual}=\tau_{top} -\rho_ah(1-\frac{c}{m_ph})\frac{d\hat{U}a}{dt} - \hat{M}_{drag}$')
#         plt.plot(t, MD_all_S[index_byS], '.', label=r'DPM $\hat{M}_{drag}$')
#         plt.plot(t, np.ones(len(t))*rho_a*u_star[j]**2, label=r'$\tau_{top}$')
#         M_areo = 0.5*rho_a*0.0037*Ua_all_S[index_byS]**2
#         plt.plot(t, M_areo, label=r'$M_{aero}=0.5\rho_aC_{D,bed}{\hat{U}_a}^2$')
#         # plt.plot(t, (RHS_all_S[index_byS] - M_areo)/MD_all_S[index_byS], label='(Mbed-Maero)/Mdrag')
#         # plt.plot(t, Mbedfriction, '.', label=r'Computed $M_{bedfriction}$')
#         plt.title(fr"$\tilde{{\Theta}}$=0.0{j+2}, $\Omega$={Omega[i]}%")
#         plt.xlabel("t [s]")
#         plt.ylabel(r"Momentum terms for air [N/m$^2$]")
#         plt.ylim(-0.5,2.5)
#         plt.xlim(0,5)
#         plt.grid(True)
#         if j == 0:
#             plt.legend(fontsize=10, loc='upper right')
#     plt.tight_layout()
#     plt.show()
    
# for i in range(5): #Omega
#     plt.figure(figsize=(10, 8))
#     for j in range(5): #Shields
#         plt.subplot(3, 2, j+1)
#         index_byS = i*5+j 

#         N = len(MD_all_S[index_byS])
#         t = np.linspace(0, 5, N)  # actual timestamps if uniformly sampled

#         sc = plt.scatter(
#             MD_all_S[index_byS], 
#             RHS_all_S[index_byS], 
#             c=t, 
#             cmap='viridis', 
#             s=10
#         )

#         plt.title(fr"$\tilde{{\Theta}}$=0.0{j+2}, $\Omega$={Omega[i]}%")
#         plt.xlabel(r"$\hat{M}_{drag}$ [N/m$^2$]")
#         plt.ylabel(r"$\hat{M}_{bedfriction}$ [N/m$^2$]")
#         plt.ylim(0,2.5)
#         plt.xlim(0,2.5)
#         plt.grid(True)

#         if j == 0:
#             cbar = plt.colorbar(sc)
#             cbar.set_label("time [s]")
#     plt.tight_layout()
#     plt.show()

plt.figure(figsize=(6,10))
plt.subplot(2,1,1)
plt.plot(t, Mres_all_S[4], '.', label=r'DPM $\hat{M}_{residual}=\tau_{top} -\rho_ah(1-\frac{c}{m_ph})\frac{d\hat{U}a}{dt} - \hat{M}_{drag}$')
plt.plot(t, MD_all_S[4], '.', label=r'DPM $\hat{M}_{drag}$')
M_areo = 0.5*rho_a*0.0037*Ua_all_S[4]**2
plt.plot(t, M_areo, '.', label=r'$M_{aero}=0.5\rho_aC_{D,bed}{\hat{U}_a}^2$')
plt.plot(t, Mres_all_S[4]-M_areo, '.', label=r'$M_{creep} = \hat{M}_{residual} - M_{aero}$')
plt.plot(t, (Mres_all_S[4] - M_areo)/MD_all_S[4], label=r'$M_{creep}/\hat{M}_{drag}$')
plt.xlabel('t [s]')
plt.title('Dry')
plt.legend()
plt.subplot(2,1,2)
plt.plot(t, Mres_all_S[24], '.')
plt.plot(t, MD_all_S[24], '.')
M_areo = 0.5*rho_a*0.0037*Ua_all_S[24]**2
plt.plot(t, M_areo, '.')
plt.plot(t, Mres_all_S[24]-M_areo, '.')
plt.plot(t, (Mres_all_S[24] - M_areo)/MD_all_S[24])
plt.xlabel('t [s]')
plt.title(r'$\Omega$=20$\%$')
plt.suptitle('Shields=0.06')
plt.tight_layout()
     

# def BintaubUa(Ua, U, c, RHS, Uabin):
#     Ua = np.asarray(Ua, dtype=float)
#     RHS = np.asarray(RHS, dtype=float)
#     edges = np.asarray(Uabin, dtype=float)

#     if Ua.shape != RHS.shape:
#         raise ValueError("Ua and RHS must have the same shape.")
#     if edges.ndim != 1 or edges.size < 2:
#         raise ValueError("Uabin must be a 1D array of length >= 2 (bin edges).")
#     if not np.all(np.diff(edges) > 0):
#         raise ValueError("Uabin must be strictly increasing.")

#     # keep only finite pairs
#     m = np.isfinite(Ua) & np.isfinite(RHS)
#     Ua = Ua[m]
#     RHS = RHS[m]

#     nbins = edges.size - 1
#     RHS_mean   = np.full(nbins, np.nan, dtype=float)
#     RHS_se     = np.full(nbins, np.nan, dtype=float)
#     Ua_mean  = np.full(nbins, np.nan, dtype=float)
#     U_mean     = np.full(nbins, np.nan, dtype=float)
#     c_mean     = np.full(nbins, np.nan, dtype=float)

#     for i in range(nbins):
#         lo, hi = edges[i], edges[i+1]
#         # right-inclusive on the last bin
#         if i < nbins - 1:
#             sel = (Ua >= lo) & (Ua < hi)
#         else:
#             sel = (Ua >= lo) & (Ua <= hi)

#         if not np.any(sel):
#             continue

#         RHS_i = RHS[sel]
#         Ua_i = Ua[sel]
#         U_i, c_i = U[sel], c[sel]
#         n = RHS_i.size

#         RHS_mean[i] = np.mean(RHS_i)
#         if n > 1:
#             sd = np.std(RHS_i, ddof=1)
#             se = sd / np.sqrt(n)
#             RHS_se[i] = np.nan if se == 0 else se
#         else:
#             RHS_se[i] = np.nan

#         Ua_mean[i] = np.mean(Ua_i)
#         U_mean[i]  = np.mean(U_i)
#         c_mean[i]  = np.mean(c_i)
#     return RHS_mean, RHS_se, U_mean, c_mean, Ua_mean