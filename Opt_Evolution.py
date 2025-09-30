import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

# -------------------- constants & inputs --------------------
g = 9.81
d = 0.00025
const = np.sqrt(g*d)
h = 0.197
u_star = 0.56

rho_a  = 1.225
nu_a   = 1.46e-5
rho_p  = 2650.0
mp = rho_p * (np.pi/6.0) * d**3

UE_mag = 4.53 * const
thetaE_deg = 40.0
cos_thetaE = np.cos(np.deg2rad(thetaE_deg))

# 若这些系数尚未定义，先设为1；若也想拟合，可加入参数向量
alpha_Eim = 1.0
alpha_ED  = 1.0
gamma_drag = 1.0

# -------------------- 原 closures（保持不变的部分） --------------------
def Uim_from_U(U):
    Uim_mag = 0.04*(abs(U)/const)**1.54 + 32.23
    return np.sign(U) * (Uim_mag*const)

def UD_from_U(U):
    UD_mag = 6.66
    return np.sign(U) * (UD_mag*const)

def Tim_from_Uim(Uim):
    return 0.04 * (abs(Uim))**0.84

def TD_from_UD(UD):
    return 0.07 * (abs(UD))**0.66

def theta_im_from_Uim(Uim):
    x = 50.40 / (abs(Uim)/const + 159.33)
    return np.arcsin(np.clip(x, -1.0, 1.0))

def theta_D_from_UD(UD):
    x = 163.68 / (np.abs(UD) / const + 156.65)
    x = np.clip(x, 0.0, 1.0)
    theta = 0.28 * np.arcsin(x)
    return np.clip(theta, 0.0, 0.5*np.pi)

def theta_reb_from_Uim(Uim):
    x = -0.0003*(abs(Uim)/const) + 0.52
    return np.arcsin(np.clip(x, -1.0, 1.0))

def tau_top(u_star):
    return rho_a * u_star**2

# def Mdrag(c, Uair, U):
#     b = 0.55
#     C_D = 0.1037
#     Ueff = b * Uair
#     dU = Ueff - U
#     fdrag = np.pi/8 * d**2 * rho_a * C_D * abs(dU) * dU
#     return (c * fdrag) / mp

def MD_eff(Ua, U, c):
    Uaeff = 0.85*Ua
    return rho_a * 0.12 * c/(rho_p*d) * abs(Uaeff - U) * (Uaeff - U)

# -------------------- 可调参数化 closures --------------------
# params = [Pmax, Uc, pshape, A_NE]
def Preff_of_U_param(U, params):
    Pmax, Uc, pshape, _A_NE, _A_e, _B_e, _b, _C_D = params
    Pmin = 0.0
    Uabs = np.abs(U)
    return np.clip(Pmin + (Pmax - Pmin)*(1.0 - np.exp(-(Uabs/max(Uc,1e-6))**pshape)), 0.0, 1.0)

def NE_from_Uinc_param(Uinc, params):
    _Pmax, _Uc, _pshape, A_NE, _A_e, _B_e, _b, _C_D = params
    return A_NE * (np.abs(Uinc)/const)

def e_COR_from_Uim(Uim, params):
    _Pmax, _Uc, _pshape, _A_NE, A_e, B_e, _b, _C_D = params
    return A_e * (abs(Uim)/const + 1e-12)**(-B_e)

def Mdrag(c, Uair, U, params):
    _,_,_,_,_,_, b, C_D = params
    Ueff = b * Uair
    dU = Ueff - U
    fdrag = np.pi/8 * d**2 * rho_a * C_D * abs(dU) * dU
    return (c * fdrag) / mp

# -------------------- RHS 工厂（把 params 显式传入） --------------------
def make_rhs(u_star, params, c_floor=1e-3, U_switch=1e-3):
    def rhs(t, y):
        c, U, Uair = y

        dir = np.tanh(U / max(U_switch,1e-9))  # 平滑方向

        Uim = Uim_from_U(U); UD = UD_from_U(U)
        th_im = theta_im_from_Uim(Uim)
        th_D  = theta_D_from_UD(UD)
        th_re = theta_reb_from_Uim(Uim)
        eCOR  = e_COR_from_Uim(Uim, params)
        Ure   = Uim * eCOR

        Tim = Tim_from_Uim(Uim)
        TD  = TD_from_UD(UD)

        Pr    = Preff_of_U_param(U, params)            # ← 用拟合参数
        phi_im = (Pr*Tim) / (Pr*Tim + (1.0-Pr)*TD + 1e-12)
        cim = c * phi_im
        cD  = c - cim

        r_im  = cim / Tim
        r_dep = cD  / TD

        NE_im = NE_from_Uinc_param(Uim, params)        # ← 用拟合参数
        NE_D  = NE_from_Uinc_param(UD,  params)

        E = alpha_Eim * r_im * NE_im + alpha_ED * r_dep * NE_D
        D = r_dep

        M_drag = gamma_drag * Mdrag(c, Uair, U, params)
        M_eje  = (alpha_Eim * r_im * NE_im * UE_mag * cos_thetaE
                  + alpha_ED  * r_dep * NE_D  * UE_mag * cos_thetaE) * dir
        M_re   = r_im  * ( Ure * np.cos(th_re) ) * dir
        M_im   = r_im  * ( Uim * np.cos(th_im) ) * dir
        M_dep  = r_dep * ( UD  * np.cos(th_D)  ) * dir

        dc_dt = E - D

        S_U   = M_drag + M_eje + M_re - M_im - M_dep
        c_den = max(c, c_floor)
        dU_dt = (S_U / c_den) - (U * dc_dt / c_den)

        chi        = max(1e-6, 1.0 - c/(rho_p*h))
        m_air_eff  = rho_a * h * chi
        dUair_dt   = (tau_top(u_star) - MD_eff(Uair, U, c)) / m_air_eff

        return [dc_dt, dU_dt, dUair_dt]
    return rhs

# -------------------- 运行一次给定参数的模型并插值到 DPM 时间 --------------------
def simulate_and_sample_case(u_star_case, params, T, y0, t_sample):
    rhs = make_rhs(u_star_case, params)
    sol = solve_ivp(rhs, (0.0, T), y0, method='Radau',
                    rtol=1e-4, atol=[1e-9, 1e-8, 1e-8], max_step=1e-2, dense_output=True)
    y_samp = sol.sol(t_sample)  # shape: (3, len(t_sample))
    c_mod, U_mod, Ua_mod = y_samp[0], y_samp[1], y_samp[2]
    return c_mod, U_mod, Ua_mod

# -------------------- load DPM for a given Shields --------------------
def load_dpm_case(shields_value):
    # Map Shields to your file naming, e.g., 0.02 → S002, ..., 0.06 → S006
    label = int(round(shields_value*100))  # 2,3,4,5,6
    cg = np.loadtxt(f"CGdata/hb=12d/Shields{label:03d}dry.txt")
    C_dpm = cg[:,1]; U_dpm = cg[:,2]
    data_ua = np.loadtxt(f"TotalDragForce/Uair_ave-tS{label:03d}Dryh02.txt", delimiter="\t")
    Ua_dpm = data_ua[:,1]
    return C_dpm, U_dpm, Ua_dpm

# -------------------- per-case residual --------------------
def make_residuals_time_single(u_star_case, y0, T, t_dpm, C_dpm, U_dpm, Ua_dpm,
                               wC=2.0, wU=2.0, wUa=0.5, normalize=True):
    sC  = max(np.mean(np.abs(C_dpm)),  1e-6)
    sU  = max(np.mean(np.abs(U_dpm)),  1e-6)
    sUa = max(np.mean(np.abs(Ua_dpm)), 1e-6)
    def residuals(params):
        c_m, U_m, Ua_m = simulate_and_sample_case(u_star_case, params, T, y0, t_dpm)
        rC  = (c_m  - C_dpm)  / (sC  if normalize else 1.0)
        rU  = (U_m  - U_dpm)  / (sU  if normalize else 1.0)
        rUa = (Ua_m - Ua_dpm) / (sUa if normalize else 1.0)
        return np.r_[wC*rC, wU*rU, wUa*rUa]
    return residuals

# -------------------- main: fit each Shields independently --------------------
Shields_all = np.linspace(0.02, 0.06, 5)
u_star_all  = np.sqrt(Shields_all * (rho_p - rho_a) * g * d / rho_a)

T = 5.0
t_dpm = np.linspace(0, T, 501)

# parameter boxes (same for each case; you can customize per case)
p0 = np.array([0.6, 2.0, 0.7, 0.05, 3.0, 0.5, 0.5, 0.1])    # [Pmax, Uc, p, A_NE, A_e, B_e, b, C_D]
lb = np.array([0.3, 0.1, 0.2, 0.005, 0.5, 0.1, 0.01, 0.001])
ub = np.array([0.95, 8.0, 3.0, 0.2, 10.0, 1.0, 1.0, 0.5])

params_fit = []
fits_C, fits_U, fits_Ua = [], [], []
DPM_C, DPM_U, DPM_Ua    = [], [], []

for k, Theta in enumerate(Shields_all):
    ustar = u_star_all[k]
    C_dpm, U_dpm, Ua_dpm = load_dpm_case(Theta)
    y0 = (C_dpm[0], U_dpm[0], Ua_dpm[0])  # per-case initial condition

    resfn = make_residuals_time_single(ustar, y0, T, t_dpm, C_dpm, U_dpm, Ua_dpm,
                                       wC=1.0, wU=1.0, wUa=1.0, normalize=True)

    opt = least_squares(resfn, p0, bounds=(lb, ub),
                        method='trf', loss='soft_l1', f_scale=1.0,
                        xtol=1e-7, ftol=1e-7, gtol=1e-7)

    params_fit.append(opt.x)

    # simulate with best-fit for plotting
    c_m, U_m, Ua_m = simulate_and_sample_case(ustar, opt.x, T, y0, t_dpm)
    fits_C.append(c_m); fits_U.append(U_m); fits_Ua.append(Ua_m)
    DPM_C.append(C_dpm); DPM_U.append(U_dpm); DPM_Ua.append(Ua_dpm)

# -------------------- print fitted parameters --------------------
print("\nBest-fit params per Shields [Pmax, Uc, p, A_NE, A_e, B_e]:")
for k, Theta in enumerate(Shields_all):
    print(f"Theta={Theta:.2f}: {params_fit[k]}")

# -------------------- plot: overlay 5 cases (C, U, Ua) --------------------
colors = plt.cm.RdBu(np.linspace(0, 1, 5))

plt.figure(figsize=(14,4))

# C
plt.subplot(1,3,1)
for k, Theta in enumerate(Shields_all):
    plt.plot(t_dpm, DPM_C[k], '.', ms=2, alpha=0.7, label=f'Θ={Theta:.2f}', color=colors[k])
    plt.plot(t_dpm, fits_C[k], '-', lw=1.5, alpha=0.9, color=colors[k])
plt.xlabel('t [s]'); plt.ylabel('C [-]'); plt.grid(True)
plt.legend()
# U
plt.subplot(1,3,2)
for k, Theta in enumerate(Shields_all):
    plt.plot(t_dpm, DPM_U[k], '.', ms=2, alpha=0.7, label=None, color=colors[k])
    plt.plot(t_dpm, fits_U[k], '-', lw=1.5, alpha=0.9, label=None, color=colors[k])
plt.xlabel('t [s]'); plt.ylabel('U [m/s]'); plt.grid(True)

# Ua
plt.subplot(1,3,3)
for k, Theta in enumerate(Shields_all):
    plt.plot(t_dpm, DPM_Ua[k], '.', ms=2, alpha=0.7, label=None, color=colors[k])
    plt.plot(t_dpm, fits_Ua[k], '-', lw=1.5, alpha=0.9, label=None, color=colors[k])
plt.xlabel('t [s]'); plt.ylabel('Ua [m/s]'); plt.grid(True)

# simple legend (two entries) outside
lines = [plt.Line2D([0],[0], color='k', marker='.', linestyle='None', label='DPM'),
         plt.Line2D([0],[0], color='C0', linestyle='-', label='Model')]
plt.legend(handles=lines, loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=2)
plt.tight_layout()
plt.show()