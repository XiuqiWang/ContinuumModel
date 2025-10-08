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

Omega = 0.0 

# -------------------- closures --------------------
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

def calc_N_E_test(Uinc):
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

def e_COR_from_Uim(Uim):
    return 3.05 * (abs(Uim)/const + 1e-12)**(-0.47)       
    # return 3.0932 * (abs(Uim)/const + 1e-12)**(-0.4689) # tuned for steady solutions                 

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

def tau_top(u_star):
    return rho_a * u_star**2

# -------------------- closures to be optimized --------------------
# params = [Pmax, Uc, pshape, A_NE]
def Preff_of_U_param(U, params):
    Pmax, Uc, pshape, _alpha, _K, _CD_bed, _alpha_drag, _A_NE = params
    Pmin = 0.0
    Uabs = np.abs(U)
    return np.clip(Pmin + (Pmax - Pmin)*(1.0 - np.exp(-(Uabs/max(Uc,1e-6))**pshape)), 0.0, 1.0)

def MD_eff_params(Ua, U, c, params):
    _Pmax, _Uc, _pshape, alpha, K, CD_bed, _alpha_drag, _A_NE = params
    alpha, K = 1.20, 0.040
    Uaeff = alpha*Ua
    MD_eff = rho_a * K * c/(rho_p*d) *abs(Uaeff - U) * (Uaeff - U)
    CD_bed = 0.053
    tau_basic = 0.5 * rho_a * CD_bed * Ua * abs(Ua)
    B, p = 1.07e+25, 0.033
    M_tune = tau_basic * (1/(1+(B*MD_eff)**p)) # to guarantee that there is a balancing term with tau_top when c=0
    M_final = MD_eff + M_tune
    return M_final

def calc_Mdrag_params(c, Uair, U, params):
    _Pmax, _Uc, _pshape, _alpha, _K, _CD_bed, alpha_drag, _A_NE = params
    Ngrains = c/mp
    alpha = 0.32
    Ueff = alpha*Uair
    Urel = Ueff-U
    Re = abs(Urel)*d/nu_a
    Ruc = 24
    Cd_inf = 0.5
    Cd = (np.sqrt(Cd_inf)+np.sqrt(Ruc/Re))**2
    
    Agrain = np.pi*(d/2)**2
    Mdrag = 0.5*rho_a*Urel*abs(Urel)*Cd*Agrain*Ngrains # drag term based on uniform velocity
    return Mdrag

def NE_from_Uinc(Uinc, params):
    Pmax, Uc, pshape, _alpha, _K, _CD_bed, _alpha_drag, A_NE = params
    return A_NE * (abs(Uinc)/const)     
    # return 0.0635 * (abs(Uinc)/const)   
    
def calc_N_E_test3(Uinc, params):
    N_E_Xiuqi = NE_from_Uinc(Uinc, params)
    
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

# -------------------- RHS --------------------
def rhs(t, y, eps, u_star, params):
    c, m, Ua = y
    U = m/(c + eps)

    Uim, UD = Uim_from_U(U), UD_from_U(U)
    Tim, TD  = calc_T_jump_Test(Uim), calc_T_jump_Test(UD) # changed 
    Pr = Preff_of_U_param(U, params) # changed

    # mixing fractions and rates
    phi_im = (Pr*Tim) / (Pr*Tim + (1.0-Pr)*TD)
    cim, cD = c*phi_im, c*(1.0-phi_im)

    if cD<0:
            print('cD=',cD)
    r_im, r_dep = cim/Tim, cD/TD

    # ejection numbers and rebound kinematics
    NEim, NEd = calc_N_E_test3(Uim, params), calc_N_E_test3(UD, params)# changed
    eCOR = e_COR_from_Uim(Uim); Ure = Uim*eCOR
    th_im, th_D, th_re = theta_im_from_Uim(Uim), theta_D_from_UD(UD), theta_reb_from_Uim(Uim)

    # scalar sources
    E = r_im*NEim + r_dep*NEd
    D = r_dep
    
    if E<0:
            print('E = ',E)

    # momentum sources (streamwise)
    M_drag = calc_Mdrag_params(c, Ua, U, params) # changed
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

    phi_term  = 1.0 - c/(rho_p*h)
    m_air_eff = rho_a*h*phi_term
    dUa_dt    = (tau_top(u_star) - MD_eff_params(Ua, U, c, params)) / m_air_eff

    return [dc_dt, dm_dt, dUa_dt]

# ---------- simple Euler–forward integrator ----------
def euler_forward(rhs, y0, t_span, dt, u_star, params):
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
        eps=1e-16
        f = rhs(tk, yk, eps, u_star, params)
        y_next = yk + dt * np.asarray(f, dtype=float)

        t[k+1]   = min(tk + dt, t1)
        y[:,k+1] = y_next

    return t, y

# -------------------- 运行一次给定参数的模型并插值到 DPM 时间 --------------------
def simulate_and_sample_case(u_star_case, params, T, y0, dt):
    t, y_samp = euler_forward(rhs, y0, (0, T), dt, u_star_case, params)
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
def make_residuals_time_single(u_star_case, y0, T, dt, C_dpm, U_dpm, Ua_dpm,
                               wC=1.0, wU=1.0, wUa=1.0, normalize=True):
    sC  = max(np.mean(np.abs(C_dpm)),  1e-6)
    sU  = max(np.mean(np.abs(U_dpm)),  1e-6)
    sUa = max(np.mean(np.abs(Ua_dpm)), 1e-6)
    def residuals(params):
        c_m, U_m, Ua_m = simulate_and_sample_case(u_star_case, params, T, y0, dt)
        rC  = (c_m  - C_dpm)  / (sC  if normalize else 1.0)
        rU  = (U_m  - U_dpm)  / (sU  if normalize else 1.0)
        rUa = (Ua_m - Ua_dpm) / (sUa if normalize else 1.0)
        return np.r_[wC*rC, wU*rU, wUa*rUa]
    return residuals

# -------------------- main: fit each Shields independently --------------------
Shields_all = np.linspace(0.05, 0.06, 2)
u_star_all  = np.sqrt(Shields_all * (rho_p - rho_a) * g * d / rho_a)

T = 5.0
dt = 0.01
t_dpm = np.linspace(0, 5, 501)

# parameter boxes (same for each case; you can customize per case)
p0 = np.array([0.999, 3.5, 0.7, 1.2, 0.04, 0.05, 0.4, 0.04])    # [Pmax, Uc, p, alpha, K, CD_bed, alpha_drag]
lb = np.array([0.3, 0.1, 0.2, 0.005, 0.001, 0.001, 0.001, 0.001])
ub = np.array([1.0, 8.0, 3.0, 5.0, 1.0, 1.0, 2.0, 1.0])

params_fit = []
fits_C, fits_U, fits_Ua = [], [], []
DPM_C, DPM_U, DPM_Ua    = [], [], []

for k, Theta in enumerate(Shields_all):
    ustar = u_star_all[k]
    C_dpm, U_dpm, Ua_dpm = load_dpm_case(Theta)
    y0 = (C_dpm[0], U_dpm[0], Ua_dpm[0])  # per-case initial condition

    resfn = make_residuals_time_single(ustar, y0, T, dt, C_dpm, U_dpm, Ua_dpm,
                                       wC=1.0, wU=1.0, wUa=1.0, normalize=True)

    opt = least_squares(resfn, p0, bounds=(lb, ub),
                        method='trf', loss='soft_l1', f_scale=1.0,
                        xtol=1e-7, ftol=1e-7, gtol=1e-7)

    params_fit.append(opt.x)

    # simulate with best-fit for plotting
    c_m, U_m, Ua_m = simulate_and_sample_case(ustar, opt.x, T, y0, dt)
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