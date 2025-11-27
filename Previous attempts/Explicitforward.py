import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import pandas as pd

# Constants
D = 0.00025
constant = np.sqrt(9.81 * D)
Shields = np.linspace(0.01, 0.06, 6)
u_star = np.sqrt(Shields * (2650 - 1.225) * 9.81 * D / 1.225)

# Mass of air per unit area
h = 0.2 - 0.00025 * 10
mass_air = 1.225 * h
rho_a = 1.225
rho_p = 2650
nu_a = 1.46e-5
CD_air = 8e-3
CD_bed = 3e-4
CD_drag_reduce = 1
alpha_ero = 1
alpha_dep = 1
alpha_im = 1
cos_thetaej = np.cos(47 / 180 * np.pi)


def solveUim(u_sal):
    # magnitude fit (what you already had), then give it the saltation sign
    # Umag = 0.94*(abs(u_sal)/constant) + 10.50
    Umag = (0.14 * (np.abs(u_sal)/constant) + 26) * constant #32.15
    return np.sign(u_sal) * Umag

def solveUD(u_sal):
    Umag = 9.08 * constant
    # Umag = 0.66*(abs(u_sal)/constant) + 1.34
    return np.sign(u_sal) * Umag

def solveUinc(Uim, Udep, Tim, Tdep):
    def equation(Uinc):
        speed_inc = abs(Uinc)
        Pr = 0.94 * np.exp(-7.11 * np.exp(-0.11 * speed_inc / constant))
        denom = Pr * Tim + (1 - Pr) * Tdep
        weight_im = (Pr * Tim) / denom
        weight_D = 1 - weight_im
        Uinc_model = weight_im * abs(Uim) + weight_D * abs(Udep)
        return speed_inc - Uinc_model
    Uinc_guess = Uim
    Uinc_solution, = fsolve(equation, Uinc_guess)
    return Uinc_solution

def Calfd(u_air, u_sal):
    hs = 6.514e-4
    z0 = 2.386e-6
    u_eff = u_air * (np.log(hs/z0)-1)/(np.log(h/z0)-1)
    k = 3.216e-9
    n = 2.194
    Δu = u_eff - u_sal
    fd = k * np.abs(Δu)**(n-1) * Δu
    # u_eff = 0.3*u_air
    # Re = abs(u_eff - u_sal) * D/(1.45e-6) + 1e-12
    # C_D = (np.sqrt(0.5) + np.sqrt(24 / Re))**2
    # fd = 0.5* np.pi/8 * 1.225 * D**2 * C_D * (u_eff - u_sal)* abs(u_eff - u_sal)
    return fd

def taub_minusphib(u_air):
    l_eff = 0.0319
    phi_b = 0.64
    tau_b = rho_a * (nu_a + l_eff**2*abs(u_air/h))*u_air/h
    return tau_b*(1-phi_b) 

def explicit_upwind_solver_debug(u_star, y0, t_span, dt=1e-3, *,
                                 step_limiter=True, max_rel_U_change=0.2,
                                 verbose_every=0):
    """
    Explicit Euler with full diagnostics.

    y = [c_sal, mom_sal, u_air]; u_sal = mom_sal / max(c_sal, eps)
    Returns: t, y, mom_drag_series, debug_df
    """
    # --- constants / tiny guards
    eps_c = 1e-12
    eps_T = 1e-12

    t_start, t_end = t_span
    n_steps = int((t_end - t_start) / dt)
    t = np.linspace(t_start, t_end, n_steps + 1)
    y = np.zeros((3, n_steps + 1))
    y[:, 0] = y0

    mom_drag_list = np.zeros(n_steps + 1)

    # ---- debug storage
    log = {
        "t": [], "c_sal": [], "u_sal": [], "u_air": [],
        "Uim": [], "Udep": [], "Tim": [], "Tdep": [],
        "Pr": [], "NE": [], "UE": [], "COR": [],
        "theta_im": [], "theta_dep": [], "theta_re": [],
        "cim": [], "cdep": [],
        "mass_ero": [], "mass_dep": [],
        "mom_drag": [], "mom_ero": [], "mom_re": [], "mom_im": [], "mom_dep": [],
        "rhs_mass": [], "rhs_mom": [], "rhs_air": [],
    }

    for i in range(n_steps):
        # --- unpack state
        c_sal, mom_sal, u_air = y[:, i]
        denom_c = max(c_sal, eps_c)
        u_sal = mom_sal / denom_c

        # --- closures
        Uim  = solveUim(u_sal)                         
        Udep = solveUD(u_sal)
        
        speed_im, speed_dep = np.abs(Uim), np.abs(Udep)

        # angles (clipped)
        arg_im  = 50.4  / (speed_im  / constant + 159.33)
        theta_im  = np.arcsin(np.clip(arg_im,  -1.0, 1.0))
        arg_dep = 163.68 / (speed_dep / constant + 154.65)
        theta_dep = 0.28 * np.arcsin(np.clip(arg_dep, -1.0, 1.0))

        # time scales
        Tim  = 0.04 * speed_im**0.84  + eps_T
        Tdep = 0.07 * speed_dep**0.66 + eps_T

        Uinc = solveUinc(Uim,Udep,Tim,Tdep)
        speed_inc = abs(Uinc)
        # splash
        Pr  = 0.94 * np.exp(-7.12 * np.exp(-0.1 * speed_inc / constant))
        NE  = 0.04 * speed_inc / constant
        UE  =  4.53 * constant if Uinc >= 0 else -4.53 * constant
        COR = 3.05 * (speed_im / constant + 1e-9) ** (-0.47)

        # rebound
        arg_re = -0.0003 * speed_im / constant + 0.52
        theta_re = np.arcsin(np.clip(arg_re, -1.0, 1.0))
        Ure = Uim * COR

        # concentrations split
        cim  = c_sal * Pr * Tim / (Pr * Tim + (1 - Pr) * Tdep)
        cdep = c_sal - cim

        # drag (signed, using relative velocity)
        mp = 2650 * np.pi / 6 * D**3
        fd_sal = Calfd(u_air, u_sal)                   # signed: >0 if air faster than grains
        mom_drag = CD_drag_reduce * c_sal * fd_sal / mp
        mom_drag_list[i] = mom_drag

        # sources/sinks
        mass_ero = alpha_ero * NE * (cim/Tim + cdep/Tdep)
        mom_ero  = mass_ero * UE * cos_thetaej
        mom_re   = cim * Ure * np.cos(theta_re) / Tim
        mass_dep = cdep / Tdep
        mom_im   = cim * Uim * np.cos(theta_im) / Tim
        mom_dep  = mass_dep * Udep * np.cos(theta_dep)
        
        # air side
        tau_top = 1.225 * u_star**2
        tau_bed_eff = taub_minusphib(u_air)
        chi = max(1e-6, 1.0 - c_sal/(rho_p*h))
        m_air_eff = rho_a * h * chi

        # RHS
        rhs_mass = (mass_ero - mass_dep)
        k_drag = 1.0      # multiply mom_drag
        k_re   = 1.0      # multiply mom_re
        k_im   = 1.0      # multiply mom_im
        k_dep  = 1.0      # multiply mom_dep
        rhs_mom = k_drag*mom_drag + mom_ero + k_re*mom_re - k_im*mom_im - k_dep*mom_dep
        # rhs_mom  = (mom_drag + mom_ero + mom_re - mom_im - mom_dep)
        rhs_air  = (tau_top - tau_bed_eff - mom_drag)/m_air_eff
        print(f"[mass] mass_ero={mass_ero:.3e}  mass_dep={mass_dep:.3e}  dC/dt={mass_ero-mass_dep:.3e}")
        print(f"[mom]  drag={mom_drag:.3e}  re={mom_re:.3e}  ero={mom_ero:.3e}  im={mom_im:.3e}  dep={mom_dep:.3e}  rhs={rhs_mom:.3e}")
        print(f"[air]  tau_top={tau_top:.3e}  tau_bed_eff={tau_bed_eff:.3e}  Mdrag={mom_drag:.3e}  dUa/dt={rhs_air:.3e}")

        # ---- diagnostics: finiteness
        if not np.isfinite(rhs_mom) or not np.isfinite(rhs_mass) or not np.isfinite(rhs_air):
            print(f"[STOP at i={i}, t={t[i]:.6g}] Non-finite RHS:",
                  f"rhs_mass={rhs_mass}, rhs_mom={rhs_mom}, rhs_air={rhs_air}")
            # truncate
            df = pd.DataFrame(log)
            return t[:i+1], y[:, :i+1], mom_drag_list[:i+1], df

        # ---- optional step limiter on U to reduce numerical blowups during debugging
        if step_limiter:
            U_now   = u_sal
            U_next  = (mom_sal + dt*rhs_mom) / max(c_sal + dt*rhs_mass, eps_c)
            dU_rel  = 0.0 if U_now == 0 else abs(U_next - U_now)/max(abs(U_now), 1e-9)
            if dU_rel > max_rel_U_change:
                # shrink dt to keep relative change bounded
                fac = max_rel_U_change / dU_rel
                dt_eff = dt * fac
            else:
                dt_eff = dt
        else:
            dt_eff = dt

        # ---- store debug
        log["t"].append(t[i]); log["c_sal"].append(c_sal); log["u_sal"].append(u_sal); log["u_air"].append(u_air)
        log["Uim"].append(Uim); log["Udep"].append(Udep); log["Tim"].append(Tim); log["Tdep"].append(Tdep)
        log["Pr"].append(Pr); log["NE"].append(NE); log["UE"].append(UE); log["COR"].append(COR)
        log["theta_im"].append(theta_im); log["theta_dep"].append(theta_dep); log["theta_re"].append(theta_re)
        log["cim"].append(cim); log["cdep"].append(cdep)
        log["mass_ero"].append(mass_ero); log["mass_dep"].append(mass_dep)
        log["mom_drag"].append(mom_drag); log["mom_ero"].append(mom_ero); log["mom_re"].append(mom_re)
        log["mom_im"].append(mom_im); log["mom_dep"].append(mom_dep)
        log["rhs_mass"].append(rhs_mass); log["rhs_mom"].append(rhs_mom); log["rhs_air"].append(rhs_air)

        # ---- update (forward Euler with possibly reduced dt)
        y[0, i+1] = c_sal + dt_eff * rhs_mass
        y[0, i+1] = max(y[0, i+1], 0)
        y[1, i+1] = mom_sal + dt_eff * rhs_mom
        y[2, i+1] = u_air   + dt_eff * rhs_air

        if verbose_every and (i % verbose_every == 0):
            print(f"t={t[i]:.3f}  U={u_sal:.3f}  rhs_mom={rhs_mom:.3e}  ",
                  f"drag={mom_drag:.3e}  re={mom_re:.3e}  im={mom_im:.3e}  dep={mom_dep:.3e}")
            
        r_im  = cim/Tim
        r_dep = cdep/Tdep
        if dt * max(r_im, r_dep) > 0.1:
            print("Warning: dt too large for hop rates", i, t[i], r_im, r_dep)

    df = pd.DataFrame(log)
    return t, y, mom_drag_list, df

data = np.loadtxt('CGdata/Shields006dry.txt')
Q_dpm = data[:, 0]
C_dpm = data[:, 1]
U_dpm = data[:, 2]
t_dpm = np.linspace(0,5,501)
data_ua = np.loadtxt('TotalDragForce/Uair_ave-tS006Dryh02.txt', delimiter='\t')
# Ua_dpm = np.insert(data_ua[20:,1], 0, Uair0[-1])
Ua_dpm = data_ua[0:,1]   
file_fd = 'TotalDragForce/FD_S006dry.txt'
data_FD = np.loadtxt(file_fd)
FD_dpm = data_FD / (100 * D * 2 * D)

# Initial conditions
c0 =  C_dpm[0] # 0.147 # 0.0139
Usal0 = U_dpm[0] # 0.55 # 2.9279
Uair0 = Ua_dpm[0]
y0 = [c0, c0 * Usal0, 13]
t_span = [0, 5]
dt = 0.001  # Time step (adjust for stability)

t, y, mom_drag_series, dbg = explicit_upwind_solver_debug(u_star=u_star[2], y0=[c0, c0*Usal0, Uair0], t_span=(0, 5), dt=1e-3, verbose_every=100)

# Quick triage: who dominates rhs_mom?
cols = ["mom_drag","mom_ero","mom_re","mom_im","mom_dep","rhs_mom"]
print(dbg[cols].head(10))
print(dbg[cols].tail(10))

# Plot contributions over time to see which term flips the sign:
# (drag vs. im/dep sinks are common culprits)


C = y[0]
U = y[1]/(y[0]+1e-9)
Q = y[1]
Ua = y[2]

plt.close('all')
plt.figure(figsize=(8, 6))
# Data for plotting: (x_main, y_main, x_dpm, y_dpm, label)
plots = [
    (t, Q,    t_dpm, Q_dpm,  "Q"),
    (t, C,    t_dpm, C_dpm,  "C"),
    (t, U,    t_dpm, U_dpm,  "U"),
    (t, Ua,   t_dpm, Ua_dpm, "Ua"),
]
for i, (x1, y1, x2, y2, label) in enumerate(plots, start=1):
    plt.subplot(2, 2, i)
    plt.plot(x1, y1, label=label)
    plt.plot(x2, y2, label=f"{label} dpm")
    plt.xlabel("t [s]")
    plt.ylabel(label)
    plt.legend()
plt.tight_layout()

# plt.figure()
# plt.plot(t, mom_drag)
# plt.plot(t_dpm, FD_dpm)
# plt.xlabel('t [s]')
# plt.ylabel(r'$M_{drag}$ [N/m$^2$]')