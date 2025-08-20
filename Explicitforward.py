import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

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


def solveUim(u_sal): #R2=0.74
    u_sal = np.array(u_sal)
    # Uim = 0.96*(abs(u_sal)/constant) + 15.07
    Uim = 0.14 * abs(u_sal)/constant + 32.15
    return Uim*constant

def solveUD(u_sal): #R2=0.83
    # UD = 0.96*(abs(u_sal)/constant) + 3.41
    UD = 9.08
    return UD*constant

def solveUinc(Uim, Udep, Tim, Tdep):
    def equation(Uinc):
        Pr = 0.94 * np.exp(-7.11 * np.exp(-0.11 * Uinc / constant))
        denom = Pr * Tim + (1 - Pr) * Tdep
        weight_im = (Pr * Tim) / denom
        weight_D = 1 - weight_im
        Uinc_model = weight_im * Uim + weight_D * Udep
        return Uinc - Uinc_model
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

# def CalCDbed(u_air):
#     Re = u_air * D / (1.46e-5)
#     CD_bed = 1.05e-6 * Re**2
#     return CD_bed

# Add a history buffer for u_sal
u_sal_history = []

# def get_delayed_Uim(u_sal_current, t_current, t_prev, Tim):
#     # Store current u_sal and time
#     u_sal_history.append((t_current, u_sal_current))
    
#     # Remove entries older than Tim/2
#     u_sal_history[:] = [(t, u) for (t, u) in u_sal_history if t_current - t <= Tim/2]
    
#     # Compute time-weighted average 
#     if not u_sal_history:
#         Uim = solveUim(u_sal_current)
#         return Uim
    
#     # Target delay time
#     target_delay = Tim / 2
#     # Compute weights based on how close each timestamp is to Tim/2 ago
#     times = np.array([t_current - t for (t, u) in u_sal_history])
#     values = np.array([u for (t, u) in u_sal_history])
#     # Gaussian weight: peak at target_delay, width = target_delay/2
#     sigma = target_delay / 2
#     weights = np.exp(-0.5 * ((times - target_delay) / sigma) ** 2)
#     # Avoid divide-by-zero
#     if weights.sum() == 0:
#         return solveUim(u_sal_current)
#     weighted_mean_u = np.sum(values * weights) / np.sum(weights)
#     return solveUim(weighted_mean_u)

def explicit_upwind_solver(u_star, y0, t_span, dt=0.001):
    t_start, t_end = t_span
    n_steps = int((t_end - t_start) / dt)
    t = np.linspace(t_start, t_end, n_steps + 1)
    y = np.zeros((3, n_steps + 1))
    y[:, 0] = y0

    mom_drag_list = np.zeros(n_steps + 1)

    for i in range(n_steps):
        # Current state
        c_sal, mom_sal, u_air = y[:, i]
        u_sal = mom_sal / (c_sal + 1e-9)

        # Solve Uim and related quantities
        Uim = solveUim(u_sal)
        Udep = solveUD(u_sal)
        
        # Impact and deposition angles (clipped to avoid domain errors)
        arg_im = 50.4 / (abs(Uim) / constant + 159.33)
        arg_im_clipped = np.clip(arg_im, -1.0, 1.0)
        theta_im = np.arcsin(arg_im_clipped)

        arg_dep = 163.68 / (abs(Udep) / constant + 154.65)
        arg_dep_clipped = np.clip(arg_dep, -1.0, 1.0)
        theta_dep = 0.28 * np.arcsin(arg_dep_clipped)

        # Time scales
        # Tim = 1e-9 + 2 * abs(Uim) * np.sin(theta_im) / 9.81
        # Tdep = 1e-9 + 2 * abs(Udep) * np.sin(theta_dep) / 9.81
        Tim = 0.04*Uim**0.84 + 1e-9
        Tdep = 0.07*Udep**0.66 + 1e-9
        
        # # Inside the time loop:
        # Uim = get_delayed_Uim(u_sal, t[i], t[i-1], Tim)
        # # Udep = get_delayed_Uim(u_sal, t[i], t[i-1], Tdep)
        
        # # Impact and deposition angles (clipped to avoid domain errors)
        # arg_im = 39.21 / (abs(Uim) / constant + 105.73)
        # arg_im_clipped = np.clip(arg_im, -1.0, 1.0)
        # theta_im = np.arcsin(arg_im_clipped)

        # arg_dep = 96.36 / (abs(Udep) / constant + 83.42)
        # arg_dep_clipped = np.clip(arg_dep, -1.0, 1.0)
        # theta_dep = 0.33 * np.arcsin(arg_dep_clipped)
        
        # # Time scales
        # Tim = 1e-9 + 2 * abs(Uim) * np.sin(theta_im) / 9.81
        # Tdep = 1e-9 + 2 * abs(Udep) * np.sin(theta_dep) / 9.81
         
        # Splash functions
        # Uinc = solveUinc(Uim, Udep, Tim, Tdep)
        Pr = 0.94 * np.exp(-7.12 * np.exp(-0.1 * abs(Uim) / constant))
        NE = 0.04 * abs(Uim) / constant
        UE = 4.53 * constant if Uim >= 0 else -4.53 * constant
        COR = 3.05 * (abs(Uim) / constant + 1e-9) ** (-0.47)

        # Rebound angle
        arg_re = -0.0003 * abs(Uim) / constant + 0.52
        arg_re_clipped = np.clip(arg_re, -1.0, 1.0)
        theta_re = np.arcsin(arg_re_clipped)
        Ure = Uim * COR
        cos_thetare = np.cos(theta_re)

        # Concentrations
        cim = c_sal * Pr * Tim / (Pr * Tim + (1 - Pr) * Tdep)
        cdep = c_sal - cim

        # drag forces and momenta
        mp = 2650 * np.pi / 6 * D ** 3
        fd_sal = Calfd(u_air, u_sal)
        mom_drag = CD_drag_reduce * c_sal * fd_sal / mp
        mom_drag_list[i+1] = mom_drag

        # Source terms (explicit upwind)
        mass_ero = alpha_ero * NE * (cim/Tim + cdep/Tdep)
        mom_ero = mass_ero * UE * cos_thetaej
        mom_re = cim * Ure * cos_thetare / Tim
        mass_dep = cdep / Tdep
        mom_im = cim * Uim * np.cos(theta_im) / Tim
        mom_dep = mass_dep * Udep * np.cos(theta_dep)

        # Air momentum sources
        mom_air_gain = 1.225 * u_star ** 2
        mom_air_loss = taub_minusphib(u_air)
        # CD_bed = CalCDbed(u_air)
        # mom_air_loss = 0.5* 1.225 * CD_bed * u_air * abs(u_air) 
        # Effective air mass per area
        chi = max(1e-6, 1.0 - c_sal/(rho_p*h))           # keep positive
        m_air_eff = rho_a * h * chi

        # Update state (forward Euler)
        y[0, i + 1] = c_sal + dt * (mass_ero - mass_dep)
        y[1, i + 1] = mom_sal + dt * (mom_drag + mom_ero + mom_re - mom_im - mom_dep)
        y[2, i + 1] = u_air + dt * (mom_air_gain - mom_air_loss - mom_drag)/m_air_eff

    return t, y, mom_drag_list

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

t, y, mom_drag = explicit_upwind_solver(u_star[5], y0, t_span, dt)

C = y[0]
U = y[1]/(y[0]+1e-9)
Q = y[1]
Ua = y[2]/mass_air

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

plt.figure()
plt.plot(t, mom_drag)
plt.plot(t_dpm, FD_dpm)
plt.xlabel('t [s]')
plt.ylabel(r'$M_{drag}$ [N/m$^2$]')