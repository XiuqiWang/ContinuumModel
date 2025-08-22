# -*- coding: utf-8 -*-
"""
Created on Fri Jul  4 15:28:23 2025

@author: WangX3
"""

#read job_test.log and Data files to calculate the total drag force in each time step
#To use this script, please copy the right input files to this dir, change the name of input and output files
import numpy as np
from read_data import read_data
from ReadFlowLog import ReadFlowLog
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

##read flow velocities
flow_velocities = ReadFlowLog('job_test_4679.log')
##read particle data
data_p = read_data('S006DryLBIni.data', 14, (1,6.01)) #6.01 to have 501 steps
##read saltation concentration and velocity
data_CG = np.loadtxt('../Shields006dry.txt')
C_dpm = data_CG[1:, 1]
U_dpm = data_CG[1:, 2]

##calculate the time series of total drag force
def Calfd(u_air, u_sal, D):
    # Ensure all inputs are numpy arrays
    u_diff = u_air - u_sal
    nu = 1.45e-6  # kinematic viscosity (m²/s)
    # Prevent divide-by-zero by adding a small epsilon
    epsilon = 1e-12
    Re = np.abs(u_diff) * D / nu + epsilon
    C_D = (np.sqrt(0.5) + np.sqrt(24 / Re))**2
    fd = 0.5 * np.pi / 8 * 1.225 * D**2 * C_D * u_diff * np.abs(u_diff)
    return fd

FD = []
height_threshold = 12*0.00025
for t in range(len(flow_velocities)):
    # Get the flow profile at this time
    flow_z = flow_velocities[t][:,0]  # elevations
    flow_u = flow_velocities[t][:,1]  # velocities

    # Interpolate flow at each particle z
    z_p = data_p[t]['Position'][:,2]  # particle elevation
    u_px = data_p[t]['Velocity'][:,0] #particle horinzontal velocity
    D_p = data_p[t]['Radius']*2 #particle diameter
    u_inter_flow = np.interp(z_p, flow_z, flow_u)
    fd = Calfd(u_inter_flow, u_px, D_p)
    # Create a mask for particles above the threshold
    mask = z_p >= height_threshold
    FD.append(np.sum(fd[mask]))
FD = np.array(FD)

##store in a txt file
np.savetxt("FD_S006dry.txt", FD)

##calculate the time series of representative Uair from FD, Csal and Usal
# def compute_u_air_from_FD(FD, D_mean, C_sal, u_sal):
#     rho_air = 1.225       # kg/m^3
#     rho_p = 2650          # kg/m^3
#     nu = 1.45e-6          # m^2/s (kinematic viscosity of air)
    
#     # Particle mass
#     mp = (np.pi / 6) * rho_p * D_mean**3

#     # Domain size
#     Lx = 100 * D_mean
#     Ly = 2 * D_mean

#     # Known total force per unit area
#     FD_per_area = FD / (Lx * Ly)

#     u_air_solutions = []

#     for i in range(len(FD)):
#         FD_per_area = FD[i] / (Lx * Ly)

#         def equation(u_air):
#             delta_u = u_air - u_sal[i]
#             Re_p = abs(delta_u) * D_mean / nu
#             CD = (np.sqrt(0.5) + np.sqrt(24 / Re_p))**2
#             fd = 0.5 * np.pi / 8 * rho_air * D_mean**2 * CD * delta_u * abs(delta_u)
#             return (C_sal[i] * fd / mp) - FD_per_area

#         # Initial guess
#         u_guess = u_sal[i] + 0.5

#         u_air_i, = fsolve(equation, u_guess)
#         u_air_solutions.append(u_air_i)

#     return np.array(u_air_solutions)
    
# U_air = compute_u_air_from_FD(FD, 0.00025, C_dpm, U_dpm)
        
        

