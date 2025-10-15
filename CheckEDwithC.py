# -*- coding: utf-8 -*-
"""
Created on Wed Oct 15 13:48:51 2025

@author: WangX3
"""

import numpy as np
import matplotlib.pyplot as plt

cases = range(2, 7)  # S002..S006
C_list, dcdt_list, E_list, D_list = [], [], [], []
t_dpm = np.linspace(0.01, 5, 501)
dt = 0.01
for i in cases:
    # CG data: columns -> Q, C, U (501 rows)
    cg = np.loadtxt(f"CGdata/hb=12d/Shields{i:03d}dry.txt")
    C_list.append(cg[:, 1])
    dcdt = np.gradient(cg[:, 1], dt)
    dcdt_list.append(dcdt)
    # E and D
    ED = np.loadtxt(f"S{i:03d}EandD.txt")
    # t_Dis = np.linspace(5/len(ED[:,0]), 5, len(ED[:,0]))
    # # make E and D same length as U and C
    # E_on_U = np.interp(t_dpm, t_Dis, ED[:,0])
    # D_on_U = np.interp(t_dpm, t_Dis, ED[:,1])
    E, D = ED[:,0], ED[:,1]
    E_list.append(E)
    D_list.append(D)

c_com_list = []
N = len(C_list[0])    

for i in range(5):    
    c0 = C_list[i][5]
    c = np.zeros(N-5)
    c[0] = c0
    for j in range(N-6):
        dcdt = E_list[i][j] - D_list[i][j]
        c[j+1] = c[j] + dcdt * dt
    c_com_list.append(c)


plt.close('all')
plt.figure(figsize=(10,9))
for i in range(5):
    plt.subplot(3,2,i+1)
    plt.plot(t_dpm, C_list[i], label='DPM CG data')
    plt.plot(t_dpm[5:], c_com_list[i], label='computed from E and D')
    plt.xlabel('t [s]')
    plt.ylabel(r'$C$ [kg/m$^2$]')
plt.legend()
plt.tight_layout()
    
plt.figure(figsize=(10,9))
for i in range(5):
    plt.subplot(3,2,i+1)
    plt.plot(t_dpm, dcdt_list[i], label='DPM CG data')
    plt.plot(t_dpm[5:], E_list[i][5:]-D_list[i][5:], label='computed from E and D')
    plt.xlabel('t [s]')
    plt.ylabel(r'$dC/dt$ [kg/m$^2$/s]')
plt.legend()
plt.tight_layout()