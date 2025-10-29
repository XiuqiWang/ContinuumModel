# -*- coding: utf-8 -*-
"""
Created on Wed Oct 29 13:58:55 2025

@author: WangX3
"""
import numpy as np
from read_data import read_data
import matplotlib.pyplot as plt

##read particle data
data_dict = {}  # Dictionary to store data
for i in range(5):
    data_p = read_data(f'../TotalDragForce/S00{i+2}M20LBIni.data', 14) #6.01 to have 501 steps 
    data_dict[i] = data_p

def volume_above_dh(Z, R, dh, *, eps=0.0):
    """
    计算每个时刻 z > dh 的总颗粒体积（部分高于dh的颗粒计入球冠体积）

    Parameters
    ----------
    Z : ndarray, shape (T, N)
        各时间步每个颗粒的球心高度
    R : ndarray, shape (N,)
        每个颗粒的半径
    dh : float
        截面高度 z = dh
    eps : float, optional
        数值容差，用于边界判定（默认0）

    Returns
    -------
    V_total : ndarray, shape (T,)
        每个时间步 dh 以上的总颗粒体积
    """
    Z = np.asarray(Z, dtype=float)          # (T, N)
    R = np.asarray(R, dtype=float)          # (N,)
    Rb = R[None, :]                         # (1, N) 广播用

    # 判定三种情况
    full_above = (Z - Rb) >= (dh - eps)     # 完全在上方
    full_below = (Z + Rb) <= (dh + eps)     # 完全在下方
    partial    = ~(full_above | full_below) # 部分截断

    # 每个颗粒的整体体积 (N,) → 广播成 (T, N)
    V_full_particle = (4.0/3.0) * np.pi * R**3
    V_full = V_full_particle[None, :]       # (1, N) → 广播

    # 初始化体积矩阵
    V = np.zeros_like(Z, dtype=float)

    # 完全在上方：直接取整球体积（用 where 广播，避免布尔索引赋值不匹配）
    V = np.where(full_above, V_full, 0.0)

    # 部分截断：球冠体积
    # 球冠高度 h = z + r - dh，仅对 partial 有效；其余位置设为0
    h_all  = Z + Rb - dh
    # 为稳健可裁剪到 [0, 2r]，虽然 partial 掩码已保证范围
    h_cap  = np.where(partial, np.clip(h_all, 0.0, 2.0*Rb), 0.0)
    V_cap  = (np.pi * h_cap**2 * (3.0*Rb - h_cap)) / 3.0

    # 合并
    V += np.where(partial, V_cap, 0.0)

    # 每个时间步求和
    V_total = V.sum(axis=1)
    return V_total

def avg_horizontal_velocity_above_dh(Z, R, U, dh, *, eps=0.0):
    """
    计算每个时间步中，z > dh 区域内颗粒的体积加权平均水平速度。
    （部分高于 dh 的颗粒按球冠体积计权）

    Parameters
    ----------
    Z : ndarray, shape (T, N)
        每个时间步各颗粒的球心高度
    R : ndarray, shape (N,)
        每个颗粒的半径
    U : ndarray, shape (T, N)
        每个时间步各颗粒的水平速度（可为标量速度或分量）
    dh : float
        截面高度 z = dh
    eps : float, optional
        数值容差，用于边界判定（默认 0）

    Returns
    -------
    U_avg : ndarray, shape (T,)
        每个时间步的体积加权平均水平速度
    """
    Z = np.asarray(Z, dtype=float)
    R = np.asarray(R, dtype=float)
    U = np.asarray(U, dtype=float)
    Rb = R[None, :]   # 广播 (1, N)

    # --- 判定位置关系 ---
    full_above = (Z - Rb) >= (dh - eps)
    full_below = (Z + Rb) <= (dh + eps)
    partial    = ~(full_above | full_below)

    # --- 球体总体积 (N,) ---
    V_full_particle = (4/3) * np.pi * R**3
    V_full = V_full_particle[None, :]  # (1, N)

    # --- 初始化体积矩阵 ---
    V = np.zeros_like(Z, dtype=float)
    V = np.where(full_above, V_full, 0.0)

    # --- 部分截断：球冠体积 ---
    h_all = Z + Rb - dh
    h_cap = np.where(partial, np.clip(h_all, 0.0, 2.0*Rb), 0.0)
    V_cap = (np.pi * h_cap**2 * (3.0*Rb - h_cap)) / 3.0
    V += np.where(partial, V_cap, 0.0)

    # --- 加权平均 ---
    numerator   = np.sum(V * U, axis=1)   # Σ (V_i * u_i)
    denominator = np.sum(V, axis=1)       # Σ V_i
    U_avg = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator>0)

    return U_avg


t_dpm = np.linspace(0.01, 5, 501)
D = 0.00025
dh = 13.5*D
                    
for i in range(5):
    data_p = data_dict[i]
    Z = np.array([d['Position'][:,2] for d in data_p])
    R = data_p[0]['Radius'] #particle radius
    Ux = np.array([d['Velocity'][:,0] for d in data_p])
    C = volume_above_dh(Z, R, dh) * 2650 / (100*2*D**2)
    U = avg_horizontal_velocity_above_dh(Z, R, Ux, dh)
    
    # plt.figure()
    # plt.plot(t_dpm, U)
    # plt.title(f'S{i+2}')
    
    out = np.column_stack([C, U])
    fname = f"S00{i+2}M20discrete.txt"
    np.savetxt(fname, out, fmt="%.6e", delimiter="\t",
                header="", comments="")

    print(f"Wrote {fname}.")
