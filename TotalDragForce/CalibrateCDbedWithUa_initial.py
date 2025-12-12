# -*- coding: utf-8 -*-
"""
Created on Fri Oct 10 18:41:36 2025

@author: WangX3
"""

import numpy as np
import matplotlib.pyplot as plt

def depth_avg_from_file(filename, z_min, z_max, lines_per_block, col_z=0, col_u=1):
    """
    从 filename 读取按块的 (z, u, …) 数据，计算每块在 [z_min, z_max] 内的深度平均速度。
    返回：长度为块数的一维数组，每个元素是对应块的 u_avg（缺数据则为 np.nan）
    """
    # 读入所有数值行（至少 3 列），忽略非数值行
    rows = []
    with open(filename, "r") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 3:
                try:
                    rows.append([float(x) for x in parts[:3]])
                except ValueError:
                    pass  # 跳过非数值行
    data = np.asarray(rows)
    if data.size == 0:
        raise ValueError("没有读取到有效数值行。")

    nrows = data.shape[0]
    if nrows % lines_per_block != 0:
        raise ValueError(f"总行数 {nrows} 不是 lines_per_block={lines_per_block} 的整数倍。")

    nblocks = nrows // lines_per_block
    u_avg_list = []

    for b in range(nblocks):
        block = data[b*lines_per_block : (b+1)*lines_per_block]
        z = block[:, col_z]
        u = block[:, col_u]

        # 按 z 排序
        idx = np.argsort(z)
        z = z[idx]; u = u[idx]

        # 取区间内的点
        mask = (z >= z_min) & (z <= z_max)
        z_clip = z[mask]
        u_clip = u[mask]

        # 在线性插值下补上边界 z_min / z_max（若原数据覆盖到这些高度）
        def interp_at(z0):
            return np.interp(z0, z, u)  # 只要 z_min/max 在 [z.min, z.max] 内即可插值

        if z_min >= z[0] and z_min <= z[-1]:
            if len(z_clip) == 0 or z_clip[0] > z_min:
                z_clip = np.insert(z_clip, 0, z_min)
                u_clip = np.insert(u_clip, 0, interp_at(z_min))

        if z_max >= z[0] and z_max <= z[-1]:
            if len(z_clip) == 0 or z_clip[-1] < z_max:
                z_clip = np.append(z_clip, z_max)
                u_clip = np.append(u_clip, interp_at(z_max))

        # 计算深度平均速度  ⟨u⟩ = ( ∫_{z_min}^{z_max} u(z) dz ) / (z_max - z_min)
        if len(z_clip) >= 2:
            H = z_max - z_min
            u_avg = np.trapz(u_clip, z_clip) / H
        else:
            u_avg = np.nan  # 该块在指定高度范围内没有足够数据

        u_avg_list.append(u_avg)

    return np.array(u_avg_list)

g = 9.8
d = 0.00025
rho_a  = 1.225             # air density [kg/m^3] 
rho_p  = 2650.0            # particle density [kg/m^3]   
Shields_all = np.linspace(0.02, 0.06, 5)
u_star_all  = np.sqrt(Shields_all * (rho_p - rho_a) * g * d / rho_a)
tau_top = rho_a * u_star_all**2 

Ua0_all = []
for i in range(5):
    filename = f"FinalS00{i+2}.txt"
    z_min = 13.5*d #12D
    z_max = 0.2
    lines_per_block = 1600

    u_avg_per_block = depth_avg_from_file(filename, z_min, z_max, lines_per_block)
    print(f"S00{i+2}：", u_avg_per_block)
    Ua0_all.append(u_avg_per_block[0])

# tau_basic = 0.5 * rho_a * CD_bed * Ua * abs(Ua)
Ua0_all = np.array(Ua0_all)
CD_bed = tau_top*2/(rho_a*Ua0_all**2)
print('CD_bed', CD_bed)

plt.figure(figsize=(6,5))
plt.plot(Shields_all, CD_bed, 'o')
plt.xlabel(r'$\tilde{\Theta}$ [-]')
plt.ylabel(r'$C_\mathrm{D,bed}$ [-]')
plt.xticks([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07])
plt.ylim(0.003, 0.0045)
plt.tight_layout()