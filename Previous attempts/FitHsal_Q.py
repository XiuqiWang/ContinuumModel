# -*- coding: utf-8 -*-
"""
Created on Thu Aug 14 16:33:05 2025

@author: WangX3
"""

import pandas as pd
import numpy as np
from scipy import stats

# Load file (tab or comma separated; change 'sep' if needed)
df = pd.read_csv("Hsals_Qs_timeseries.csv", sep=",")  

# Filter: Shields between 0.02 and 0.06, dry condition (Liquid=0)
df_filtered = df[(df["Shields"] >= 0.02) & 
                 (df["Shields"] <= 0.06) &
                 (df["Liquid"] == 0)]

# Remove any NaNs or zeros
df_filtered = df_filtered.dropna(subset=["Q", "Hsal"])
df_filtered = df_filtered[(df_filtered["Q"] > 0) & (df_filtered["Hsal"] > 0)]


Q = df_filtered["Q"].values
Hs = df_filtered["Hsal"].values

# log-log regression
logQ = np.log(Q)
logHs = np.log(Hs)

slope, intercept, r_value, p_value, std_err = stats.linregress(logQ, logHs)
A = np.exp(intercept)
B = slope

print(f"Fit results: h_s = {A:.4e} * Q^{B:.4f}")
print(f"R^2 = {r_value**2:.4f}")

import matplotlib.pyplot as plt

plt.figure(figsize=(6,4))
plt.scatter(Q, Hs, s=20, alpha=0.7, label="DEM data (dry)")

Qfit = np.linspace(Q.min(), Q.max(), 100)
plt.plot(Qfit, A*Qfit**B, 'r-', label=f"Fit: A={A:.2e}, B={B:.2f}")
# plt.plot(Qfit, A_fixed*Qfit**B_fixed, 'g--', label=f"Phys: B=2/3, A={A_fixed:.2e}")

plt.xscale("log")
plt.yscale("log")
plt.xlabel("Q [m²/s]", fontsize=12)
plt.ylabel("Hsal [m]", fontsize=12)
plt.legend()
plt.tight_layout()
plt.show()

