# -*- coding: utf-8 -*-
"""
Created on Tue Dec 16 14:54:36 2025

@author: WangX3
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatter


# -----------------------------
# User settings
# -----------------------------
base_dir = "vtuData"   # <-- change this
moisture_tags = ["M1", "M5", "M10", "M20"]
file_indices = range(204, 705)        # 204 to 704 inclusive

# -----------------------------
# Output container
# -----------------------------
fullLiquidVolume_data = {tag: [] for tag in moisture_tags}

z_position_data = {tag: [] for tag in moisture_tags}

volume_data = {tag: [] for tag in moisture_tags}

# -----------------------------
# Loop over files
# -----------------------------
for tag in moisture_tags:
    for idx in file_indices:

        filename = f"S005{tag}LBIniParticle_{idx}.vtu"
        filepath = os.path.join(base_dir, filename)

        if not os.path.isfile(filepath):
            print(f"Warning: file not found: {filename}")
            continue

        with open(filepath, "r") as f:
            lines = f.readlines()

        # Extract liquid volume from Position DataArray
        # -------------------------------------------------
        start_line = None
        for i, line in enumerate(lines):
            if 'Name="fullLiquidVolume"' in line:
                start_line = i + 1  # data starts on next line
                break

        if start_line is None:
            print(f"Warning: fullLiquidVolume not found in {filename}")
            continue

        # --- read the next 2720 values ---
        values = []
        for line in lines[start_line:]:
            if "</DataArray>" in line:
                break
            values.extend(float(x) for x in line.split())

        if len(values) < 2725:
            print(f"Warning: only {len(values)} values in {filename}")
            continue

        # keep exactly 2720 values
        fullLiquidVolume_data[tag].append(values[:2725])
        
        # Extract z-position from Position DataArray
        # -------------------------------------------------
        pos_start = None
        for i, line in enumerate(lines):
            if 'Name="Position"' in line and 'NumberOfComponents="3"' in line:
                pos_start = i + 1
                break

        if pos_start is None:
            print(f"Warning: Position array not found in {filename}")
            continue

        z_values = []
        for line in lines[pos_start:]:
            if "</DataArray>" in line:
                break

            nums = [float(x) for x in line.split()]
            # take every 3rd value → z-component
            z_values.extend(nums[2::3])

        if len(z_values) < 2725:
            print(f"Warning: only {len(z_values)} z-values in {filename}")
            continue

        z_position_data[tag].append(z_values[:2725])
        
        # Extract Radius and compute particle volume
        # -------------------------------------------------
        rad_start = None
        for i, line in enumerate(lines):
            if 'Name="Radius"' in line:
                rad_start = i + 1
                break

        if rad_start is None:
            print(f"Warning: Radius array not found in {filename}")
            continue

        R_values = []
        for line in lines[rad_start:]:
            if "</DataArray>" in line:
                break
            R_values.extend([float(x) for x in line.split()])

        if len(R_values) < 2725:
            print(f"Warning: only {len(R_values)} radii in {filename}")
            continue

        R_values = np.asarray(R_values[:2725])

        # Particle volume: V = 4/3 π R³
        Vp_values = (4.0 / 3.0) * np.pi * R_values**3

        volume_data[tag].append(Vp_values)


from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# -----------------------------
# Constants and settings
# -----------------------------
D = 0.00025  # particle diameter [m]

time_windows = [
    (0, 100, "0–1 s"),
    (200, 300, "2–3 s"),
    (400, 500, "4–5 s"),
]

colors = {
    "0–1 s": "tab:blue",
    "2–3 s": "tab:orange",
    "4–5 s": "tab:green",
}

tag = "M20"

# Height bins (in z/D)
z_max = 800
dz = 2
z_bins = np.arange(0, z_max + dz, dz)
z_centers = 0.5 * (z_bins[:-1] + z_bins[1:])

# -------------------------------------------------
# Helper: compute mean vertical profile
# -------------------------------------------------
def compute_mean_profile(i0, i1):

    sum_liquid = np.zeros(len(z_centers))
    count = np.zeros(len(z_centers))

    for idx in range(i0, i1):

        V_liq = np.asarray(fullLiquidVolume_data[tag][idx])
        V_p   = np.asarray(volume_data[tag][idx])
        z     = np.asarray(z_position_data[tag][idx])

        Omega_p = V_liq / V_p * 100.0   # [%]
        zD = z / D

        bin_idx = np.digitize(zD, z_bins) - 1
        valid = (bin_idx >= 0) & (bin_idx < len(z_centers))

        for b in np.unique(bin_idx[valid]):
            mask = bin_idx == b
            sum_liquid[b] += Omega_p[mask].sum()
            count[b] += np.sum(mask)

    mean_liquid = np.full_like(z_centers, np.nan, dtype=float)
    m = count > 0
    mean_liquid[m] = sum_liquid[m] / count[m]

    return mean_liquid

# -------------------------------------------------
# Precompute profiles for all stages (needed for inset)
# -------------------------------------------------
profiles = {}
for i0, i1, label in time_windows:
    profiles[label] = compute_mean_profile(i0, i1)

# -----------------------------
# Figure
# -----------------------------
plt.close('all')
fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)

for ax, (_, _, label) in zip(axes, time_windows):

    mean_liquid = profiles[label]

    ax.plot(
        mean_liquid,
        z_centers,
        'k.',
        label=label
    )

    ax.set_xscale("log")
    ax.set_title(label)
    ax.set_xlabel(r"Mean particle liquid content $V_\ell/V_p$ [%]")
    ax.set_xlim(1e-4, 30)
    ax.set_ylim(0, z_max)
    ax.grid(True, which="both", linestyle="--", alpha=0.5)

# Shared y-axis
axes[0].set_ylabel(r"Elevation $z/d$")

# -------------------------------------------------
# Inset in the middle panel (2–3 s)
# -------------------------------------------------
ax_mid = axes[1]

ax_inset = inset_axes(
    ax_mid,
    width="40%",
    height="40%",
    loc="lower left",
    bbox_to_anchor=(0.15, 0.15, 0.75, 0.75),  # ← shift right
    bbox_transform=ax_mid.transAxes,
    borderpad=0
)

z_inset_max = 12.5
mask_inset = z_centers <= z_inset_max

for label, color in colors.items():
    ax_inset.plot(
        profiles[label][mask_inset],
        z_centers[mask_inset],
        '-o',
        color=color,
        label=label,
        markersize=3
    )

ax_inset.set_xscale("log")
ax_inset.set_xlim(9, 25)
ax_inset.set_ylim(0, z_inset_max)
ax_inset.set_xlabel(r"$V_\ell/V_p$ [%]", fontsize=8)
ax_inset.set_ylabel(r"$z/d$", fontsize=8)
ax_inset.tick_params(axis='both', labelsize=7)
ax_inset.grid(True, which="both", linestyle="--", alpha=0.4)
ax_inset.legend(fontsize=7, frameon=False)
ax_inset.set_xticks([10, 20])
ax_inset.set_xticklabels(['10', '20'], fontsize=8)

# Optional visual cue for zoomed region
ax_mid.axhspan(0, z_inset_max, color="grey", alpha=0.08)

plt.tight_layout()
plt.show()

