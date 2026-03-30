# -*- coding: utf-8 -*-
"""
Created on Wed Dec 17 11:00:18 2025

@author: WangX3
"""
import matplotlib.pyplot as plt
import pickle
import numpy as np

# read data from pikle file
with open("z_data.pkl", "rb") as f:
    data = pickle.load(f)

particle_z_all = data["particle_z_all"]
bridge_z_all   = data["bridge_z_all"]


# def mean_bridges_per_particle_z_mean_of_ratios(
#     particle_z_all,
#     bridge_z_all,
#     file_indices,
#     index_range,
#     d=0.00025,
#     z_max=800,
#     dz_factor=2
# ):
#     """
#     Compute mean number of liquid bridges per particle as a function of height,
#     using mean over time of (Nb/Np) per bin, multiplied by 2.

#     This returns:
#         mean_bridges(z) = 2 * mean_t [ Nb(z,t) / Np(z,t) ]
#     where times with Np(z,t)=0 are ignored (NaN).

#     Parameters
#     ----------
#     particle_z_all : list of arrays
#     bridge_z_all   : list of arrays
#     file_indices   : list of file indices (unused; kept for consistency)
#     time_range     : (i0, i1) index range
#     d              : particle diameter
#     z_max          : max height in z/d
#     dz_factor      : bin size in units of d

#     Returns
#     -------
#     z_centers : array (z/d)
#     mean_bridges : array
#     """

#     dz = dz_factor
#     z_bins = np.arange(0, z_max + dz, dz)
#     z_centers = 0.5 * (z_bins[:-1] + z_bins[1:])

#     i0, i1 = index_range
#     nT = i1 - i0
#     nZ = len(z_centers)

#     # Store per-time per-bin ratios
#     ratios = np.full((nT, nZ), np.nan, dtype=float)

#     for it, i in enumerate(range(i0, i1)):
#         z_p = np.asarray(particle_z_all[i]) / d
#         z_b = np.asarray(bridge_z_all[i]) / d

#         # Per-time counts
#         Np = np.zeros(nZ, dtype=float)
#         Nb = np.zeros(nZ, dtype=float)

#         # Bin particles (counts per bin)
#         bin_p = np.digitize(z_p, z_bins) - 1
#         valid_p = (bin_p >= 0) & (bin_p < nZ)
#         if np.any(valid_p):
#             Np += np.bincount(bin_p[valid_p], minlength=nZ)

#         # Bin bridges (counts per bin)
#         bin_b = np.digitize(z_b, z_bins) - 1
#         valid_b = (bin_b >= 0) & (bin_b < nZ)
#         if np.any(valid_b):
#             Nb += np.bincount(bin_b[valid_b], minlength=nZ)

#         # Ratio for this time step (leave NaN where Np==0)
#         mask = Np > 0
#         ratios[it, mask] = Nb[mask] / Np[mask]

#     # Mean over time, ignoring NaNs (times with no particles in the bin)
#     mean_bridges = 2.0 * np.nanmean(ratios, axis=0)
    

#     return z_centers, mean_bridges


def mean_bridges_per_particle_z_mean_of_ratios(
    particle_z_all,
    bridge_z_all,
    file_indices,
    index_range,
    d=0.00025,
    z_max=800,
    dz_factor=2,
    z_target=12
):
    """
    Compute mean number of liquid bridges per particle as a function of height
    using mean over time of (Nb/Np) per bin * 2.

    Additionally returns the time series of Nlb at z = z_target (in units of d).

    Returns
    -------
    z_centers : array (z/d)
    mean_bridges : array
    time_series : array of time indices
    Nlb_z_target : array of Nlb(t) at z = z_target
    """

    dz = dz_factor
    z_bins = np.arange(0, z_max + dz, dz)
    z_centers = 0.5 * (z_bins[:-1] + z_bins[1:])

    # Find bin index corresponding to z_target
    target_bin = np.digitize([z_target], z_bins)[0] - 1

    i0, i1 = index_range
    nT = i1 - i0
    nZ = len(z_centers)

    ratios = np.full((nT, nZ), np.nan, dtype=float)
    Nlb_z_target = np.full(nT, np.nan)

    for it, i in enumerate(range(i0, i1)):
        z_p = np.asarray(particle_z_all[i]) / d
        z_b = np.asarray(bridge_z_all[i]) / d

        Np = np.zeros(nZ)
        Nb = np.zeros(nZ)

        # Bin particles
        bin_p = np.digitize(z_p, z_bins) - 1
        valid_p = (bin_p >= 0) & (bin_p < nZ)
        if np.any(valid_p):
            Np += np.bincount(bin_p[valid_p], minlength=nZ)

        # Bin bridges
        bin_b = np.digitize(z_b, z_bins) - 1
        valid_b = (bin_b >= 0) & (bin_b < nZ)
        if np.any(valid_b):
            Nb += np.bincount(bin_b[valid_b], minlength=nZ)

        mask = Np > 0
        ratios[it, mask] = Nb[mask] / Np[mask]

        # Extract time-varying value at z_target
        if 0 <= target_bin < nZ and Np[target_bin] > 0:
            Nlb_z_target[it] = 2 * Nb[target_bin] / Np[target_bin]

    mean_bridges = 2.0 * np.nanmean(ratios, axis=0)

    return z_centers, mean_bridges, Nlb_z_target

# from typing import List, Tuple, Optional

# def split_three_windows_by_threshold_index(
#     C,
#     threshold: float = 0.075,
# ) -> List[Tuple[int, int]]:
#     """
#     Split a uniformly-sampled C array into 3 index windows:
#       1) 0 -> first index where C crosses upward to >= threshold
#       2) overshoot: indices where C > threshold
#       3) after overshoot until end

#     Returns index windows (start_idx, end_idx) with end exclusive.

#     If no up-crossing occurs: returns [(0, n)].
#     If up-crossing occurs but no down-crossing after:
#         returns [(0, up_idx+1), (up_idx+1, n)].
#     """

#     C = np.asarray(C, dtype=float)
#     n = C.size

#     if n < 2:
#         raise ValueError("C must have at least 2 samples.")

#     # ---- Find first up-crossing ----
#     up_idx: Optional[int] = None
#     for i in range(n - 1):
#         if C[i] < threshold <= C[i + 1]:
#             up_idx = i + 1  # first index >= threshold
#             break

#     if up_idx is None:
#         return [(0, n)]

#     # ---- Find first down-crossing after overshoot ----
#     down_idx: Optional[int] = None
#     for i in range(up_idx, n - 1):
#         if C[i] > threshold >= C[i + 1]:
#             down_idx = i + 1  # first index <= threshold
#             break

#     if down_idx is None:
#         return [
#             (0, up_idx),
#             (up_idx, n)
#         ]

#     return [
#         (0, up_idx),
#         (up_idx, down_idx),
#         (down_idx, n)
#     ]

from typing import List, Tuple, Optional

def split_three_windows_by_overshoot_recovery(
    C,
    threshold: float = 0.075,
    final_fraction: float = 0.30,
    recovery_tol: float = 0.15,
) -> List[Tuple[int, int]]:
    """
    Split a uniformly sampled C array into 3 index windows:
      1) pre-overshoot: from 0 to the first index where C >= threshold
      2) overshoot: from that index until C returns to within ± recovery_tol
         of the reference mean computed over the final `final_fraction` of the record
      3) post-overshoot / steady: from the recovery index to the end

    Returns
    -------
    List[Tuple[int, int]]
        Index windows (start_idx, end_idx), with end exclusive.

    Notes
    -----
    - The reference mean is computed over the final `final_fraction` of the record.
    - "Within ±15%" means:
          abs(C[i] - C_ref) <= recovery_tol * C_ref
      if C_ref > 0.
    - If C_ref == 0, absolute equality is used.
    - If no threshold crossing occurs, returns [(0, n)].
    - If threshold crossing occurs but no recovery is found, returns:
          [(0, up_idx), (up_idx, n)]
    """

    C = np.asarray(C, dtype=float)
    n = C.size

    if n < 2:
        raise ValueError("C must have at least 2 samples.")

    if not (0 < final_fraction <= 1):
        raise ValueError("final_fraction must be in the interval (0, 1].")

    if recovery_tol < 0:
        raise ValueError("recovery_tol must be non-negative.")

    # ---- Compute reference mean over final fraction of the record ----
    tail_len = max(1, int(np.ceil(final_fraction * n)))
    tail_start = n - tail_len
    C_ref = np.mean(C[tail_start:])

    # ---- Find first index where C >= threshold ----
    up_idx: Optional[int] = None
    for i in range(n):
        if C[i] >= threshold:
            up_idx = i
            break

    if up_idx is None:
        return [(0, n)]

    # ---- Define recovery band around reference mean ----
    if C_ref == 0:
        def in_recovery_band(x: float) -> bool:
            return x == 0
    else:
        lower = (1.0 - recovery_tol) * C_ref
        upper = (1.0 + recovery_tol) * C_ref

        def in_recovery_band(x: float) -> bool:
            return lower <= x <= upper

    # ---- Find first recovery after overshoot begins ----
    recovery_idx: Optional[int] = None
    for i in range(up_idx + 1, n):
        if in_recovery_band(C[i]):
            recovery_idx = i
            break

    if recovery_idx is None:
        return [
            (0, up_idx),
            (up_idx, n),
        ]

    return [
        (0, up_idx),
        (up_idx, recovery_idx),
        (recovery_idx, n),
    ]

moisture_tags = ["M1LB", "M5LBR1", "M10LBR1", "M20LB"]
file_indices = range(204, 705)

# Loop over all moisture levels
omega_labels = ['M1', 'M5', 'M10', 'M20']
C_dpm = []
for label in omega_labels:
    # --- Load sediment transport data ---
    file_path = f'../CGdata/hb=13.5d/Shields00{5}{label}-135d.txt'
    data = np.loadtxt(file_path)
    C = data[:, 1]
    # Append to lists
    C_dpm.append(C)

profiles = {tag: [] for tag in moisture_tags}
index_store = {tag: [] for tag in moisture_tags}

for i, tag in enumerate(moisture_tags):
        index_windows_by_moisture = split_three_windows_by_overshoot_recovery(C_dpm[i], threshold=0.075)
        # index_by_moisture = [(0, 1)] + index_windows_by_moisture
        print(index_windows_by_moisture)
        index_store[tag].append(index_windows_by_moisture)
        for i0, i1 in index_windows_by_moisture:
            
            zc, mean_b, _ = mean_bridges_per_particle_z_mean_of_ratios(
                particle_z_all[tag],
                bridge_z_all[tag],
                file_indices,
                index_range=(i0, i1),
                d=0.00025,
                dz_factor=2,
                z_target=11
            )
            
            
            profiles[tag].append((zc, mean_b))


fig, axes = plt.subplots(1, 4, figsize=(14, 6), sharey=True)

time_labels = ["Pre-overshoot state", "Overshoot state", "Steady state"]
colors = {
    "Pre-overshoot state": "tab:blue",
    "Overshoot state": "tab:orange",
    "Steady state": "tab:green",
}

Omegalist = [1, 5, 10, 20]

for ax, (moisture, prof_list), omega in zip(axes, profiles.items(), Omegalist):
    for label, (zc, mean_b) in zip(time_labels, prof_list):
        ax.plot(
            mean_b, zc,
            marker='o', ms=3, lw=1.5,
            color=colors[label],
            label=label
        )

    ax.set_title(f"$\Omega$ = {omega}%", fontsize=14)
    ax.set_xlabel(r"$N_\mathrm{lb}$ [-]", fontsize=14)
    ax.set_ylim(0, 20)
    ax.grid(True, linestyle="--", alpha=0.5)
    
    ax.axhline(
    y=11,
    color='grey',
    linestyle='--',
    linewidth=1.5,
    alpha=0.8
    )
    
    ax.axhline(
    y=13.5,
    color='grey',
    linestyle='-',
    linewidth=1.5,
    alpha=0.8
    )

axes[0].set_ylabel(r"Elevation $z/d$ [-]", fontsize=14)
axes[0].legend(frameon=False)

plt.tight_layout()
plt.show()

# second graph showing time-varying Nlb and C
profiles_tv = {tag: [] for tag in moisture_tags}

for tag in moisture_tags:
        _, _, N_lb = mean_bridges_per_particle_z_mean_of_ratios(
            particle_z_all[tag],
            bridge_z_all[tag],
            file_indices,
            index_range=(0, len(particle_z_all[tag])-1),
            d=0.00025,
            dz_factor=2,
            z_target=11
        )
        profiles_tv[tag].append((N_lb))
        
time_c = np.linspace(0, 5, len(C))
time = np.linspace(0, 5, len(profiles_tv["M20LB"][0]))

def smooth_moving_average_preserve_edges(data, window=11):
    data = np.asarray(data)
    window = int(window)

    if window % 2 == 0:
        window += 1  # enforce odd window

    half = window // 2
    smoothed = data.copy()

    # Apply smoothing only to interior
    kernel = np.ones(window) / window
    interior = np.convolve(data, kernel, mode='valid')

    smoothed[half:-half] = interior

    return smoothed

Nlb = smooth_moving_average_preserve_edges(profiles_tv["M20LB"][0])

# Overshoot interval: (start_idx, end_idx) with end exclusive
i0, i1 = index_store['M20LB'][0][1]
# Convert indices to times
t_start = time_c[i0]
t_end = time_c[i1 - 1] if i1 > 0 else time_c[i1]   # last point inside the interval

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=False)

ax1.plot(time_c, C)
ax1.axvline(t_start, linestyle='--', color='grey', linewidth=1.5)
ax1.axvline(t_end, linestyle='-.', color='grey', linewidth=1.5)
ax1.grid()
ax1.set_xlabel("t [s]", fontsize=14)
ax1.set_xlim([0, 5])
ax1.set_ylabel(r"$c$ [kg/m$^2$]", fontsize=14)

ax2.plot(time, Nlb)
ax2.axvline(t_start, linestyle='--', color='grey', linewidth=1.5)
ax2.axvline(t_end, linestyle='-.', color='grey', linewidth=1.5)
ax2.grid()
ax2.set_xlim([0, 5])
ax2.set_xlabel("t [s]", fontsize=14)
ax2.set_ylabel(r"$N_{lb}$ at $z=11d$", fontsize=14)

plt.tight_layout()
plt.show()


# # for NCK days 2026 poster
# plt.figure(figsize=(6, 8))
# for label, (zc, mean_b) in zip(time_labels, profiles["M20LB"]): 
#     plt.plot(
#                 mean_b, zc,
#                 marker='o', ms=3, lw=1.5,
#                 color=colors[label],
#                 label=label
#             )
        
#     plt.axhline(
#     y=11,
#     color='grey',
#     linestyle='--',
#     linewidth=1.5,
#     alpha=0.8
#     )
        
# plt.xlabel(r"$N_\mathrm{lb}$ [-]", fontsize=16)
# plt.ylim(0, 20)
# plt.grid(True, linestyle="--", alpha=0.5)
# plt.ylabel(r"Elevation $z/d$ [-]", fontsize=16)
# plt.legend(frameon=False, fontsize=13)
# plt.tight_layout()
# plt.show()


# plt.figure(figsize=(8, 8))
# plt.subplot(2,1,1)
# plt.plot(time_c, C)
# plt.grid()
# plt.xlabel("$t$ [s]", fontsize=16)
# plt.xlim([0, 5])
# plt.ylabel(r"$C$ [kg/m$^2$]", fontsize=16)
# plt.subplot(2,1,2)
# plt.plot(time, Nlb)
# plt.grid()
# plt.xlim([0, 5])
# plt.xlabel("$t$ [s]", fontsize=16)
# plt.ylabel(r"$N_\mathrm{lb}$ at $z=11d$", fontsize=16)
# plt.tight_layout()
# plt.show()
