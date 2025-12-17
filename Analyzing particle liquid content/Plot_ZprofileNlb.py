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


def mean_bridges_per_particle_z(
    particle_z_all,
    bridge_z_all,
    file_indices,
    time_range,
    d=0.00025,
    z_max=800,
    dz_factor=2
):
    """
    Compute mean number of liquid bridges per particle as a function of height.

    Parameters
    ----------
    particle_z_all : list of arrays
    bridge_z_all   : list of arrays
    file_indices   : list of file indices (for consistency)
    time_range     : (i0, i1) index range
    d              : particle diameter
    z_max          : max height in z/d
    dz_factor      : bin size in units of d

    Returns
    -------
    z_centers : array (z/d)
    mean_bridges : array
    """

    dz = dz_factor
    z_bins = np.arange(0, z_max + dz, dz)
    z_centers = 0.5 * (z_bins[:-1] + z_bins[1:])

    bridge_count = np.zeros(len(z_centers))
    particle_count = np.zeros(len(z_centers))

    i0, i1 = time_range

    for i in range(i0, i1):

        z_p = np.asarray(particle_z_all[i]) / d
        z_b = np.asarray(bridge_z_all[i]) / d

        # Bin particles
        bin_p = np.digitize(z_p, z_bins) - 1
        valid_p = (bin_p >= 0) & (bin_p < len(z_centers))
        for b in bin_p[valid_p]:
            particle_count[b] += 1

        # Bin bridges
        bin_b = np.digitize(z_b, z_bins) - 1
        valid_b = (bin_b >= 0) & (bin_b < len(z_centers))
        for b in bin_b[valid_b]:
            bridge_count[b] += 1

    mean_bridges = np.full_like(z_centers, np.nan, dtype=float)
    mask = particle_count > 0
    mean_bridges[mask] = bridge_count[mask] / particle_count[mask]

    return z_centers, mean_bridges


moisture_tags = ["M1LB", "M5LBR1", "M10LBR1", "M20LB"]
file_indices = range(204, 705)
time_windows_by_moisture = {
    "M1LB": [
        (0.0, 0.2, "0–0.2 s"),
        (0.3, 1.2, "0.3–1.2 s"),
        (4.0, 5.0, "4–5 s"),
    ],
    "M5LBR1": [
        (0.0, 0.5, "0–0.5 s"),
        (0.5, 1.5, "0.5–1.5 s"),
        (4.0, 5.0, "4–5 s"),
    ],
    "M10LBR1": [
        (0.0, 0.8, "0–0.8 s"),
        (1.0, 2.0, "1–2 s"),
        (4.0, 5.0, "4–5 s"),
    ],
    "M20LB": [
        (0.0, 1.0, "0–1 s"),
        (2.0, 3.0, "2–3 s"),
        (4.0, 5.0, "4–5 s"),
    ],
}

dt = 0.01
def time_to_indices(t0, t1, dt):
    i0 = int(t0 / dt)
    i1 = int(t1 / dt)
    return i0, i1

profiles = {tag: [] for tag in moisture_tags}
profile_labels = {tag: [] for tag in moisture_tags}

for tag in moisture_tags:
    for t0, t1, label in time_windows_by_moisture[tag]:
        i0, i1 = time_to_indices(t0, t1, dt)

        zc, mean_b = mean_bridges_per_particle_z(
            particle_z_all[tag],
            bridge_z_all[tag],
            file_indices,
            time_range=(i0, i1),
            d=0.00025,
            dz_factor=2,
        )

        profiles[tag].append((zc, mean_b))
        profile_labels[tag].append(label)


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

    ax.set_title(f"$\Omega$ = {omega}%")
    ax.set_xlabel(r"Mean number of liquid bridges / particle")
    ax.set_ylim(0, 40)
    ax.grid(True, linestyle="--", alpha=0.5)

axes[0].set_ylabel(r"Elevation $z/d$")
axes[0].legend(frameon=False)

plt.tight_layout()
plt.show()


