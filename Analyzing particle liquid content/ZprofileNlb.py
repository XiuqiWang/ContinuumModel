# -*- coding: utf-8 -*-
"""
Created on Tue Dec 16 17:27:51 2025

@author: WangX3
"""
import os
import numpy as np
import matplotlib.pyplot as plt

def extract_particle_z_positions(filename):
    """
    Extract vertical positions (z) of particles from LiquidFilmParticle lines.
    """
    z_positions = []

    with open(filename, "r") as f:
        for line in f:
            if line.startswith("LiquidFilmParticle id"):
                tokens = line.split()

                if "position" in tokens:
                    pos_idx = tokens.index("position")
                    # position x y z → z is pos_idx + 3
                    try:
                        z = float(tokens[pos_idx + 3])
                        z_positions.append(z)
                    except (IndexError, ValueError):
                        continue

    return z_positions

def extract_liquid_bridge_z_positions(filename):
    """
    Extract vertical positions (z) of liquid bridges that are in contact
    (wasInContact == 1).
    """
    z_positions = []

    with open(filename, "r") as f:
        for line in f:
            if line.startswith(
                "LinearViscoelasticFrictionLiquidMigrationWilletViscousInteraction"
            ):
                tokens = line.split()

                # Check contact status
                if "wasInContact" not in tokens:
                    continue

                try:
                    contact_idx = tokens.index("wasInContact")
                    was_in_contact = int(tokens[contact_idx + 1])
                except (IndexError, ValueError):
                    continue

                if was_in_contact != 1:
                    continue  # skip non-contacting bridges

                # Extract contact-point z-position
                if "contactPoint" in tokens:
                    try:
                        cp_idx = tokens.index("contactPoint")
                        z = float(tokens[cp_idx + 3])  # x y z
                        z_positions.append(z)
                    except (IndexError, ValueError):
                        continue

    return z_positions



base_name = "S005M20LBIniOTR.restart"
file_indices = range(204, 705)

particle_z_all = [None] * len(file_indices)
bridge_z_all   = [None] * len(file_indices)

for i, idx in enumerate(file_indices):
    filename = f"vtuData/{base_name}.{idx}"

    if not os.path.isfile(filename):
        print(f"Missing file: {filename}")
        particle_z_all[i] = []
        bridge_z_all[i]   = []
        continue

    particle_z = extract_particle_z_positions(filename)
    bridge_z   = extract_liquid_bridge_z_positions(filename)

    particle_z_all[i] = particle_z
    bridge_z_all[i]   = bridge_z

    print(f"{filename}: particles={len(particle_z)}, bridges={len(bridge_z)}")


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

time_windows = [
    (0, 100, "0–1 s"),
    (200, 300, "2–3 s"),
    (400, 500, "4–5 s"),
]

profiles = {}

for i0, i1, label in time_windows:
    zc, mean_b = mean_bridges_per_particle_z(
        particle_z_all,
        bridge_z_all,
        file_indices,
        time_range=(i0, i1),
        d=0.00025,
        dz_factor=2
    )
    profiles[label] = (zc, mean_b)


fig, ax = plt.subplots(figsize=(4.5, 6))

colors = {
    "0–1 s": "tab:blue",
    "2–3 s": "tab:orange",
    "4–5 s": "tab:green"
}

for label, (zc, mean_b) in profiles.items():
    ax.plot(mean_b, zc, marker='o', ms=3, lw=1.5,
            label=label, color=colors[label])

ax.set_xlabel(r"Mean number of liquid bridges per particle")
ax.set_ylabel(r"Elevation $z/d$")
ax.set_ylim(0, 60)     # near-bed focus (adjust if needed)
ax.grid(True, linestyle="--", alpha=0.5)
ax.legend(frameon=False)

plt.tight_layout()
plt.show()


