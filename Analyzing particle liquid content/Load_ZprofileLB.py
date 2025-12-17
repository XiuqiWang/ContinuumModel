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



base_name = "IniOTR.restart"
moisture_tags = ["M1LB", "M5LBR1", "M10LBR1", "M20LB"]
file_indices = range(204, 705)

particle_z_all = {tag: [] for tag in moisture_tags}
bridge_z_all   = {tag: [] for tag in moisture_tags}

for tag in moisture_tags:
    for i, idx in enumerate(file_indices):
        filename = f"vtuData/restart/S005{tag}IniOTR.restart.{idx}"
    
        if not os.path.isfile(filename):
            print(f"Missing file: {filename}")
            particle_z_all[i] = []
            bridge_z_all[i]   = []
            continue
    
        particle_z = extract_particle_z_positions(filename)
        bridge_z   = extract_liquid_bridge_z_positions(filename)
    
        particle_z_all[tag].append(particle_z)
        bridge_z_all[tag].append(bridge_z)
    
        # print(f"{filename}: particles={len(particle_z)}, bridges={len(bridge_z)}")

import pickle

with open("z_data.pkl", "wb") as f:
    pickle.dump(
        {
            "particle_z_all": particle_z_all,
            "bridge_z_all": bridge_z_all,
            "moisture_tags": moisture_tags,
            "file_indices": list(file_indices),
        },
        f,
        protocol=pickle.HIGHEST_PROTOCOL,
    )



