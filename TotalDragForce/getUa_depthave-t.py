# -*- coding: utf-8 -*-
"""
Created on Fri Jul 11 14:23:07 2025

@author: WangX3
"""
#read log file and calculate the depth-averaged Uair - t
import numpy as np

filename = 'job_test_4679.log'
output_filename = 'Uair_ave-tS006Dryh02.txt'

z_min = 0.003 #12D
z_max = 0.2
lines_per_block = 1600

time_all = []
u_avg_all = []

with open(filename, 'r') as file, open(output_filename, 'w') as outfile:
    lines = file.readlines()
    i = 0

    while i < len(lines):
        line = lines[i].strip()

        if line.startswith('t='):
            # Extract time value
            t = float(line.split('=')[1])
            i += 1

            # Read next `lines_per_block` lines
            block_data = []
            for _ in range(lines_per_block):
                if i >= len(lines):
                    break
                parts = lines[i].strip().split()
                if len(parts) == 3:
                    block_data.append([float(val) for val in parts])
                i += 1

            block_data = np.array(block_data)
            z = block_data[:, 0]
            u = block_data[:, 1]

            # Filter by z range
            mask = (z >= z_min) & (z <= z_max)
            z_filtered = z[mask]
            u_filtered = u[mask]

            if len(z_filtered) > 1:
                # Sort by height
                sort_idx = np.argsort(z_filtered)
                z_filtered = z_filtered[sort_idx]
                u_filtered = u_filtered[sort_idx]

                # Compute depth-averaged velocity
                H = z_max - z_min
                u_avg = np.trapz(u_filtered, z_filtered) / H

                # Store and write
                time_all.append(t)
                u_avg_all.append(u_avg)
                outfile.write(f"{t:.5f}\t{u_avg:.6f}\n")

        else:
            i += 1

print("Depth-averaged wind velocities saved to Uair_ave-t.txt")
