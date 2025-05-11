#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is an example code to be included as supplementary material to the following article: 
"AutoHood3D: A Multi-Modal Benchmark for Automotive Hood Design and Fluid–Structure Interaction".

This is a demonstration intended to provide a working example with data provided in the repo.
For application on other datasets, the requirement is to configure the settings. 

Dependencies: 
    - Python package list provided: package.list

Running the code: 
    - assuming above dependencies are configured, "python <this_file.py>" will run the demo code. 
    NOTE - It is important to check README prior to running this code.

Contact: 
    - Vansh Sharma at vanshs@umich.edu
    - Venkat Raman at ramanvr@umich.edu

Affiliation: 
    - APCL Group 
    - Dept. of Aerospace Engg., University of Michigan, Ann Arbor
"""
import os
import json
import random
import shutil
import numpy as np
import madcad as mc
from madcad import *
from concurrent.futures import ProcessPoolExecutor, as_completed

# Reproducible 
random.seed(108)

# --- Helper functions ---
def arrange_points_in_circular_order(points, clockwise=True):
    centroid = np.mean(points[:, :2], axis=0)
    angles = np.arctan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])
    indices = np.argsort(-angles) if clockwise else np.argsort(angles)
    return points[indices]

def random_translate_curve_to_region(curve, x_min, x_max, y_min, y_max):
    cx_min, cx_max = curve[:, 0].min(), curve[:, 0].max()
    cy_min, cy_max = curve[:, 1].min(), curve[:, 1].max()
    x_range = x_max - (cx_max - cx_min)
    y_range = y_max - (cy_max - cy_min)
    ox = np.random.uniform(x_min, x_range)
    oy = np.random.uniform(y_min, y_range)
    tr = curve.copy()
    tr[:, 0] += ox - cx_min
    tr[:, 1] += oy - cy_min
    return tr

def check_overlap(candidate, existing, min_dist):
    for ex in existing:
        dists = np.linalg.norm(candidate[:, None, :2] - ex[None, :, :2], axis=2)
        if dists.min() < min_dist:
            return True
    return False

def fit_curve_with_random_position(curve, existing, min_dist, x0, x1, y0, y1, max_attempts=1000):
    for _ in range(max_attempts):
        trial = random_translate_curve_to_region(curve, x0, x1, y0, y1)
        if not check_overlap(trial, existing, min_dist):
            return trial
    print("Warning: could not place a curve after max attempts.")
    return None

def fit_curves_into_region(curves, min_dist, x0, x1, y0, y1):
    placed = []
    for curve in curves:
        p = fit_curve_with_random_position(curve, placed, min_dist, x0, x1, y0, y1)
        if p is not None:
            placed.append(p)
    return placed


#%% User Input
# --- Paths & parameters ---
default_save_dir = "OUTPUT_FOLDER"
shell_path = "PATH_TO_BaseSkins"
processed_folder = 'FOLDER_TO_PUT_PROCESSED_BaseSkins'
filenames = sorted(f for f in os.listdir(shell_path) if f.endswith('.stl'))

# Range for the offset of the cutout plane from the hood’s centerline (min, max)
center_dist_range = (0.022, 0.045)

# Range for the minimum allowable spacing between adjacent cutout curves (min, max)
min_dist_range = (0.012, 0.045)

# Number of curve‐pair sets to sample from each cluster per repetition
num_samples_outer = 8

# For each selected curve‐pair set, cp_nums[i] defines how many curves to pick 
# (per side, before mirroring) at the i‑th sampling stage
cp_nums = [1, 2, 3, 4]

# Number of times to repeat sampling cp_nums[i] curves from the available set
cp_reps = [3, 3, 3, 3]

# Number of times to resample an independent set of outer curve pairs per cluster
num_repetitions_outer = 1

# --- Load clustering results ---
JSON_file = 'clustering_output_clusters_9.json'
with open(JSON_file, 'r') as f:
    config = json.load(f)
labels = np.array(config['labels'], dtype=int)
file_paths = config['file_path']
list_peri = config['list_peri']
list_area = config['list_area']

# Build curves_list and group by label
grouped_data = {}
for idx, path in enumerate(file_paths):
    pts = np.loadtxt(path, skiprows=1, delimiter=',') * 0.001 ## units adjustment
    pts[:, 2] = 0.0
    pts = arrange_points_in_circular_order(pts)
    lbl = labels[idx]
    grouped_data.setdefault(lbl, []).append({
        'index': idx,
        'curve': pts,
        'perimeter': list_peri[idx],
        'area': list_area[idx]
    })

#%%
# --- Shell processing ---
def process_shell(stl_fname):
    shellID = int(stl_fname[11:-9])
    print(f"Processing shell {shellID}")
    out_folder = os.path.join(default_save_dir, f'geo_{shellID:03d}')
    os.makedirs(out_folder, exist_ok=True)

    # Compute bounding box once from base mesh - curves placed within this
    base = mc.read(os.path.join(shell_path, stl_fname))
    base.mergeclose()
    bbox = base.box()
    x0, x1 = 0.55 * bbox.min[0], 0.75 * bbox.max[0] ## 0.55 and 0.75 are for calibration
    y_min_base, y_max = 0.8 * bbox.min[1], 0.8 * bbox.max[1] ## 0.8 and 0.8 are for calibration
    z0, z1 = bbox.min[2], bbox.max[2]

    # Outer sampling per cluster
    for cluster_label, items in grouped_data.items():
        # sample sets of curvePair indices
        avail = list(range(len(items)))
        outer_sets = []
        for _ in range(num_repetitions_outer):
            if len(avail) < num_samples_outer:
                avail = list(range(len(items)))
            sel = random.sample(avail, num_samples_outer)
            outer_sets.append(sel)
            avail = [i for i in avail if i not in sel]

        # For each outer sample
        for curvePair in outer_sets:
            # CP sampling
            cp_avail = list(range(len(curvePair)))
            cp_sets = []
            for cnt, rep in zip(cp_nums, cp_reps):
                for _ in range(rep):
                    if len(cp_avail) < cnt:
                        cp_avail = list(range(len(curvePair)))
                    pick = random.sample(cp_avail, cnt)
                    global_idxs = [curvePair[i] for i in pick]
                    cp_sets.append(global_idxs)
                    cp_avail = [i for i in cp_avail if i not in pick]

            # Perform cuts per cp_set
            for global_idxs in cp_sets:
                cd = random.uniform(*center_dist_range)
                md = random.uniform(*min_dist_range)
                curves_to_fit = [items[i]['curve'] for i in global_idxs]
                fitted = fit_curves_into_region(curves_to_fit, md, x0, x1, cd, y_max)

                # Reload and subdivide mesh for each cut set
                mesh = mc.read(os.path.join(shell_path, stl_fname))
                mesh.mergeclose()
                mesh = mesh.subdivide(2)

                for curve in fitted:
                    pts1 = [vec3([p[0], p[1], z0]) for p in curve]
                    pts2 = [vec3([p[0], -p[1], z0]) for p in curve]
                    flat1 = web(wire(pts1).close().segmented())
                    flat2 = web(wire(pts2).close().segmented())
                    hole1 = extrusion(flat1, -0.6 * (z1 - z0) * Z, 0.75)
                    hole2 = extrusion(flat2, -0.6 * (z1 - z0) * Z, 0.75)
                    try:
                        mesh = mc.boolean.difference(mesh, hole1)
                        mesh = intersection(mesh, hole2)
                    except:
                        continue

                # Filename construction
                ids_str = "".join(f"_{items[idx]['index']:04d}" for idx in global_idxs)
                fname = (
                    f"geo_{shellID:03d}"
                    f"_clusterID_{cluster_label}"
                    f"_crvCount_{len(curves_to_fit)}"
                    f"{ids_str}"
                    f"_cd_{cd:.3f}"
                    f"_md_{md:.3f}.stl"
                )
                mc.write(mesh, os.path.join(out_folder, fname))

    # Move processed STL
    try:
        shutil.move(os.path.join(shell_path, stl_fname), processed_folder)
    except Exception as e:
        print(f"Failed to move {stl_fname}: {e}")
    print(f"Completed shell {shellID}")

#%%
# --- Parallel execution ---
if __name__ == '__main__':
    num_workers = 10
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_shell, fn): fn for fn in filenames}
        for fut in as_completed(futures):
            fn = futures[fut]
            try:
                fut.result()
            except Exception as exc:
                print(f"{fn} generated exception: {exc}")
    print("All shells processed.")
    
    
    
    
    
    
    
    
    
    