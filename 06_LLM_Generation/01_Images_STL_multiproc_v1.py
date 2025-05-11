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
import os, time, sys
import pyvista as pv
from multiprocessing import Pool


#%% USER INPUT
local = os.getcwd()  ## Get local dir
os.chdir(local)      ## shift the work dir to local dir
# print('Current working directory: {}'.format(local))
    
dirPath = 'STL_FILES'
outDir = 'OUTPUT_FOLDER'

# Set the number of worker processes (max procs in the CPU Job)
num_workers = 27

#%% Discover Subfolders and Build List of Tasks
# Read all the main folders 
exclude_prefixes = ('._', '.D')  # exclusion prefixes
include_prefixes = ('geo_')       # inclusion prefixes

# Get only directories that start with include_prefixes
for (dirpath, dirnamesH, filenames) in os.walk(dirPath):
    dirnamesH = [f for f in dirnamesH if f.startswith(include_prefixes)]
    break
del filenames, dirpath
dirnamesH.sort()

# Find hood names and their corresponding subfolder paths
hood_paths_main={}
for hood_data_dir in dirnamesH:
    hoodPath = os.path.join(dirPath, hood_data_dir)
    # For each geo_ folder, look at its subdirectories
    for (dirpath, dirnames, filenames) in os.walk(hoodPath):
        hood_paths = [f for f in filenames if f.endswith('.stl')]
        hood_paths.sort()
        break
    del filenames, dirpath, dirnames
    hood_paths_main[hood_data_dir] = hood_paths


# Build a list of tasks for multiprocessing.
# Each task will be a tuple:
# (stl_file, output_file)
tasks = []
for geoname, contents in hood_paths_main.items():
    geo_imngs_path = os.path.join(outDir,geoname)
    if not os.path.exists(geo_imngs_path):
        os.makedirs(geo_imngs_path)
        print(f"Directory '{geo_imngs_path}' created.")
    else:
        print(f"Error: Directory '{geo_imngs_path}' already exists.")
        sys.exit(1)
    
    stlbase_path = os.path.join(dirPath, geoname)
    for stls in contents:
        output_filename = os.path.join(geo_imngs_path, f"{stls[:-4]}.png")
        tasks.append( ((os.path.join(stlbase_path, stls)), output_filename ))

#%% Worker Function
def process_task(task):
    fileF, output_filename = task
    try:
        # Read the stl from the files
        point_cloudV = pv.PolyData(fileF)
        
        p = pv.Plotter(shape=(1, 2), border=False, off_screen=True)
        p.subplot(0, 0)
        edges = point_cloudV.extract_feature_edges(
            boundary_edges=False,
            non_manifold_edges=False,
            feature_angle=15,
            manifold_edges=False,
        )
        p.add_mesh(point_cloudV,
                    color=True)
        p.enable_eye_dome_lighting()
        p.add_text("Inner side", font_size=24)
        p.camera_position = 'yx'
        
        # No eye-dome lighting
        p.subplot(0, 1)
        p.add_mesh(point_cloudV, color=True)
        p.add_mesh(edges, color='k', line_width=1)
        p.add_text("Outer side", font_size=24)
        p.camera_position = 'xy'
        p.camera.roll += 90
        
        p.screenshot(filename = output_filename,
                      scale = 1,
                      return_img=False,)
        p.close()
        time.sleep(1)
        #print(f"Saved merged file: {output_filename}")
    except Exception as e:
        print(f"Failed to process {fileF}. Error: {e}")

#%% Multiprocessing Execution
if __name__ == '__main__':
    # Create a pool of workers and process the tasks in parallel.
    with Pool(processes=num_workers) as pool:
        pool.map(process_task, tasks)












    
    

