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
import pyvista as pv
from multiprocessing import Pool

#%% USER INPUT
local = os.getcwd()  ## Get local dir
os.chdir(local)      ## shift the work dir to local dir
# print('Current working directory: {}'.format(local))
    
dirPath = 'PATH_TO_SIMULAION_OUTPUT'
outdir = 'OUTPUT_FOLDER'

# Set the number of worker processes (max procs in the CPU Job)
num_workers = 50

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
hood_names = []
hood_names_pathFS = []  # each element will be a list of two paths (fluid and structure)
for hood_data_dir in dirnamesH:
    hoodPath = os.path.join(dirPath, hood_data_dir)
    # For each geo_ folder, look at its subdirectories
    for (dirpath, dirnames, filenames) in os.walk(hoodPath):
        hood_paths = []
        for subdir in dirnames:
            hood_names.append(subdir)
            hood_paths.append(os.path.join(hoodPath, subdir))
        hood_names_pathFS.append(hood_paths)
        break
    del filenames, dirpath, dirnames

# Build a list of tasks for multiprocessing.
# Each task will be a tuple:
# (fluid_file, structure_file, output_file)
tasks = []
hood_index = 0
for hood_paths in hood_names_pathFS:
    # Create output directory for the current geo folder if it doesn't exist
    outDir_data = os.path.join(outdir, dirnamesH[hood_index])
    if not os.path.exists(outDir_data):
        os.makedirs(outDir_data, exist_ok=True)
    
    # For each of the two subfolders, get the list of .vtp files
    namesF, namesS = [], []
    for folder in hood_paths:
        # Check folder name to decide if it is fluid or structure
        if 'fluid' in folder:
            for (dirpath, dirnames, filenames) in os.walk(folder):
                namesF = [os.path.join(folder, f) for f in filenames if f.endswith('.vtp')]
                break
        else:
            for (dirpath, dirnames, filenames) in os.walk(folder):
                namesS = [os.path.join(folder, f) for f in filenames if f.endswith('.vtp')]
                break

    namesF.sort()
    namesS.sort()
    
    # For each fluid file, find its matching structure file and create a task.
    for fileF in namesF:
        # Match based on a portion of the filename (adjust the slicing as needed)
        matching_files = [fs for fs in namesS if fileF[-6:] in fs[-10:]]
        if not matching_files:
            print(f"No matching structure file found for {fileF}")
            continue
        fileS = matching_files[0]
        # Build the output file name. Here we use part of fileF's name to generate a suffix.
        output_filename = os.path.join(outDir_data, f"{dirnamesH[hood_index]}{fileF[-7:-4]}.vtk")
        tasks.append((fileF, fileS, output_filename))
    hood_index += 1

#%% Worker Function
def process_task(task):
    fileF, fileS, output_filename = task
    try:
        # Read the meshes from the fluid and structure files
        meshF = pv.read(fileF)
        meshS = pv.read(fileS)
        
        # Remove the 'TimeValue' field data if present
        if 'TimeValue' in meshF.field_data:
            meshF.field_data.remove('TimeValue')
        if 'TimeValue' in meshS.field_data:
            meshS.field_data.remove('TimeValue')
        
        # Interpolate and process the mesh
        meshF = meshF.interpolate(meshS, sharpness=4, n_points=4)
        meshF = meshF.triangulate()
        meshF = meshF.compute_derivative(scalars="p")
        
        # Save the processed mesh to the output file
        meshF.save(output_filename)
        #print(f"Saved merged file: {output_filename}")
    except Exception as e:
        print(f"Failed to process {fileF} and {fileS}. Error: {e}")

#%% Multiprocessing Execution
if __name__ == '__main__':
    # Create a pool of workers and process the tasks in parallel.
    with Pool(processes=num_workers) as pool:
        pool.map(process_task, tasks)
