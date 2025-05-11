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
import numpy as np
import pyvista as pv
import concurrent.futures

def normalize_arrays(polydata):
    """
    Normalize all scalar and vector arrays in a PyVista PolyData object to the range [-1, 1].
    """
    normalized_polydata = polydata.copy()

    # Normalize point data
    for array_name in polydata.point_data:
        data = polydata[array_name]
        if data.ndim == 1:  # Scalar array
            min_val = np.min(data)
            max_val = np.max(data)
            range_val = max_val - min_val
            if range_val > 0:
                normalized_data = 2 * (data - min_val) / range_val - 1
            else:
                # If all values are equal, set the normalized array to 1.
                normalized_data = np.ones_like(data)
            normalized_polydata[array_name] = normalized_data
        elif data.ndim == 2 and data.shape[1] == 3:  # Vector array
            min_vals = np.min(data, axis=0)
            max_vals = np.max(data, axis=0)
            range_vals = max_vals - min_vals
            # Avoid division by zero for any component by using np.where
            normalized_data = 2 * (data - min_vals) / np.where(range_vals==0, 1, range_vals) - 1
            normalized_polydata[array_name] = normalized_data

    # Normalize cell data (if any)
    for array_name in polydata.cell_data:
        data = polydata.cell_data[array_name]
        if data.ndim == 1:  # Scalar array
            min_val = np.min(data)
            max_val = np.max(data)
            range_val = max_val - min_val
            if range_val > 0:
                normalized_data = 2 * (data - min_val) / range_val - 1
            else:
                normalized_data = np.ones_like(data)
            normalized_polydata.cell_data[array_name] = normalized_data
        elif data.ndim == 2 and data.shape[1] == 3:  # Vector array
            min_vals = np.min(data, axis=0)
            max_vals = np.max(data, axis=0)
            range_vals = max_vals - min_vals
            normalized_data = 2 * (data - min_vals) / np.where(range_vals==0, 1, range_vals) - 1
            normalized_polydata.cell_data[array_name] = normalized_data

    return normalized_polydata

def arrange_points_in_circular_order(points, clockwise=True):
    """
    Arrange points (x, y, z) in clockwise or counterclockwise order around their centroid.
    """
    centroid = np.mean(points[:, :2], axis=0)
    angles = np.arctan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])
    sorted_indices = np.argsort(-angles) if clockwise else np.argsort(angles)
    return points[sorted_indices]

def process_folder(i, dirnamesVTK, dirpathsVTK, stlDir, outDir, filenamesOUT):
    log_of_errors = []
    dirVtk = dirnamesVTK[i]
    outDir_data = os.path.join(outDir, dirVtk)
    if not os.path.exists(outDir_data):
        os.makedirs(outDir_data, exist_ok=True)
        # Get VTK files in current folder
        for (dirpath, dirnamesH, filenames) in os.walk(dirpathsVTK[i]):
            filenamesVTK = [f for f in filenames if f.endswith('.vtk')]
            break
        filenamesVTK.sort()

        for fileVtk in filenamesVTK:
            if fileVtk not in filenamesOUT:
                try:
                    fileV = os.path.join(dirpathsVTK[i], fileVtk)
                    point_cloudV = pv.PolyData(fileV)
    
                    # Construct STL filename from VTK file name (assuming pattern fileVtk[:-7] + '.stl')
                    fileS = os.path.join(stlDir, fileVtk[:-7] + '.stl')
                    point_cloudS = pv.PolyData(fileS)
                    
                    point_cloudV = normalize_arrays(point_cloudV)
                    
                    try:
                        refined_stl = point_cloudS.subdivide_adaptive(max_n_passes=20, max_tri_area=0.00020, max_edge_len=0.032)
                        result = refined_stl.interpolate(point_cloudV, n_points=15)
                        
                        bounds = result.bounds
                        padding = 2.0
                        grid = pv.Plane()
                        grid.spacing = (0.01, 0.01, 0.01)
                        grid.dimensions = np.array([
                            int((bounds[1] - bounds[0] + 2 * padding) / grid.spacing[0]),
                            int((bounds[3] - bounds[2] + 2 * padding) / grid.spacing[1]),
                            int((bounds[5] - bounds[4] + 2 * padding) / grid.spacing[2]),
                        ]) + 1
                        grid.origin = (
                            bounds[0] - padding,
                            bounds[2] - padding,
                            bounds[4] - padding,
                        )
                        
                        sdf = result.compute_implicit_distance(grid, inplace=True)
                        out_filename = os.path.join(outDir_data, 'SIM_' + fileVtk[:-4] + '.vtk')
                        result.save(out_filename)
                    except Exception as e:
                        print(f'Failed {e} processing file:', fileVtk[:-4])
                        log_of_errors.append(fileVtk[:-4])
                except Exception as e:
                    print(f"Failed {e} loading:", fileVtk)
                    continue
    return log_of_errors

if __name__ == '__main__':
    #%% USER INPUT
    local = os.getcwd()
    os.chdir(local)
    print('Current working directory: {}'.format(local))
    
    vtkdir = 'PATH_TO_STEP1_DATA'
    stlDir = 'PATH_TO_HOOD_STLs'
    outDir = 'OUTPUT_FOLDER'
    proc_count = 50 ## set the number of worker pools

    # Read all main folders in the VTK directory
    for (dirpath, dirnamesH, filenames) in os.walk(vtkdir):
        dirpathsVTK = [os.path.join(vtkdir, f) for f in dirnamesH]
        dirnamesVTK = sorted(dirnamesH)
        break
    del dirnamesH, dirpath, filenames
    dirpathsVTK.sort()
    dirnamesVTK.sort()
    
    # Get STL file names from the STL directory
    for (dirpath, dirnamesH, filenames) in os.walk(stlDir):
        filenamesSTL = [f for f in filenames if f.endswith('.stl')]
        break
    del dirnamesH, dirpath, filenames
    filenamesSTL.sort()

    # Get file names in the OUT folder to check if the solution exists
    for (dirpath, dirnamesO, filenames) in os.walk(outDir):
        filenamesOUT = [f[4:] for f in filenames if f.endswith('.vtk')]
        break
    del dirnamesO, dirpath, filenames
    filenamesOUT.sort()


    # Parallelize processing over folders using ProcessPoolExecutor
    all_errors = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=proc_count) as executor:
        futures = [
            executor.submit(process_folder, i, dirnamesVTK, dirpathsVTK, stlDir, outDir, filenamesOUT)
            for i in range(len(dirnamesVTK))
        ]
        for future in concurrent.futures.as_completed(futures):
            folder_errors = future.result()
            all_errors.extend(folder_errors)
    
    print("Log of errors:", all_errors)
