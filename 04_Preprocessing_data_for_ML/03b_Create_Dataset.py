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
import os.path as osp
import numpy as np
import pyvista as pv
import torch
from torch_geometric.data import Data
from multiprocessing import Pool
from tqdm import tqdm
from functools import partial
import json
import torch_geometric.nn as nng

#%% Function

def process_file(file_path, set_edge_index=False, name_mod='MLP', radius=0.05, max_neighbors=16):
    """
    Process a single mesh file and return a PyTorch Geometric Data object.
    
    Args:
        file_path (str): Path to the mesh file.
        set_edge_index (bool): If True, compute the edge index using radius_graph.
        name_mod (str): Model name, which can affect whether to compute edge_index.
        radius (float): Radius for the neighborhood graph.
        max_neighbors (int): Maximum number of neighbors for the graph.
    """
    try:
        internal = pv.PolyData(file_path)
        vertices = np.array(internal.points)
        internal = internal.compute_normals(point_normals=True, cell_normals=False)
        normals = np.array(internal.point_data["Normals"])
        
        # Compute additional cell sizes and extract the signed distance function.
        internal = internal.compute_cell_sizes(length=False, volume=False)
        geom = -internal.point_data['implicit_distance'][:, None]  # Signed distance function

        # Concatenate attributes: vertices, signed distance, normals.
        attr = np.concatenate([vertices, geom, normals], axis=-1)
        target = np.concatenate([
            internal.point_data['hoodU'],
            internal.point_data['p'][:, None],
            internal.point_data['D']
        ], axis=-1)
        pos = vertices
        init = np.concatenate([pos, attr], axis=1)
        
        # Convert to torch tensors.
        pos = torch.tensor(pos, dtype=torch.float)
        x = torch.tensor(init, dtype=torch.float)
        y = torch.tensor(target, dtype=torch.float)
        
        # Compute edge_index if requested.
        if set_edge_index:
            if name_mod not in ['PointNet', 'MLP']:
                edge_index = nng.radius_graph(x=pos, r=float(radius), loop=True, max_num_neighbors=int(max_neighbors))
                data = Data(pos=pos, x=x, y=y, edge_index=edge_index)
            else:
                data = Data(pos=pos, x=x, y=y)
        else:
            data = Data(pos=pos, x=x, y=y)
        
        return data
    except Exception as e:
        print(f"Failed to process {file_path}. Error: {e}")
        return None

def create_dataset_for_split(file_list, main_dir, output_folder, batch_size, num_workers,
                             set_edge_index, name_mod, radius, max_neighbors, split_name):
    """
    Process the given file list (relative paths) for a single split, and save batches
    as soon as they are processed so as not to load all data in memory.
    """
    # Build full file paths.
    full_file_list = [osp.join(main_dir, rel_path) for rel_path in file_list]
    
    if not full_file_list:
        print(f"No files found for output folder {output_folder}")
        return

    os.makedirs(output_folder, exist_ok=True)
    print(f"Processing {len(full_file_list)} files for split output in: {output_folder}")

    process_file_partial = partial(process_file, set_edge_index=set_edge_index,
                                   name_mod=name_mod, radius=radius, max_neighbors=max_neighbors)
    
    batch_idx = 0
    current_batch = []
    # Use a multiprocessing pool and process files in a streaming manner.
    with Pool(processes=num_workers) as pool:
        for data in tqdm(pool.imap(process_file_partial, full_file_list), total=len(full_file_list)):
            if data is not None:
                current_batch.append(data)
            if len(current_batch) >= batch_size:
                batch_file = osp.join(output_folder, f"batch_{batch_idx:02d}.pt")
                torch.save(current_batch, batch_file)
                print(f"Saved batch {batch_idx} with {len(current_batch)} items to {batch_file}")
                batch_idx += 1
                current_batch = []
    # Save any remaining data.
    if current_batch:
        if split_name=='validation':
            batch_file = osp.join(output_folder, f"val_batch_{batch_idx:02d}.pt")
        else:
            batch_file = osp.join(output_folder, f"batch_{batch_idx:02d}.pt")
        torch.save(current_batch, batch_file)
        print(f"Saved final batch {batch_idx} with {len(current_batch)} items to {batch_file}")

def create_all_datasets_from_json(json_path, output_base_folder, batch_size=4, num_workers=4,
                                  set_edge_index=False, name_mod='MLP', radius=0.05, max_neighbors=16):
    """
    Read the JSON file with file splits and create datasets for each split.
    Each split is saved in a separate subfolder within the output_base_folder.
    
    Args:
        json_path (str): Path to the JSON file containing file splits.
        output_base_folder (str): Base directory where subfolders for each split will be created.
        batch_size (int): Number of Data objects per batch.
        num_workers (int): Number of parallel worker processes.
        set_edge_index (bool): Whether to compute edge indices.
        name_mod (str): Model name to determine if edge indices should be computed.
        radius (float): Radius for the neighborhood graph.
        max_neighbors (int): Maximum number of neighbors for the graph.
    """
    with open(json_path, "r") as f:
        file_splits = json.load(f)
    
    main_dir = file_splits.get("main_dir", "")
    splits = file_splits.get("splits", {})
    #splits = splits.get("ood_testing", [])
    
    if not splits:
        print("No file splits found in the JSON file.")
        return
    
    for split_name, file_list in splits.items():
        #if split_name =='ood_testing':
        split_output_folder = osp.join(output_base_folder, split_name)
        print(f"\nCreating dataset for split '{split_name}'...")
        create_dataset_for_split(file_list, main_dir, split_output_folder, batch_size,
                                 num_workers, set_edge_index, name_mod, radius, max_neighbors, split_name)
        
    
#%% Main
if __name__ == '__main__':
    # JSON file containing the splits.
    json_path = "Files_for_ML_TrainingSplits.json"
    
    # Base output folder for the batched datasets.
    output_base_folder = "PATH_TO_OUTPUT_FOLDER"
    
    # This script is ran twice - once for point based models and 
    # next for graph based models. This way separate dataset shards are made for 
    # different architectures. To create a dataset for graph models - use set_edge_index=True
    # Set parameters below.
    batch_size = 128
    num_workers = 90
    set_edge_index = True
    name_mod = 'GraphSAGE'
    radius = 0.05
    max_neighbors = 4
    
    create_all_datasets_from_json(json_path, output_base_folder, batch_size=batch_size,
                                  num_workers=num_workers, set_edge_index=set_edge_index,
                                  name_mod=name_mod, radius=radius, max_neighbors=max_neighbors)

















