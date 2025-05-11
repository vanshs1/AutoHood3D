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
    - Harish Jai Ganesh at harishjg@umich.edu
    - Venkat Raman at ramanvr@umich.edu

Affiliation: 
    - APCL Group 
    - Dept. of Aerospace Engg., University of Michigan, Ann Arbor
"""

import os
import shutil
import math
import sys

def split_items_into_files(source_dir, output_dirs, num_files):
    items = sorted(os.listdir(source_dir))
    num_items = len(items)
    items_per_file = math.ceil(num_items / num_files)
    
    for i, output_dir in enumerate(output_dirs):
        output_file = os.path.join(output_dir, f"items_{i+1}.txt")
        if os.path.exists(output_file):
            os.remove(output_file)
        
        subset = items[i * items_per_file : min((i + 1) * items_per_file, num_items)]
        
        with open(output_file, 'w') as f:
            f.writelines(f"{item}\n" for item in subset)

def manage_run_dirs(base_dir, run_dirs, num_pools):
    existing_dirs = sorted(
    [d for d in os.listdir(os.path.dirname(run_dirs[0])) if d.startswith("run_dir_")],
    key=lambda x: int(x.split('_')[-1])
    )
    existing_count = len(existing_dirs)
    
    if existing_count > num_pools:
        remove_dirs = existing_dirs[num_pools:]
        for extra_dir in remove_dirs:
            print(f"Removing: {extra_dir}")
            shutil.rmtree(os.path.join(os.path.dirname(run_dirs[0]), extra_dir))

    
    # Ensure all necessary directories exist
    for i in range(num_pools):
        run_dir = run_dirs[i]
        if not os.path.exists(run_dir):
            print(f"Copying: {run_dir}")
            shutil.copytree(base_dir, run_dir)

        items_file = os.path.join(run_dir, f"items_{i+1}.txt")
        if os.path.exists(items_file):
            os.remove(items_file)

if __name__ == "__main__":
    hood_dir = '/path/to/stl/directory'
    solution_dir = '/path/to/solution/directory'
    base_dir = '/path/to/template/case'
    num_pools = int(sys.argv[1])
    
    run_dirs = [f"/path/to/run_dir_{i+1}" for i in range(num_pools)]
    
    manage_run_dirs(base_dir, run_dirs, num_pools)
    split_items_into_files(hood_dir, run_dirs, num_pools)

