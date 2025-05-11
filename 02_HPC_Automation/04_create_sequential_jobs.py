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
import sys

def read_file_to_dict(file_path):
    data_dict = {}

    with open(file_path, mode='r') as file:
        for line in file:
            key, value = line.strip().split(',')
            data_dict[key] = int(value)

    return data_dict

def create_and_modify_slurm_files(template_file, directory, pools_list, num_files, num_procs):
    # Check if the template file exists
    if not os.path.isfile(template_file):
        print(f"Error: Template file {template_file} does not exist.")
        return

    if not os.path.isfile(pools_list):
        print(f"Error: Pools list file does not exist.")
        return
    
    pools_dict = read_file_to_dict(pools_list)

    # Create new SLURM files and modify them
    i = 1
    for node, pool in pools_dict.items():
        for k in range(pool):
            slurm_file = os.path.join(directory, f"srun{i}.sh")

            shutil.copy(template_file, slurm_file)

            with open(slurm_file, 'r') as file:
                lines = file.readlines()

            for idx, line in enumerate(lines):
                if line.startswith("#SBATCH --job-name="):
                    lines[idx] = f"#SBATCH --job-name=srun{i}\n"
                elif line.strip().startswith("python3"):
                    script_name = f"python3 /scratch/ramanvr_root/ramanvr/harishjg/parallel_run/srun{i}.py"
                    lines[idx] = f"{script_name}\n"
                elif line.startswith("#SBATCH --ntasks-per-node="):
                    lines[idx] = f"#SBATCH --ntasks-per-node={num_procs}\n"
                elif line.startswith("#SBATCH --nodelist="):
                    lines[idx] = f"#SBATCH --nodelist={node}\n"

            with open(slurm_file, 'w') as file:
                file.writelines(lines)
            i = i + 1

template_file = "/path/to/sequential_job.sh"
directory = "/path/to/output/directory"
pools_list = "/path/to/nodes_pools_list.txt"
num_pools = int(sys.argv[1])
num_procs = int(sys.argv[2])
create_and_modify_slurm_files(template_file, directory, pools_list, num_pools, num_procs)

