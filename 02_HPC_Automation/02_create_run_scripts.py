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

def create_and_modify_python_files(template_file, output_dir, num_files, num_procs):
    num_procs = int(num_procs/2)

    # Check if the template file exists
    if not os.path.isfile(template_file):
        print(f"Error: Template file {template_file} does not exist.")
        return

    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # Loop over the number of files to create and modify
    for i in range(1, num_files + 1):
        python_file = os.path.join(output_dir, f"srun{i}.py")

        shutil.copy(template_file, python_file)

        with open(python_file, 'r') as file:
            lines = file.readlines()

        for idx, line in enumerate(lines):
            if 'run_dir' in line and "/path/to/run_dir_1" in line:
                lines[idx] = line.split('=')[0] + "= '/path/to/run_dir_" + str(i) + "'\n"

            if 'list_file' in line and "os.path.join(run_dir, 'items_1.txt')" in line:
                lines[idx] = line.replace('items_1.txt', f'items_{i}.txt') + '\n'
            
            if 'num_procs =' in line:
                lines[idx] = line.split('=')[0] + "= " + str(num_procs) + "\n"
                        
        with open(python_file, 'w') as file:
            file.writelines(lines)

template_file = "/path/to/run_cases.py"
output_dir = "/path/to/output/directory"
num_pools = int(sys.argv[1])
num_procs = int(sys.argv[2])
create_and_modify_python_files(template_file, output_dir, num_pools, num_procs)

