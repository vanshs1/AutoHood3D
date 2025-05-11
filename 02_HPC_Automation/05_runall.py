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
import subprocess

def submit_sbatch_jobs():
    # Get the list of all .sh files in the current directory
    sh_files = [f for f in os.listdir('.') if f.endswith('.sh')]

    if not sh_files:
        print("No .sh files found in the current directory.")
        return

    # Submit each .sh file using sbatch
    for sh_file in sh_files:
        try:
            subprocess.run(['sbatch', sh_file], check=True)
            print(f"Successfully submitted: {sh_file}")
        except subprocess.CalledProcessError as e:
            print(f"Failed to submit {sh_file}: {e}")

if __name__ == "__main__":
    submit_sbatch_jobs()

