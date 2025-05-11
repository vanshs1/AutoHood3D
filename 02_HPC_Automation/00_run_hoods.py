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

import subprocess
import os
import re
import glob

# === User Settings ===
MAX_POOLS = 30
PROCS_PER_JOB = 12
MAX_CORES_PER_NODE = 80
ALL_NODES = [f"lh{str(i).zfill(4)}" for i in range(900, 912)]

# === Paths ===
CREATE_RUNS_PY = "/path/to/create_runs.py"
CREATE_RUN_SCRIPTS_PY = "/path/to/create_run_scripts.py"
CREATE_SEQUENTIAL_JOBS_PY = "/path/to/create_sequential_jobs.py"
RUNALL_PY = "/path/to/runall.py"
CLEANUP_PATTERNS = [
    "/path/to/srun*",
    "/path/to/slurm*",
    "/path/to/srun*"
]
NODE_POOL_LIST_PATH = "/path/to/nodes_pools_list.txt"

# === Step 1: Get Available Nodes ===
def expand_nodelist(nodelist):
    if '[' not in nodelist:
        return [nodelist]
    match = re.match(r'([^\[]+)\[(.+)\]', nodelist)
    if not match:
        return [nodelist]
    prefix, ranges = match.groups()
    nodes = []
    for part in ranges.split(','):
        if '-' in part:
            start, end = part.split('-')
            width = len(start)
            for i in range(int(start), int(end)+1):
                nodes.append(prefix + str(i).zfill(width))
        else:
            nodes.append(prefix + part)
    return nodes

def get_available_nodes():
    candidates = set(ALL_NODES)

    result = subprocess.run(['sinfo', '--noheader', '--format=%P %t %N'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    if result.returncode != 0:
        raise RuntimeError(f"sinfo failed:\n{result.stderr}")

    sinfo_output = result.stdout
    valid_states = {'mix', 'idle'}
    lines = sinfo_output.strip().split('\n')

    available_nodes = set()
    for line in lines:
        parts = line.split()
        if len(parts) < 3:
            continue
        partition, state, nodelist = parts[0], parts[1], parts[2]
        if state.lower() in valid_states:
            expanded_nodes = expand_nodelist(nodelist)
            available_nodes.update(expanded_nodes)

    rejected_sinfo = candidates - available_nodes
    candidates &= available_nodes

    sstate_result = subprocess.run(['sstate'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    if sstate_result.returncode != 0:
        raise RuntimeError(f"sstate failed:\n{sstate_result.stderr}")

    sstate_output = sstate_result.stdout
    lines = sstate_output.strip().split('\n')

    final_nodes = {}
    rejected_sstate = set()
    node_alloccpu = {}
    for line in lines:
        if line.startswith('Node') or line.startswith('------'):
            continue
        parts = line.split()
        if len(parts) < 15:
            continue
        node = parts[0]
        try:
            alloc_cpu = int(parts[1])
        except ValueError:
            continue

        if node in candidates:
            if alloc_cpu < MAX_CORES_PER_NODE:
                node_alloccpu[node] = alloc_cpu
            else:
                rejected_sstate.add(node)

    print("Nodes rejected by sinfo (not MIX/IDLE):", sorted(rejected_sinfo))
    print("Nodes rejected by sstate (AllocCPU >= 80):", sorted(rejected_sstate))

    return node_alloccpu

# === Step 2: Calculate Pools per Node ===
def calculate_pools(nodes):
    node_pool_mapping = {}
    total_pools = 0
    rejected_no_space = set()
    for node, alloc_cpu in nodes.items():
        available_cores = MAX_CORES_PER_NODE - alloc_cpu
        pools_for_node = available_cores // PROCS_PER_JOB
        if pools_for_node > 0 and total_pools < MAX_POOLS:
            pools_to_allocate = min(pools_for_node, MAX_POOLS - total_pools)
            node_pool_mapping[node] = pools_to_allocate
            total_pools += pools_to_allocate
        else:
            rejected_no_space.add(node)
    return node_pool_mapping, total_pools, rejected_no_space

# === Step 3: Write Node-Pool Mapping to File ===
def write_node_pool_list(mapping):
    with open(NODE_POOL_LIST_PATH, 'w') as f:
        for node, pools in mapping.items():
            f.write(f"{node}, {pools}\n")

# === Step 4: Clean Up Old Files ===
def clean_old_files():
    if os.path.exists(NODE_POOL_LIST_PATH):
        os.remove(NODE_POOL_LIST_PATH)
    for pattern in CLEANUP_PATTERNS:
        for file in glob.glob(pattern):
            os.remove(file)

# === Step 5: Run Python Scripts ===
def run_python_scripts(num_pools):
    subprocess.run(['python3', CREATE_RUNS_PY, str(num_pools)], check=True)
    subprocess.run(['python3', CREATE_RUN_SCRIPTS_PY, str(num_pools), str(PROCS_PER_JOB)], check=True)
    subprocess.run(['python3', CREATE_SEQUENTIAL_JOBS_PY, str(num_pools), str(PROCS_PER_JOB)], check=True)
    runall_dir = os.path.dirname(RUNALL_PY)
    subprocess.run(['python3', RUNALL_PY], check=True, cwd=runall_dir)

# === Main Execution ===
if __name__ == "__main__":
    nodes = get_available_nodes()
    if not nodes:
        raise RuntimeError("No usable nodes available.")

    node_pool_mapping, usable_pools, rejected_no_space = calculate_pools(nodes)

    pools_accum = 0
    trimmed_mapping = {}
    for node, pools in sorted(node_pool_mapping.items()):
        if pools_accum >= usable_pools:
            break
        pools_to_use = min(pools, usable_pools - pools_accum)
        if pools_to_use > 0:
            trimmed_mapping[node] = pools_to_use
            pools_accum += pools_to_use

    print("Nodes rejected because no pools fit (not enough cores):", sorted(rejected_no_space))
    print("Selected nodes and pools:")
    for node, pools in trimmed_mapping.items():
        print(f"  {node}: {pools} pools")

    clean_old_files()
    write_node_pool_list(trimmed_mapping)
    run_python_scripts(pools_accum)

    print(f"Setup complete with {pools_accum} pools.")
