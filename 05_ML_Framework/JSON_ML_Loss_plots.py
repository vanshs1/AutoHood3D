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
import glob
import json
import re
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# =============================================================================
# Configuration and file search
# =============================================================================

#### Parent folder containing multiple experiment folders.
PARENT_FOLDER = "PATH_TO_MODEL_FOLDER"
run_num = "run01"
PARENT_FOLDER = PARENT_FOLDER+run_num+"/"
# Recursively search for JSON files.
# Assumption: Each experiment folder (first level) contains a subfolder that holds JSON files.
json_files = glob.glob(os.path.join(PARENT_FOLDER, "*", "*", "model_run_*_log.json"), recursive=True)

# Group JSON files by the experiment folder (the first directory name relative to PARENT_FOLDER).
experiment_groups = defaultdict(list)
for file_path in json_files:
    rel_path = os.path.relpath(file_path, PARENT_FOLDER)
    experiment = rel_path.split(os.sep)[0]  # first folder as experiment label
    experiment_groups[experiment].append(file_path)

# =============================================================================
# Plot 1: Overall Train vs. Validation Loss for Each Experiment
# =============================================================================

plt.figure(figsize=(18, 8), dpi=450)
# Assign each experiment a unique color using a colormap.
experiments = sorted(experiment_groups.keys())
cmap = plt.get_cmap('tab20')  # Up to 10 distinct colors; use modulo if >10.
# %%
color_mapping = {exp: cmap(i % 20) for i, exp in enumerate(experiments)}

experiment_group_data_losses = {}
for experiment, files in sorted(experiment_groups.items()):
    # Sort JSON files by epoch, assuming filename pattern: model_run_<epoch>_log.json.
    files.sort(key=lambda f: int(re.search(r"model_run_(\d+)_log.json", os.path.basename(f)).group(1)))
    
    data_list = []
    for f in files:
        match = re.search(r"model_run_(\d+)_log.json", os.path.basename(f))
        if match:
            epoch = int(match.group(1))
        else:
            continue

        with open(f, "r") as jf:
            data = json.load(jf)
        
        current_lr = data.get("current_lr")
        avg_train_loss = data.get("Average train loss")
        val_loss = data.get("Val loss")
        data_list.append({
            "epoch": epoch,
            "avg_train_loss": avg_train_loss,
            "val_loss": val_loss,
            "current_lr": current_lr
        })
    
    if not data_list:
        continue

    # Sort by epoch.
    data_list.sort(key=lambda x: x["epoch"])
    epochs = np.array([d["epoch"] for d in data_list])
    avg_train_losses = np.array([d["avg_train_loss"] for d in data_list])
    val_losses = np.array([d["val_loss"] for d in data_list])
    lr_data = np.array([d["current_lr"] for d in data_list])
    
    experiment_group_data_losses[experiment] = {
        "epochs": epochs,
        "avg_train_loss": avg_train_losses,
        "val_loss": val_losses,
        "current_lr": lr_data
    }
    
    col = color_mapping[experiment]
    # Plot training loss (circle markers) and validation loss (square markers), using the same color.
    plt.plot(epochs, avg_train_losses, marker='o', linestyle='-', color=col, label=f"{experiment} Train")
    plt.plot(epochs, val_losses, marker='s', linestyle='--', color=col, label=f"{experiment} Val")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Train vs. Validation Loss for Different Experiments")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.grid(True, linestyle='--')
plt.tight_layout()
plt.savefig(f"{PARENT_FOLDER}{run_num}_Train_Val_Loss.png")
plt.show()

# =============================================================================
plt.figure(figsize=(18, 8), dpi=450)
for experiment, dat in sorted(experiment_group_data_losses.items()):
    col = color_mapping.get(experiment, 'black')
    plt.semilogy(dat["epochs"], dat["current_lr"], marker='o', linestyle='-', color=col, label=experiment)
plt.xlabel("Epoch")
plt.ylabel("Learning Rate")
plt.title("LR vs. Epochs for Different Experiments")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.grid(True, linestyle='--')
plt.tight_layout()
plt.savefig(f"{PARENT_FOLDER}{run_num}_dyn_LR.png")
plt.show()

# =============================================================================
# Process Grouped "Val loss per var" for each experiment
# =============================================================================

# For each experiment, we will extract epochs and compute the three groups:
#  - Group 1: average of the first three values (velocity)
#  - Group 2: the 4th value (pressure)
#  - Group 3: average of the 5th-7th values (displacement)
experiment_group_data = {}

for experiment, files in sorted(experiment_groups.items()):
    files.sort(key=lambda f: int(re.search(r"model_run_(\d+)_log.json", os.path.basename(f)).group(1)))
    
    data_list = []
    for f in files:
        match = re.search(r"model_run_(\d+)_log.json", os.path.basename(f))
        if match:
            epoch = int(match.group(1))
        else:
            continue

        with open(f, "r") as jf:
            data = json.load(jf)
        
        val_loss_per_var = data.get("Val loss per var", [])
        data_list.append({
            "epoch": epoch,
            "val_loss_per_var": val_loss_per_var
        })
    if not data_list:
        continue

    data_list.sort(key=lambda x: x["epoch"])
    epochs = np.array([d["epoch"] for d in data_list])
    group1, group2, group3 = [], [], []
    for d in data_list:
        losses = d["val_loss_per_var"]
        # Ensure we have enough values; otherwise assign NaN.
        if isinstance(losses, list) and len(losses) >= 7:
            group1.append(np.mean(losses[0:3]))
            group2.append(losses[3])
            group3.append(np.mean(losses[4:7]))
        else:
            group1.append(np.nan)
            group2.append(np.nan)
            group3.append(np.nan)
    experiment_group_data[experiment] = {
        "epochs": epochs,
        "group1": np.array(group1),
        "group2": np.array(group2),
        "group3": np.array(group3)
        }

# =============================================================================
# Plot 2: Group 1 (Avg of Vars 1-3) for Each Experiment
# =============================================================================
plt.figure(figsize=(18, 8), dpi=450)
for experiment, dat in sorted(experiment_group_data.items()):
    col = color_mapping.get(experiment, 'black')
    plt.plot(dat["epochs"], dat["group1"], marker='o', linestyle='-', color=col, label=experiment)
plt.xlabel("Epoch")
plt.ylabel("Group 1 Loss (Avg Vars 1-3)")
plt.title("Grouped Validation Loss per Var: Group 1 (Vars 1-3 Average)")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.grid(True, linestyle='--')
plt.tight_layout()
plt.savefig(f"{PARENT_FOLDER}{run_num}_Grouped_Val_Loss_Group_1.png")
plt.show()

# =============================================================================
# Plot 3: Group 2 (Var 4) for Each Experiment
# =============================================================================
plt.figure(figsize=(18, 8), dpi=450)
for experiment, dat in sorted(experiment_group_data.items()):
    col = color_mapping.get(experiment, 'black')
    plt.plot(dat["epochs"], dat["group2"], marker='s', linestyle='--', color=col, label=experiment)
plt.xlabel("Epoch")
plt.ylabel("Group 2 Loss (Var 4)")
plt.title("Grouped Validation Loss per Var: Group 2 (Var 4)")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.grid(True, linestyle='--')
plt.tight_layout()
plt.savefig(f"{PARENT_FOLDER}{run_num}_Grouped_Val_Loss_Group_2.png")
plt.show()

# =============================================================================
# Plot 4: Group 3 (Avg of Vars 5-7) for Each Experiment
# =============================================================================
plt.figure(figsize=(18, 8), dpi=450)
for experiment, dat in sorted(experiment_group_data.items()):
    col = color_mapping.get(experiment, 'black')
    plt.plot(dat["epochs"], dat["group3"], marker='^', linestyle='-.', color=col, label=experiment)
plt.xlabel("Epoch")
plt.ylabel("Group 3 Loss (Avg Vars 5-7)")
plt.title("Grouped Validation Loss per Var: Group 3 (Vars 5-7 Average)")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.grid(True, linestyle='--')
plt.tight_layout()
plt.savefig(f"{PARENT_FOLDER}{run_num}_Grouped_Val_Loss_Group_3.png")
plt.show()


