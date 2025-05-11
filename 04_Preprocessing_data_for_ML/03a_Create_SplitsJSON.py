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
import json
import re
import math
import numpy as np
from collections import defaultdict

#%% Functions

def parse_file_name(file_path):
    """
    Extract the geo and crvCount from the file name.
    Example file name (basename):
        SIM_geo_063_crvCount_3_0824_0731_1600_cd_0.022_md_0.035_08.vtk
    returns geo = "063" and crv_count = 3.
    """
    base = os.path.basename(file_path)
    match = re.match(r".*geo_(\d+)_clusterID_(\d+)_crvCount_(\d+).*", base)
    if match:
        geo = match.group(1)
        crv_count = int(match.group(3))
        return geo, crv_count
    return None, None

def gather_solution_files(folder_path, suffix="_08.vtk"):
    """
    Traverse the given folder (which contains subfolders) and return a list of relative file paths 
    (in the format <subfolder>/<file_name>) that end with the given suffix.
    """
    files = []
    for subfolder in os.listdir(folder_path):
        subfolder_path = os.path.join(folder_path, subfolder)
        if os.path.isdir(subfolder_path):
            for f in os.listdir(subfolder_path):
                if f.endswith(suffix):
                    relative_path = os.path.join(subfolder, f)
                    files.append(relative_path)
    return files

def stratified_sample(sorted_list, sample_count):
    """
    Given a sorted list and a desired sample_count, return a list of items chosen
    at evenly spaced indices.
    """
    if len(sorted_list) == 0 or sample_count <= 0:
        return []
    indices = np.linspace(0, len(sorted_list)-1, num=sample_count, dtype=int)
    return [sorted_list[i] for i in indices]

def organize_files(folder_path, output_json="file_splits.json", ood_geo="030", ood_pattern=None,
                   id_test_pct=0.1, train_pct=0.6, valid_pct=0.3):
    """
    Organize solution files into training, validation, and testing sets.
    
    Steps:
      1. Gather all solution files (files ending with "_08.vtk") from subfolders.
         Files are stored with their relative path (<subfolder>/<file>).
      2. Parse the geo and curve count from each file name.
      3. For each geo, sort files in ascending order by curve count.
      4. Remove files for OOD testing:
           - If the geo equals ood_geo, or 
           - If ood_pattern is provided and the geo starts with that pattern.
      5. For each remaining geo:
         - Reserve a percentage (id_test_pct) of the files for in-distribution testing,
           chosen via stratified sampling so that the selected files span the range of curve counts.
         - From the remaining files, select training files (train_pct) via stratified sampling,
           and assign the rest to validation (valid_pct).
    
    The percentages id_test_pct, train_pct, and valid_pct should sum to 1.0 for the non-OOD files.
    
    The resulting splits are saved into a JSON file with the following structure:
    
        {
            "main_dir": "<folder_path>",
            "splits": {
                "training": [ "<subfolder>/<file>", ... ],
                "validation": [ ... ],
                "id_testing": [ ... ],
                "ood_testing": [ ... ]
            }
        }
    """
    # Gather files from all subfolders.
    files = gather_solution_files(folder_path, suffix="_08.vtk")
    if not files:
        print("No solution files found in subfolders of", folder_path)
        return {}
    
    # Group files by (geo, crvCount)
    file_info = defaultdict(list)
    for file in files:
        geo, crv_count = parse_file_name(file)
        if geo is not None:
            file_info[(geo, crv_count)].append(file)
    
    # Within each (geo, crvCount) group, sort the files (if duplicates exist)
    for key in file_info:
        file_info[key].sort()

    # Group files by geo and then sort by curve count (ascending)
    geo_files = defaultdict(list)
    for (geo, crv_count), file_list in file_info.items():
        geo_files[geo].extend(file_list)
    for geo in geo_files:
        geo_files[geo].sort(key=lambda f: parse_file_name(f)[1])
    
    # Initialize split lists.
    training_files = []
    validation_files = []
    id_testing_files = []
    ood_testing_files = []

    # Process each geo group.
    for geo, file_list in geo_files.items():
        total_files = len(file_list)
        # Determine if this geo is OOD using either the exact match or pattern.
        if geo == ood_geo or (ood_pattern is not None and geo.startswith(ood_pattern)):
            ood_testing_files.extend(file_list)
        else:
            # Calculate number of files for in-distribution testing.
            test_count = max(1, math.ceil(id_test_pct * total_files))
            # Use stratified sampling over the sorted list to pick test samples.
            id_test_samples = stratified_sample(file_list, test_count)
            
            # Remove the selected test samples from the full list to get remaining files.
            remaining_files = [f for f in file_list if f not in id_test_samples]
            remaining_count = len(remaining_files)
            
            if remaining_count > 0:
                # For training, sample a fraction of the remaining files.
                train_count = max(1, math.ceil(train_pct * remaining_count))
                training_samples = stratified_sample(remaining_files, train_count)
                # The remaining (unsampled) files become the validation set.
                validation_samples = [f for f in remaining_files if f not in training_samples]
            else:
                training_samples = []
                validation_samples = []
            
            # Append the samples for this geo.
            id_testing_files.extend(id_test_samples)
            training_files.extend(training_samples)
            validation_files.extend(validation_samples)
    
    # Sort each split for consistency (sorting key: (geo, crvCount)).
    training_files.sort(key=lambda f: parse_file_name(f))
    validation_files.sort(key=lambda f: parse_file_name(f))
    id_testing_files.sort(key=lambda f: parse_file_name(f))
    ood_testing_files.sort(key=lambda f: parse_file_name(f))

    # Build final JSON data.
    file_splits = {
        "main_dir": folder_path,
        "splits": {
            "training": training_files,
            "validation": validation_files,
            "id_testing": id_testing_files,
            "ood_testing": ood_testing_files
        }
    }

    # Save the splits to a JSON file.
    with open(output_json, "w") as json_file:
        json.dump(file_splits, json_file, indent=4)
    print(f"File splits saved to {output_json}")

    return file_splits

def load_file_splits(json_path):
    """
    Load the file splits from a JSON file.
    """
    with open(json_path, "r") as json_file:
        file_splits = json.load(json_file)
    return file_splits

#%% Main
if __name__ == "__main__":
    # Set the folder that contains subfolders with solution files.
    folder_path = "PATH_TO_DATA_FROM_STEP2"
    output_json = "Files_for_ML_TrainingSplits.json"

    # Set the percentages.
    id_test_pct = 0.10  # % of the files per geo (except OOD) will be used for in-distribution testing.
    train_pct = 0.75    # % of the remaining files go to training.
    valid_pct = 0.15    # % of the remaining files go to validation.

    # For OOD, choose a mode: either a single geo (ood_geo) or a pattern (ood_pattern)
    ood_geo = "005"          # This will remove geo "030"
    ood_pattern = "005"       # This will remove all geos starting with "03" (e.g. "030", "031", ...)

    # Uncomment specifc selection mode:
    # Option 1: Single geo removal
    # file_splits = organize_files(folder_path, output_json, ood_geo=ood_geo,
    #                              id_test_pct=id_test_pct, train_pct=train_pct, valid_pct=valid_pct)
    
    # Option 2: Pattern removal (all geos starting with "03")
    file_splits = organize_files(folder_path, output_json, ood_geo="", ood_pattern=ood_pattern,
                                 id_test_pct=id_test_pct, train_pct=train_pct, valid_pct=valid_pct)

    # Load splits during runtime (for later use in training/testing).
    ## Note - we only use _08.vtk solution for this script - can be changed in the code
    loaded_splits = load_file_splits(output_json)
    print("Loaded file splits:", json.dumps(loaded_splits, indent=4))















