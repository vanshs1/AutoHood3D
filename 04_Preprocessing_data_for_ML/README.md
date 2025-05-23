# Dataset Preparation Pipeline

This directory contains the scripts and metadata required to assemble and partition the AutoHood3D dataset for ML training.

## Contents

- **01_Consolidate_hood_Data.py**  
  Merges raw CFD and FEA outputs (pressure, velocity, displacement fields) with geometry metadata into unified per‑sample files.

- **02_Consolidate_hood_Data_to_STL.py**  
  Projects the merged field data onto high‑resolution STL meshes, producing STL files with per‑vertex attributes.

- **03a_Create_SplitsJSON.py**  
  Generates `Files_for_ML_TrainingSplits.json`, which lists filenames and assigns each sample to one of:  
  - `train`  
  - `validation`  
  - `test`  
  - `ood_test`

- **03b_Create_Dataset.py**  
  Reads `Files_for_ML_TrainingSplits.json` and produces sharded data packages for point‑cloud and graph‑based model inputs (PyTorch `.pt` files).

- **Files_for_ML_TrainingSplits.json**  
  Auto‑generated by `03a_Create_SplitsJSON.py`. Defines the split assignment for every sample.

## Note on Validation Set

After running 03b_Create_Dataset.py, please rename the validation shard files by adding the val_ prefix (e.g., val_batch_01.pt) and move them into the train/ directory for model training.

## Dependencies

- Python 3.8+  
- PyVista (or your preferred STL‐rendering library)  
- Multiprocessing (built‑in)  
- Any JSON utilities (e.g. built‑in `json` module)
- `pyMadCAD` (for STL handling)  
- `PyTorch Geometric` (for PyTorch dataset shards)

Install required packages:
Use packages similar to the ML model framework

---

##  License

CC BY-NC 4.0 License. See the `LICENSE` file for details.

---



