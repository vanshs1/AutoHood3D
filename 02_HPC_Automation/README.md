# HPC FSI Automation Scripts

This directory contains six scripts to automate case setup, job‐script generation, and SLURM submission for distributed OpenFOAM/preCICE co‑simulations. 
These scripts are generalized and can be extended to different applications workflows beyond the current study.

## Scripts

0. **00_runhoods.py**  
   - Orchestrator: discovers idle HPC nodes (via `sinfo`/`sstate`), computes pool distribution  
   - Cleans previous SLURM scripts, run scripts, and logs  
   - Launches `create_runs.py` → `create_run_scripts.py` → `create_sequential_jobs.py` → `runall.py`  

1. **01_create_runs.py**  
   - Inputs: number of pools, path to hood STL directory  
   - Creates (or prunes) `run_<pool_id>` directories from `baseCase_demo/`  
   - Generates `hood_list.txt` for each pool, assigning STL files evenly  

2. **02_create_run_scripts.py**  
   - Inputs: number of pools, procs per pool  
   - Clones `run_cases.py` into each `run_<pool_id>` directory  
   - Edits CPU counts and file paths in each script to match its pool  
  
3. **03_run_cases.py**  
   - Per‑pool execution script  
   - Reads `hood_list.txt`, updates OpenFOAM case (STL placement, `locationInMesh`)  
   - Runs `surfaceFeatureExtract`, `blockMesh`, `snappyHexMesh`, `decomposePar`  
   - Launches `mpirun UM_pimpleFoam` & `UM_solidDisplacementFoam`  
   - Moves results to `solutions/` and iterates over assigned hoods  

4. **04_create_sequential_jobs.py**  
   - Inputs: number of pools, procs per pool, `node_assignments.txt`  
   - Clones and configures SLURM submission scripts (`.sh`) for each pool  
   - Updates `#SBATCH` directives and module loads, pointing to its `run_cases.py`  

5. **05_runall.py**  
   - Submits all generated SLURM scripts in the current directory via `sbatch`  

## Usage Workflow

1. Adjust parameters in `00_run_hoods.py` (max pools, CPUs per pool, paths).  
2. Execute `python 00_run_hoods.py`.  

The orchestrator will prepare cases, generate and submit jobs, and distribute work across available nodes.

## Dependencies

- Python 3.8+  
- Multiprocessing (built‑in)  
- Any JSON utilities (e.g. built‑in `json` module)

Install required packages:
OpenFoam, preCICE (with adapter)

---

##  License

CC BY-NC 4.0 License. See the `LICENSE` file for details.

---



