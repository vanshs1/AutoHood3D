# ML Framework

This directory contains scripts and configuration files for training and evaluating machine‑learning surrogate models on the AutoHood3D dataset. Two model families are supported—point‑cloud and graph‑based—and utilities for visualizing training losses.

---

## Directory Contents

- **models/**  
  Model architecture definitions.

- **package.list**  
  Pinfile listing required Python packages and versions.

- **ML_train_pointModels.py**  
  Training pipeline for point‑cloud architectures (e.g., MLP, PointNet).  
  - Loads point‑cloud inputs and target fields (U, p, D)  
  - Configures dataset splits, data loaders, and augmentation  
  - Trains models with Adam optimizer and ReduceLROnPlateau scheduler  
  - Outputs model weights and training logs to user defined folder

- **ML_train_graphModels.py**  
  Training pipeline for graph‑based architectures (e.g., GraphSAGE, Graph U‑Net, PointGNNConv).  
  - Constructs `torch_geometric.data.Data` graphs from mesh connectivity  
  - Follows analogous training loop with graph‑specific modules  
  - Outputs model weights and training logs to user defined folder

- **JSON_ML_Loss_plots.py**  
  Utility to parse training/validation loss JSON logs and generate comparative loss curves.  
  - Produces `.png` or `.pdf` figures summarizing convergence behavior  
  - Supports multiple model runs and custom legend labels

---

## Dependencies

- Cuda v12.1.1
- Python 3.11.5
- PyVista (or your preferred STL‐rendering library)  
- Multiprocessing (built‑in)  
- Any JSON utilities (e.g. built‑in `json` module)

Install dependencies from `package.list`

---

## Usage
- **Edit the training hyperparameters at the top of each file** 
```bash
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 ML_train_<model>.py >& output.log 2>&1
```

- **Plot training losses** 
```bash
python JSON_ML_Loss_plots.py
```
Trained models and loss plots will appear under your specified output directory.

---

##  License

CC BY-NC 4.0 License. See the `LICENSE` file for details.

---
