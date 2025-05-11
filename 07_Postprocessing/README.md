# Visualize Model Outputs

This directory contains Python scripts for post‑training visualization, producing comparative performance and field‑output plots for each model.

## Contents

- `Viz_demo_pointNet.py`  
   Creates the single field plots for user defined model. This demo is applicable for point-based models.
 
- `Viz_demo_GraphSAGE.py`  
  Creates the multiple plots for different fields based on user defined model. This demo is applicable for graph-based models.
  
  Note - these scripts are provided for demonstration, and should be customized based on specific use cases.

## Dependencies

- Python 3.8+ 
- scikit-learn  
- pyMadCAD  
- PyVista (or your preferred STL‐rendering library)  
- Multiprocessing (built‑in)  
- Any JSON utilities (e.g. built‑in `json` module)

Install required packages:
packages from ML Framework package.list


---

##  License

CC BY-NC 4.0 License. See the `LICENSE` file for details.

---



