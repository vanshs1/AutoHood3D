# LLM Prompt Generation for Hood Geometry

This directory provides a set of Python scripts and associated data for generating, organizing, and consolidating natural‑language prompts paired with 3D automotive‑hood models. These prompts can be used for supervised fine‑tuning of vision‑language models in CAD‑driven generative‑AI workflows.

## Contents

- `01_Images_STL_multiproc_v1.py`  
  Loads all STL files in `dataset/`, renders orthographic projections (or other view angles) in parallel, and writes the resulting images into `sample_data/`.

- `02_CreateDataset_Hood_FB_v1.py`  
  Parses rendered images and base parameters (e.g., curve count, spacing, symmetry) to generate initial per‑model JSON prompts.

- `03_Consolidate_all_prompts_v1.py`  
  Merges individual JSON prompt files into a single `Final_consolidated_prompts.json` ready for LLM fine‑tuning.

- `sample_data/`  
  Example STL files and parameter definitions for testing the pipeline.

- `output_baseSkins/`  
  Generated images from the base STL models (used for creating baseSkins_prompts.json).

- `baseSkins_prompts.json`  
  Intermediate prompt dataset used by script 02.

- `Final_consolidated_prompts.json`  
  Final, merged prompt dataset suitable for direct LLM ingestion.

## Dependencies
- Cuda v12.1.1
- Python 3.11.5
- PyVista (or your preferred STL‐rendering library)  
- Multiprocessing (built‑in)  
- Any JSON utilities (e.g. built‑in `json` module)

Install required packages:
Gemma 3 is supported starting from transformers 4.50.0.

```bash
pip install pyvista
pip install -U transformers
```
---

##  License

CC BY-NC 4.0 License. See the `LICENSE` file for details.

---



