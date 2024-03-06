# Optimized persistence homological scaffolds of hemodynamic networks covary with MEG theta-alpha aperiodic dynamics

## 1. Install requirements
```
pip install -r requirements.txt
```

## 2. Repo structure
- **data**: Contains numpy matrices used in the manuscript.
    - `regions_info.npy`: a (3, 360) array containing the names (row 0), functional networks (row 1), and myelin content (row 2) of 360 Glasser regions.
- **src**: Source code.
     - `fmri_pipeline.py`: functions to compute persistence homological scaffolds, persistence centrality vectors, and degree centrality vectors. Assume functional connectivity matrices are already computed.
     - `meg_pipeline.py`: function to perform source localization and IRASA decomposition on MEG data. Assume all MEG data has been downloaded from the HCP database.
     - `utils.py`: helper functions.
- **results**: Contains Jupyter notebooks (`.ipynb` files) documenting the experimental results and analysis.

