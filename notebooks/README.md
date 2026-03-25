# Jupyter Notebooks (`notebooks/`)

This directory acts as the research and visualization staging ground. While `src/` strictly handles the rigid machine learning architecture, `notebooks/` allows dynamic and visual interpretation of the inner workings of the model.

## Available Notebooks

### 1. `01_data_exploration.ipynb`
Used strictly prior to model execution. Heavily manipulates the CSV files in `data/` using pandas.
- Implements visual scatter plots of the raw light curves.
- Applies error-bar visualizations on top of photometric flux measurements over the Modified Julian Date (MJD).
- Visually establishes the problem of extreme uneven sampling in time-domain astronomy.

### 2. `02_attention_heatmaps.ipynb`
Introspects the trained `best_model.pth` artifact.
- Traces a single light curve through the custom `sCTRDT_Attention` block.
- Plots 2-dimensional Seaborn heatmaps extracting the softmax representation matrix $A$.
- Empirically verifies if the **PEG (Pairwise Error Gating)** kernel correctly suppresses attention on high-noise timestamp pairs and checks if **LTDK (Time-Decay)** visually isolates attention exclusively to contiguous explosive periods (such as across a 30-day Supernovae phase).
