# Data Engine (`src/data_engine/`)

This directory handles the extremely unevenly-sampled, incomplete, and noisy astrophysical light curve datasets (such as PLAsTiCC or Kepler). The core goal is to map continuous timestamps, heterogeneous measurement cadences, and discrete pass-bands into unified tensors suitable for transformer analysis.

## Core Modules and Implementation

### 1. `dataset.py`
Contains the `AstroDataset` class inheriting from `torch.utils.data.Dataset`.
- **CSV and Grouping Pipeline**: Designed to ingest massive relational tables (CSV formats) by grouping rows over an `object_id`. This directly structures varying-length sequence data.
- **Synthetic Fallback**: To allow code portability and immediate developer testing, the dataset degrades to generating mock continuous timestamps and harmonic signals mimicking astronomical observations.
- **Robust Normalization**: 
  - *Time ($t$)* is min-max normalized $\tilde{\tau}_i = \frac{t_i - min}{max - min + \epsilon}$.
  - *Flux & Errors ($f, \epsilon$)* are normalized using IQR (Inter-Quartile Range) and Median scaling rather than Standard Deviation processing ($Z$-norm). This handles extreme outliers typical in transient astronomy (such as catastrophic Supernovae explosions that shatter the mean).
- **Sequence Handling**: Variable length sequences are either heavily zero-padded to `max_seq_len` or aggressively truncated. Crucially, a binary boolean `padding_mask` is generated to block attention aggregation over these zero-pads inside the transformer model.

### 2. `masking.py`
Astronomical observations are plagued with systemic missing data issues, ranging from weather interruptions to multi-month telescope failures.
The occlusion module synthesizes these real-world disruptions for robustness benchmarking:
- **`50%_random`**: Mimics poor weather/day-night cycles uniformly dropping exactly 50% of continuous measurements.
- **`75%_block`**: Simulates catastrophic long-term failure (e.g., 3-month shutdown) dropping 75% of the adjacent temporally contiguous observations. The dataset class seamlessly pads back to correct tensor geometry even during this extreme simulation.
