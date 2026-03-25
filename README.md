# sCT-RDT: Scaled Continuous-Time Rotary Damped Transformer

**A novel Physics-Informed Transformer for highly irregular astrophysical time-series.**

Welcome to the `sCT-RDT` (Scaled Continuous-Time Rotary Damped Transformer) repository. This project introduces a state-of-the-art transformer architecture designed to classify and analyze highly irregular, multi-band astrophysical time-series data (e.g., classifying Supernovae, observing variable stars). Time-domain astronomy data is notoriously difficult to model because observations are sporadically sampled, unevenly spaced in time, and plagued with varying observational errors. The sCT-RDT architecture tackles these issues head-on through custom attention mechanisms.

## Solving Real-World Astrophysical Challenges

The sCT-RDT transformer is built to address critical bottlenecks in modern time-domain astronomy and large-scale sky surveys (such as the upcoming Vera C. Rubin Observatory's LSST, which will generate terabytes of data nightly). It helps solve these real-world problems:

1. **Handling Severe Observation Gaps**: Ground-based telescopes suffer from diurnal cycles (daylight), lunar interference, weather disruptions, and seasonal observability limits. Traditional RNNs or standard Transformers struggle when data points are separated by months of silence. sCT-RDT's continuous-time embeddings natively ingest the exact time of observation, allowing the model to seamlessly bridge massive temporal gaps without resorting to inaccurate imputation or interpolation methods.
2. **Robustness to Heteroscedastic Noise**: Astronomical measurements are inherently noisy, and the error varies significantly between observations (e.g., due to atmospheric conditions). By explicitly incorporating flux uncertainty into the attention matrix via Pairwise Error Gating, the model learns to prioritize high signal-to-noise data and down-weight highly uncertain observations, preventing noise from corrupting the learned representations.
3. **Rapid Transient Discovery**: Events like Supernovae or Kilonovae are ephemeral. Classifying them quickly is critical to triggering follow-up observations from expensive space-based or spectroscopic telescopes. sCT-RDT is structured to identify the signatures of these transient events early in their light curves, acting as a highly accurate, automated early-warning filter.
4. **Cosmological Calibration**: Accurate classification of Type Ia Supernovae is fundamental to measuring the expansion rate of the universe. By providing a highly scalable, robust classifier capable of handling imperfect, real-world photometric data pipelines out-of-the-box, sCT-RDT serves as a vital tool for cosmologists working with massive, uncurated data releases.

## System Architecture

The overarching architecture of this project is organized into modular components. At the highest level, the software is divided into:
- **`src/models/`**: Contains the PyTorch implementation of the `sCT-RDT` neural network, including the custom attention mechanism and transformer blocks.
- **`src/data_engine/`**: Handles loading, standardizing, and simulating astronomical data. Features robust normalization, handling of unevenly spaced padding, and synthetic occlusion for benchmarking robustness.
- **`src/utils/`**: Implements training utilities like custom loss functions (Focal Loss for class imbalance) and metric calculations.
- **`scripts/`**: Shell scripts for data acquisition.
- **`train.py` & `evaluate.py`**: Top-level entry points for training the model and evaluating its occlusion resilience.

## Core Mathematical Pillars

The key innovation of sCT-RDT lies in the three additive kernels within the Attention matrix equation $S_{ij} = K_{periodic} + K_{decay} + K_{noise}$.

1. **scRoPE (Scaled Continuous RoPE) - $K_{periodic}$**:
   Traditional RoPE uses absolute token positions. In astronomy, timestamps ($t$) are continuous. We define $\tilde{\tau}_i$ as the min-max normalized time scaled by $L_{max}$. We apply rotary embeddings directly on $\tilde{\tau}_i$, allowing the attention mechanism to act as a Fourier spectral analyzer capturing periodic orbital/rotational frequencies of stars.
2. **LTDK (Light Curve Time-Decay Kernel) - $K_{decay}$**:
   Many transient events (like Supernovae) are highly localized in time. LTDK applies a learnable decay penalty based on the temporal distance $|\tilde{\tau}_i - \tilde{\tau}_j|$. The decay rate is bounded by a softplus function to ensure stability, penalizing attention between observations that are too far apart in time.
3. **PEG (Pairwise Error Gating) - $K_{noise}$**:
   Astrophysical observations include inherent flux errors ($\epsilon$). PEG calculates a pairwise error tensor and passes it through a learned linear gate with a $\gamma$-stabilized log-sigmoid activation. This allows the model to mathematically suppress attention weights between highly noisy observations over reliable ones.

## Training and Evaluation Pipeline

### 1. Data Processing and Handling Missing Data
The `AstroDataset` (in `src/data_engine/`) manages incoming variable-length series. It bounds the observations to `max_seq_len` and heavily utilizes `pad_mask`ing. If a CSV dataset isn't found, it gracefully degrades to generating synthetic harmonic and transient events, allowing immediate testing of the mathematical integrity of the transformer.

### 2. Training (`train.py`)
Uses `PyTorch` with the `AdamW` optimizer. Since astronomical datasets exhibit massive class imbalances (millions of background stars vs a few hundred Supernovae), we utilize a `FocalLoss` objective to force the model to focus on hard, rare transient events.

### 3. Masking and Evaluation (`evaluate.py`)
To prove the architecture's resilience to weather issues, daylight, or telescope failures, `evaluate.py` benchmarks the model against extreme conditions:
- **50% Random Masking**: Dropping 50% of the observations uniformly.
- **75% Block Masking**: Simulating a contiguous 3-month telescope shutdown where a massive chunk of the light curve is permanently missing.

## Getting Started

1. **Install dependencies**: `pip install -r requirements.txt`
2. **Download datasets**: Optionally run `bash scripts/download_data.sh` to acquire real PLAsTiCC/Kepler datasets to `data/`.
3. **Train the model**: `python train.py` (Will auto-generate mock data if datasets are missing).
4. **Run the Occlusion Benchmark**: `python evaluate.py` to see how the model handles severe data dropouts.

---
*For more granular details on the code implementations, please refer to the `README.md` files localized within each respective subdirectory (`src/models`, `src/data_engine`, etc.).*
