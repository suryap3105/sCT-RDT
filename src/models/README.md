# Neural Network Models (`src/models/`)

This directory contains the custom PyTorch implementations of the `sCT-RDT` (Scaled Continuous-Time Rotary Damped Transformer) architecture. The objective of this neural network is to classify highly irregular astrophysical time-series data using physics-informed attention mechanisms.

## Architecture and Components

The model is highly modular and broken into the following files:

### 1. `attention.py`
This is the heart of the architecture. It implements the `sCTRDT_Attention` module and the `ScaledContinuousRoPE` embedding layer.
- **`ScaledContinuousRoPE` ($K_{periodic}$)**: Replaces standard Positional Embeddings with continuous-time rotary embeddings. Instead of absolute tokens $(1, 2, 3)$, it uses normalized timestamps ($\tilde{\tau}_i$). A $d_k$-dimensional rotation matrix is computed using a fractional power base to allow the attention mechanism to capture Fourier spectral frequencies.
- **`sCTRDT_Attention`**: Implements the additive attention master equation: $S_{ij} = K_{periodic} + K_{decay} + K_{noise}$
  - The base scaled-dot product $S_{case}$ is computed down to $\frac{QK^T}{\sqrt{d_k}}$.
  - **$K_{decay}$ (LTDK)** parses the temporal absolute difference $|\tilde{\tau}_i - \tilde{\tau}_j|$ passing it through a softplus $\hat{\lambda}_h = \log(1 + \exp(\lambda_h))$ head-specific parameter.
  - **$K_{noise}$ (PEG)** parses the concatenated errors $[\epsilon_i \oplus \epsilon_j]$ using a linear layer gate $W_g$ followed by a $\gamma$-stabilized log-sigmoid function ($\log( \sigma(W_g [\epsilon_i \oplus \epsilon_j] + b_g) + \gamma )$) effectively suppressing attention to noisy observations.

### 2. `transformer.py`
Defines the `sCTRDT_EncoderBlock` a single transformer layer wrapping the `sCTRDT_Attention` block.
- Implements standard Pre-Layer Normalization (Pre-LN) for both the Attention and Feed-Forward Neural Network (FFN) sublayers.
- Houses a 2-layer FFN projecting from $d_{model}$ to $4 \times d_{model}$ using a `GELU` activation and `Dropout` regularization.

### 3. `sct_rdt.py`
The top-level wrapper defining `Full_sCTRDT_Model`.
- **Input Projection**: Projects the 1D astronomical flux to $d_{model}$ while adding a 6-dimensional discrete passed band embedding via `nn.Embedding(num_embeddings=6)`.
- **Deep Routing**: Routes the raw float tensors (timestamps, errors) alongside feature representations through exactly $N$ stacked `sCTRDT_EncoderBlock` layers.
- **Pooling strategy**: Implements **Masked Global Average Pooling** $\frac{\sum H_{L,i} \cdot m_i}{\sum m_i}$. This ensures zero-padded timestep entries do not degrade the sequence representation $Z$ before passing to the final classification head.
- **Classifier Head**: The pooled representation maps the sequence features linearly to `config['num_classes']` (the astronomical transient classes).
