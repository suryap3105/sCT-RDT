# Source Code (`src/`)

This directory houses the core Python implementation of the `sCT-RDT` (Scaled Continuous-Time Rotary Damped Transformer) project. It is strictly organized into functional submodules corresponding to the main architectural boundaries of the machine learning pipeline.

## Directory Layout & Responsibilities

- **`models/`**: 
  The neural network architecture. Contains the custom `sCTRDT_Attention` mechanism, transformer encoder blocks, and the high-level full model wrapper that maps raw features to classification logits.
  
- **`data_engine/`**: 
  The data pipeline. Contains the custom PyTorch `Dataset` definition, synthetic data generation, robust feature normalization, and the synthetic occlusion simulation mechanisms used for rigorous benchmarking.

- **`utils/`**: 
  Training and evaluation components. Includes domain-specific loss functions (like `FocalLoss` to combat transient event class imbalance) and metric computations (Accuracy, Macro F1).

## Implementation Design Principles

The code inside `src/` adheres to the following principles:
1. **PyTorch Idiomacy**: Heavy reliance on vectorized operations, custom `nn.Module` classes, and standard `torch.utils.data` interfaces.
2. **Numerical Stability**: When dealing with continuous variables like time and float errors, equations utilize min-max clamping, softplus activations (to guarantee positivity for decay rates), and $\gamma$-epsilons within mathematical operations (like `torch.log`) to prevent NaNs during backpropagation.
3. **Modularity**: The attention mechanism (`attention.py`) is decoupled from the transformer block (`transformer.py`), which is decoupled from the downstream classification head (`sct_rdt.py`). This allows researchers to easily inject the `sCTRDT_Attention` module into other non-classification architectures.
