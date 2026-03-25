# Utils (`src/utils/`)

This directory houses auxiliary training infrastructure. In astrophysics machine learning, the extreme rarity of critical events requires custom loss and metric functions. Standard cross-entropy and accuracy fail to evaluate transient detection architectures reliably.

## Implementations

### 1. `focal_loss.py`
The `FocalLoss` class tackles massive dataset imbalances. Standard star observations might have 1,000,000 positive samples, while a Type Ia Supernovae event might appear only 300 times.
- By defining loss as $(1 - p_t)^{\gamma} \log(p_t)$, easy classifications (background stars with $p_t > 0.9$) have their loss contribution heavily down-weighted. The network is violently incentivized to learn features distinguishing the extremely difficult, rare transient occurrences.
- Supports an `alpha` array parameter to provide hardcoded class distribution weight scalers.

### 2. `metrics.py`
Wrapper utilizing `sklearn.metrics`.
- Calculates generic **Accuracy** to estimate the majority class hit rate.
- Uses **Macro F1-Score**. Unlike micro F1-score which favors the majority background class, a macro F1 calculates F1 scores exclusively within respective classes individually and averages the scores. This treats a 100-sample Supernovae class with the identical weight as a 1,000,000-sample Star class, forcing absolute validation metrics to reflect multi-class discrimination evenly.
