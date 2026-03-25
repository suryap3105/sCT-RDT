# Datasets (`data/`)

This directory serves as the massive storage locus for raw, unadulterated astrophysical time-series data acquired from external telescope surveys. The data is deliberately excluded from version control systems via `.gitignore` due to volumetric constraints.

## Supported Surveys

The `sCT-RDT` architecture relies on structured CSV outputs heavily featuring the schema: `[object_id, mjd, passband, flux, flux_err, target]`.
- **`plasticc/`**: Holds the *Photometric LSST Astronomical Time-Series Classification Challenge* dataset. Features wildly varying extragalactic transient events with massive noise footprints across 6 simulated telescope filter passbands.
- **`kepler/`**: Stores data from the Kepler Space Telescope. Characterized by continuous, evenly sampled (high-cadence), single-band white-light photometry primarily aiming at classifying eclipsing binaries and exoplanet transits.

## Mock Data Architecture

If these directories remain unpopulated, the pipeline (`AstroDataset`) detects the `FileNotFoundError` and gracefully falls back to dynamic runtime tensor generation. It mimics sparse cadence, stochastic gaps, and randomized error profiles mathematically, bypassing the storage constraints to allow immediate ML systems validation.
