# Scripts (`scripts/`)

This directory is designated for automation pipelines dealing tightly with system shells, specifically regarding external dependency fulfillment, data acquisition, and automated hardware management scripts outside the Python neural network ecosystem.

## Scripts Overview

### `download_data.sh`
This bash script interacts with external platforms. It is intended to handle the acquisition, massive ZIP-file decompression, and data pipelining needed prior to utilizing PyTorch. Often integrating tools like curl, wget or kaggle CLI interfaces. This handles large storage orchestration before handing execution over to `src/data_engine/dataset.py`.
