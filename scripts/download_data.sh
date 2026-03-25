#!/bin/bash
# Download and Extract Kaggle Datasets for sCT-RDT

echo "Checking for Kaggle API..."
pip install -q kaggle

# Make the directories if they don't exist
mkdir -p data/plasticc data/kepler

echo "Downloading PLAsTiCC (2018) Dataset..."
kaggle competitions download -c PLAsTiCC-2018 -p data/plasticc
unzip -q data/plasticc/PLAsTiCC-2018.zip -d data/plasticc/
rm data/plasticc/PLAsTiCC-2018.zip

echo "Downloading Kepler Exoplanet Search Results Dataset..."
kaggle datasets download -d keplersmachines/kepler-labelled-time-series-data -p data/kepler
unzip -q data/kepler/kepler-labelled-time-series-data.zip -d data/kepler/
rm data/kepler/kepler-labelled-time-series-data.zip

echo "Data successfully downloaded and extracted into the 'data' directory."
