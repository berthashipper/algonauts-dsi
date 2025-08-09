# Multimodal fMRI Encoding Model Pipeline

This repository contains the core scripts for extracting multimodal features, training encoding models, and evaluating brain response predictions using fMRI data and naturalistic stimuli.

> **Note:** This repo contains only the code. It does **not** include the fMRI data, raw stimuli (videos/audio), or processed PCA features.

## Script Overview & Typical Workflow

## Step 1: Load Data and Video/Transcript Utilities

The `load_files.py` file contains utility functions to load and inspect raw movie (.mkv) files and transcript (.tsv) files, including:

- Reading video properties (FPS, resolution, duration) using OpenCV and MoviePy
- Extracting audio information from video files
- Loading language transcripts as pandas DataFrames
- Splitting movies into fixed-length chunks for feature extraction
- Extracting video segments with audio for inspection or debugging

The script sets the root data directory for fMRI and stimuli files as:
`root_data_dir = '/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/data'`
`fmri_dir = root_data_dir + "/fmri"`

## Step 2: Extract Multimodal Features

Before performing PCA and encoding modeling, raw stimulus features must be extracted separately for each modality: visual, audio, and language. This step processes raw video/audio files into time-resolved feature arrays stored as `.npy` files.

### General Extraction Workflow (applies to all modalities):
- Identify raw stimulus files (e.g., `.mkv` videos) in the specified directory.
- Segment stimuli into fixed-length chunks (duration varies by modality and TR alignment).
- Extract features chunk-wise using pretrained models or feature extractors appropriate for each modality:
  - **Visual:** Often uses CNN embeddings or pretrained image/video models.
  - **Audio:** Uses models like HuBERT or Wav2Vec.
  - **Language:** Uses pretrained language models (e.g., RoBERTa, BERT).
- Save features chunk-wise as `.npy` arrays per episode, preserving temporal order.

### Audio Feature Extraction (Example with HuBERT)
- Uses the `scripts/hubert_audio_feature_extraction_batch.py` script.
- For each video episode (`.mkv`), audio is extracted in fixed-length chunks (e.g., 1.49 seconds).
- Each audio chunk is saved as a temporary `.wav` file, loaded, and zero-padded if shorter than expected.
- Features are extracted using a pretrained HuBERT transformer model (`facebook/hubert-large-ls960-ft`) via HuggingFace.
- Extracted features correspond to a selected hidden layer (e.g., layer 19) averaged over time frames.
- All chunk features for an episode are saved as a single `.npy` array.

### Running Audio Feature Extraction on HPC (Example)
A SLURM batch script `scripts/run_hubert_audio_features_array.sh` runs extraction jobs in parallel over folders:

```bash
sbatch scripts/run_hubert_audio_features_array.sh
```

## Step 3: Dimensionality Reduction with PCA

Scripts `pca_audio_batch.py`, `pca_language_batch.py`, and `pca_visual_batch.py` perform PCA on pre-extracted raw features (numpy .npy arrays) for audio, language, and visual modalities respectively.

### How PCA Works:

1. Recursively loads all `.npy` feature files from the specified input directories.
2. Concatenates features from all episodes into a single dataset and standardizes them (zero mean, unit variance).
3. Fits PCA on the standardized data to reduce dimensionality.  
   - The number of principal components to retain can be set via a parameter.  
   - This controls the dimensionality of the output features and balances compression with retained variance.
4. Transforms the original features into the reduced PCA space and saves both:  
   - The PCA model parameters (e.g., mean, components) for later use.  
   - The reduced-dimension features, saved back to disk while preserving the original episode and chunk structure for consistency.

### Running PCA on Visual Features (example):

To run PCA jobs on visual features in parallel on a High Performance Computing (HPC) cluster using SLURM, use the provided batch script:

```bash
sbatch scripts/run_pca_visual_features.sh
```

## Step 4: Prepare Encoding Model

The `scripts/prepare_encoding_model.py` script implements a full pipeline to train and evaluate ridge regression encoding models that predict fMRI voxel responses from PCA-reduced multimodal stimulus features (visual, audio, language).

### Key Functionalities

#### 1. Data Loading
- **PCA Features**: Loads `.npy` PCA feature files from modality directories
- **fMRI Data**: Loads preprocessed fMRI data from HDF5 files for specified subjects

#### 2. Feature-fMRI Alignment
- Aligns time series features with fMRI responses considering hemodynamic delay
- Supports multi-modal concatenation per timepoint

#### 3. Model Training
- Fits ridge regression with cross-validated alpha hyperparameters
- Predicts voxel responses from features

#### 4. Accuracy Computation & Visualization
- Computes Pearson correlations per voxel
- Plots results on glass brains (individual and group level)
- Provides network-specific accuracy breakdowns (Yeo-7 networks)
- Comparative visualizations across temporal contexts

#### 5. Additional Visualizations
- Schaefer-1000 parcellation views
- Network-specific glass brains (e.g., Somatomotor network)
- Grouped bar plots for network comparisons

### Command Line Interface

Supports two modes:
1. **Single subject**: Trains and evaluates model, saves accuracy plots
2. **Group mode**: Aggregates results across subjects, plots group statistics

### Usage Example

```bash
python scripts/prepare_encoding_model.py \
    --fmri_root "/path/to/fmri_data" \
    --subject "1,2,3" \
    --train_movies "friends-s01,friends-s02,movie10-bourne" \
    --val_movies "friends-s04,movie10-life" \
    --modality_paths "visual:/path/to/visual,audio:/path/to/audio,language:/path/to/language" \
    --version_name "v_final" \
    --output_model_dir "./models" \
    --group_plot \
    --plot_networks
```
