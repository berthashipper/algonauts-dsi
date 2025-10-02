# Predicting Neural Responses to Multimodal Stimuli with Machine Learning Models

**See research symposium poster [here](algonauts-dsi/Bertha%20Shipper%20-%20DSI%20Symposium%20Poster.pdf).**

This repository contains the scripts for a machine learning project aimed at predicting human brain responses to naturalistic stimuli using fMRI data.
The project was part of the 2025 Algonauts Challenge, an international competition focused on linking brain activity with complex audiovisual inputs such as movies and TV shows.


For descriptions of the important scripts and the overall pipeline workflow, please see the [USAGE.md](./USAGE.md) file.

## Project Overview

We developed a multimodal machine learning pipeline that integrates features extracted from video, audio, and language data to model how different regions of the brain respond during natural viewing experiences.
Using pre-trained neural networks for feature extraction, dimensionality reduction techniques, and regularized regression models, the pipeline predicts voxelwise fMRI responses to unseen stimuli.

### Key goals:
- Extract meaningful features from visual, auditory, and linguistic modalities using neural networks.
- Reduce feature dimensionality to optimize model training and performance.
- Train and validate encoding models to predict brain activity across various regions.
- Tune model parameters and evaluate generalization on held-out movie clips.

## Technologies and Tools
- Python, PyTorch, scikit-learn, NumPy, nilearn  
- High-performance computing environment with Slurm job scheduling  
- Pretrained CNNs, transformers, and audio processing models  

---

## About This Project

This project was completed in the Computational Affective and Neuroscience Lab at the University of Chicago, under the mentorship of [Yuan Chang (YC) Leong](https://github.com/ycleong), [Monica Rosenberg](https://github.com/monicadrosenberg), and alongside collaborator [Olivia Sopala](https://github.com/buggy1135).
Our goal was to build machine learning models that predict brain responses to naturalistic experiences using fMRI data collected from participants watching movies and TV shows.

Our final model, which integrated all three sensory modalities and employed a longer context window for language features, ranked 27th out of over 200 teams worldwide in the [2025 Algonauts Challenge](https://algonautsproject.com/), a global competition on predicting brain responses from complex natural stimuli.
