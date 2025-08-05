# Light imports (fast to import, common utilities)
import os
from pathlib import Path
import glob
import re
import numpy as np
import pandas as pd
import h5py
import ast
import string
import zipfile
from tqdm.notebook import tqdm
from sklearn.linear_model import RidgeCV, Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import pearsonr

# Heavy imports deferred inside functions
def import_torch():
    import torch
    return torch

def import_librosa():
    import librosa
    return librosa

def import_cv2():
    import cv2
    return cv2

def import_moviepy():
    import moviepy
    from moviepy.editor import VideoFileClip
    return moviepy, VideoFileClip

def import_nibabel_and_nilearn():
    import nibabel as nib
    from nilearn import plotting
    from nilearn.maskers import NiftiLabelsMasker
    return nib, plotting, NiftiLabelsMasker

def import_transformers():
    from transformers.models.bert import BertTokenizer, BertModel
    return BertTokenizer, BertModel

def import_torchvision_and_pytorchvideo():
    from torchvision.transforms import Compose, Lambda, CenterCrop
    from torchvision.models.feature_extraction import create_feature_extractor
    from pytorchvideo.transforms import Normalize, UniformTemporalSubsample, ShortSideScale
    return Compose, Lambda, CenterCrop, create_feature_extractor, Normalize, UniformTemporalSubsample, ShortSideScale

def import_sklearn_pca():
    from sklearn.decomposition import PCA
    return PCA

def import_sklearn_scaler():
    from sklearn.preprocessing import StandardScaler
    return StandardScaler
