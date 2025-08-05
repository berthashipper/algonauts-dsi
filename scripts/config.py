import os
import torch
from pathlib import Path

# Base paths
class Paths:
    fMRI_BASE = "/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/data/fmri"
    STIMULI_BASE = "/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/data/stimuli"
    RESULTS = str(Path(__file__).parent.parent / "results")
    
    @staticmethod
    def fmri(subject):
        return f"{Paths.fMRI_BASE}/sub-{subject:02d}/func"
    
    @staticmethod
    def stimuli_friends(season):
        return f"{Paths.STIMULI_BASE}/movies/friends/s{season}"

# Device configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Constants
TR = 1.49  # Repetition time in seconds
HRF_DELAY = 3  # Hemodynamic response delay
