# HRF delay parameter
hrf_delay = 4
print(f"HRF Delay is set to {hrf_delay}")

# Define file paths and dataset name
root_data_dir = '/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/data'
movie_path = root_data_dir + "/stimuli/movies/friends/s1/friends_s01e01a.mkv"
transcript_path = root_data_dir + "/stimuli/transcripts/friends/s1/friends_s01e01a.tsv"
fmri_file_path = root_data_dir + "/fmri/sub-01/func/sub-01_task-friends_space-MNI152NLin2009cAsym_atlas-Schaefer18_parcel-1000Par7Net_desc-s123456_bold.h5"
atlas_path = root_data_dir + "/fmri/sub-01/atlas/sub-01_space-MNI152NLin2009cAsym_atlas-Schaefer18_parcel-1000Par7Net_desc-dseg_parcellation.nii.gz"

print(f"Movie file path:       {movie_path}")
print(f"Transcript file path:  {transcript_path}")
print(f"fMRI file path:        {fmri_file_path}")
