import os
from imports import np, h5py, import_sklearn_pca, import_sklearn_scaler
from load_files import root_data_dir

StandardScaler = import_sklearn_scaler()
PCA = import_sklearn_pca()

def load_features(root_data_dir, modality):
    """
    Load the extracted features from the .npy file.

    Parameters
    ----------
    root_data_dir : str
        Root data directory.
    modality : str
        The modality of the features ('visual', 'audio', or 'language').

    Returns
    -------
    features : float
        Stimulus features.
    """
    # Path to .npy features saved by visual_feature_extraction.py
    npy_path = os.path.join(
    	os.path.dirname(os.path.abspath(__file__)),  # base = /scripts
    	'../results/visual_features',
    	f'friends_s01e01a_features_{modality}.npy'
    )
    npy_path = os.path.normpath(npy_path)

    print(f"Loading features from: {npy_path}")
    features = np.load(npy_path)
    print(f"{modality} features original shape: {features.shape}")
    print('(Movie samples × Features)')
    return features



"""
def load_features(root_data_dir, modality):
    
    Load the extracted features from the HDF5 file.

    Parameters
    ----------
    root_data_dir : str
        Root data directory.
    modality : str
        The modality of the features ('visual', 'audio', or 'language').

    Returns
    -------
    features : float
        Stimulus features.


    ### Get the stimulus features file directory ###
    data_dir = os.path.join(root_data_dir, 'stimulus_features', 'raw', modality,
        'friends_s01e01a_features_'+modality+'.h5')

    ### Load the stimulus features ###
    with h5py.File(data_dir, 'r') as data:
        for episode in data.keys():
            if modality != 'language':
                features = np.asarray(data[episode][modality])
            else:
                # Vectorize and append pooler_output and last_hidden_state
                # language features
                pooler_output = np.asarray(
                    data[episode][modality+'_pooler_output'])
                last_hidden = np.asarray(np.reshape(
                    data[episode][modality+'_last_hidden_state'],
                    (len(pooler_output), -1)))
                features = np.append(pooler_output, last_hidden, axis=1)
    print(f"{modality} features original shape: {features.shape}")
    print('(Movie samples × Features)')

    ### Output ###
    return features
"""


def preprocess_features(features):
    """
    Rplaces NaN values in the stimulus features with zeros, and z-score the
    features.

    Parameters
    ----------
    features : float
        Stimulus features.

    Returns
    -------
    prepr_features : float
        Preprocessed stimulus features.

    """

    ### Convert NaN values to zeros ###
    features = np.nan_to_num(features)

    ### Z-score the features ###
    scaler = StandardScaler()
    prepr_features = scaler.fit_transform(features)

    ### Output ###
    return prepr_features

def perform_pca(prepr_features, n_components):
    """
    Perform PCA on the standardized features.

    Parameters
    ----------
    prepr_features : float
        Preprocessed stimulus features.
    n_components : int
        Number of components to keep

    Returns
    -------
    features_pca : float
        PCA-downsampled stimulus features.

    """

    ### Set the number of principal components to keep ###
    # If number of PCs is larger than the number of features, set the PC number
    # to the number of features
    if n_components > prepr_features.shape[1]:
        n_components = prepr_features.shape[1]

    ### Perform PCA ###n_init=4, max_iter=300
    pca = PCA(n_components, random_state=20200220)
    features_pca = pca.fit_transform(prepr_features)
    print(f"\n{modality} features PCA shape: {features_pca.shape}")
    print('(Movie samples × Principal components)')

    ### Output ###
    return features_pca


# Choose modality and PCs
modality = "visual"
n_components = 250

# Load the stimulus features
features = load_features(root_data_dir, modality)

# Preprocess the stimulus features
prepr_features = preprocess_features(features)

# Perform PCA
features_pca = perform_pca(prepr_features, n_components)

# Save PCA-reduced features as .npy
pca_out_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '../results/visual_features'
)
os.makedirs(pca_out_dir, exist_ok=True)

pca_out_path = os.path.join(pca_out_dir, f'friends_s01e01a_features_{modality}_pca{n_components}.npy')
np.save(pca_out_path, features_pca)
print(f"Saved PCA-reduced features to {pca_out_path}")
