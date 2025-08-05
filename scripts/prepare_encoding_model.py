import os
import numpy as np
import h5py
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV
from scipy.stats import pearsonr
from nilearn.maskers import NiftiLabelsMasker
from nilearn import plotting
import joblib

def load_pca_features(modality_path):
    """Load PCA features from specified directory, skipping param files"""
    features = {}
    if not os.path.exists(modality_path):
        raise FileNotFoundError(f"PCA modality directory not found: {modality_path}")

    for root, dirs, files in os.walk(modality_path):
        for fname in files:
            if not fname.endswith(".npy"):
                continue
            if fname in ["pca_param.npy", "scaler_param.npy"]:
                continue  # Skip PCA/scaler metadata files
            full_path = os.path.join(root, fname)
            key = os.path.relpath(full_path, modality_path)
            features[key] = np.load(full_path, allow_pickle=True)

    if not features:
        raise ValueError(f"No valid .npy feature files found in {modality_path}")
    
    print(f"Loaded {len(features)} feature files from {modality_path}")
    return features


def load_fmri(root_data_dir, subject):
    """Load fMRI responses for the given subject."""
    fmri = {}

    # Load Friends fMRI
    fmri_file = f'sub-0{subject}_task-friends_space-MNI152NLin2009cAsym_atlas-Schaefer18_parcel-1000Par7Net_desc-s123456_bold.h5'
    fmri_path = os.path.join(root_data_dir, f'sub-0{subject}', 'func', fmri_file)
    with h5py.File(fmri_path, 'r') as f:
        for key, val in f.items():
            fmri[str(key[13:])] = val[:].astype(np.float32)

    # Load Movie10 fMRI
    fmri_file = f'sub-0{subject}_task-movie10_space-MNI152NLin2009cAsym_atlas-Schaefer18_parcel-1000Par7Net_bold.h5'
    fmri_path = os.path.join(root_data_dir, f'sub-0{subject}', 'func', fmri_file)
    with h5py.File(fmri_path, 'r') as f:
        for key, val in f.items():
            fmri[str(key[13:])] = val[:].astype(np.float32)

    # Average repeats
    for movie_prefix in ['figures', 'life']:
        keys_all = list(fmri.keys())
        for s in range(12 if movie_prefix == 'figures' else 5):
            movie = f'{movie_prefix}{str(s+1).zfill(2)}'
            reps = [k for k in keys_all if movie in k]
            if len(reps) == 2:
                fmri[movie] = ((fmri[reps[0]] + fmri[reps[1]]) / 2).astype(np.float32)
                del fmri[reps[0]]
                del fmri[reps[1]]

    return fmri


def align_features_and_fmri_samples(features, fmri, excluded_samples_start, excluded_samples_end,
                                  hrf_delay, stimulus_window, movies):
    """Align features with fMRI samples"""
    aligned_features = []
    aligned_fmri = np.empty((0, 1000), dtype=np.float32)
    expected_length = None  # Track expected feature vector length for consistency

    for movie in movies:
        if movie.startswith('friends'):
            season_num = movie.split('-')[1][1:]
            season_prefix = f"s{season_num}e"
            movie_splits = [key for key in fmri if key.startswith(season_prefix)]
        elif movie.startswith('movie10'):
            movie_folder = movie.split('-')[1]
            movie_splits = [key for key in fmri if key.startswith(movie_folder)]
        else:
            movie_splits = [key for key in fmri if key.startswith(movie)]

        if not movie_splits:
            print(f"Warning: no fmri keys found for movie '{movie}'")
            continue

        for split in movie_splits:
            fmri_split = fmri[split][excluded_samples_start:-excluded_samples_end]
            aligned_fmri = np.append(aligned_fmri, fmri_split, axis=0)

            for s in range(len(fmri_split)):
                f_all = []
                for mod in features.keys():
                    matching_keys = [k for k in features[mod].keys() if split in k]
                    if not matching_keys:
                        raise KeyError(f"Could not match split '{split}' to any key in {mod} features")
                    full_key = matching_keys[0]

                    if mod in ['visual', 'audio']:
                        if s < (stimulus_window + hrf_delay):
                            idx_start = excluded_samples_start
                            idx_end = idx_start + stimulus_window
                        else:
                            idx_start = s + excluded_samples_start - hrf_delay - stimulus_window + 1
                            idx_end = idx_start + stimulus_window

                        feat_len = len(features[mod][full_key])
                        if idx_end > feat_len:
                            idx_end = feat_len
                            idx_start = idx_end - stimulus_window

                        f = features[mod][full_key][idx_start:idx_end].flatten()
                        f_all.append(f)

                    elif mod == 'language':
                        idx = excluded_samples_start if s < hrf_delay else s + excluded_samples_start - hrf_delay
                        lang_feat_len = len(features[mod][full_key])
                        f = features[mod][full_key][-1] if idx >= lang_feat_len - hrf_delay else features[mod][full_key][idx]
                        f_all.append(f.flatten())

                if f_all:
                    feature_vector = np.concatenate(f_all)
                    if expected_length is None:
                        expected_length = len(feature_vector)
                    if len(feature_vector) != expected_length:
                        print(f"\n[‼️ FEATURE LENGTH MISMATCH]")
                        print(f"  Split: {split} | Timepoint: {s}")
                        print(f"  Expected: {expected_length} | Got: {len(feature_vector)}")
                        print(f"  Modality component shapes: {[f.shape for f in f_all]}")
                        raise ValueError("Feature vector length mismatch detected.")
                    aligned_features.append(feature_vector)
                else:
                    aligned_features.append(np.array([]))

    return np.asarray(aligned_features, dtype=np.float32), aligned_fmri


def train_encoding(features_train, fmri_train):
    """Train ridge regression encoding model"""
    alphas = np.array([1e6, 1e5, 1e4, 1e3, 1e2, 10, 1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 1e-4, 1e-5])
    model = RidgeCV(alphas=alphas, cv=None, alpha_per_target=True)
    model.fit(features_train, fmri_train)
    return model

def compute_encoding_accuracy(fmri_val, fmri_val_pred, subject, modality, root_data_dir):
    """Compute and plot encoding accuracy"""
    try:
        encoding_accuracy = np.array([pearsonr(fmri_val[:, p], fmri_val_pred[:, p])[0] 
                                    for p in range(fmri_val.shape[1])], dtype=np.float32)
        mean_acc = np.round(np.mean(encoding_accuracy), 3)

        atlas_file = f'sub-0{subject}_space-MNI152NLin2009cAsym_atlas-Schaefer18_parcel-1000Par7Net_desc-dseg_parcellation.nii.gz'
        atlas_path = os.path.join(root_data_dir, f'sub-0{subject}', 'atlas', atlas_file)
        
        masker = NiftiLabelsMasker(labels_img=atlas_path)
        masker.fit()
        acc_nii = masker.inverse_transform(encoding_accuracy)

        display = plotting.plot_glass_brain(
            acc_nii, display_mode="lyrz", cmap='hot_r', colorbar=True,
            plot_abs=False, symmetric_cbar=False,
            vmin=-1e-5, vmax=0.7,
            title=f"Encoding accuracy, sub-0{subject}, modality: language, mean: {mean_acc}"
        )
        
        os.makedirs("output_plots", exist_ok=True)
        display.savefig(f"output_plots/encoding_accuracy_sub{subject}.png")
        plt.close()
        
    except Exception as e:
        print(f"Error plotting accuracy: {e}")
    
    print(f"MEAN_ACCURACY: {mean_acc}")

def compute_encoding_accuracy_group(accuracy_dict, root_data_dir, modality, output_dir="output_plots"):
    """
    Compute and plot mean encoding accuracy across multiple subjects.

    Args:
        accuracy_dict (dict): Keys=subject id, values=np.array encoding accuracies (voxels,)
        root_data_dir (str): Base path for atlas files per subject.
        modality (str): modality name for plot title.
        output_dir (str): Where to save combined plot.
    """
    import matplotlib.pyplot as plt
    from nilearn.maskers import NiftiLabelsMasker
    from nilearn import plotting
    import numpy as np
    import os

    # Print mean accuracy per subject
    print("Mean encoding accuracies per subject:")
    for subject, acc in accuracy_dict.items():
        mean_acc = np.round(np.mean(acc), 3)
        print(f"  Subject {subject}: {mean_acc}")

    # Stack accuracies: shape (subjects, voxels)
    all_acc = np.array(list(accuracy_dict.values()))
    mean_acc_all = np.round(np.mean(all_acc), 3)
    print(f"Mean encoding accuracy across all {len(accuracy_dict)} subjects: {mean_acc_all}")

    # Use atlas of first subject as reference (assuming all same space)
    first_sub = list(accuracy_dict.keys())[0]
    atlas_file = f'sub-0{first_sub}_space-MNI152NLin2009cAsym_atlas-Schaefer18_parcel-1000Par7Net_desc-dseg_parcellation.nii.gz'
    atlas_path = os.path.join(root_data_dir, f'sub-0{first_sub}', 'atlas', atlas_file)

    masker = NiftiLabelsMasker(labels_img=atlas_path)
    masker.fit()
    acc_nii = masker.inverse_transform(np.mean(all_acc, axis=0))

    os.makedirs(output_dir, exist_ok=True)

    display = plotting.plot_glass_brain(
        acc_nii, display_mode="lyrz", cmap='hot_r', colorbar=True,
        plot_abs=False, symmetric_cbar=False,
        vmin=-1e-5, vmax=0.7,
        title=f"Mean Encoding Accuracy, modality: {modality}, mean: {mean_acc_all}"
    )
    save_path = os.path.join(output_dir, "encoding_accuracy_mean_all_subjects.png")
    display.savefig(save_path)
    plt.close()
    print(f"Saved group mean accuracy plot: {save_path}")

def compute_encoding_accuracy_group_schaefer(accuracy_dict, root_data_dir, modality, output_dir="output_plots"):
    """
    Compute and plot mean encoding accuracy across multiple subjects with region labels.
    """
    import matplotlib.pyplot as plt
    from nilearn.maskers import NiftiLabelsMasker
    from nilearn import plotting
    from nilearn.datasets import fetch_atlas_schaefer_2018
    from collections import defaultdict
    import numpy as np
    import os
    from nilearn.plotting import find_xyz_cut_coords

    # Print mean accuracy per subject
    print("Mean encoding accuracies per subject:")
    for subject, acc in accuracy_dict.items():
        mean_acc = np.round(np.mean(acc), 3)
        print(f"  Subject {subject}: {mean_acc}")

    # Stack accuracies: shape (subjects, voxels)
    all_acc = np.array(list(accuracy_dict.values()))
    mean_acc_all = np.round(np.mean(all_acc), 3)
    print(f"Mean encoding accuracy across all {len(accuracy_dict)} subjects: {mean_acc_all}")

    # Use atlas of first subject as reference
    first_sub = list(accuracy_dict.keys())[0]
    atlas_file = f'sub-0{first_sub}_space-MNI152NLin2009cAsym_atlas-Schaefer18_parcel-1000Par7Net_desc-dseg_parcellation.nii.gz'
    atlas_path = os.path.join(root_data_dir, f'sub-0{first_sub}', 'atlas', atlas_file)

    # Load Schaefer atlas for region names
    schaefer = fetch_atlas_schaefer_2018(n_rois=1000, yeo_networks=7)
    schaefer_labels = schaefer['labels']
    
    masker = NiftiLabelsMasker(labels_img=atlas_path)
    masker.fit()
    acc_mean = np.mean(all_acc, axis=0)
    acc_nii = masker.inverse_transform(acc_mean)

    # ===========================================
    # NEW: Find and print the ROI with highest accuracy
    # ===========================================
    max_roi_idx = np.argmax(acc_mean)
    max_accuracy = acc_mean[max_roi_idx]
    max_roi_label = schaefer_labels[max_roi_idx + 1]  # +1 because labels start at 1
    
    # Parse the label to get network and region name
    label_parts = max_roi_label.split('_')
    network = label_parts[2]
    region = ' '.join(label_parts[3:-1])
    
    print("\n" + "="*60)
    print(f"ROI with highest encoding accuracy:")
    print(f"  Accuracy: {max_accuracy:.4f}")
    print(f"  Network: {network}")
    print(f"  Region: {region}")
    print(f"  Full label: {max_roi_label}")
    print("="*60 + "\n")

    os.makedirs(output_dir, exist_ok=True)

    # Rest of your existing plotting code...
    # Create glass brain plot with more informative display
    fig = plt.figure(figsize=(16, 8))
    
    # Plot glass brain
    display = plotting.plot_glass_brain(
        acc_nii, display_mode="ortho", cmap='hot_r', colorbar=True,
        plot_abs=False, symmetric_cbar=False, figure=fig,
        vmin=-1e-5, vmax=0.7,
        title=f"Mean Encoding Accuracy ({modality})\nMean across {len(accuracy_dict)} subjects: {mean_acc_all:.3f}"
    )
    
    # Find top N most activated regions to label
    top_n = 15  # Number of regions to label
    threshold = np.percentile(acc_mean, 95)  # Only label regions above 95th percentile
    
    # Get coordinates and names for top regions
    for roi_index in np.argsort(acc_mean)[-top_n:][::-1]:
        if acc_mean[roi_index] < threshold:
            continue
            
        # Get region coordinates
        roi_mask = (masker.transform(atlas_path) == roi_index + 1)  # +1 because labels start at 1
        roi_img = masker.inverse_transform(roi_mask.astype(float))
        coords = find_xyz_cut_coords(roi_img)
        
        # Get region name and network
        label_parts = schaefer_labels[roi_index + 1].split('_')
        network = label_parts[2]
        region = ' '.join(label_parts[3:-1])
        
        # Format label text
        label_text = f"{region}\n({network}): {acc_mean[roi_index]:.3f}"
        
        # Add label to plot
        display.add_markers(
            marker_coords=[coords],
            marker_color='cyan',
            marker_size=30,
            alpha=0.7
        )
        display.annotate(
            label_text,
            coords,
            color='black',
            size=8
        )

    # Save the enhanced plot
    save_path = os.path.join(output_dir, "encoding_accuracy_mean_all_subjects_annotated.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved annotated group mean accuracy plot: {save_path}")

    # ==============================
    # Show per-network accuracy
    # ==============================
    network_to_acc = defaultdict(list)
    for roi_index, label in enumerate(schaefer_labels[1:]):  # skip background (first label)
        net_name = label.split('_')[2]  # e.g., "Default", "Visual"
        network_to_acc[net_name].append(acc_mean[roi_index])

    print("\nMean accuracy per Yeo-7 network:")
    network_means = {}
    for net, accs in network_to_acc.items():
        mean_net_acc = np.mean(accs)
        network_means[net] = mean_net_acc
        print(f"  {net:15s}: {mean_net_acc:.3f}")

    # Create a more informative network accuracy plot
    plt.figure(figsize=(12, 6))
    nets = list(network_means.keys())
    accs = [network_means[n] for n in nets]
    
    bars = plt.bar(nets, accs, color='tomato')
    plt.ylabel("Mean Encoding Accuracy")
    plt.title(f"Accuracy per Yeo-7 Network\nMean across {len(accuracy_dict)} subjects: {mean_acc_all:.3f}")
    plt.xticks(rotation=45)
    
    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "accuracy_by_network_annotated.png"), dpi=300)
    plt.close()


def plot_accuracy_by_network_and_context(accuracy_dicts, root_data_dir, context_windows, output_dir="output_plots"):
    """
    Plot mean encoding accuracy per Yeo-7 network comparing different context windows.
    """
    from nilearn.datasets import fetch_atlas_schaefer_2018
    from collections import defaultdict
    import numpy as np
    import matplotlib.pyplot as plt
    import os

    # Load Schaefer atlas
    schaefer = fetch_atlas_schaefer_2018(n_rois=1000, yeo_networks=7)
    schaefer_labels = schaefer['labels']
    
    # Prepare figure
    plt.figure(figsize=(16, 8))
    
    # Define colors for different context windows
    colors = plt.cm.viridis(np.linspace(0, 1, len(context_windows)))
    
    # Get network order
    network_order = ['Vis', 'SomMot', 'DorsAttn', 'SalVentAttn', 'Limbic', 'Cont', 'Default']
    network_names = ['Visual', 'SomMot', 'DorsAttn', 'SalVentAttn', 'Limbic', 'Control', 'Default']
    
    # Calculate mean accuracy per network per context window
    network_results = {net: [] for net in network_order}
    
    for ctx, acc_dict in zip(context_windows, accuracy_dicts):
        # Calculate mean accuracy across subjects
        all_acc = np.array(list(acc_dict.values()))
        mean_acc = np.mean(all_acc, axis=0)
        
        # Group by network
        network_acc = defaultdict(list)
        for roi_idx, label in enumerate(schaefer_labels[1:]):  # Skip background
            net_name = label.split('_')[2]
            network_acc[net_name].append(mean_acc[roi_idx])
        
        # Store results
        for net in network_order:
            network_results[net].append(np.mean(network_acc[net]))
    
    # Plot results
    x = np.arange(len(network_order))
    width = 0.8 / len(context_windows)
    
    for i, ctx in enumerate(context_windows):
        offsets = x + (i - len(context_windows)/2) * width + width/2
        plt.bar(offsets, 
               [network_results[net][i] for net in network_order],
               width=width,
               color=colors[i],
               label=f'Context {ctx}')
    
    plt.xticks(x, network_names, rotation=45)
    plt.ylabel('Mean Encoding Accuracy (r)')
    plt.title('Encoding Accuracy by Yeo-7 Network and Context Window Size')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f"network_accuracy_by_context.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved network accuracy comparison plot: {plot_path}")

def plot_network_atlas_glass_brain(output_dir="output_plots"):
    """
    Plot the Schaefer-1000 Yeo-7 network parcellation over glass brain.
    Each network will be shown in a different color.
    """
    from nilearn.datasets import fetch_atlas_schaefer_2018
    from nilearn import plotting
    import os

    os.makedirs(output_dir, exist_ok=True)

    # Load atlas with Yeo-7 network labels
    atlas = fetch_atlas_schaefer_2018(n_rois=1000, yeo_networks=7)
    atlas_img = atlas['maps']

    display = plotting.plot_roi(
        atlas_img, title="Yeo-7 Networks (Schaefer-1000)", display_mode="ortho",
        cmap='tab10', colorbar=False
    )
    display.savefig(os.path.join(output_dir, "yeo7_networks_glass_brain.png"))
    display.close()

def plot_somMot_network(output_dir="output_plots"):
    """
    Plot only the SomMot (Somatomotor) network from Schaefer-1000 atlas on glass brain.
    """
    from nilearn.datasets import fetch_atlas_schaefer_2018
    from nilearn import plotting
    from nilearn.maskers import NiftiLabelsMasker
    import numpy as np
    import os

    os.makedirs(output_dir, exist_ok=True)

    # Load Schaefer atlas
    schaefer = fetch_atlas_schaefer_2018(n_rois=1000, yeo_networks=7)
    atlas_img = schaefer['maps']
    labels = schaefer['labels']

    # Create a mask for SomMot network only
    somMot_mask = np.zeros(1000)  # Schaefer-1000 has 1000 ROIs
    for i, label in enumerate(labels[1:]):  # Skip background
        if 'SomMot' in label:
            somMot_mask[i] = 1

    # Apply the mask to the atlas
    masker = NiftiLabelsMasker(labels_img=atlas_img)
    masker.fit()
    somMot_img = masker.inverse_transform(somMot_mask)

    # Plot the SomMot network
    display = plotting.plot_glass_brain(
        somMot_img, 
        display_mode="ortho", 
        cmap='autumn',  # Using a different colormap for clarity
        colorbar=False,
        title="SomMot Network (Schaefer-1000 Atlas)"
    )
    
    save_path = os.path.join(output_dir, "somMot_network_glass_brain.png")
    display.savefig(save_path)
    display.close()
    print(f"Saved SomMot network plot: {save_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fmri_root", type=str, required=True)
    parser.add_argument("--subject", type=str, required=True)  # <-- string to allow comma-separated
    parser.add_argument("--train_movies", type=str, required=True)
    parser.add_argument("--val_movies", type=str, required=True)
    parser.add_argument("--modality_paths", type=str, required=True,
                  help="Comma-separated modality:path pairs, e.g. 'visual:/path1,audio:/path2'")
    parser.add_argument("--version_name", type=str, required=True,
                  help="Version tag to append to saved model filename")
    parser.add_argument("--output_model_dir", type=str, default="models",
                  help="Directory to save trained model .pkl")
    parser.add_argument("--group_plot", action="store_true",
                  help="If set, compute one mean accuracy plot over all subjects instead of per-subject plots")
    parser.add_argument("--plot_networks", action="store_true",
                  help="If set, include glass brain plots of Yeo-7 networks and SomMot network")
    parser.add_argument("--plot_somMot_only", action="store_true",
                  help="If set, plot only the SomMot network (overrides --plot_networks if both are set)")

    args = parser.parse_args()

    modality_paths = dict(item.split(":") for item in args.modality_paths.split(","))
    movies_train = args.train_movies.split(",")
    movies_val = args.val_movies.split(",")

    if args.group_plot:
        subjects = [int(s) for s in args.subject.split(",")]
        accuracy_dict = {}

        for subject in subjects:
            print(f"Processing subject {subject} for group mean plot")

            features = {mod: load_pca_features(modality_paths[mod]) for mod in modality_paths}
            fmri = load_fmri(args.fmri_root, subject)

            features_val, fmri_val = align_features_and_fmri_samples(
                features, fmri,
                excluded_samples_start=5,
                excluded_samples_end=5,
                hrf_delay=2,
                stimulus_window=10,
                movies=movies_val
            )
            if features_val.shape[0] == 0:
                print(f"No validation features for subject {subject}, skipping.")
                continue

            features_train, fmri_train = align_features_and_fmri_samples(
                features, fmri,
                excluded_samples_start=5,
                excluded_samples_end=5,
                hrf_delay=2,
                stimulus_window=10,
                movies=movies_train
            )
            model = train_encoding(features_train, fmri_train)
            fmri_val_pred = model.predict(features_val)

            encoding_accuracy = np.array([
                pearsonr(fmri_val[:, p], fmri_val_pred[:, p])[0]
                for p in range(fmri_val.shape[1])
            ], dtype=np.float32)

            accuracy_dict[subject] = encoding_accuracy

        compute_encoding_accuracy_group(accuracy_dict, args.fmri_root, modality="all")
        
        # Handle network plotting options
        if args.plot_somMot_only:
            plot_somMot_network()
        elif args.plot_networks:
            plot_network_atlas_glass_brain()
            plot_somMot_network()

    else:
        features = {mod: load_pca_features(modality_paths[mod]) for mod in modality_paths}
        fmri = load_fmri(args.fmri_root, int(args.subject))

        features_train, fmri_train = align_features_and_fmri_samples(
            features, fmri,
            excluded_samples_start=5,
            excluded_samples_end=5,
            hrf_delay=2,
            stimulus_window=10,
            movies=movies_train
        )
        model = train_encoding(features_train, fmri_train)

        features_val, fmri_val = align_features_and_fmri_samples(
            features, fmri,
            excluded_samples_start=5,
            excluded_samples_end=5,
            hrf_delay=2,
            stimulus_window=10,
            movies=movies_val
        )
        fmri_val_pred = model.predict(features_val)

        compute_encoding_accuracy(fmri_val, fmri_val_pred, int(args.subject), "language", args.fmri_root)


if __name__ == "__main__":
    main()
