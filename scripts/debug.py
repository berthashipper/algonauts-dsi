import os
import numpy as np

# Path to top-level directory where all CLIP visual features are stored
clip_root = "/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/results/clip_visual_features"

bad_files = []

for dirpath, dirnames, filenames in os.walk(clip_root):
    for fname in filenames:
        if fname.endswith(".npy"):
            fpath = os.path.join(dirpath, fname)
            try:
                data = np.load(fpath, allow_pickle=True)
                if data.ndim != 2 or data.shape[0] <= 1:
                    print(f"❌ BAD SHAPE: {fpath} → shape={data.shape}")
                    bad_files.append(fpath)
            except Exception as e:
                print(f"‼️ ERROR reading {fpath}: {e}")
                bad_files.append(fpath)

print("\n=== SUMMARY ===")
print(f"Found {len(bad_files)} bad CLIP feature files.")

# Confirm before deleting
for f in bad_files:
    try:
        os.remove(f)
        print(f"🗑️ Deleted: {f}")
    except Exception as e:
        print(f"‼️ Could not delete {f}: {e}")
