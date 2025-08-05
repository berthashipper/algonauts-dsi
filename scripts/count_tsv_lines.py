import glob
from pathlib import Path

pattern = "/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/data/stimuli/transcripts/**/*.tsv"
for f in glob.glob(pattern, recursive=True):
    fname = Path(f).name
    with open(f, 'r') as file:
        n_lines = sum(1 for _ in file) - 1  # subtract header
    print(f'"{fname}": {n_lines},')
