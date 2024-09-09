import os
import shutil
from pathlib import Path

# Define paths
results_dir = Path('results')
old_dir = Path('old_results')
old_dir.mkdir(parents=True, exist_ok=True)

# Iterate through all files in the results directory
for filename in os.listdir(results_dir):
    file_path = results_dir / filename
    if file_path.is_file() and filename.endswith('.csv'):
        # Check if the file name contains a date other than 20240819
        if '20240909' not in filename:
            # Move the file to the old_results directory
            shutil.move(file_path, old_dir / filename)
