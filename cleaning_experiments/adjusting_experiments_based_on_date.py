import os
import shutil
from pathlib import Path

# Define paths
results_dir = Path('results')
old_dir = Path('old_results')
old_dir.mkdir(parents=True, exist_ok=True)

# Iterate through all files and directories in the results directory
for filename in os.listdir(results_dir):
    file_path = results_dir / filename
    # Check if it's a file or a directory (in the case of TensorBoard logs)
    if file_path.is_dir() or file_path.is_file():
        # Exclude 'runs_history.csv' from being moved
        if filename == 'runs_history.csv':
            print(f"Skipping file: {filename} (runs_history.csv is excluded)")
            continue  # Skip the rest of the loop for this file
        
        # Check if it's tensorboard_logs or a .csv file
        if filename.startswith('tensorboard_logs') or filename.endswith('.csv'):
            print(f"Processing file: {filename}")  # Add logging to see which files are being processed
            # Check if the file name contains dates other than 09.09 or 10.09 (20240909 or 20240910)
            if '20240909' not in filename and '20240910' not in filename:
                # Move the file or folder to the old_results directory
                shutil.move(file_path, old_dir / filename)
                print(f"Moved file: {filename}")
            else:
                print(f"Skipping file (date matches 09.09 or 10.09): {filename}")
    else:
        print(f"Not a valid file or directory: {filename}")

print("Proces zakończony.")


