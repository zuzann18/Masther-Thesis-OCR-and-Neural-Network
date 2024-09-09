import pandas as pd

# Define the path to the runs_history.csv file
runs_history_path = 'results/runs_history.csv'

# Read the CSV file with a valid argument for on_bad_lines
runs_history = pd.read_csv(runs_history_path, on_bad_lines='skip')

# Initialize the actual_epochs column if it doesn't exist
if 'actual_epochs' not in runs_history.columns:
    runs_history['actual_epochs'] = None

# Save the updated DataFrame back to the CSV file
runs_history.to_csv(runs_history_path, index=False)

# Iterate through each row to update the actual_epochs
for index, row in runs_history.iterrows():
    history_file_path = row['history_csv_file']
    history_df = pd.read_csv(history_file_path)
    actual_epochs = len(history_df['loss'])
    runs_history.at[index, 'actual_epochs'] = actual_epochs

# Save the updated runs_history.csv file
runs_history.to_csv(runs_history_path, index=False)
