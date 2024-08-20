import pandas as pd
import os

# Path to the runs history CSV file
runs_history_csv_path = 'results/runs_history.csv'

# Ensure the 'results' directory exists
os.makedirs(os.path.dirname(runs_history_csv_path), exist_ok=True)

# Load the existing runs history CSV file
if os.path.exists(runs_history_csv_path):
    runs_history_df = pd.read_csv(runs_history_csv_path)
else:
    runs_history_df = pd.DataFrame(columns=[
        'experiment_id', 'timestamp', 'model_name', 'epochs', 'actual_epochs', 'batch_size',
        'dropout_rate', 'learning_rate', 'optimizer', 'augmentation', 'total_seconds',
        'best_train_accuracy', 'best_val_accuracy', 'best_train_loss', 'best_val_loss', 'history_csv_file'
    ])


# Function to get training history for an experiment from a CSV file
def get_training_history_from_csv(file_path):
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        return pd.DataFrame()


# Iterate through experiments 0 to 59
for experiment_id in range(60):
    if experiment_id not in runs_history_df['experiment_id'].unique():
        # Construct the file path for the training history CSV file
        training_history_file = f'results/training_history_{experiment_id}.csv'

        # Get the training history for the experiment
        training_history_df = get_training_history_from_csv(training_history_file)

        if not training_history_df.empty:
            # Extract summary information
            summary_info = {
                'experiment_id': experiment_id,
                'timestamp': training_history_df['timestamp'].iloc[0],
                'model_name': training_history_df['model_name'].iloc[0],
                'epochs': training_history_df['epochs'].iloc[0],
                'actual_epochs': training_history_df['actual_epochs'].iloc[0],
                'batch_size': training_history_df['batch_size'].iloc[0],
                'dropout_rate': training_history_df['dropout_rate'].iloc[0],
                'learning_rate': training_history_df['learning_rate'].iloc[0],
                'optimizer': training_history_df['optimizer'].iloc[0],
                'augmentation': training_history_df['augmentation'].iloc[0],
                'total_seconds': training_history_df['total_seconds'].iloc[0],
                'best_train_accuracy': training_history_df['best_train_accuracy'].max(),
                'best_val_accuracy': training_history_df['best_val_accuracy'].max(),
                'best_train_loss': training_history_df['best_train_loss'].min(),
                'best_val_loss': training_history_df['best_val_loss'].min(),
                'history_csv_file': training_history_file,
            }

            # Append the new row to the runs history DataFrame
            runs_history_df = pd.concat([runs_history_df, pd.DataFrame([summary_info])], ignore_index=True)

# Save the updated runs history DataFrame to the CSV file
runs_history_df.to_csv(runs_history_csv_path, index=False)

print(runs_history_df)
