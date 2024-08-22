import os
import pandas as pd
import matplotlib.pyplot as plt

# Function to find the latest history CSV file
def find_latest_history_csv(results_dir):
    csv_files = [f for f in os.listdir(results_dir) if f.startswith('training_history') and f.endswith('.csv')]
    latest_file = max(csv_files, key=lambda x: os.path.getctime(os.path.join(results_dir, x)))
    return os.path.join(results_dir, latest_file)

# Function to generate plots and save them as images
def generate_plots(history_df, output_dir):
    metrics = ['accuracy', 'val_accuracy', 'loss', 'val_loss']
    for metric in metrics:
        plt.figure(figsize=(10, 5))
        plt.plot(history_df[metric])
        plt.title(f'Model {metric}')
        plt.ylabel(metric)
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'] if 'val' in metric else ['Train'], loc='upper left')
        plot_path = f'{output_dir}/{metric}_plot.jpg'
        plt.savefig(plot_path)
        plt.close()

# Function to find the best results from runs_history.csv
def find_best_results(runs_history_df):
    best_run = runs_history_df.loc[runs_history_df['best_val_accuracy'].idxmax()]
    return best_run

# Function to generate the report content
def generate_report(output_dir):
    try:
        runs_history_df = pd.read_csv('results/runs_history.csv')
    except pd.errors.ParserError as e:
        print(f"Error reading CSV file: {e}")
        return
    history_csv_file = find_latest_history_csv(output_dir)
    print(f"Latest history CSV file: {history_csv_file}")

    if not os.path.exists(history_csv_file):
        raise FileNotFoundError(f"The file {history_csv_file} does not exist.")

    history_df = pd.read_csv(history_csv_file)
    generate_plots(history_df, output_dir)

    # Read the run details from the CSV file
    runs_history_df = pd.read_csv('results/runs_history.csv')
    print(f"Entries in runs_history.csv:\n{runs_history_df}")

    matching_rows = runs_history_df[runs_history_df['history_csv_file'] == history_csv_file]
    if not matching_rows.empty:
        run_details = matching_rows.iloc[0]
    else:
        raise ValueError(f"No matching entry found for {history_csv_file} in runs_history.csv")

    # Find the best results
    best_run = find_best_results(runs_history_df)

    report_content = f"""
# Experiment Report

## Parameters
- **Experiment ID**: {run_details['experiment_id']}
- **Timestamp**: {run_details['timestamp']}
- **Model Name**: {run_details['model_name']}
- **Epochs**: {run_details['epochs']}
- **Batch Size**: {run_details['batch_size']}
- **Dropout Rate**: {run_details['dropout_rate']}
- **Learning Rate**: {run_details['learning_rate']}
- **Optimizer**: {run_details['optimizer']}
- **Augmentation**: {run_details['augmentation']}
  - **Zoom Range**: {run_details['zoom_range']}
  - **Rotation Range**: {run_details['rotation_range']}
  - **Width Shift Range**: {run_details['width_shift_range']}
  - **Height Shift Range**: {run_details['height_shift_range']}
  - **Shear Range**: {run_details['shear_range']}
- **Number of Layers**: {run_details['num_layers']}
- **Total Seconds**: {run_details['total_seconds']}

## Metrics
### Model Accuracy
![Accuracy Plot]({output_dir}/accuracy_plot.jpg)

### Model Validation Accuracy
![Validation Accuracy Plot]({output_dir}/val_accuracy_plot.jpg)

### Model Loss
![Loss Plot]({output_dir}/loss_plot.jpg)

### Model Validation Loss
![Validation Loss Plot]({output_dir}/val_loss_plot.jpg)

## Best Results
- **Experiment ID**: {best_run['experiment_id']}
- **Timestamp**: {best_run['timestamp']}
- **Model Name**: {best_run['model_name']}
- **Best Validation Accuracy**: {best_run['best_val_accuracy']}
- **Best Validation Loss**: {best_run['best_val_loss']}
"""

    with open(f'{output_dir}/Report.md', 'w') as report_file:
        report_file.write(report_content)

# Example usage
output_dir = 'results'
generate_report(output_dir)