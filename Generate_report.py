import os
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import tensorflow as tf
import csv

def debug_csv_file(file_path, expected_columns):
    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)
        num_columns = len(header)
        print(f"Oczekujemy kolumn: {num_columns}")

        for i, row in enumerate(reader, start=2):
            if len(row) != num_columns:
                print(f"Problem linia {i}. Oczekjemy {num_columns} kolumn, ale mamy {len(row)}")
                print(f"Zawartość linii: {row}")
                
csv_file_path = 'results/runs_history.csv'
debug_csv_file(csv_file_path, expected_columns=18)

def find_latest_tensorboard_log(logs_dir):
    log_dirs = [f for f in os.listdir(logs_dir) if os.path.isdir(os.path.join(logs_dir, f))]
    latest_dir = max(log_dirs, key=lambda x: os.path.getctime(os.path.join(logs_dir, x)))
    return os.path.join(logs_dir, latest_dir)

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


def load_tensorboard_logs(log_dir):
    log_data = []

    for root, dirs, files in os.walk(log_dir):
        for file in files:
            if "tfevents" in file:
                file_path = os.path.join(root, file)
                for event in tf.compat.v1.train.summary_iterator(file_path):
                    for value in event.summary.value:
                        log_data.append({
                            'wall_time': event.wall_time,
                            'step': event.step,
                            'tag': value.tag,
                            'value': tf.make_ndarray(value.tensor).item(),
                            'run': root.split('/')[-1]
                        })

    df = pd.DataFrame(log_data)
    return df

# Function to generate the report content
def generate_report(output_dir):
# finding_new_experiments.py
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

    matching_rows = runs_history_df[runs_history_df['history_csv_file'] == history_csv_file]
    if not matching_rows.empty:
        run_details = matching_rows.iloc[0]
    else:
        raise ValueError(f"No matching entry found for {history_csv_file} in runs_history.csv")

    best_run = find_best_results(runs_history_df)

    model_summary = []
    model_summary_str = "\n".join(model_summary)

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
    - **Total Seconds**: {run_details['total_seconds']}

    ## Model Architecture
    {model_summary_str}

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
logs_dir = 'results'

tensorboard_log_dir = find_latest_tensorboard_log(logs_dir)

generate_report(output_dir)