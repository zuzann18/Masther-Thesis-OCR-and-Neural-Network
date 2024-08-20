import os
import pandas as pd
import matplotlib.pyplot as plt

# Directory containing the result files
results_dir = 'results'
# Directory to save the best_results_visualizations
visualizations_dir = 'best_results_visualizations'
os.makedirs(visualizations_dir, exist_ok=True)

# List to store the results
results = []

# Iterate through all files in the directory
for filename in os.listdir(results_dir):
    if filename.endswith('.csv'):
        file_path = os.path.join(results_dir, filename)
        try:
            if os.path.getsize(file_path) > 0:  # Check if the file is not empty
                df = pd.read_csv(file_path)
                # Extract accuracy and loss columns
                if 'accuracy' in df.columns and 'loss' in df.columns:
                    train_accuracy = df['accuracy'].tolist()
                    test_accuracy = df['val_accuracy'].tolist()
                    train_loss = df['loss'].tolist()
                    test_loss = df['val_loss'].tolist()
                    results.append({
                        'filename': filename,
                        'train_accuracy': train_accuracy,
                        'test_accuracy': test_accuracy,
                        'train_loss': train_loss,
                        'test_loss': test_loss
                    })
            else:
                print(f"Skipping empty file: {file_path}")
        except pd.errors.EmptyDataError:
            print(f"Skipping file due to EmptyDataError: {file_path}")
        except Exception as e:
            print(f"An error occurred while processing {file_path}: {e}")
    else:
        print(f"File not found: {file_path}")

# Sort results by highest test accuracy and lowest test loss
sorted_by_test_accuracy = sorted(results, key=lambda x: max(x['test_accuracy']), reverse=True)[:3]
sorted_by_test_loss = sorted(results, key=lambda x: min(x['test_loss']))[:3]

# Print the top 3 highest test accuracy
print("Top 3 Highest Test Accuracy:")
for result in sorted_by_test_accuracy:
    print(f"File: {result['filename']}")
    print(f"Test Accuracy: {result['test_accuracy']}")
    print()

# Print the top 3 lowest test loss
print("Top 3 Lowest Test Loss:")
for result in sorted_by_test_loss:
    print(f"File: {result['filename']}")
    print(f"Test Loss: {result['test_loss']}")
    print()


# Function to plot and save best_results_visualizations
def plot_accuracy(result, metric, index):
    plt.figure(figsize=(10, 5))
    plt.plot(result['train_accuracy'], label='Train Accuracy')
    plt.plot(result['test_accuracy'], label='Test Accuracy')
    plt.title(f"File: {result['filename']} - Top {metric}")
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt_path = os.path.join(visualizations_dir, f"{metric}_{index}_{result['filename']}.png")
    plt.savefig(plt_path)
    plt.show()


def plot_loss(result, metric, index):
    plt.figure(figsize=(10, 5))
    plt.plot(result['train_loss'], label='Train Loss')
    plt.plot(result['test_loss'], label='Test Loss')
    plt.title(f"File: {result['filename']} - Top {metric}")
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt_path = os.path.join(visualizations_dir, f"{metric}_{index}_{result['filename']}.png")
    plt.savefig(plt_path)
    plt.show()


# Visualize and save the top 3 highest test accuracy
for index, result in enumerate(sorted_by_test_accuracy):
    plot_accuracy(result, 'Test Accuracy', index)

# Visualize and save the top 3 lowest test loss
for index, result in enumerate(sorted_by_test_loss):
    plot_loss(result, 'Test Loss', index)
