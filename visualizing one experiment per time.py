import click
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_PATH = Path("results")
CHARTS_PATH = RESULTS_PATH / "best_results_visualizations"
CHARTS_PATH.mkdir(parents=True, exist_ok=True)


@click.command()
@click.option('--experiment_id', type=int, help='Experiment ID', required=True)
def visualize_experiment(experiment_id):
    runs_history = pd.read_csv(RESULTS_PATH / "runs_history.csv")
    row = runs_history[runs_history['experiment_id'] == experiment_id].iloc[0]
    history_file = pd.read_csv(row['history_csv_file'])

    fig, axs = plt.subplots(2)
    fig.suptitle(f"Experiment {row['experiment_id']} - {row['model_name']} - {row['timestamp']}")

    axs[0].plot(history_file['loss'], label='train')
    axs[0].plot(history_file['val_loss'], label='test')

    axs[1].plot(history_file['accuracy'], label='train')
    axs[1].plot(history_file['val_accuracy'], label='test')

    axs[0].set_title('Loss')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].legend()

    axs[1].set_title('Accuracy')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Accuracy')
    axs[1].legend()
    plt.tight_layout()
    plt.savefig(
        str(CHARTS_PATH / f"experiment_{row['experiment_id']}_model_{row['model_name']}_{row['timestamp']}.png"))
    plt.show()
    plt.close(fig)  # Close the figure to free up memory


if __name__ == '__main__':
    visualize_experiment()
