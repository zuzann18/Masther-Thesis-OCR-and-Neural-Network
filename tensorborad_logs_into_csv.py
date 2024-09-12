
import os
import csv
from tensorboard.backend.event_processing import event_accumulator

log_dir = 'results/tensorboard_logs'

def export_tensorboard_to_csv(log_dir, output_csv):
    event_acc = event_accumulator.EventAccumulator(log_dir)
    event_acc.Reload()

    metrics = event_acc.Tags()['scalars']

    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Step', 'Metric', 'Value'])

        for metric in metrics:
            events = event_acc.Scalars(metric)
            for event in events:
                writer.writerow([event.step, metric, event.value])

    print(f"Dane zostały zapisane w pliku {output_csv}")

export_tensorboard_to_csv(log_dir, 'tensorboard_metrics.csv')