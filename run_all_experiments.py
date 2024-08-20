import subprocess
from experimental_config import EXPERIMENTAL_CONFIG


def run_all_experiments(start_id=0, experiment_ids=None):
    if experiment_ids is not None:
        configs_to_run = [config for config in EXPERIMENTAL_CONFIG if config['experiment_id'] in experiment_ids]
    else:
        configs_to_run = [config for config in EXPERIMENTAL_CONFIG if config['experiment_id'] >= start_id]

    for config in configs_to_run:
        experiment_id = config['experiment_id']
        epochs = config.get('epochs', 350)  # Default to 350 epochs if not specified
        try:
            subprocess.run(
                ['python', 'run_experiments.py', '--experiment_id', str(experiment_id), '--epochs', str(epochs)],
                check=True
            )
        except subprocess.CalledProcessError as e:
            print(f"Experiment {experiment_id} failed with error: {e}")

if __name__ == '__main__':
    import sys
    import ast

    if len(sys.argv) > 1:
        experiment_ids = ast.literal_eval(sys.argv[1])
        run_all_experiments(experiment_ids=experiment_ids)
    else:
        run_all_experiments()