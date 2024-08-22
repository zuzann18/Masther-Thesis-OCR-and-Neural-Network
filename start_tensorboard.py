import subprocess
import os


def start_tensorboard(logs_base_dir='results'):
    print(f"Uruchamianie TensorBoard dla logów w katalogu: {logs_base_dir}")
    subprocess.Popen(['tensorboard', '--logdir', logs_base_dir])


if __name__ == '__main__':
    logs_base_dir = 'results'
    start_tensorboard(logs_base_dir)
