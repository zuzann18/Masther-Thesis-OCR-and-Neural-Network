import subprocess
import os

# Define the path to the TensorBoard logs folder
tensorboard_logs_folder = "C:\\Users\\zuzan\\OneDrive\\Pulpit\\Dokumenty\\GitHub\\Masther Thesis  OCR and Neural Network\\tensorboard_logs"

# Start TensorBoard using the subprocess module
subprocess.run(["tensorboard", "--logdir", tensorboard_logs_folder])
