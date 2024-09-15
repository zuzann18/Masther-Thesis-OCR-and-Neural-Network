import csv
from datetime import datetime
import click
import pandas as pd
from keras.callbacks import CSVLogger, EarlyStopping, LearningRateScheduler, TensorBoard
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from constants import RESULTS_PATH
from data_processing import load_training_test_data
from experimental_config import EXPERIMENTAL_CONFIG
from models import get_model
import os
import subprocess
import xgboost as xgb
from sklearn.metrics import accuracy_score

def lr_schedule(epoch, lr):
    if epoch > 10:
        lr = 1e-4
    elif epoch > 20:
        lr = 1e-5
    else:
        lr = 1e-3
    return lr

@click.command()
@click.option('--experiment_id', type=int, help='Experiment ID', required=True)
@click.option('--epochs', type=int, help='Number of epochs', default=2)
def main(experiment_id, epochs):
    """
    Example:
    python run_experiments.py --experiment_id 0 --epochs 2
    """
    config = [c for c in EXPERIMENTAL_CONFIG if c['experiment_id'] == experiment_id][0]
    config['epochs'] = epochs
    run_experiment(**config)

def run_tensorboard(logs_base_dir='results'):
    print(f"Starting TensorBoard for logs in directory: {logs_base_dir}")
    subprocess.Popen(['tensorboard', '--logdir', logs_base_dir])

def run_experiment(
        experiment_id,
        model_name,
        epochs,
        batch_size=64,
        dropout_rate=0,
        learning_rate=0.0001,
        optimizer='adam',
        augmentation=False,
        zoom_range=0.2,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        learning_rate_scheduler=None,
        extra_layers=None,
        num_residual_blocks=None
):
    print(f"{augmentation=}")
    print(f"{dropout_rate=}")
    print(f"{experiment_id=}")
    print(f"{model_name=}")

    start_time = datetime.now()
    train_images, test_images, train_labels, test_labels = load_training_test_data()
    test_batches = (test_images, test_labels)

    if model_name == 'xgboost':
        # Flatten the images
        x_train_flat = train_images.reshape(train_images.shape[0], -1)
        x_test_flat = test_images.reshape(test_images.shape[0], -1)

        # Convert to DMatrix
        dtrain = xgb.DMatrix(x_train_flat, label=train_labels.argmax(axis=1))
        dtest = xgb.DMatrix(x_test_flat, label=test_labels.argmax(axis=1))

        # Set parameters
        params = {
            'objective': 'multi:softmax',
            'num_class': 10,
            'eval_metric': 'mlogloss',
            'learning_rate': learning_rate
        }

        # Train model
        bst = xgb.train(params, dtrain, num_boost_round=100)

        # Predict
        y_pred = bst.predict(dtest)
        accuracy = accuracy_score(test_labels.argmax(axis=1), y_pred)
        print(f'XGBoost Accuracy: {accuracy}')

        # Save results
        run_details = {
            'experiment_id': experiment_id,
            'timestamp': datetime.now().strftime("%Y%m%d-%H%M%S"),
            'model_name': model_name,
            'accuracy': accuracy
        }

        result_path = RESULTS_PATH / f"runs_history.csv"
        file_exists = os.path.isfile(result_path)
        with open(result_path, 'a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=run_details.keys())
            if not file_exists or os.stat(result_path).st_size == 0:
                writer.writeheader()
            writer.writerow(run_details)

    else:
        model = get_model(
            model_name=model_name,
            dropout_rate=dropout_rate,
            learning_rate=learning_rate,
            optimizer=optimizer,
            augmentation=augmentation,
            extra_layers=extra_layers,
            num_residual_blocks=num_residual_blocks
        )

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        history_csv_file = RESULTS_PATH / f"training_history_{experiment_id}_{timestamp}.csv"
        tensorboard_log_dir = RESULTS_PATH / f"tensorboard_logs_{experiment_id}_{timestamp}"

        early_stopping = EarlyStopping(monitor='val_loss', patience=8)
        csv_logger = CSVLogger(history_csv_file)
        tensorboard = TensorBoard(log_dir=tensorboard_log_dir)
        callbacks = [early_stopping, csv_logger, tensorboard]

        # Ensure the directory exists or is created during the training process
        if not os.path.exists(tensorboard_log_dir):
            os.makedirs(tensorboard_log_dir)

        # Start TensorBoard before model training
        run_tensorboard(tensorboard_log_dir)

        if learning_rate_scheduler:
            if learning_rate_scheduler['type'] == "ReduceLROnPlateau":
                lr_scheduler = LearningRateScheduler(lr_schedule)
            elif learning_rate_scheduler['type'] == "ExponentialDecay":
                lr_scheduler = LearningRateScheduler(lr_schedule)

            callbacks.append(lr_scheduler)

        if augmentation:
            print("Data augmentation enabled")
            datagen = ImageDataGenerator(
                rotation_range=rotation_range,
                width_shift_range=width_shift_range,
                height_shift_range=height_shift_range,
                horizontal_flip=False,
                vertical_flip=False,
                zoom_range=zoom_range,
                shear_range=shear_range,
                fill_mode='nearest'
            )
            train_data = datagen.flow(train_images, train_labels, batch_size=batch_size)
        else:
            train_data = (train_images, train_labels)

        history = model.fit(
            train_data,
            epochs=epochs,
            validation_data=test_batches,
            callbacks=callbacks
        )

        pd.DataFrame(history.history).to_csv(history_csv_file, index=False)

        total_seconds = (datetime.now() - start_time).total_seconds()
        actual_epochs = len(history.history['loss'])
        best_train_accuracy = max(history.history['accuracy'])
        best_val_accuracy = max(history.history['val_accuracy'])
        best_train_loss = min(history.history['loss'])
        best_val_loss = min(history.history['val_loss'])

        run_details = {
            'experiment_id': experiment_id,
            'timestamp': timestamp,
            'model_name': model_name,
            'epochs': epochs,
            'actual_epochs': actual_epochs,
            'batch_size': batch_size,
            'dropout_rate': dropout_rate,
            'learning_rate': learning_rate,
            'optimizer': optimizer,
            'augmentation': augmentation,
            'zoom_range': zoom_range,
            'rotation_range': rotation_range,
            'width_shift_range': width_shift_range,
            'height_shift_range': height_shift_range,
            'shear_range': shear_range,
            'learning_rate_scheduler': learning_rate_scheduler,
            'extra_layers': extra_layers,
            'num_residual_blocks': num_residual_blocks,
            'total_seconds': total_seconds,
            'best_train_accuracy': best_train_accuracy,
            'best_val_accuracy': best_val_accuracy,
            'best_train_loss': best_train_loss,
            'best_val_loss': best_val_loss,
            'history_csv_file': history_csv_file,
            'tensorboard_log_dir': tensorboard_log_dir
        }

        result_path = RESULTS_PATH / f"runs_history.csv"
        file_exists = os.path.isfile(result_path)
        with open(result_path, 'a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=run_details.keys())
            if not file_exists or os.stat(result_path).st_size == 0:
                writer.writeheader()
            writer.writerow(run_details)

        # Check if the directory exists and list its contents
        log_dir = str(tensorboard_log_dir)
        if os.path.exists(log_dir):
            print(f"Directory '{log_dir}' exists.")

            # List all files in the directory
            files = os.listdir(log_dir)

            if files:
                print(f"Log files in '{log_dir}':")
                for file in files:
                    print(file)
            else:
                print(f"No log files found in '{log_dir}'.")
        else:
            print(f"Directory '{log_dir}' does not exist.")

if __name__ == '__main__':
    main()