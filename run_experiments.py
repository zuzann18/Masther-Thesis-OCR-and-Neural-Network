import csv
from datetime import datetime

import click
import pandas as pd
from keras.callbacks import CSVLogger
from keras.src.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from constants import RESULTS_PATH
from data_processing import load_training_test_data
from experimental_config import EXPERIMENTAL_CONFIG
from models import get_model
from tensorflow.keras.callbacks import LearningRateScheduler


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
        learning_rate_scheduler=False,
):
    print(f"{augmentation=}")
    print(f"{dropout_rate=}")
    # start measuring time
    start_time = datetime.now()
    train_images, test_images, train_labels, test_labels = load_training_test_data()
    test_batches = test_images, test_labels
    model = get_model(
        model_name=model_name,
        dropout_rate=dropout_rate,
        learning_rate=learning_rate,
        optimizer=optimizer,
        augmentation=augmentation,
    )

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    history_csv_file = RESULTS_PATH / f"training_history_{experiment_id}_{timestamp}.csv"

    early_stopping = EarlyStopping(monitor='val_loss', patience=8)
    csv_logger = CSVLogger(history_csv_file)
    callbacks = [early_stopping, csv_logger]

    if learning_rate_scheduler:
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
        history = model.fit(
            train_data,
            steps_per_epoch=int(len(train_images) / batch_size),
            epochs=epochs,
            validation_data=test_batches,
            callbacks=[early_stopping, csv_logger]
        )
    else:
        history = model.fit(
            train_images, train_labels,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=test_batches,
            callbacks=[early_stopping, csv_logger]
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
        'total_seconds': total_seconds,
        'best_train_accuracy': best_train_accuracy,
        'best_val_accuracy': best_val_accuracy,
        'best_train_loss': best_train_loss,
        'best_val_loss': best_val_loss,
        'history_csv_file': history_csv_file,
    }
    result_path = RESULTS_PATH / f"runs_history.csv"

    with open(result_path, 'a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=run_details.keys())
        if file.tell() == 0:
            writer.writeheader()
        writer.writerow(run_details)


if __name__ == '__main__':
    main()
