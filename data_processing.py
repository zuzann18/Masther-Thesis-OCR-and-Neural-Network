import numpy as np

from constants import DATA_PATH


def load_dataset(dataset: str):
    """
    Loads dataset

    Parameters:
    - dataset (str): The name of the dataset to load.

    Returns:
    - training/testing data ((train_images, train_labels), (test_images, test_labels))
    """

    available_datasets = ['small', 'big']
    assert dataset in available_datasets, f"Dataset '{dataset}' should be one of {available_datasets}"
    dataset_path = DATA_PATH / dataset

    images = np.load(dataset_path / 'images.npy')
    labels = np.load(dataset_path / 'labels.npy')

    images = images.astype('float32') / 255.0
    return images, labels


def load_training_test_data():
    images_test, labels_test = load_dataset(dataset='small')
    images_train, labels_train = load_dataset(dataset='big')
    return images_train, images_test, labels_train, labels_test


if __name__ == '__main__':
    images_train, images_test, labels_train, labels_test = load_training_test_data()
    print(images_train.shape, images_test.shape, labels_train.shape, labels_test.shape)
