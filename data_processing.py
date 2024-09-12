import numpy as np
import os
from constants import DATA_PATH
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

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

def load_iam_dataset(level='words'):
    """
    Loads the IAM dataset. Supports word, line, sentence, and form levels.

    Parameters:
    - level (str): The level of the dataset to load (e.g., 'words', 'lines', 'sentences', 'forms').

    Returns:
    - (train_images, test_images, train_labels, test_labels)
    """
    iam_data_path = os.path.join(DATA_PATH, 'IAM')
    
    # Load different levels of data
    if level == 'words':
        train_images = np.load(os.path.join(iam_data_path, 'preprocessed_words.npy'))
    elif level == 'lines':
        train_images = np.load(os.path.join(iam_data_path, 'preprocessed_lines.npy'))
    elif level == 'sentences':
        train_images = np.load(os.path.join(iam_data_path, 'preprocessed_sentences.npy'))
    elif level == 'forms':
        train_images = np.load(os.path.join(iam_data_path, 'preprocessed_forms.npy'))
    else:
        raise ValueError("Invalid level! Choose from 'words', 'lines', 'sentences', 'forms'")
    
    # For simplicity, splitting the data into 80% train and 20% test
    split_idx = int(0.8 * len(train_images))
    test_images = train_images[split_idx:]
    train_images = train_images[:split_idx]

    # Labels need to be processed similarly (this is a placeholder)
    train_labels = np.zeros(len(train_images))  # Dummy labels, replace with actual label processing
    test_labels = np.zeros(len(test_images))    # Dummy labels, replace with actual label processing

    return train_images, test_images, train_labels, test_labels

def load_training_test_data():
    images_test, labels_test = load_dataset(dataset='small')
    images_train, labels_train = load_dataset(dataset='big')
    return images_train, images_test, labels_train, labels_test

def visualize_images(images, labels, num_samples=5):
    """
    Visualizes a few sample images and their labels.

    Parameters:
    - images (numpy array): Array of images to visualize.
    - labels (numpy array): Array of labels corresponding to the images.
    - num_samples (int): Number of samples to visualize.
    """
    plt.figure(figsize=(10, 2 * num_samples))
    for i in range(num_samples):
        plt.subplot(num_samples, 1, i + 1)
        plt.imshow(images[i].squeeze(), cmap='gray')
        plt.title(f"Label: {labels[i]}")
        plt.axis('off')
    plt.tight_layout()

if __name__ == '__main__':
    images_train, images_test, labels_train, labels_test = load_training_test_data()
    print(images_train.shape, images_test.shape, labels_train.shape, labels_test.shape)

      # Print sample data to verify
    print("Sample training image:", images_train[0])
    print("Sample training label:", labels_train[0])
    print("Sample testing image:", images_test[0])
    print("Sample testing label:", labels_test[0])