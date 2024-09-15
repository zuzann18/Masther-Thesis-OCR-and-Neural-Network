import os
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from constants import DATA_PATH
import matplotlib.pyplot as plt

def load_dataset(dataset: str):
    """
    Loads dataset.

    Parameters:
    - dataset (str): The name of the dataset to load.

    Returns:
    - images (numpy array): Array of images.
    - labels (numpy array): Array of labels.
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
        images = np.load(os.path.join(iam_data_path, 'preprocessed_words.npy'))
        labels = np.load(os.path.join(iam_data_path, 'labels_words.npy'))
    elif level == 'lines':
        images = np.load(os.path.join(iam_data_path, 'preprocessed_lines.npy'))
        labels = np.load(os.path.join(iam_data_path, 'labels_lines.npy'))
    elif level == 'sentences':
        images = np.load(os.path.join(iam_data_path, 'preprocessed_sentences.npy'))
        labels = np.load(os.path.join(iam_data_path, 'labels_sentences.npy'))
    elif level == 'forms':
        images = np.load(os.path.join(iam_data_path, 'preprocessed_forms.npy'))
        labels = np.load(os.path.join(iam_data_path, 'labels_forms.npy'))
    else:
        raise ValueError("Invalid level! Choose from 'words', 'lines', 'sentences', 'forms'")
    
    # Normalize images
    images = images.astype('float32') / 255.0
    
    # Encode labels
    label_encoder = LabelEncoder()
    integer_encoded_labels = label_encoder.fit_transform(labels)
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded_labels = integer_encoded_labels.reshape(len(integer_encoded_labels), 1)
    onehot_encoded_labels = onehot_encoder.fit_transform(integer_encoded_labels)
    
    # For simplicity, splitting the data into 80% train and 20% test
    split_idx = int(0.8 * len(images))
    train_images = images[:split_idx]
    test_images = images[split_idx:]
    train_labels = onehot_encoded_labels[:split_idx]
    test_labels = onehot_encoded_labels[split_idx:]

    return train_images, test_images, train_labels, test_labels

def load_training_test_data(dataset='small'):
    """
    Loads training and testing data for the specified dataset.

    Parameters:
    - dataset (str): The name of the dataset to load.

    Returns:
    - (train_images, test_images, train_labels, test_labels)
    """
    images, labels = load_dataset(dataset)
    split_idx = int(0.8 * len(images))
    train_images = images[:split_idx]
    test_images = images[split_idx:]
    train_labels = labels[:split_idx]
    test_labels = labels[split_idx:]
    return train_images, test_images, train_labels, test_labels
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
if __name__ == '__main__':
    train_images, test_images, train_labels, test_labels = load_training_test_data(dataset='small')
    print(train_images.shape, test_images.shape, train_labels.shape, test_labels.shape)
    
    
  