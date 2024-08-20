import numpy as np
import requests
from datasets import load_dataset

from constants import DATA_PATH, NUM_CLASSES


def download_big_dataset():
    dataset = load_dataset("pittawat/letter_recognition")

    images = np.array([
        np.expand_dims(
            np.array(item['image'].convert('1')),
            axis=2
        )
        for item in dataset['train']]
    )
    labels = np.array([item['label'] for item in dataset['train']])
    one_hot_labels = np.zeros((labels.shape[0], NUM_CLASSES))

    for idx, label in enumerate(labels):
        one_hot_labels[idx, label] = 1

    dataset_path = DATA_PATH / 'big'
    dataset_path.mkdir(parents=True, exist_ok=True)

    np.save(dataset_path / 'images.npy', images)
    np.save(dataset_path / 'labels.npy', one_hot_labels)

    # for item in dataset['train']:
    #     img = Image.fromarray(item['image'])
    #     img_resized = img.resize(desired_size)
    #     images_resized.append(np.array(img_resized))
    #
    # # Convert to numpy arrays
    # images_np = np.array(images_resized)
    # labels_np = np.array([item['label'] for item in dataset['train']])
    #
    # # Define the dataset path
    # DATA_PATH = Path('data/big')
    # DATA_PATH.mkdir(parents=True, exist_ok=True)
    #
    # # Save images and labels as .npy files
    # np.save(DATA_PATH / 'images.npy', images_np)
    # np.save(DATA_PATH / 'labels.npy', labels_np)


def download_small_dataset():
    download_from_google_drive(
        file_id="1UXrjOg9Q5xmJMMj6frBT11mZIlq0Z7UT",
        file_name='images.npy',
        directory='small'
    )
    download_from_google_drive(
        file_id="1U4ohJaydU7ERryHFph9km_T4Po4rYmWv",
        file_name='labels.npy',
        directory='small'
    )


def download_from_google_drive(file_id, directory, file_name):
    dataset_path = DATA_PATH / directory
    file_path = dataset_path / file_name
    dataset_path.mkdir(parents=True, exist_ok=True)
    URL = "https://drive.google.com/uc?id=" + file_id
    response = requests.get(URL, stream=True)
    with open(file_path, "wb") as file:
        for chunk in response.iter_content(chunk_size=32768):
            if chunk:  # filter out keep-alive new chunks
                file.write(chunk)


if __name__ == '__main__':
    download_big_dataset()
    download_small_dataset()
