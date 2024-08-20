import os

import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

from constants import FOLDER_PATH, IMAGES_SAVE_PATH, LABELS_SAVE_PATH


def bmp_to_npy(folder_path, images_save_path, labels_save_path):
    """
    Convert the BMP files to NPY with labels from file name
    """
    file_list = [file for file in os.listdir(folder_path) if file.endswith(".bmp")]

    if len(file_list) == 0:
        print("No BMP files in the folder.")
        return

    data = []
    labels = []

    for bmp_file in file_list:
        file_path = os.path.join(folder_path, bmp_file)
        image = Image.open(file_path)
        image_gray = image.convert("L")
        image_array = np.array(image_gray)
        image_normalized = image_array / 255.0  # Normalization
        data.append(image_normalized)

        # Extract the label (first character of the file name)
        label = os.path.splitext(bmp_file)[0][0]
        labels.append(label)

    # Convert the list of images to a numpy array and add an extra dimension
    data_np = np.expand_dims(np.array(data), axis=-1)

    # Encode labels
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)
    labels_encoded = to_categorical(labels_encoded)

    # Save the numpy arrays to .npy files
    np.save(images_save_path, data_np)
    np.save(labels_save_path, labels_encoded)
    print(f"Images saved to {images_save_path}")
    print(f"Labels saved to {labels_save_path}")


def print_bmp_zero_one_string():
    """
    This function loads all BMP files and print them as a string consisting of zeros and ones
    """
    folder_path = r"C:\Users\zuzan\OneDrive\Pulpit\literki"  # Ścieżka do folderu zawierającego plik BMP

    # Pobierz listę plików z folderu
    file_list = [file for file in os.listdir(folder_path) if file.endswith(".bmp")]

    if len(file_list) == 0:
        print("Brak plików BMP w folderze.")
        exit()

    for bmp_file in file_list:
        # Pełna ścieżka do pliku BMP
        file_path = os.path.join(folder_path, bmp_file)

        # Wczytaj plik BMP
        image = Image.open(file_path)

        # Przekształć obraz na skalę szarości
        image_gray = image.convert("L")

        # Przekształć obraz na macierz numpy
        image_array = np.array(image_gray)

        # Przekształć macierz obrazu na ciąg zer i jedynek
        binary_string = "".join(str(int(pixel / 255)) for row in image_array for pixel in row)

        print("Plik:", bmp_file)
        print("Binary string:", binary_string)
        print()


if __name__ == "__main__":
    bmp_to_npy(
        folder_path=FOLDER_PATH,
        images_save_path=IMAGES_SAVE_PATH,
        labels_save_path=LABELS_SAVE_PATH,
    )
