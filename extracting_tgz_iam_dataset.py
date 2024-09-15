import tarfile
import os
from PIL import Image
import numpy as np
import xml.etree.ElementTree as ET

# Function to extract any .tgz or .gz file
def extract_tgz(tgz_file, extract_path):
    """
    Extracts a .tgz or .gz file to a specified directory.

    Parameters:
    - tgz_file: The path to the .tgz/.gz file.
    - extract_path: The directory where the files should be extracted.
    """
    if tgz_file.endswith(("tgz", "tar.gz")):  # Handle both .tgz and .tar.gz
        if not os.path.exists(extract_path):
            os.makedirs(extract_path)
        with tarfile.open(tgz_file, "r:gz") as tar:
            tar.extractall(path=extract_path)
            print(f"Extracted {tgz_file} to {extract_path}")
    else:
        print(f"The file {tgz_file} is not a valid .tgz or .tar.gz archive.")

# Function to preprocess images (resize, normalize, and save as .npy)
def preprocess_images(image_dir, output_file, img_size=(128, 32)):
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]
    images = []

    for image_file in image_files:
        img_path = os.path.join(image_dir, image_file)
        img = Image.open(img_path).convert('L')  # Convert to grayscale
        img = img.resize(img_size)               # Resize to a fixed size
        img_array = np.array(img) / 255.0        # Normalize pixel values
        images.append(img_array)

    # Save as a NumPy array
    images = np.array(images)
    np.save(output_file, images)
    print(f"Preprocessed images saved to {output_file}")

# Function to parse XML metadata (for bounding boxes, word coordinates, etc.)
def parse_xml(xml_dir):
    """
    Parses XML metadata for transcriptions and bounding boxes.

    Parameters:
    - xml_dir: The directory containing the extracted XML files.

    Returns:
    - metadata: A list of dictionaries containing bounding box info and transcriptions.
    """
    metadata = []
    
    for xml_file in os.listdir(xml_dir):
        if xml_file.endswith('.xml'):
            file_path = os.path.join(xml_dir, xml_file)
            tree = ET.parse(file_path)
            root = tree.getroot()

            for form in root.findall('form'):
                form_id = form.get('id')
                for line in form.findall('line'):
                    line_id = line.get('id')
                    for word in line.findall('word'):
                        word_text = word.get('text')
                        bounding_box = word.get('bounding_box')
                        metadata.append({
                            'form_id': form_id,
                            'line_id': line_id,
                            'word_text': word_text,
                            'bounding_box': bounding_box
                        })
    return metadata

# Function to preprocess all ASCII files and save the encoded transcriptions
def preprocess_ascii(ascii_dir, output_file):
    """
    Preprocesses all ASCII files in a directory and saves the encoded transcriptions.

    Parameters:
    - ascii_dir: The directory containing the extracted ASCII files.
    - output_file: Path to save the preprocessed transcriptions as a NumPy file.
    """
    transcriptions = []

    for ascii_file in os.listdir(ascii_dir):
        if ascii_file.endswith('.txt'):  # Modify this according to the actual file extensions
            file_path = os.path.join(ascii_dir, ascii_file)
            with open(file_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    if "FORM:" in line:
                        transcription = line.split(":")[-1].strip()
                        transcriptions.append(transcription)

    # Encode transcriptions into numerical format
    char_to_int = {char: idx for idx, char in enumerate('abcdefghijklmnopqrstuvwxyz ')}
    encoded_transcriptions = [[char_to_int.get(char, 0) for char in transcription] for transcription in transcriptions]
    
    # Save the preprocessed transcriptions as a NumPy array
    np.save(output_file, np.array(encoded_transcriptions))
    print(f"Preprocessed ASCII data saved to {output_file}")

#  extracting and preprocessing IAM dataset, including XML
tgz_files = {
    'forms': 'data/I_am_data/formsI-Z.tgz',
    'words': 'data/I_am_data/words.tgz',
    'sentences': 'data/I_am_data/sentences.tgz',
    'ascii': 'data/I_am_data/ascii.tgz',
    'xml': 'data/I_am_data/xml.tgz'  # Added XML archive for metadata
}

# Extract and preprocess images, ASCII files, and XML metadata
for dataset_type, tgz_file in tgz_files.items():
    extract_path = f'data/I_am_data/{dataset_type}/'
    extract_tgz(tgz_file, extract_path)
    
    # Preprocess extracted images and save them as .npy (skip ASCII and XML here)
    if dataset_type == 'ascii':
        preprocess_ascii(extract_path, f'data/I_am_data/preprocessed_{dataset_type}.npy')
    elif dataset_type == 'xml':
        metadata = parse_xml(extract_path)
        print(f"Parsed XML metadata: {metadata[:2]}")  # Example of printing parsed metadata
    else:
        preprocess_images(extract_path, f'data/I_am_data/preprocessed_{dataset_type}.npy')
