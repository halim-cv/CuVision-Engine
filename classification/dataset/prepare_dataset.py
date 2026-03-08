import os
import glob
import numpy as np
from PIL import Image
import requests
import tarfile

# Dataset Configuration
# Using Oxford 17 Flowers as a source for 10 classes
DATASET_URL = "https://www.robots.ox.ac.uk/~vgg/data/flowers/17/17flowers.tgz"
DATASET_FILE = "17flowers.tgz"
EXTRACT_DIR = "flower_data"
OUTPUT_FILE = "flowers10.bin"

def download_data():
    if not os.path.exists(DATASET_FILE):
        print(f"Downloading dataset from {DATASET_URL}...")
        try:
            r = requests.get(DATASET_URL, stream=True)
            r.raise_for_status()
            with open(DATASET_FILE, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("Download complete.")
        except Exception as e:
            print(f"Error downloading: {e}")
            return False
    
    if not os.path.exists(EXTRACT_DIR):
        print("Extracting dataset...")
        try:
            with tarfile.open(DATASET_FILE, "r:gz") as tar:
                tar.extractall(path=EXTRACT_DIR)
            print("Extraction complete.")
        except Exception as e:
            print(f"Error extracting: {e}")
            return False
    return True

def process_data(num_classes=10, images_per_class=80):
    img_dir = os.path.join(EXTRACT_DIR, "jpg")
    if not os.path.exists(img_dir):
        print(f"Error: Image directory {img_dir} not found.")
        return

    img_files = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
    
    if len(img_files) < num_classes * images_per_class:
        print("Error: Not enough images in dataset.")
        return

    print(f"Processing {num_classes} classes, {images_per_class} images each...")
    
    # Format: [Label (1 byte)][R (1024)][G (1024)][B (1024)]
    with open(OUTPUT_FILE, 'wb') as f:
        # Write total number of images as first 4 bytes (int)
        total_images = num_classes * images_per_class
        f.write(np.array([total_images], dtype=np.int32).tobytes())

        for class_id in range(num_classes):
            start_idx = class_id * images_per_class
            for i in range(images_per_class):
                img_path = img_files[start_idx + i]
                try:
                    img = Image.open(img_path).convert('RGB').resize((32, 32))
                    img_data = np.array(img).transpose(2, 0, 1) # HWC to CHW
                    
                    # Write Label
                    f.write(np.array([class_id], dtype=np.uint8).tobytes())
                    # Write Pixels
                    f.write(img_data.tobytes())
                except Exception as e:
                    print(f"Skipping {img_path} due to error: {e}")
                
                # Progress indicator
                processed = class_id * images_per_class + i + 1
                if processed % 100 == 0 or processed == total_images:
                    print(f"\rProgress: {processed}/{total_images} images processed...", end="", flush=True)
        print()

    print(f"Successfully created {OUTPUT_FILE} with {total_images} images.")

if __name__ == "__main__":
    if download_data():
        process_data()
    else:
        print("Dataset preparation aborted.")
