# ============================================================
#  CuVision-Engine | Object Detection
#  prepare_dataset.py
#
#  Dataset:  Pascal VOC 2007  (trainval, ~5,000 images)
#  Source:   http://host.robots.ox.ac.uk/pascal/VOC/
#  Mirror:   pjreddie.com (reliable for small downloads)
#
#  Output binary format  (od_voc2007.bin):
#  ┌─────────────────────────────────────────────────────┐
#  │ Header: [num_images (int32)] [IMG_H (int32)] [IMG_W (int32)]
#  │         [num_classes (int32)]
#  │ Per image record:
#  │   [num_boxes (int32)]
#  │   [num_boxes × 5 floats: cls x1 y1 x2 y2 (normalised)]
#  │   [C × H × W uint8 pixels  (RGB, CHW layout)]
#  └─────────────────────────────────────────────────────┘
#
#  Usage:   python prepare_dataset.py
#  Deps:    pip install requests numpy pillow
# ============================================================

import os
import glob
import struct
import tarfile
import xml.etree.ElementTree as ET

import numpy as np
import requests
from PIL import Image

# -------------------------------------------------------
#  Configuration
# -------------------------------------------------------
IMG_H, IMG_W = 300, 300          # resize target
MAX_BOXES    = 30                 # pad / cap per image
OUTPUT_FILE  = "od_voc2007.bin"

# VOC 2007 trainval tar (≈ 439 MB)
VOC_URL  = "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar"
VOC_TAR  = "VOCtrainval_06-Nov-2007.tar"
VOC_ROOT = "VOCdevkit"

VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]
CLASS_TO_IDX = {c: i for i, c in enumerate(VOC_CLASSES)}
NUM_CLASSES  = len(VOC_CLASSES)


# -------------------------------------------------------
#  Step 1: Download + Extract
# -------------------------------------------------------
def download_dataset():
    if not os.path.exists(VOC_TAR):
        print(f"Downloading Pascal VOC 2007 trainval (~439 MB)...")
        try:
            with requests.get(VOC_URL, stream=True) as r:
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0))
                downloaded = 0
                with open(VOC_TAR, "wb") as f:
                    for chunk in r.iter_content(chunk_size=65536):
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total:
                            pct = downloaded / total * 100
                            print(f"\r  {pct:.1f}%  ({downloaded//(1<<20)} MB / {total//(1<<20)} MB)",
                                  end="", flush=True)
            print("\nDownload complete.")
        except Exception as e:
            print(f"\n[ERROR] Download failed: {e}")
            return False
    else:
        print(f"Archive already present: {VOC_TAR}")

    if not os.path.exists(VOC_ROOT):
        print("Extracting archive...")
        try:
            with tarfile.open(VOC_TAR, "r") as tar:
                tar.extractall()
            print("Extraction complete.")
        except Exception as e:
            print(f"[ERROR] Extraction failed: {e}")
            return False
    else:
        print(f"Already extracted: {VOC_ROOT}/")

    return True


# -------------------------------------------------------
#  Step 2: Parse VOC XML annotation
# -------------------------------------------------------
def parse_annotation(xml_path, img_w, img_h):
    """
    Returns list of [class_idx, x1_norm, y1_norm, x2_norm, y2_norm]
    Coordinates are normalised to [0, 1].
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    boxes = []
    for obj in root.findall("object"):
        cls_name = obj.find("name").text.strip().lower()
        if cls_name not in CLASS_TO_IDX:
            continue
        diff = obj.find("difficult")
        if diff is not None and int(diff.text) == 1:
            continue          # skip difficult objects
        bnd = obj.find("bndbox")
        x1 = max(0.0, float(bnd.find("xmin").text)) / img_w
        y1 = max(0.0, float(bnd.find("ymin").text)) / img_h
        x2 = min(1.0, float(bnd.find("xmax").text)) / img_w
        y2 = min(1.0, float(bnd.find("ymax").text)) / img_h
        if x2 > x1 and y2 > y1:
            boxes.append([CLASS_TO_IDX[cls_name], x1, y1, x2, y2])
    return boxes[:MAX_BOXES]


# -------------------------------------------------------
#  Step 3: Build binary dataset
# -------------------------------------------------------
def build_binary():
    voc_path  = os.path.join(VOC_ROOT, "VOC2007")
    img_dir   = os.path.join(voc_path, "JPEGImages")
    ann_dir   = os.path.join(voc_path, "Annotations")
    split_txt = os.path.join(voc_path, "ImageSets", "Main", "trainval.txt")

    if not os.path.exists(split_txt):
        print(f"[ERROR] Split file not found: {split_txt}")
        return

    with open(split_txt) as f:
        ids = [l.strip() for l in f if l.strip()]

    print(f"Found {len(ids)} trainval images.")

    # First pass: collect valid records
    records = []
    for img_id in ids:
        img_path = os.path.join(img_dir, img_id + ".jpg")
        xml_path = os.path.join(ann_dir, img_id + ".xml")
        if not os.path.exists(img_path) or not os.path.exists(xml_path):
            continue
        try:
            img = Image.open(img_path).convert("RGB")
            orig_w, orig_h = img.size
            boxes = parse_annotation(xml_path, orig_w, orig_h)
            if len(boxes) == 0:
                continue          # skip background-only images
            img_resized = img.resize((IMG_W, IMG_H), Image.BILINEAR)
            records.append((img_resized, boxes))
        except Exception as e:
            print(f"  [WARN] Skipping {img_id}: {e}")

    num_images = len(records)
    print(f"Writing {num_images} valid images → {OUTPUT_FILE}")

    with open(OUTPUT_FILE, "wb") as f:
        # ---- Header ----
        f.write(struct.pack("iiii", num_images, IMG_H, IMG_W, NUM_CLASSES))

        for idx, (img, boxes) in enumerate(records):
            # ---- Bounding boxes ----
            f.write(struct.pack("i", len(boxes)))
            for box in boxes:
                f.write(struct.pack("fffff",
                    float(box[0]),  # class index (stored as float for padding ease)
                    box[1], box[2], box[3], box[4]))  # x1 y1 x2 y2

            # ---- Image pixels: uint8 CHW ----
            arr = np.array(img, dtype=np.uint8)   # HWC
            arr = arr.transpose(2, 0, 1)           # CHW
            f.write(arr.tobytes())

            # Progress
            if (idx + 1) % 200 == 0 or (idx + 1) == num_images:
                print(f"\r  {idx+1}/{num_images} images written...", end="", flush=True)

    print(f"\nDone!  Saved → {OUTPUT_FILE}  ({os.path.getsize(OUTPUT_FILE)//(1<<20)} MB)")

    # Print class distribution summary
    all_cls = [b[0] for _, boxes in records for b in boxes]
    print(f"\nClass distribution ({NUM_CLASSES} classes):")
    for cls_idx, cls_name in enumerate(VOC_CLASSES):
        count = all_cls.count(cls_idx)
        print(f"  [{cls_idx:2d}] {cls_name:<15s}: {count} boxes")


# -------------------------------------------------------
#  Entry point
# -------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("  CuVision-Engine | Object Detection Dataset Preparation")
    print("  Dataset: Pascal VOC 2007 trainval")
    print("=" * 60)

    if download_dataset():
        build_binary()
    else:
        print("[ABORT] Dataset preparation failed.")
