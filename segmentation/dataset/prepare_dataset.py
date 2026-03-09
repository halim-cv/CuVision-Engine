# ============================================================
#  CuVision-Engine | Segmentation
#  prepare_dataset.py
#
#  Dataset:  Oxford-IIIT Pet Dataset  (37 breeds, ~7,390 images)
#  Source:   https://www.robots.ox.ac.uk/~vgg/data/pets/
#
#  Segmentation task:
#    3-class pixel labelling per image
#      0 → background / border
#      1 → pet foreground
#      2 → pet silhouette boundary  (optional; merged into 1 if NUM_CLASSES=2)
#
#  Output binary format  (seg_pets.bin):
#  ┌────────────────────────────────────────────────────────┐
#  │ Header: [num_images (int32)] [IMG_H (int32)] [IMG_W (int32)]
#  │         [num_classes (int32)]
#  │ Per image record:
#  │   Image pixels : C × H × W  uint8  (RGB, CHW)
#  │   Mask pixels  :     H × W  uint8  (class index per pixel)
#  └────────────────────────────────────────────────────────┘
#
#  Usage:   python prepare_dataset.py
#  Deps:    pip install requests numpy pillow
# ============================================================

import os
import struct
import tarfile

import numpy as np
import requests
from PIL import Image

# -------------------------------------------------------
#  Configuration
# -------------------------------------------------------
IMG_H, IMG_W  = 256, 256
NUM_CLASSES   = 3          # background=0, pet=1, boundary=2
OUTPUT_FILE   = "seg_pets.bin"

# Oxford-IIIT Pet: images + annotations (trimaps) — total ≈ 800 MB
IMAGES_URL  = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"
ANNOTS_URL  = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz"
IMAGES_TAR  = "pet_images.tar.gz"
ANNOTS_TAR  = "pet_annotations.tar.gz"
IMAGES_DIR  = "images"
ANNOTS_DIR  = "annotations"


# -------------------------------------------------------
#  Step 1: Download + Extract
# -------------------------------------------------------
def _download(url: str, dest: str):
    if os.path.exists(dest):
        print(f"  Already downloaded: {dest}")
        return True
    print(f"  Downloading {os.path.basename(url)}...")
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            done  = 0
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=65536):
                    f.write(chunk)
                    done += len(chunk)
                    if total:
                        print(f"\r    {done/total*100:.1f}%  ({done//(1<<20)} / {total//(1<<20)} MB)",
                              end="", flush=True)
        print()
        return True
    except Exception as e:
        print(f"\n  [ERROR] {e}")
        return False


def _extract(tar_path: str, check_dir: str):
    if os.path.exists(check_dir):
        print(f"  Already extracted: {check_dir}/")
        return True
    print(f"  Extracting {tar_path}...")
    try:
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall()
        return True
    except Exception as e:
        print(f"  [ERROR] {e}")
        return False


def download_dataset():
    print("-- Images --")
    if not _download(IMAGES_URL, IMAGES_TAR): return False
    if not _extract(IMAGES_TAR, IMAGES_DIR):  return False
    print("-- Annotations --")
    if not _download(ANNOTS_URL, ANNOTS_TAR): return False
    if not _extract(ANNOTS_TAR, ANNOTS_DIR):  return False
    return True


# -------------------------------------------------------
#  Step 2: Trimap → integer mask
#
#  Oxford Pet trimap pixel values:
#    1 → foreground (pet body)
#    2 → background
#    3 → boundary / not classified
#
#  We remap to our convention:
#    0 → background  (VOC:2)
#    1 → pet         (VOC:1)
#    2 → boundary    (VOC:3)
# -------------------------------------------------------
TRIMAP_REMAP = {1: 1, 2: 0, 3: 2}   # trimap pixel → our class index

def trimap_to_mask(trimap_path: str) -> np.ndarray:
    """Load a .png trimap and return uint8 class-index mask (H, W)."""
    tri = np.array(Image.open(trimap_path).convert("L"))
    mask = np.zeros_like(tri, dtype=np.uint8)
    for src, dst in TRIMAP_REMAP.items():
        mask[tri == src] = dst
    return mask


# -------------------------------------------------------
#  Step 3: Build binary dataset
# -------------------------------------------------------
def build_binary():
    trimap_dir = os.path.join(ANNOTS_DIR, "trimaps")
    if not os.path.exists(trimap_dir):
        print(f"[ERROR] Trimap directory not found: {trimap_dir}")
        return

    trimap_paths = sorted([
        os.path.join(trimap_dir, f)
        for f in os.listdir(trimap_dir)
        if f.lower().endswith(".png")
    ])
    print(f"Found {len(trimap_paths)} trimap annotations.")

    records = []
    for tri_path in trimap_paths:
        # Image has same stem, stored in IMAGES_DIR/<breed>/<stem>.jpg
        stem = os.path.splitext(os.path.basename(tri_path))[0]
        # pets images are flat in images/ directory
        img_path = os.path.join(IMAGES_DIR, stem + ".jpg")
        if not os.path.exists(img_path):
            # Try PNG fallback
            img_path_png = os.path.join(IMAGES_DIR, stem + ".png")
            if os.path.exists(img_path_png):
                img_path = img_path_png
            else:
                continue

        try:
            img  = Image.open(img_path).convert("RGB").resize(
                       (IMG_W, IMG_H), Image.BILINEAR)
            tri  = Image.open(tri_path).resize(
                       (IMG_W, IMG_H), Image.NEAREST)   # nearest for labels
            mask = trimap_to_mask(tri_path)
            mask = np.array(
                Image.fromarray(mask).resize((IMG_W, IMG_H), Image.NEAREST)
            )
            records.append((img, mask))
        except Exception as e:
            print(f"  [WARN] Skipping {stem}: {e}")

    num_images = len(records)
    print(f"Writing {num_images} valid image-mask pairs → {OUTPUT_FILE}")

    with open(OUTPUT_FILE, "wb") as f:
        # ---- Header ----
        f.write(struct.pack("iiii", num_images, IMG_H, IMG_W, NUM_CLASSES))

        for idx, (img, mask) in enumerate(records):
            # ---- Image pixels: uint8 CHW ----
            arr = np.array(img, dtype=np.uint8).transpose(2, 0, 1)  # CHW
            f.write(arr.tobytes())

            # ---- Mask pixels: uint8 HW ----
            f.write(mask.astype(np.uint8).tobytes())

            if (idx + 1) % 500 == 0 or (idx + 1) == num_images:
                print(f"\r  {idx+1}/{num_images} pairs written...", end="", flush=True)

    print(f"\nDone!  Saved → {OUTPUT_FILE}  ({os.path.getsize(OUTPUT_FILE)//(1<<20)} MB)")

    # Mask class distribution
    all_masks = np.array([mask for _, mask in records], dtype=np.uint8)
    total_px  = all_masks.size
    print(f"\nMask class distribution:")
    labels    = ["background (0)", "pet body  (1)", "boundary  (2)"]
    for c in range(NUM_CLASSES):
        cnt = int((all_masks == c).sum())
        pct = cnt / total_px * 100
        print(f"  {labels[c]}: {cnt:>10,} px  ({pct:.1f}%)")


# -------------------------------------------------------
#  Entry point
# -------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("  CuVision-Engine | Segmentation Dataset Preparation")
    print("  Dataset: Oxford-IIIT Pet  (37 breeds, ~7.4k images)")
    print(f"  Output : {IMG_H}×{IMG_W} images + trimap masks → {OUTPUT_FILE}")
    print("=" * 60)

    if download_dataset():
        build_binary()
    else:
        print("[ABORT] Dataset preparation failed.")
