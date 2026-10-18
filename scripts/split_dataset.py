import os
import random
import shutil
from pathlib import Path

DATASET_DIR = Path("../dataset") 
IMAGE_DIR = DATASET_DIR / "images"
LABEL_DIR = DATASET_DIR / "txts (YOLO)" 
OUTPUT_IMAGE_DIR = IMAGE_DIR
OUTPUT_LABEL_DIR = LABEL_DIR 

# Percentage of data to use for validation
VAL_SPLIT_PERCENT = 0.20 # 20%


print("Creating train/val directories...")
for split in ["train", "val"]:
    (OUTPUT_IMAGE_DIR / split).mkdir(parents=True, exist_ok=True)
    (OUTPUT_LABEL_DIR / split).mkdir(parents=True, exist_ok=True)


print("Getting list of image files...")
image_files = sorted([f for f in IMAGE_DIR.glob("*.jpg")]) 
if not image_files:
    print(f"Error: No .jpg images found in {IMAGE_DIR}")
    exit(1)

print(f"Found {len(image_files)} images.")

# Shuffle and Split 
random.seed(42) # for reproducible splits
random.shuffle(image_files)

split_index = int(len(image_files) * (1 - VAL_SPLIT_PERCENT))
train_files = image_files[:split_index]
val_files = image_files[split_index:]

print(f"Splitting into {len(train_files)} training and {len(val_files)} validation images.")


def move_files(file_list, split_name):
    print(f"Moving {split_name} files...")
    moved_count = 0
    skipped_labels = 0
    for img_path in file_list:
        
        label_filename = img_path.stem + ".txt"
        label_path = LABEL_DIR / label_filename

        
        dest_img_path = OUTPUT_IMAGE_DIR / split_name / img_path.name
        dest_label_path = OUTPUT_LABEL_DIR / split_name / label_filename

        # Move image
        try:
             shutil.move(str(img_path), str(dest_img_path))
        except Exception as e:
            print(f"  Error moving image {img_path.name}: {e}")
            continue # Skip this image if error

        # Move corresponding label file if it exists
        if label_path.exists():
            try:
                shutil.move(str(label_path), str(dest_label_path))
                moved_count += 1
            except Exception as e:
                 print(f"  Error moving label {label_filename}: {e}")
        else:
            
            skipped_labels += 1 # Count missing labels silently

    print(f"Moved {moved_count} image/label pairs to '{split_name}'.")
    if skipped_labels > 0:
        print(f"  (Skipped {skipped_labels} missing label files for {split_name})")



move_files(train_files, "train")
move_files(val_files, "val")

print("\ Dataset splitting complete!")
