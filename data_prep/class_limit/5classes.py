import os
import json
import shutil
import numpy as np
from PIL import Image
from collections import defaultdict

GTFINE_PATH   = "path"
LEFTIMG_PATH  = "path"
OUTPUT_PATH   = "path"

# Official Cityscapes labelId → new sequential ID
# background=0, car=1, person=2, vegetation=3, traffic sign=4, traffic light=5
LABEL_REMAP = {
    26: 1,  # car
    24: 2,  # person
    21: 3,  # vegetation
    20: 4,  # traffic sign
    19: 5,  # traffic light
}
TARGET_CLASSES = {"car", "person", "vegetation", "traffic sign", "traffic light"}
CLASS_NAMES = {0: "background", 1: "car", 2: "person", 3: "vegetation", 4: "traffic sign", 5: "traffic light"}

# Build uint8 lookup table: index = original labelId, value = new ID (default 0)
LUT = np.zeros(256, dtype=np.uint8)
for src, dst in LABEL_REMAP.items():
    LUT[src] = dst

# Official Cityscapes RGB colors indexed by new class ID
COLOR_PALETTE = np.array([
    [  0,   0,   0],   # 0 background — black
    [  0,   0, 142],   # 1 car        — blue
    [220,  20,  60],   # 2 person     — red
    [107, 142,  35],   # 3 vegetation — green
    [220, 220,   0],   # 4 traffic sign — yellow
    [250, 170,  30],   # 5 traffic light — orange
], dtype=np.uint8)

split_counts = defaultdict(int)
pixel_counts = defaultdict(int)

for split in ["train", "val"]:
    out_images = os.path.join(OUTPUT_PATH, split, "images")
    out_masks  = os.path.join(OUTPUT_PATH, split, "masks")
    os.makedirs(out_images, exist_ok=True)
    os.makedirs(out_masks,  exist_ok=True)

    split_path = os.path.join(GTFINE_PATH, split)
    if not os.path.exists(split_path):
        print(f"Skipping {split} — not found")
        continue

    for city in sorted(os.listdir(split_path)):
        city_path = os.path.join(split_path, city)
        if not os.path.isdir(city_path):
            continue

        for filename in os.listdir(city_path):
            if not filename.endswith("_gtFine_polygons.json"):
                continue

            # check class presence via polygon JSON
            with open(os.path.join(city_path, filename)) as f:
                data = json.load(f)
            labels = {obj.get("label") for obj in data.get("objects", [])}
            if not TARGET_CLASSES & labels:
                continue

            base = filename.replace("_gtFine_polygons.json", "")

            # remap mask using LUT
            mask_src = os.path.join(city_path, f"{base}_gtFine_labelIds.png")
            mask = np.array(Image.open(mask_src), dtype=np.uint8)
            remapped = LUT[mask]

            # store raw class IDs (0-5) for training
            Image.fromarray(remapped).save(os.path.join(out_masks, f"{base}_mask.png"))
            # save RGB color visualization using official Cityscapes colors
            Image.fromarray(COLOR_PALETTE[remapped]).save(os.path.join(out_masks, f"{base}_mask_viz.png"))

            img_src = os.path.join(LEFTIMG_PATH, split, city, f"{base}_leftImg8bit.png")
            if os.path.exists(img_src):
                shutil.copy2(img_src, os.path.join(out_images, f"{base}_leftImg8bit.png"))

            split_counts[split] += 1
            for cls_id in range(6):
                pixel_counts[cls_id] += int(np.sum(remapped == cls_id))

    print(f"  {split}: {split_counts[split]} images processed")

total = sum(split_counts.values())
print(f"\n=== Summary ===")
print(f"Total images saved: {total}  (train={split_counts['train']}, val={split_counts['val']})")
print(f"\nPixel counts per class:")
total_pixels = sum(pixel_counts.values())
for cls_id, name in CLASS_NAMES.items():
    px = pixel_counts[cls_id]
    print(f"  {cls_id} {name:<15} {px:>14,}  ({100*px/total_pixels:.2f}%)")
print(f"\nOutput: {OUTPUT_PATH}")
