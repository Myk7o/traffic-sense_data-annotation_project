import os
import json
import shutil
import numpy as np
from PIL import Image
from collections import defaultdict

GTFINE_PATH  = "/Users/mykola/Downloads/gtFine_trainvaltest/gtFine"
LEFTIMG_PATH = "/Users/mykola/Downloads/leftImg8bit_trainvaltest/leftImg8bit"
OUTPUT_PATH  = "/Volumes/PortableSSD/2026 Summer project /cityscapes_3class"

# background=0, car=1, person=2, cyclist=3
# cyclist = bicycle(33) + motorcycle(32) + rider(25)
LUT = np.zeros(256, dtype=np.uint8)
LUT[26] = 1  # car
LUT[24] = 2  # person
LUT[25] = 3  # rider     -> cyclist
LUT[32] = 3  # motorcycle -> cyclist
LUT[33] = 3  # bicycle   -> cyclist

COLOR_PALETTE = np.array([
    [  0,   0,   0],   # 0 background — black
    [  0,   0, 142],   # 1 car        — blue
    [255, 182, 193],   # 2 person     — pink
    [255, 255,   0],   # 3 cyclist    — yellow
], dtype=np.uint8)

CLASS_NAMES = {0: "background", 1: "car", 2: "person", 3: "cyclist"}

split_counts  = defaultdict(int)
pixel_counts  = defaultdict(int)
skipped       = 0

for split in ["train", "val"]:
    gt_split  = os.path.join(GTFINE_PATH,  split)
    img_split = os.path.join(LEFTIMG_PATH, split)
    if not os.path.exists(gt_split):
        print(f"Skipping {split} — not found")
        continue

    for city in sorted(os.listdir(gt_split)):
        city_gt  = os.path.join(gt_split,  city)
        city_img = os.path.join(img_split, city)
        if not os.path.isdir(city_gt):
            continue

        for fname in os.listdir(city_gt):
            if not fname.endswith("_gtFine_labelIds.png"):
                continue

            base = fname.replace("_gtFine_labelIds.png", "")

            # remap mask
            mask   = np.array(Image.open(os.path.join(city_gt, fname)), dtype=np.uint8)
            remapped = LUT[mask]

            # skip if no target class pixel present
            if not np.any(remapped > 0):
                skipped += 1
                continue

            # output dirs
            out_img  = os.path.join(OUTPUT_PATH, split, "images",        city)
            out_mask = os.path.join(OUTPUT_PATH, split, "masks",         city)
            out_col  = os.path.join(OUTPUT_PATH, split, "colored_masks", city)
            os.makedirs(out_img,  exist_ok=True)
            os.makedirs(out_mask, exist_ok=True)
            os.makedirs(out_col,  exist_ok=True)

            # save grayscale mask
            Image.fromarray(remapped).save(os.path.join(out_mask, f"{base}_mask.png"))

            # save colorized mask
            Image.fromarray(COLOR_PALETTE[remapped]).save(os.path.join(out_col, f"{base}_colored.png"))

            # copy original image
            src_img = os.path.join(city_img, f"{base}_leftImg8bit.png")
            if os.path.exists(src_img):
                shutil.copy2(src_img, os.path.join(out_img, f"{base}_leftImg8bit.png"))

            split_counts[split] += 1
            for cls_id in range(4):
                pixel_counts[cls_id] += int(np.sum(remapped == cls_id))

    print(f"  {split}: {split_counts[split]} images saved  ({skipped} skipped so far)")

# summary
total       = sum(split_counts.values())
total_pixels = sum(pixel_counts.values())

print(f"\n{'─'*52}")
print(f"  Total images:  {total}  (train={split_counts['train']}, val={split_counts['val']})")
print(f"  Skipped:       {skipped}  (no car/person/cyclist pixels)")
print(f"\n  {'Class':<15} {'Pixels':>16}  {'%':>8}")
print(f"  {'─'*42}")
for cls_id, name in CLASS_NAMES.items():
    px = pixel_counts[cls_id]
    print(f"  {cls_id} {name:<13} {px:>16,}  {100*px/total_pixels:>7.3f}%")
print(f"  {'─'*42}")
print(f"  {'TOTAL':<15} {total_pixels:>16,}  100.000%")
print(f"\n  Output: {OUTPUT_PATH}")
