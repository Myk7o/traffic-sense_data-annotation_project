import os
import json
import numpy as np
from PIL import Image

GTFINE_PATH = "/Users/mykola/Downloads/gtFine_trainvaltest/gtFine"

# JSON label names grouped with their variants -> display name
CLASS_GROUPS = {
    "person":     {"person", "persongroup"},
    "rider":      {"rider", "ridergroup"},
    "car":        {"car", "cargroup"},
    "truck":      {"truck", "truckgroup"},
    "bus":        {"bus"},
    "train":      {"train"},
    "motorcycle": {"motorcycle", "motorcyclegroup"},
    "bicycle":    {"bicycle", "bicyclegroup"},
    "caravan":    {"caravan"},
    "trailer":    {"trailer"},
    "vegetation":     {"vegetation"},
    "traffic sign":   {"traffic sign"},
    "traffic light":  {"traffic light"},
}

# Official Cityscapes labelIds for pixel counting
LABEL_IDS = {
    "person":        {24},
    "rider":         {25},
    "car":           {26},
    "truck":         {27},
    "bus":           {28},
    "train":         {31},
    "motorcycle":    {32},
    "bicycle":       {33},
    "caravan":       {29},
    "trailer":       {30},
    "vegetation":    {21},
    "traffic sign":  {20},
    "traffic light": {19},
}

# Table 1: image counts from polygon JSONs
image_counts = {cls: 0 for cls in CLASS_GROUPS}
total_images = 0

for split in ["train", "val"]:
    split_path = os.path.join(GTFINE_PATH, split)
    if not os.path.exists(split_path):
        continue
    for city in os.listdir(split_path):
        city_path = os.path.join(split_path, city)
        if not os.path.isdir(city_path):
            continue
        for fname in os.listdir(city_path):
            if not fname.endswith("_gtFine_polygons.json"):
                continue
            total_images += 1
            with open(os.path.join(city_path, fname)) as f:
                data = json.load(f)
            labels = {obj.get("label", "") for obj in data.get("objects", [])}
            for cls, variants in CLASS_GROUPS.items():
                if labels & variants:
                    image_counts[cls] += 1

print(f"\n── Table 1: Images containing each class (out of {total_images} total) ──")
print(f"{'Class':<15} {'Images':>8}  {'%':>7}  {'Without':>9}  {'%':>7}")
print("─" * 52)
for cls in CLASS_GROUPS:
    w  = image_counts[cls]
    wo = total_images - w
    print(f"{cls:<15} {w:>8,}  {100*w/total_images:>6.1f}%  {wo:>9,}  {100*wo/total_images:>6.1f}%")

# Table 2: pixel counts from original labelIds masks
pixel_counts = {cls: 0 for cls in LABEL_IDS}
total_pixels = 0

for split in ["train", "val"]:
    split_path = os.path.join(GTFINE_PATH, split)
    if not os.path.exists(split_path):
        continue
    for city in os.listdir(split_path):
        city_path = os.path.join(split_path, city)
        if not os.path.isdir(city_path):
            continue
        for fname in os.listdir(city_path):
            if not fname.endswith("_gtFine_labelIds.png"):
                continue
            mask = np.array(Image.open(os.path.join(city_path, fname)), dtype=np.uint8)
            total_pixels += mask.size
            for cls, ids in LABEL_IDS.items():
                for lid in ids:
                    pixel_counts[cls] += int(np.sum(mask == lid))

print(f"\n── Table 2: Pixel counts per class across all {total_images} masks ──")
print(f"{'Class':<15} {'Pixels':>16}  {'% of total':>11}")
print("─" * 46)
for cls in LABEL_IDS:
    px = pixel_counts[cls]
    print(f"{cls:<15} {px:>16,}  {100*px/total_pixels:>10.3f}%")
print("─" * 46)
print(f"{'TOTAL PIXELS':<15} {total_pixels:>16,}  {'100.000%':>11}")
