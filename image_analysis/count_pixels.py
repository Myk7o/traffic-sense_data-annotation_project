import os
import numpy as np
from PIL import Image

MASKS_ROOT = "/Volumes/PortableSSD/2026 Summer project /5classes_dataset"

CLASS_NAMES = {
    0: "background",
    1: "car",
    2: "person",
    3: "vegetation",
    4: "traffic sign",
    5: "traffic light",
}

pixel_counts = np.zeros(6, dtype=np.int64)

for split in ["train", "val"]:
    mask_dir = os.path.join(MASKS_ROOT, split, "masks")
    for fname in os.listdir(mask_dir):
        if fname.startswith(".") or not fname.endswith("_mask.png") or fname.endswith("_mask_viz.png"):
            continue
        mask = np.array(Image.open(os.path.join(mask_dir, fname)), dtype=np.uint8)
        for cls_id in range(6):
            pixel_counts[cls_id] += np.sum(mask == cls_id)

total = pixel_counts.sum()

print(f"{'Class':<20} {'Pixels':>15}  {'%':>7}")
print("-" * 46)
for cls_id, name in CLASS_NAMES.items():
    px = pixel_counts[cls_id]
    print(f"{cls_id} {name:<18} {px:>15,}  {100*px/total:>6.2f}%")
print("-" * 46)
print(f"{'TOTAL':<20} {total:>15,}  100.00%")
