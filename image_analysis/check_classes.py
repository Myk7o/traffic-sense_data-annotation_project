import os
import json

GTFINE_PATH = "path"

TARGET = ["car", "traffic sign", "person", "vegetation", "traffic light"]

counts = {cls: 0 for cls in TARGET}
none_of_five = 0
total = 0

for split in ["train", "val"]:
    split_path = os.path.join(GTFINE_PATH, split)
    if not os.path.exists(split_path):
        continue

    for city in os.listdir(split_path):
        city_path = os.path.join(split_path, city)
        if not os.path.isdir(city_path):
            continue

        for filename in os.listdir(city_path):
            if not filename.endswith("_polygons.json"):
                continue

            total += 1
            with open(os.path.join(city_path, filename)) as f:
                data = json.load(f)

            labels = {obj.get("label") for obj in data.get("objects", [])}
            for cls in TARGET:
                if cls in labels:
                    counts[cls] += 1
            if not any(cls in labels for cls in TARGET):
                none_of_five += 1

print(f"\nTotal images: {total}\n")
print(f"{'Class':<20} {'With':>6}  {'%':>6}    {'Without':>8}  {'%':>6}")
print("-" * 56)
for cls in TARGET:
    w = counts[cls]
    wo = total - w
    print(f"{cls:<20} {w:>6}  {100*w/total:>5.1f}%    {wo:>8}  {100*wo/total:>5.1f}%")
print(f"\nImages with none of the 5 classes: {none_of_five}  ({100*none_of_five/total:.1f}%)")
