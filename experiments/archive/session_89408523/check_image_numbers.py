from pathlib import Path
import re
from collections import Counter

data_dir = Path(r"D:\PAEP_AI_Program\05_Deep_Learning\Projects_and_Homeworks\Pneumonia_Chest_XRay_Kit\data\chest_xray_3class")

img_nums_bact = []
img_nums_viral = []

for split in ["train", "val", "test"]:
    for f in (data_dir / split / "BACTERIAL").glob("*.*"):
        m = re.match(r"person\d+_bacteria_(\d+)", f.name)
        if m:
            img_nums_bact.append(int(m.group(1)))
    for f in (data_dir / split / "VIRAL").glob("*.*"):
        m = re.match(r"person\d+_virus_(\d+)", f.name)
        if m:
            img_nums_viral.append(int(m.group(1)))

print(f"Bacterial image numbers: min={min(img_nums_bact)}, max={max(img_nums_bact)}, count={len(img_nums_bact)}, unique={len(set(img_nums_bact))}")
print(f"Viral image numbers:     min={min(img_nums_viral)}, max={max(img_nums_viral)}, count={len(img_nums_viral)}, unique={len(set(img_nums_viral))}")

# Check if any image number overlaps between bacterial and viral
overlap_img_nums = set(img_nums_bact).intersection(set(img_nums_viral))
print(f"Overlap between bacterial image numbers and viral image numbers: {len(overlap_img_nums)}")
if overlap_img_nums:
    print(f"Sample overlapping image numbers: {sorted(list(overlap_img_nums))[:10]}")
