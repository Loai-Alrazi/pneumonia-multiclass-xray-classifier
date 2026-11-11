from pathlib import Path
import re
from collections import defaultdict

data_dir = Path(r"D:\PAEP_AI_Program\05_Deep_Learning\Projects_and_Homeworks\Pneumonia_Chest_XRay_Kit\data\chest_xray_3class")

normal_splits = defaultdict(lambda: defaultdict(list))

for split in ["train", "val", "test"]:
    folder = data_dir / split / "NORMAL"
    for f in folder.glob("*.*"):
        stem = f.stem
        # In NORMAL, filenames are like IM-0115-0001 or NORMAL2-IM-0865-0001
        # Let's extract patient ID:
        # e.g. IM-0115 or NORMAL2-IM-0865
        parts = stem.split("-")
        if len(parts) >= 2:
            pid = "-".join(parts[:-1]) # patient is everything except the image sequence number
        else:
            pid = stem
        normal_splits[pid][split].append(f.name)

cross_split_normal = {pid: splits for pid, splits in normal_splits.items() if len(splits) > 1}
print(f"Total unique Normal patient prefixes: {len(normal_splits)}")
print(f"Number of Normal patient prefixes in MULTIPLE splits: {len(cross_split_normal)}")
if cross_split_normal:
    for pid, splits in list(cross_split_normal.items())[:10]:
        print(f"  {pid}: {splits}")
