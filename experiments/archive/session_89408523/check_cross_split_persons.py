from pathlib import Path
import re
from collections import defaultdict

data_dir = Path(r"D:\PAEP_AI_Program\05_Deep_Learning\Projects_and_Homeworks\Pneumonia_Chest_XRay_Kit\data\chest_xray_3class")

person_splits = defaultdict(lambda: defaultdict(list))

for split in ["train", "val", "test"]:
    for cls in ["BACTERIAL", "VIRAL"]:
        folder = data_dir / split / cls
        for f in folder.glob("*.*"):
            m = re.match(r"(person\d+)_([a-z]+)_(\d+)", f.name)
            if m:
                pid, subtype, img_id = m.groups()
                person_splits[pid][split].append((cls, f.name))

# Check for cross-split leakage of person IDs!
cross_split_persons = {pid: splits for pid, splits in person_splits.items() if len(splits) > 1}
print(f"Total person IDs in Pneumonia: {len(person_splits)}")
print(f"Number of person IDs that appear in MULTIPLE splits: {len(cross_split_persons)}")

for pid, splits in list(cross_split_persons.items())[:20]:
    print(f"\n{pid} appears in splits: {list(splits.keys())}")
    for s, flist in splits.items():
        print(f"  {s}: {[x[1] for x in flist]}")
